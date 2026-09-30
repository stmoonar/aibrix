/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/go-redis/v9"
	"github.com/tidwall/gjson"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	"github.com/vllm-project/aibrix/pkg/types"
)

// TRE-PATCH(BL-GW-001): per-request event stream for the baseline autoscalers
// (TokenScale, PreServe). Opt-in; off by default.
//
// Stream: tre:v2:bl:req:<model>, XADD with automatic IDs and MAXLEN ~ N. The stream ID
// (Redis server clock at write time) is the authoritative timestamp; events are batched,
// so an ID can trail the event by up to one flush interval plus the Redis round trip.
// gw_local_ms (gateway wall clock) and dt_arr_ms (gateway monotonic clock) are
// informational. Every value is a string.
//
//	kind=arr   once the target pod is chosen: req_id reissue gw_local_ms pod in_tokens
//	           in_src(header|estimate|token_ids) max_tokens stream
//	kind=ft    first non-empty response body frame of a non-error response:
//	           req_id reissue gw_local_ms pod dt_arr_ms
//	kind=done  exactly once per request that had arr, when its ext_proc stream ends
//	           (completion, upstream error, abort, shutdown): req_id reissue gw_local_ms
//	           pod status out_tokens out_src(usage|none) dt_arr_ms
//
// The request path never blocks on Redis: events go through a bounded channel drained by
// one goroutine that writes them in pipelined batches; a full channel drops the event.

const (
	envBLReqEvents         = "TRE_BL_REQ_EVENTS"
	envBLReqEventsMaxLen   = "TRE_BL_REQ_EVENTS_MAXLEN"
	envBLReqEventsBuffer   = "TRE_BL_REQ_EVENTS_BUFFER"
	envBLCharsPerToken     = "TRE_BL_CHARS_PER_TOKEN"
	BLReqEventsKeyPrefix   = "tre:v2:bl:req:"
	HeaderBLInTokens       = "x-tre-bl-in-tokens"
	HeaderTREContinued     = "x-tre-continued"
	HeaderTRERetried       = "x-tre-retried"
	defaultBLReqMaxLen     = int64(200000)
	defaultBLReqBuffer     = 10000
	defaultBLCharsPerToken = 4.0
	blCharsPerTokenDefault = "*"

	blFlushInterval = 50 * time.Millisecond
	blFlushBatch    = 256
	blRedisTimeout  = 5 * time.Second
	blCloseTimeout  = 2 * time.Second
	blLogEvery      = 1000

	blKindArr  = "arr"
	blKindFT   = "ft"
	blKindDone = "done"
)

var (
	blReqEventsDropped = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tre_gateway_bl_req_events_dropped_total",
		Help: "Baseline request events not written (reason=buffer_full|redis_error).",
	}, []string{"reason"})
	blReqEventsWritten = promauto.NewCounter(prometheus.CounterOpts{
		Name: "tre_gateway_bl_req_events_written_total",
		Help: "Baseline request events written to Redis.",
	})
)

// ---------------------------------------------------------------------------------------
// configuration

type blReqConfig struct {
	maxLen        int64
	buffer        int
	charsPerToken map[string]float64 // per model; "*" is the default
	flushInterval time.Duration
	flushBatch    int
}

func blReqEventsEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(envBLReqEvents))) {
	case "true", "1":
		return true
	}
	return false
}

func loadBLReqConfig() blReqConfig {
	cfg := blReqConfig{
		maxLen:        defaultBLReqMaxLen,
		buffer:        defaultBLReqBuffer,
		charsPerToken: map[string]float64{blCharsPerTokenDefault: defaultBLCharsPerToken},
		flushInterval: blFlushInterval,
		flushBatch:    blFlushBatch,
	}
	if v := strings.TrimSpace(os.Getenv(envBLReqEventsMaxLen)); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			cfg.maxLen = n
		} else {
			klog.ErrorS(err, "ignoring invalid env; using default", "env", envBLReqEventsMaxLen, "value", v, "default", defaultBLReqMaxLen)
		}
	}
	if v := strings.TrimSpace(os.Getenv(envBLReqEventsBuffer)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.buffer = n
		} else {
			klog.ErrorS(err, "ignoring invalid env; using default", "env", envBLReqEventsBuffer, "value", v, "default", defaultBLReqBuffer)
		}
	}
	if v := strings.TrimSpace(os.Getenv(envBLCharsPerToken)); v != "" {
		cfg.charsPerToken = parseBLCharsPerToken(v)
	}
	return cfg
}

// parseBLCharsPerToken parses {"<model>": <chars per token>, "*": <default>}. Invalid JSON
// or non-positive values are ignored; the "*" default is always present.
func parseBLCharsPerToken(v string) map[string]float64 {
	out := map[string]float64{blCharsPerTokenDefault: defaultBLCharsPerToken}
	var m map[string]float64
	if err := json.Unmarshal([]byte(v), &m); err != nil {
		klog.ErrorS(err, "ignoring invalid env; using default chars/token", "env", envBLCharsPerToken, "default", defaultBLCharsPerToken)
		return out
	}
	for k, f := range m {
		if f > 0 && !math.IsInf(f, 0) && !math.IsNaN(f) {
			out[k] = f
		} else {
			klog.ErrorS(nil, "ignoring non-positive chars/token", "env", envBLCharsPerToken, "model", k, "value", f)
		}
	}
	return out
}

func (c blReqConfig) charsPerTokenFor(model string) float64 {
	if f, ok := c.charsPerToken[model]; ok {
		return f
	}
	return c.charsPerToken[blCharsPerTokenDefault]
}

// ---------------------------------------------------------------------------------------
// emitter: bounded channel + one batching writer goroutine

type blReqEvent struct {
	stream string
	values []string // field, value, field, value, ...
}

type blReqEmitter struct {
	client  redis.Cmdable
	cfg     blReqConfig
	ch      chan blReqEvent
	dropped atomic.Uint64 // buffer_full drops (also exported as a Prometheus counter)
	errs    atomic.Uint64 // failed batches

	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
}

// newBLReqEmitterFromEnv returns nil (feature off: no goroutine, no Redis writes) unless
// TRE_BL_REQ_EVENTS is "true"/"1" and a Redis client is configured.
func newBLReqEmitterFromEnv(client *redis.Client) *blReqEmitter {
	if !blReqEventsEnabled() {
		return nil
	}
	if client == nil {
		klog.ErrorS(nil, "baseline request events requested but no Redis client is configured; disabled", "env", envBLReqEvents)
		return nil
	}
	cfg := loadBLReqConfig()
	klog.InfoS("baseline request events enabled", "keyPrefix", BLReqEventsKeyPrefix, "maxLen", cfg.maxLen,
		"buffer", cfg.buffer, "charsPerToken", cfg.charsPerToken)
	return newBLReqEmitter(client, cfg)
}

func newBLReqEmitter(client redis.Cmdable, cfg blReqConfig) *blReqEmitter {
	e := &blReqEmitter{
		client: client,
		cfg:    cfg,
		ch:     make(chan blReqEvent, cfg.buffer),
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
	go e.run()
	return e
}

// enqueue never blocks: a full channel drops the event and counts it.
func (e *blReqEmitter) enqueue(ev blReqEvent) {
	select {
	case e.ch <- ev:
	default:
		n := e.dropped.Add(1)
		blReqEventsDropped.WithLabelValues("buffer_full").Inc()
		if n == 1 || n%blLogEvery == 0 {
			klog.ErrorS(nil, "baseline request event buffer full; dropping events", "droppedTotal", n, "buffer", cap(e.ch))
		}
	}
}

func (e *blReqEmitter) run() {
	defer close(e.done)
	batch := make([]blReqEvent, 0, e.cfg.flushBatch)
	var timer *time.Timer
	var timerC <-chan time.Time
	stopTimer := func() {
		if timer != nil {
			timer.Stop()
		}
		timer, timerC = nil, nil
	}
	flush := func() {
		stopTimer()
		if len(batch) > 0 {
			e.write(batch)
			batch = batch[:0]
		}
	}
	for {
		select {
		case ev := <-e.ch:
			batch = append(batch, ev)
			if len(batch) >= e.cfg.flushBatch {
				flush()
			} else if timerC == nil {
				timer = time.NewTimer(e.cfg.flushInterval)
				timerC = timer.C
			}
		case <-timerC:
			timer, timerC = nil, nil
			flush()
		case <-e.stop:
		drain: // write what is already buffered, without waiting for more
			for {
				select {
				case ev := <-e.ch:
					batch = append(batch, ev)
					if len(batch) >= e.cfg.flushBatch {
						flush()
					}
				default:
					break drain
				}
			}
			flush()
			return
		}
	}
}

func (e *blReqEmitter) write(batch []blReqEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), blRedisTimeout)
	defer cancel()
	_, err := e.client.Pipelined(ctx, func(p redis.Pipeliner) error {
		for i := range batch {
			p.XAdd(ctx, &redis.XAddArgs{
				Stream: batch[i].stream,
				MaxLen: e.cfg.maxLen,
				Approx: true,
				ID:     "*",
				Values: batch[i].values,
			})
		}
		return nil
	})
	if err != nil {
		n := e.errs.Add(1)
		blReqEventsDropped.WithLabelValues("redis_error").Add(float64(len(batch)))
		if n == 1 || n%100 == 0 {
			klog.ErrorS(err, "writing baseline request events to redis failed; batch dropped", "events", len(batch), "failedBatches", n)
		}
		return
	}
	blReqEventsWritten.Add(float64(len(batch)))
}

// close flushes what is buffered and stops the writer (bounded wait).
func (e *blReqEmitter) close() {
	if e == nil {
		return
	}
	e.stopOnce.Do(func() { close(e.stop) })
	select {
	case <-e.done:
	case <-time.After(blCloseTimeout):
		klog.ErrorS(nil, "baseline request event writer did not stop in time")
	}
}

// ---------------------------------------------------------------------------------------
// per-request state and hooks (all no-ops when the emitter is nil)

type blReqState struct {
	stream   string // Redis key
	reqID    string
	reissue  string
	pod      string
	arrAt    time.Time
	ftSent   bool
	doneSent bool
	// outTokens is the completion token count from usage, or -1 when none was seen.
	outTokens int64
	// immediateStatus is the HTTP code of an immediate response the gateway sent after
	// routing (e.g. an upstream error rewritten by the gateway).
	immediateStatus string
	respHeadersSeen bool
}

func blReissue(h map[string]string) string {
	if _, ok := h[HeaderTREContinued]; ok {
		return "continued"
	}
	if _, ok := h[HeaderTRERetried]; ok {
		return "retried"
	}
	return "none"
}

// blCaptureRequestHeader keeps the headers the event stream needs; they are only stored
// while the feature is on.
func (s *Server) blCaptureRequestHeader(headers map[string]string, key string, raw []byte) {
	if s.blEvents == nil {
		return
	}
	headers[key] = string(raw)
}

// blOnRequestBody emits arr for a request that was forwarded upstream (no immediate
// response) and keeps its event state in st.bl.
func (s *Server) blOnRequestBody(st *processState, req *extProcPb.ProcessingRequest, resp *extProcPb.ProcessingResponse) {
	e := s.blEvents
	if e == nil || st.bl != nil || resp == nil || resp.GetImmediateResponse() != nil || st.model == "" || st.routerCtx == nil {
		return
	}
	rc := st.routerCtx
	// The routing context is pooled and recycled at request end: copy what is needed now.
	pod := ""
	if rc.HasRouted() {
		if p := rc.TargetPod(); p != nil && p.Name != "" {
			pod = p.Name
		} else {
			pod = rc.TargetAddress()
		}
	}
	body := req.GetRequestBody().GetBody()
	inTokens, inSrc := blInputTokens(rc, st.model, body, e.cfg)
	maxTokens := ""
	for _, k := range [...]string{"max_tokens", "max_completion_tokens"} {
		if r := gjson.GetBytes(body, k); r.Exists() && r.Type == gjson.Number {
			maxTokens = strconv.FormatInt(r.Int(), 10)
			break
		}
	}
	b := &blReqState{
		stream:    BLReqEventsKeyPrefix + st.model,
		reqID:     st.requestID,
		reissue:   blReissue(rc.ReqHeaders),
		pod:       pod,
		arrAt:     time.Now(),
		outTokens: -1,
	}
	st.bl = b
	e.enqueue(blReqEvent{stream: b.stream, values: []string{
		"kind", blKindArr,
		"req_id", b.reqID,
		"reissue", b.reissue,
		"gw_local_ms", strconv.FormatInt(b.arrAt.UnixMilli(), 10),
		"pod", b.pod,
		"in_tokens", strconv.FormatInt(inTokens, 10),
		"in_src", inSrc,
		"max_tokens", maxTokens,
		"stream", strconv.FormatBool(st.stream),
	}})
}

// blOnResponseHeaders notes that the upstream answered (the status itself is in st).
func (s *Server) blOnResponseHeaders(st *processState) {
	if st.bl != nil {
		st.bl.respHeadersSeen = true
	}
}

// blOnResponseBody emits ft on the first non-empty body frame of a non-error response.
// It does not parse the frame.
func (s *Server) blOnResponseBody(st *processState, frame []byte) {
	b := st.bl
	if b == nil || b.ftSent || st.isRespError || len(frame) == 0 || s.blEvents == nil {
		return
	}
	b.ftSent = true
	now := time.Now()
	s.blEvents.enqueue(blReqEvent{stream: b.stream, values: []string{
		"kind", blKindFT,
		"req_id", b.reqID,
		"reissue", b.reissue,
		"gw_local_ms", strconv.FormatInt(now.UnixMilli(), 10),
		"pod", b.pod,
		"dt_arr_ms", blDurMs(now.Sub(b.arrAt)),
	}})
}

func (s *Server) blOnOutTokens(st *processState, out int64) {
	if st.bl != nil && out >= 0 {
		st.bl.outTokens = out
	}
}

// blOnResponse records the status of an immediate response sent after routing.
func (s *Server) blOnResponse(st *processState, resp *extProcPb.ProcessingResponse) {
	if st.bl == nil || st.bl.immediateStatus != "" {
		return
	}
	if ir := resp.GetImmediateResponse(); ir != nil {
		st.bl.immediateStatus = strconv.Itoa(int(ir.GetStatus().GetCode()))
	}
}

// blOnStreamEnd emits done. It is called once, from the deferred cleanup of Process, which
// runs on every exit of the ext_proc stream; doneSent makes a second call a no-op.
func (s *Server) blOnStreamEnd(st *processState, err error) {
	b := st.bl
	if b == nil || b.doneSent || s.blEvents == nil {
		return
	}
	b.doneSent = true
	outSrc := "none"
	if b.outTokens >= 0 {
		outSrc = "usage"
	}
	now := time.Now()
	s.blEvents.enqueue(blReqEvent{stream: b.stream, values: []string{
		"kind", blKindDone,
		"req_id", b.reqID,
		"reissue", b.reissue,
		"gw_local_ms", strconv.FormatInt(now.UnixMilli(), 10),
		"pod", b.pod,
		"status", blDoneStatus(st, err),
		"out_tokens", strconv.FormatInt(b.outTokens, 10),
		"out_src", outSrc,
		"dt_arr_ms", blDurMs(now.Sub(b.arrAt)),
	}})
}

// blDoneStatus: an HTTP status when the response ended (or the gateway answered with an
// error), otherwise the class of the error that ended the stream.
func blDoneStatus(st *processState, err error) string {
	b := st.bl
	switch {
	case b.immediateStatus != "":
		return b.immediateStatus
	case st.isRespError && st.respErrorCode > 0:
		return strconv.Itoa(st.respErrorCode)
	case st.completed || st.respBodyEOS:
		return "200"
	case err != nil:
		return blErrorClass(err)
	case b.respHeadersSeen:
		return "200"
	default:
		return "incomplete"
	}
}

func blErrorClass(err error) string {
	switch {
	case errors.Is(err, io.EOF):
		return "client_eof"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	}
	if se, ok := status.FromError(err); ok {
		switch se.Code() {
		case codes.Canceled:
			return "canceled"
		case codes.DeadlineExceeded:
			return "deadline"
		case codes.Unavailable:
			return "unavailable"
		}
		return "grpc_" + strings.ToLower(se.Code().String())
	}
	return "error"
}

func blDurMs(d time.Duration) string {
	return strconv.FormatFloat(float64(d.Microseconds())/1000, 'f', 3, 64)
}

// ---------------------------------------------------------------------------------------
// input token count

// blInputTokens returns the prompt length in tokens and its source:
//   - "header": x-tre-bl-in-tokens, a positive integer set by the client (exact);
//   - "token_ids": a completions prompt given as token ids (exact, the id count);
//   - "estimate": ceil(chars / chars_per_token[model]), chars counted in runes.
//
// The estimate is a fallback only. A fixed chars/token ratio is inaccurate: CJK text has
// far fewer characters per token than English (a ratio of 4 underestimates Chinese by
// roughly 2-3x), and chat templates / special tokens are not counted. Clients should send
// the header when the exact count matters.
func blInputTokens(rc *types.RoutingContext, model string, body []byte, cfg blReqConfig) (int64, string) {
	if v, ok := rc.ReqHeaders[HeaderBLInTokens]; ok {
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil && n > 0 {
			return n, "header"
		}
	}
	chars, ids := blPromptChars(rc.ReqPath, body, rc.Message)
	if ids >= 0 {
		return ids, "token_ids"
	}
	return blEstimateTokens(chars, cfg.charsPerTokenFor(model)), "estimate"
}

func blEstimateTokens(chars int64, charsPerToken float64) int64 {
	if chars <= 0 {
		return 0
	}
	if charsPerToken <= 0 {
		charsPerToken = defaultBLCharsPerToken
	}
	return int64(math.Ceil(float64(chars) / charsPerToken))
}

// blPromptChars counts the prompt characters (runes) of a request body. ids >= 0 means the
// prompt was given as token ids (completions only) and ids is their count.
func blPromptChars(path string, body []byte, message string) (chars int64, ids int64) {
	switch path {
	case PathCompletions:
		return blCompletionPromptChars(gjson.GetBytes(body, "prompt"))
	case PathChatCompletions, PathMessages:
		chars = blContentChars(gjson.GetBytes(body, "system")) // Anthropic-style top level
		gjson.GetBytes(body, "messages").ForEach(func(_, m gjson.Result) bool {
			chars += blContentChars(m.Get("content"))
			return true
		})
		return chars, -1
	case PathResponses:
		in := gjson.GetBytes(body, "input")
		if in.Type == gjson.String {
			return int64(utf8.RuneCountInString(in.String())), -1
		}
		in.ForEach(func(_, item gjson.Result) bool {
			chars += blContentChars(item.Get("content"))
			return true
		})
		return chars, -1
	default:
		return int64(utf8.RuneCountInString(message)), -1
	}
}

// blCompletionPromptChars: a string, an array of strings, an array of token ids or an array
// of token-id arrays.
func blCompletionPromptChars(p gjson.Result) (chars int64, ids int64) {
	switch {
	case p.Type == gjson.String:
		return int64(utf8.RuneCountInString(p.String())), -1
	case p.IsArray():
		var nIDs int64
		allIDs := true
		p.ForEach(func(_, el gjson.Result) bool {
			switch {
			case el.Type == gjson.String:
				chars += int64(utf8.RuneCountInString(el.String()))
				allIDs = false
			case el.Type == gjson.Number:
				nIDs++
			case el.IsArray():
				nIDs += int64(len(el.Array()))
			default:
				allIDs = false
			}
			return true
		})
		if allIDs && nIDs > 0 {
			return 0, nIDs
		}
		return chars, -1
	default:
		return 0, -1
	}
}

// blContentChars counts a message content: a string, or content parts whose "text" is a
// string (non-text parts such as images contribute nothing).
func blContentChars(c gjson.Result) int64 {
	switch {
	case c.Type == gjson.String:
		return int64(utf8.RuneCountInString(c.String()))
	case c.IsArray():
		var n int64
		c.ForEach(func(_, part gjson.Result) bool {
			if part.Type == gjson.String {
				n += int64(utf8.RuneCountInString(part.String()))
			} else if t := part.Get("text"); t.Type == gjson.String {
				n += int64(utf8.RuneCountInString(t.String()))
			}
			return true
		})
		return n
	default:
		return 0
	}
}
