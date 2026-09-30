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
	"errors"
	"io"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	miniredis "github.com/alicebob/miniredis/v2"
	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	dto "github.com/prometheus/client_model/go"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/aibrix/pkg/types"
)

// TRE-PATCH(BL-GW-001) tests. They share process-wide state with the TRE tests (treGW,
// the router registry, the cache), so none of them runs in parallel.

// blCaptureHook records every command sent through the client and can hold pipelines.
type blCaptureHook struct {
	mu    sync.Mutex
	cmds  [][]interface{}
	block chan struct{} // non-nil: pipelines wait until it is closed
}

func (h *blCaptureHook) record(cmd redis.Cmder) {
	h.mu.Lock()
	h.cmds = append(h.cmds, cmd.Args())
	h.mu.Unlock()
}

func (h *blCaptureHook) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.cmds)
}

func (h *blCaptureHook) all() [][]interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([][]interface{}(nil), h.cmds...)
}

func (h *blCaptureHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *blCaptureHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		h.record(cmd)
		return next(ctx, cmd)
	}
}

func (h *blCaptureHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		if h.block != nil {
			<-h.block
		}
		for _, c := range cmds {
			h.record(c)
		}
		return next(ctx, cmds)
	}
}

type blRedis struct {
	mr     *miniredis.Miniredis
	client *redis.Client // hooked: what the emitter writes with
	reader *redis.Client // unhooked: what the test reads with
	hook   *blCaptureHook
}

func newBLRedis(t *testing.T) *blRedis {
	t.Helper()
	mr := miniredis.RunT(t)
	hook := &blCaptureHook{}
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	client.AddHook(hook)
	reader := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close(); _ = reader.Close() })
	return &blRedis{mr: mr, client: client, reader: reader, hook: hook}
}

func testBLConfig() blReqConfig {
	return blReqConfig{
		maxLen:        200000,
		buffer:        64,
		charsPerToken: map[string]float64{blCharsPerTokenDefault: defaultBLCharsPerToken},
		flushInterval: 5 * time.Millisecond,
		flushBatch:    blFlushBatch,
	}
}

func (r *blRedis) events(t *testing.T, model string) []map[string]string {
	t.Helper()
	msgs, err := r.reader.XRange(context.Background(), BLReqEventsKeyPrefix+model, "-", "+").Result()
	require.NoError(t, err)
	out := make([]map[string]string, 0, len(msgs))
	for _, m := range msgs {
		ev := map[string]string{}
		for k, v := range m.Values {
			ev[k] = v.(string)
		}
		out = append(out, ev)
	}
	return out
}

func blKinds(evs []map[string]string) []string {
	out := make([]string, 0, len(evs))
	for _, e := range evs {
		out = append(out, e["kind"])
	}
	return out
}

func blKeys(ev map[string]string) []string {
	out := make([]string, 0, len(ev))
	for k := range ev {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sorted(s ...string) []string {
	sort.Strings(s)
	return s
}

var (
	blArrFields  = sorted("kind", "req_id", "reissue", "gw_local_ms", "pod", "in_tokens", "in_src", "max_tokens", "stream")
	blFTFields   = sorted("kind", "req_id", "reissue", "gw_local_ms", "pod", "dt_arr_ms")
	blDoneFields = sorted("kind", "req_id", "reissue", "gw_local_ms", "pod", "status", "out_tokens", "out_src", "dt_arr_ms")
)

func blRespHeaders(code string) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: ":status", RawValue: []byte(code)}}}},
	}}
}

func blRespBody(body string, eos bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
		ResponseBody: &extProcPb.HttpBody{Body: []byte(body), EndOfStream: eos},
	}}
}

// blRun routes one request through Process against pod "p" of model "m" with the given
// request headers/body, then replays the response messages / stream ending in script.
func blRun(t *testing.T, s *Server, name, path, body string, hdrs []string, script func(srv *mockProcessServer)) error {
	t.Helper()
	resetTREGateway(t)
	algo := registerTRENamedRouter(t, "bl-test-"+name, &treNamedRouter{order: []string{"p"}})
	s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
	srv := &mockProcessServer{ctx: context.Background()}
	srv.On("Recv").Return(treHeadersReq(path, append([]string{HeaderRoutingStrategy, string(algo)}, hdrs...)...), nil).Once()
	srv.On("Recv").Return(treBodyReq(body), nil).Once()
	script(srv)
	srv.On("Send", mock.Anything).Return(nil)
	return s.Process(srv)
}

func newBLServer(r *blRedis, cfg blReqConfig) *Server {
	s := newProcessTestServer(openShutdownCh(), nil)
	s.redisClient = r.client
	s.blEvents = newBLReqEmitter(r.client, cfg)
	return s
}

func counterValue(t *testing.T, c interface{ Write(*dto.Metric) error }) float64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, c.Write(&m))
	return m.GetCounter().GetValue()
}

// ---------------------------------------------------------------------------------------
// 1. switch off: no emitter, no Redis writes, no allocations in the hooks

func TestBLReqEvents_SwitchParsing(t *testing.T) {
	r := newBLRedis(t)
	for _, v := range []string{"", "false", "0", "yes", "on"} {
		t.Setenv(envBLReqEvents, v)
		assert.Nil(t, newBLReqEmitterFromEnv(r.client), "value %q must leave the feature off", v)
	}
	t.Setenv(envBLReqEvents, "true")
	assert.Nil(t, newBLReqEmitterFromEnv(nil), "no Redis client: off")
	for _, v := range []string{"true", "1", "TRUE", " True "} {
		t.Setenv(envBLReqEvents, v)
		e := newBLReqEmitterFromEnv(r.client)
		require.NotNil(t, e, "value %q enables the feature", v)
		e.close()
	}
}

func TestBLReqEvents_OffMeansZeroRedisWrites(t *testing.T) {
	t.Setenv(envBLReqEvents, "")
	r := newBLRedis(t)
	s := newProcessTestServer(openShutdownCh(), nil)
	s.redisClient = r.client
	s.blEvents = newBLReqEmitterFromEnv(r.client)
	require.Nil(t, s.blEvents)

	err := blRun(t, s, "off", PathCompletions, `{"model":"m","prompt":"hello","max_tokens":4,"stream":true}`,
		[]string{HeaderBLInTokens, "5", HeaderTREContinued, "1"}, func(srv *mockProcessServer) {
			srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
			srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"a\"}]}\n\n", false), nil).Once()
			srv.On("Recv").Return(blRespBody("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\ndata: [DONE]\n\n", true), nil).Once()
		})
	require.NoError(t, err)
	assert.Zero(t, r.hook.count(), "no command may reach Redis")
	assert.Zero(t, r.mr.CommandCount())
	assert.Empty(t, r.mr.Keys())
}

func TestBLReqEvents_OffHeadersNotCapturedAndHooksDoNotAllocate(t *testing.T) {
	s := &Server{}
	_, _, _, rc := s.HandleRequestHeaders(context.Background(), "rid",
		treHeadersReq(PathCompletions, HeaderBLInTokens, "5", HeaderTREContinued, "1", HeaderTRERetried, "1"))
	require.NotNil(t, rc)
	for _, h := range []string{HeaderBLInTokens, HeaderTREContinued, HeaderTRERetried} {
		_, ok := rc.ReqHeaders[h]
		assert.False(t, ok, "%s must not be stored while the feature is off", h)
	}
	rc.Delete()

	st := &processState{model: "m", requestID: "rid", routerCtx: types.NewRoutingContext(context.Background(), "", "m", "", "rid", "")}
	defer st.routerCtx.Delete()
	reqBody := treBodyReq(`{"model":"m","prompt":"hi"}`)
	resp := &extProcPb.ProcessingResponse{Response: &extProcPb.ProcessingResponse_RequestBody{}}
	frame := []byte("data: x\n\n")
	errEOF := io.EOF
	allocs := testing.AllocsPerRun(1000, func() {
		s.blCaptureRequestHeader(nil, HeaderBLInTokens, frame)
		s.blOnRequestBody(st, reqBody, resp)
		s.blOnResponseHeaders(st)
		s.blOnResponseBody(st, frame)
		s.blOnOutTokens(st, 3)
		s.blOnResponse(st, resp)
		s.blOnStreamEnd(st, errEOF)
	})
	assert.Zero(t, allocs)
	assert.Nil(t, st.bl)
}

// ---------------------------------------------------------------------------------------
// 2. arr -> ft -> done with all fields

func TestBLReqEvents_StreamingCompletionsSequence(t *testing.T) {
	r := newBLRedis(t)
	s := newBLServer(r, testBLConfig())
	err := blRun(t, s, "stream-cmpl", PathCompletions, `{"model":"m","prompt":"hello world!","max_tokens":16,"stream":true}`, nil,
		func(srv *mockProcessServer) {
			srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
			srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"a\"}]}\n\n", false), nil).Once()
			srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"b\",\"finish_reason\":\"length\"}]}\n\n", false), nil).Once()
			srv.On("Recv").Return(blRespBody("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n", true), nil).Once()
		})
	require.NoError(t, err)
	s.blEvents.close()

	evs := r.events(t, "m")
	require.Equal(t, []string{"arr", "ft", "done"}, blKinds(evs))
	arr, ft, done := evs[0], evs[1], evs[2]
	assert.Equal(t, blArrFields, blKeys(arr))
	assert.Equal(t, blFTFields, blKeys(ft))
	assert.Equal(t, blDoneFields, blKeys(done))

	require.NotEmpty(t, arr["req_id"])
	for _, e := range evs {
		assert.Equal(t, arr["req_id"], e["req_id"])
		assert.Equal(t, "p", e["pod"])
		assert.Equal(t, "none", e["reissue"])
		ms, err := strconv.ParseInt(e["gw_local_ms"], 10, 64)
		require.NoError(t, err)
		assert.InDelta(t, time.Now().UnixMilli(), ms, 60_000)
	}
	assert.Equal(t, "3", arr["in_tokens"], "ceil(12 chars / 4)")
	assert.Equal(t, "estimate", arr["in_src"])
	assert.Equal(t, "16", arr["max_tokens"])
	assert.Equal(t, "true", arr["stream"])

	ftMs, err := strconv.ParseFloat(ft["dt_arr_ms"], 64)
	require.NoError(t, err)
	doneMs, err := strconv.ParseFloat(done["dt_arr_ms"], 64)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, ftMs, 0.0)
	assert.GreaterOrEqual(t, doneMs, ftMs)

	assert.Equal(t, "200", done["status"])
	assert.Equal(t, "2", done["out_tokens"])
	assert.Equal(t, "usage", done["out_src"])
}

func TestBLReqEvents_ChatNonStreamingEstimateWithContentParts(t *testing.T) {
	r := newBLRedis(t)
	cfg := testBLConfig()
	cfg.charsPerToken = parseBLCharsPerToken(`{"m": 3, "*": 4}`)
	s := newBLServer(r, cfg)
	body := `{"model":"m","max_completion_tokens":7,"messages":[` +
		`{"role":"system","content":"你好"},` +
		`{"role":"user","content":[{"type":"text","text":"abcdef"},{"type":"image_url","image_url":{"url":"data:x"}}]}]}`
	err := blRun(t, s, "chat", PathChatCompletions, body, nil, func(srv *mockProcessServer) {
		srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
		srv.On("Recv").Return(blRespBody(`{"model":"m","choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`, true), nil).Once()
	})
	require.NoError(t, err)
	s.blEvents.close()

	evs := r.events(t, "m")
	require.Equal(t, []string{"arr", "ft", "done"}, blKinds(evs))
	assert.Equal(t, "3", evs[0]["in_tokens"], "ceil((2+6) runes / 3 for model m)")
	assert.Equal(t, "estimate", evs[0]["in_src"])
	assert.Equal(t, "7", evs[0]["max_tokens"])
	assert.Equal(t, "false", evs[0]["stream"])
	assert.Equal(t, "200", evs[2]["status"])
	assert.Equal(t, "4", evs[2]["out_tokens"])
	assert.Equal(t, "usage", evs[2]["out_src"])
}

// ---------------------------------------------------------------------------------------
// 3. in_tokens: header vs estimate, coefficients

func TestBLReqEvents_InTokensHeaderThroughProcess(t *testing.T) {
	r := newBLRedis(t)
	s := newBLServer(r, testBLConfig())
	err := blRun(t, s, "hdr", PathCompletions, `{"model":"m","prompt":"hello world!"}`,
		[]string{HeaderBLInTokens, "42"}, func(srv *mockProcessServer) {
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Once()
		})
	require.Error(t, err)
	s.blEvents.close()
	evs := r.events(t, "m")
	require.Equal(t, []string{"arr", "done"}, blKinds(evs))
	assert.Equal(t, "42", evs[0]["in_tokens"])
	assert.Equal(t, "header", evs[0]["in_src"])
	assert.Equal(t, "", evs[0]["max_tokens"], "absent max_tokens is empty")
	assert.Equal(t, "false", evs[0]["stream"])
}

func TestBLReqEvents_InputTokens(t *testing.T) {
	cfg := testBLConfig()
	cfg.charsPerToken = parseBLCharsPerToken(`{"a": 2.5, "*": 3}`)
	cmpl := []byte(`{"prompt":"0123456789"}`) // 10 runes
	cases := []struct {
		name, model, header string
		hasHeader           bool
		path                string
		body                []byte
		want                int64
		src                 string
	}{
		{"header", "a", "123", true, PathCompletions, cmpl, 123, "header"},
		{"header with spaces", "a", " 7 ", true, PathCompletions, cmpl, 7, "header"},
		{"zero header", "a", "0", true, PathCompletions, cmpl, 4, "estimate"},
		{"negative header", "a", "-5", true, PathCompletions, cmpl, 4, "estimate"},
		{"non-int header", "a", "abc", true, PathCompletions, cmpl, 4, "estimate"},
		{"empty header", "a", "", true, PathCompletions, cmpl, 4, "estimate"},
		{"missing header, model coefficient", "a", "", false, PathCompletions, cmpl, 4, "estimate"},            // ceil(10/2.5)
		{"missing header, default coefficient", "other", "", false, PathCompletions, cmpl, 4, "estimate"},      // ceil(10/3)
		{"prompt list", "other", "", false, PathCompletions, []byte(`{"prompt":["abc","de"]}`), 2, "estimate"}, // ceil(5/3)
		{"cjk runes not bytes", "other", "", false, PathCompletions, []byte(`{"prompt":"中文中文中文中"}`), 3, "estimate"},
		{"escaped string", "other", "", false, PathCompletions, []byte(`{"prompt":"中\n"}`), 1, "estimate"},
		{"token ids", "other", "", false, PathCompletions, []byte(`{"prompt":[1,2,3,4,5]}`), 5, "token_ids"},
		{"token id batches", "other", "", false, PathCompletions, []byte(`{"prompt":[[1,2],[3]]}`), 3, "token_ids"},
		{"empty prompt", "other", "", false, PathCompletions, []byte(`{"prompt":""}`), 0, "estimate"},
		{"chat string", "other", "", false, PathChatCompletions, []byte(`{"messages":[{"content":"abc"},{"content":"defg"}]}`), 3, "estimate"},
		{"chat parts", "other", "", false, PathChatCompletions, []byte(`{"messages":[{"content":[{"type":"text","text":"abcdef"},{"type":"image_url"}]}]}`), 2, "estimate"},
		{"responses input", "other", "", false, PathResponses, []byte(`{"input":"abcdefg"}`), 3, "estimate"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rc := types.NewRoutingContext(context.Background(), "", c.model, "", "rid", "")
			defer rc.Delete()
			rc.ReqPath = c.path
			rc.ReqHeaders = map[string]string{}
			if c.hasHeader {
				rc.ReqHeaders[HeaderBLInTokens] = c.header
			}
			n, src := blInputTokens(rc, c.model, c.body, cfg)
			assert.Equal(t, c.want, n)
			assert.Equal(t, c.src, src)
		})
	}
}

func TestBLReqEvents_CharsPerTokenAndConfig(t *testing.T) {
	m := parseBLCharsPerToken(`{"a": 2, "b": -1, "c": 0}`)
	assert.Equal(t, map[string]float64{"*": 4, "a": 2}, m, "non-positive values ignored, * defaults to 4")
	assert.Equal(t, map[string]float64{"*": 4}, parseBLCharsPerToken(`not json`))
	cfg := blReqConfig{charsPerToken: parseBLCharsPerToken(`{"*": 2}`)}
	assert.Equal(t, 2.0, cfg.charsPerTokenFor("anything"))

	assert.Equal(t, int64(0), blEstimateTokens(0, 4))
	assert.Equal(t, int64(1), blEstimateTokens(1, 4))
	assert.Equal(t, int64(1), blEstimateTokens(4, 4))
	assert.Equal(t, int64(2), blEstimateTokens(5, 4))
	assert.Equal(t, int64(4), blEstimateTokens(7, 2))

	t.Setenv(envBLReqEventsMaxLen, "1234")
	t.Setenv(envBLReqEventsBuffer, "77")
	t.Setenv(envBLCharsPerToken, `{"m": 3.5}`)
	got := loadBLReqConfig()
	assert.Equal(t, int64(1234), got.maxLen)
	assert.Equal(t, 77, got.buffer)
	assert.Equal(t, 3.5, got.charsPerTokenFor("m"))
	assert.Equal(t, 4.0, got.charsPerTokenFor("x"))

	t.Setenv(envBLReqEventsMaxLen, "-1")
	t.Setenv(envBLReqEventsBuffer, "zero")
	t.Setenv(envBLCharsPerToken, "")
	got = loadBLReqConfig()
	assert.Equal(t, defaultBLReqMaxLen, got.maxLen)
	assert.Equal(t, defaultBLReqBuffer, got.buffer)
	assert.Equal(t, 4.0, got.charsPerTokenFor("m"))
}

// ---------------------------------------------------------------------------------------
// 4. error / abort paths: done exactly once, ft at most once

func TestBLReqEvents_DoneExactlyOnceOnEveryEnding(t *testing.T) {
	endings := map[string]struct {
		script     func(srv *mockProcessServer)
		wantKinds  []string
		wantStatus string
	}{
		"upstream error response": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return(blRespHeaders("503"), nil).Once()
				srv.On("Recv").Return(blRespBody(`{"error":{"type":"EngineSleeping"}}`, true), nil).Once()
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Canceled, "done")).Once()
			},
			wantKinds: []string{"arr", "done"}, wantStatus: "503",
		},
		"client disconnect mid-stream": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
				srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"a\"}]}\n\n", false), nil).Once()
				srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"b\"}]}\n\n", false), nil).Once()
				srv.On("Recv").Return(blRespBody("data: {\"choices\":[{\"text\":\"c\"}]}\n\n", false), nil).Once()
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Once()
			},
			wantKinds: []string{"arr", "ft", "done"}, wantStatus: "client_eof",
		},
		"stream cancelled before response": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Canceled, "gone")).Once()
			},
			wantKinds: []string{"arr", "done"}, wantStatus: "canceled",
		},
		"grpc error": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Internal, "boom")).Once()
			},
			wantKinds: []string{"arr", "done"}, wantStatus: "grpc_internal",
		},
		"non-grpc error": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), errors.New("transport")).Once()
			},
			wantKinds: []string{"arr", "done"}, wantStatus: "grpc_unknown",
		},
		"malformed SSE (gateway immediate 500)": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
				srv.On("Recv").Return(blRespBody("data: {\"usage\": broken\n\n", false), nil).Once()
				srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Canceled, "done")).Once()
			},
			wantKinds: []string{"arr", "ft", "done"}, wantStatus: "500",
		},
		"completed": {
			script: func(srv *mockProcessServer) {
				srv.On("Recv").Return(blRespHeaders("200"), nil).Once()
				srv.On("Recv").Return(blRespBody(`{"model":"m","choices":[{"text":"x"}]}`, true), nil).Once()
			},
			wantKinds: []string{"arr", "ft", "done"}, wantStatus: "200",
		},
	}
	for name, c := range endings {
		t.Run(name, func(t *testing.T) {
			r := newBLRedis(t)
			s := newBLServer(r, testBLConfig())
			_ = blRun(t, s, "end-"+name, PathCompletions, `{"model":"m","prompt":"hi","stream":true}`, nil, c.script)
			s.blEvents.close()
			evs := r.events(t, "m")
			require.Equal(t, c.wantKinds, blKinds(evs))
			done := evs[len(evs)-1]
			assert.Equal(t, blDoneFields, blKeys(done))
			assert.Equal(t, c.wantStatus, done["status"])
			if name != "completed" {
				assert.Equal(t, "-1", done["out_tokens"])
				assert.Equal(t, "none", done["out_src"])
			}
		})
	}
}

func TestBLReqEvents_DoneOnShutdownAndSendFailure(t *testing.T) {
	t.Run("shutdown while waiting for the response", func(t *testing.T) {
		r := newBLRedis(t)
		shutdown := make(chan struct{})
		s := newBLServer(r, testBLConfig())
		s.shutdownCh = shutdown
		resetTREGateway(t)
		algo := registerTRENamedRouter(t, "bl-test-shutdown", &treNamedRouter{order: []string{"p"}})
		s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
		srv := &mockProcessServer{ctx: context.Background()}
		srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
		srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi"}`), nil).Once()
		block := make(chan struct{})
		defer close(block)
		srv.On("Recv").Run(func(mock.Arguments) { <-block }).Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Maybe()
		srv.On("Send", mock.Anything).Run(func(args mock.Arguments) {
			if args.Get(0).(*extProcPb.ProcessingResponse).GetRequestBody() != nil {
				close(shutdown)
			}
		}).Return(nil)
		require.Error(t, s.Process(srv))
		s.blEvents.close()
		evs := r.events(t, "m")
		require.Equal(t, []string{"arr", "done"}, blKinds(evs))
		assert.Equal(t, "unavailable", evs[1]["status"])
	})

	t.Run("send to envoy fails", func(t *testing.T) {
		r := newBLRedis(t)
		s := newBLServer(r, testBLConfig())
		resetTREGateway(t)
		algo := registerTRENamedRouter(t, "bl-test-sendfail", &treNamedRouter{order: []string{"p"}})
		s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
		srv := &mockProcessServer{ctx: context.Background()}
		srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
		srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi"}`), nil).Once()
		srv.On("Send", mock.MatchedBy(func(r *extProcPb.ProcessingResponse) bool { return r.GetRequestBody() == nil })).Return(nil)
		srv.On("Send", mock.MatchedBy(func(r *extProcPb.ProcessingResponse) bool { return r.GetRequestBody() != nil })).
			Return(status.Error(codes.Canceled, "client went away"))
		require.Error(t, s.Process(srv))
		s.blEvents.close()
		evs := r.events(t, "m")
		require.Equal(t, []string{"arr", "done"}, blKinds(evs))
		assert.Equal(t, "canceled", evs[1]["status"])
	})
}

// A request answered by the gateway before routing (unknown model) has no arr and no done.
func TestBLReqEvents_NoEventsWithoutRouting(t *testing.T) {
	r := newBLRedis(t)
	s := newBLServer(r, testBLConfig())
	err := blRun(t, s, "unknown-model", PathCompletions, `{"model":"nope","prompt":"hi"}`, nil, func(srv *mockProcessServer) {
		srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Once()
	})
	require.Error(t, err)
	s.blEvents.close()
	assert.Empty(t, r.events(t, "nope"))
	assert.Empty(t, r.events(t, "m"))
	assert.Empty(t, r.mr.Keys())
}

// The done hook is idempotent even if called again.
func TestBLReqEvents_StreamEndHookIdempotent(t *testing.T) {
	r := newBLRedis(t)
	s := &Server{blEvents: newBLReqEmitter(r.client, testBLConfig())}
	st := &processState{bl: &blReqState{stream: BLReqEventsKeyPrefix + "m", reqID: "r", reissue: "none", pod: "p", arrAt: time.Now(), outTokens: -1}}
	s.blOnStreamEnd(st, io.EOF)
	s.blOnStreamEnd(st, nil)
	s.blOnResponseBody(st, []byte("x"))
	s.blOnResponseBody(st, []byte("y"))
	s.blEvents.close()
	assert.Equal(t, []string{"done", "ft"}, blKinds(r.events(t, "m")))
}

// ---------------------------------------------------------------------------------------
// 5. reissue headers

func TestBLReqEvents_Reissue(t *testing.T) {
	assert.Equal(t, "none", blReissue(map[string]string{}))
	assert.Equal(t, "continued", blReissue(map[string]string{HeaderTREContinued: ""}))
	assert.Equal(t, "retried", blReissue(map[string]string{HeaderTRERetried: "1"}))
	assert.Equal(t, "continued", blReissue(map[string]string{HeaderTREContinued: "1", HeaderTRERetried: "1"}))

	for hdr, want := range map[string]string{"X-TRE-Continued": "continued", "x-tre-retried": "retried"} {
		t.Run(want, func(t *testing.T) {
			r := newBLRedis(t)
			s := newBLServer(r, testBLConfig())
			_ = blRun(t, s, "reissue-"+want, PathCompletions, `{"model":"m","prompt":[1,2,3]}`, []string{hdr, "1"},
				func(srv *mockProcessServer) {
					srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Once()
				})
			s.blEvents.close()
			evs := r.events(t, "m")
			require.Equal(t, []string{"arr", "done"}, blKinds(evs))
			for _, e := range evs {
				assert.Equal(t, want, e["reissue"])
			}
			assert.Equal(t, "3", evs[0]["in_tokens"])
			assert.Equal(t, "token_ids", evs[0]["in_src"])
		})
	}
}

// ---------------------------------------------------------------------------------------
// 6. full channel: drop, count, never block

func TestBLReqEvents_FullChannelDropsWithoutBlocking(t *testing.T) {
	r := newBLRedis(t)
	r.hook.block = make(chan struct{}) // Redis "hangs": the writer is stuck in its first pipeline
	cfg := testBLConfig()
	cfg.buffer = 4
	cfg.flushBatch = 1
	e := newBLReqEmitter(r.client, cfg)
	promDropped := blReqEventsDropped.WithLabelValues("buffer_full")
	before := counterValue(t, promDropped)

	ev := func(i int) blReqEvent {
		return blReqEvent{stream: BLReqEventsKeyPrefix + "m", values: []string{"kind", "arr", "i", strconv.Itoa(i)}}
	}
	e.enqueue(ev(0))
	require.Eventually(t, func() bool { return len(e.ch) == 0 }, time.Second, time.Millisecond, "writer took event 0 and blocks")
	for i := 1; i <= cfg.buffer; i++ {
		e.enqueue(ev(i))
	}
	require.Equal(t, cfg.buffer, len(e.ch))

	const extra = 100
	var worst time.Duration
	for i := 0; i < extra; i++ {
		start := time.Now()
		e.enqueue(ev(100 + i))
		if d := time.Since(start); d > worst {
			worst = d
		}
	}
	assert.Less(t, worst, time.Millisecond, "enqueue must not block")
	assert.Equal(t, uint64(extra), e.dropped.Load())
	assert.Equal(t, float64(extra), counterValue(t, promDropped)-before)

	close(r.hook.block)
	e.close()
	evs := r.events(t, "m")
	require.Len(t, evs, 1+cfg.buffer, "everything that was accepted is written, in order")
	for i, x := range evs {
		assert.Equal(t, strconv.Itoa(i), x["i"])
	}
}

// ---------------------------------------------------------------------------------------
// 7. XADD arguments: automatic ID and approximate MAXLEN

func TestBLReqEvents_XAddUsesApproxMaxLenAndAutoID(t *testing.T) {
	r := newBLRedis(t)
	cfg := testBLConfig()
	cfg.maxLen = 123
	e := newBLReqEmitter(r.client, cfg)
	e.enqueue(blReqEvent{stream: BLReqEventsKeyPrefix + "m", values: []string{"kind", "arr", "req_id", "r1"}})
	e.enqueue(blReqEvent{stream: BLReqEventsKeyPrefix + "m", values: []string{"kind", "done", "req_id", "r1"}})
	e.close()

	cmds := r.hook.all()
	require.Len(t, cmds, 2)
	for _, args := range cmds {
		require.GreaterOrEqual(t, len(args), 6)
		assert.Equal(t, []interface{}{"xadd", BLReqEventsKeyPrefix + "m", "maxlen", "~", int64(123), "*"}, args[:6])
	}
	assert.Equal(t, []interface{}{"kind", "arr", "req_id", "r1"}, cmds[0][6:])

	ids, err := r.reader.XRange(context.Background(), BLReqEventsKeyPrefix+"m", "-", "+").Result()
	require.NoError(t, err)
	require.Len(t, ids, 2)
	assert.Less(t, ids[0].ID, ids[1].ID)
}

// Batching: many events go out in few pipelines, and a quiet emitter flushes within the
// interval without being closed.
func TestBLReqEvents_BatchesAndFlushesOnInterval(t *testing.T) {
	r := newBLRedis(t)
	var pipelines int
	var mu sync.Mutex
	r.client.AddHook(blPipelineCounter{fn: func() { mu.Lock(); pipelines++; mu.Unlock() }})
	cfg := testBLConfig()
	cfg.buffer = 2000
	cfg.flushInterval = 20 * time.Millisecond
	e := newBLReqEmitter(r.client, cfg)
	defer e.close()
	for i := 0; i < 600; i++ {
		e.enqueue(blReqEvent{stream: BLReqEventsKeyPrefix + "m", values: []string{"kind", "arr", "i", strconv.Itoa(i)}})
	}
	require.Eventually(t, func() bool { return len(r.events(t, "m")) == 600 }, 2*time.Second, 5*time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	assert.LessOrEqual(t, pipelines, 600/blFlushBatch+3, "events are pipelined in batches")
}

type blPipelineCounter struct{ fn func() }

func (h blPipelineCounter) DialHook(next redis.DialHook) redis.DialHook          { return next }
func (h blPipelineCounter) ProcessHook(next redis.ProcessHook) redis.ProcessHook { return next }
func (h blPipelineCounter) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		h.fn()
		return next(ctx, cmds)
	}
}
