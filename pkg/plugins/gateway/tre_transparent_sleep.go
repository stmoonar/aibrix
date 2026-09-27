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
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bytedance/sonic"
	"github.com/redis/go-redis/v9"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P3-GW-007..011): gateway side of "transparent sleep"
// (plan 2026-09-27, decisions D3/D4/D5/D6/D10).
//
// Contract with the service-manager (SM), all keys in the TRE Redis:
//
//   tre:v2:gw:instances        ZSET  member=<instance id>  score=heartbeat unix ms
//   tre:v2:gw:seen:<pod>       HASH  field=<instance id>   value={"gen":int,"routable":bool,"ts":ms}
//   tre:v2:gw:inflight:<pod>   HASH  field=<instance id>   value={"total":int,"non_continuable":int,"ts":ms}
//
// Guarantees of this instance:
//   - seen: written only after the pod's new state is what routing enforces. Routing
//     commits a target (inflight +1) under a read lock on the route table; a table update
//     takes the write lock. So when "gen G" is acked, every request this instance routed
//     under an older state is already counted in inflight, and no later request can be
//     routed against the older state. The inflight value of the same pod is (re)written in
//     the same Redis pipeline, before the seen field, so a reader that sees the ack also
//     sees an inflight value at least as new as the ack.
//   - inflight: in-memory counts are authoritative; Redis is a write-through mirror by a
//     single writer goroutine (coalesced, never reordered). Each routed request is released
//     exactly once when its ext_proc stream ends (completion, error, client disconnect,
//     stream close, shutdown).
//   - instances: heartbeat only starts after the first full ack/inflight refresh has been
//     attempted; the member is removed on graceful shutdown.

const (
	TREGatewayInstancesKey      = "tre:v2:gw:instances"
	TREGatewaySeenKeyPrefix     = "tre:v2:gw:seen:"
	TREGatewayInflightKeyPrefix = "tre:v2:gw:inflight:"
	// TRERouteGenAnnotation is incremented by the SM in the same patch that changes the
	// routable label.
	TRERouteGenAnnotation = "tre.aibrix.io/route-gen"
	// HeaderTREExcludePod lists pod names (comma separated) that must not serve this
	// request (the sidecar's retry of a request a sleeping pod refused).
	HeaderTREExcludePod = "x-tre-exclude-pod"

	envTREGatewayCoordination      = "TRE_GW_COORDINATION"
	envTREGatewayInstanceID        = "TRE_GW_INSTANCE_ID"
	envTREGatewayHeartbeatInterval = "TRE_GW_HEARTBEAT_INTERVAL"
	envTREGatewayInstanceRetention = "TRE_GW_INSTANCE_RETENTION"
	envTREGatewayKeyTTL            = "TRE_GW_KEY_TTL"
	envTREGatewayRefreshInterval   = "TRE_GW_REFRESH_INTERVAL"
	envTREGatewayRetryAfterSeconds = "TRE_GW_RETRY_AFTER_SECONDS"
	envTREDefaultRoutingStrategy   = "TRE_DEFAULT_ROUTING_STRATEGY"

	defaultTREGatewayHeartbeatInterval = 2 * time.Second
	defaultTREGatewayInstanceRetention = 10 * time.Minute
	defaultTREGatewayKeyTTL            = 300 * time.Second
	defaultTREGatewayRefreshInterval   = 30 * time.Second
	defaultTREGatewayRetryAfterSeconds = 1
	defaultTREDefaultRoutingStrategy   = "least-gpu-cache"

	treGatewayFlushRetryBackoff = 250 * time.Millisecond
	treGatewayRedisTimeout      = 5 * time.Second
	// treCommitAttempts bounds re-routing when the chosen pod turned unroutable between
	// candidate listing and commit (a hide raced the request).
	treCommitAttempts = 3
)

var (
	errTREAllCandidatesExcluded = errors.New("all routable pods are excluded by " + HeaderTREExcludePod)
	errTRECommitRace            = errors.New("routable pods changed during routing; retry")
)

// ---------------------------------------------------------------------------------------
// configuration

type treGatewayConfig struct {
	instanceID        string
	heartbeat         time.Duration
	instanceRetention time.Duration
	keyTTL            time.Duration
	refresh           time.Duration
}

func loadTREGatewayConfig() treGatewayConfig {
	return treGatewayConfig{
		instanceID:        treGatewayInstanceID(),
		heartbeat:         envDuration(envTREGatewayHeartbeatInterval, defaultTREGatewayHeartbeatInterval),
		instanceRetention: envDuration(envTREGatewayInstanceRetention, defaultTREGatewayInstanceRetention),
		keyTTL:            envDuration(envTREGatewayKeyTTL, defaultTREGatewayKeyTTL),
		refresh:           envDuration(envTREGatewayRefreshInterval, defaultTREGatewayRefreshInterval),
	}
}

func treGatewayInstanceID() string {
	if v := strings.TrimSpace(os.Getenv(envTREGatewayInstanceID)); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("POD_NAME")); v != "" {
		return v
	}
	if h, err := os.Hostname(); err == nil && h != "" {
		return h
	}
	return "unknown-gateway-instance"
}

func envDuration(key string, def time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	if d, err := time.ParseDuration(v); err == nil && d > 0 {
		return d
	}
	klog.Warningf("invalid %s=%q, using default %s", key, v, def)
	return def
}

// treCoordinationWanted: explicit TRE_GW_COORDINATION wins; otherwise on exactly when the
// routable-label gate is on (the TRE deployment).
func treCoordinationWanted() bool {
	if v, ok := os.LookupEnv(envTREGatewayCoordination); ok && strings.TrimSpace(v) != "" {
		return utils.LoadEnvBool(envTREGatewayCoordination, false)
	}
	return utils.TRERoutableLabelFilterEnabled()
}

var treRetryAfterSeconds = func() int {
	n := utils.LoadEnvInt(envTREGatewayRetryAfterSeconds, defaultTREGatewayRetryAfterSeconds)
	if n < 0 {
		return defaultTREGatewayRetryAfterSeconds
	}
	return n
}()

// treDefaultRoutingStrategy (D10) is the strategy used when neither the request header,
// the model config profile nor ROUTING_ALGORITHM names one, so that every request goes
// through ext_proc pod selection (and its sleep awareness). Unset env = least-gpu-cache;
// "", "none" or "off" disables it (upstream behaviour: route via HTTPRoute/Service).
var treDefaultRoutingStrategy = loadTREDefaultRoutingStrategy()

func loadTREDefaultRoutingStrategy() string {
	v, ok := os.LookupEnv(envTREDefaultRoutingStrategy)
	if !ok {
		return defaultTREDefaultRoutingStrategy
	}
	v = strings.TrimSpace(v)
	switch strings.ToLower(v) {
	case "", "none", "off", "false":
		return ""
	}
	return v
}

// setTREDefaultRoutingStrategy overrides the default (tests) and returns the previous one.
func setTREDefaultRoutingStrategy(s string) string {
	prev := treDefaultRoutingStrategy
	treDefaultRoutingStrategy = s
	return prev
}

// ---------------------------------------------------------------------------------------
// request classification and exclusion

// parseTREExcludePods parses the x-tre-exclude-pod header value.
func parseTREExcludePods(v string) map[string]struct{} {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	out := map[string]struct{}{}
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out[p] = struct{}{}
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func treExcludedPods(headers map[string]string) map[string]struct{} {
	if headers == nil {
		return nil
	}
	return parseTREExcludePods(headers[HeaderTREExcludePod])
}

// treSamplingFields are the request fields that make a generation impossible to resume
// exactly from its already-emitted tokens (D6).
type treSamplingFields struct {
	N              json.RawMessage `json:"n"`
	BestOf         json.RawMessage `json:"best_of"`
	Logprobs       json.RawMessage `json:"logprobs"`
	TopLogprobs    json.RawMessage `json:"top_logprobs"`
	PromptLogprobs json.RawMessage `json:"prompt_logprobs"`
	Echo           json.RawMessage `json:"echo"`
	UseBeamSearch  json.RawMessage `json:"use_beam_search"`
	Tools          json.RawMessage `json:"tools"`
	Functions      json.RawMessage `json:"functions"`
	ToolChoice     json.RawMessage `json:"tool_choice"`
	Stream         json.RawMessage `json:"stream"`
}

// treNonContinuable reports whether a request cannot be continued after an abort and must
// therefore be drained: n>1, best_of>1, any logprobs output, echo, beam search, tool calls
// while streaming, and every non-generation endpoint (only completions and chat
// completions have a continuation path). Unparseable fields count as non-continuable.
func treNonContinuable(requestPath string, body []byte) bool {
	if requestPath != PathCompletions && requestPath != PathChatCompletions {
		return true
	}
	var f treSamplingFields
	if err := sonic.Unmarshal(body, &f); err != nil {
		return true
	}
	if rawNumberAbove(f.N, 1) || rawNumberAbove(f.BestOf, 1) {
		return true
	}
	// logprobs: completions int (0 still returns logprobs), chat bool.
	if rawPresent(f.Logprobs) && !rawIs(f.Logprobs, "false") {
		return true
	}
	if rawNumberAbove(f.TopLogprobs, 0) || rawPresent(f.PromptLogprobs) {
		return true
	}
	if rawIs(f.Echo, "true") || rawIs(f.UseBeamSearch, "true") {
		return true
	}
	if rawIs(f.Stream, "true") {
		if rawNonEmptyArray(f.Tools) || rawNonEmptyArray(f.Functions) {
			return true
		}
		if rawPresent(f.ToolChoice) && !rawIs(f.ToolChoice, `"none"`) {
			return true
		}
	}
	return false
}

func rawPresent(r json.RawMessage) bool {
	s := strings.TrimSpace(string(r))
	return s != "" && s != "null"
}

func rawIs(r json.RawMessage, lit string) bool {
	return strings.TrimSpace(string(r)) == lit
}

// rawNumberAbove: present and > limit; present but not a number counts as above.
func rawNumberAbove(r json.RawMessage, limit float64) bool {
	if !rawPresent(r) {
		return false
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(string(r)), 64)
	if err != nil {
		return true
	}
	return f > limit
}

func rawNonEmptyArray(r json.RawMessage) bool {
	if !rawPresent(r) {
		return false
	}
	var arr []json.RawMessage
	if err := sonic.Unmarshal(r, &arr); err != nil {
		return true // present but not an array: be conservative
	}
	return len(arr) > 0
}

// ---------------------------------------------------------------------------------------
// inflight accounting (D4)

type treInflightCounts struct {
	Total          int64 `json:"total"`
	NonContinuable int64 `json:"non_continuable"`
}

type treInflightTracker struct {
	mu       sync.Mutex
	counts   map[string]*treInflightCounts // pod name
	onChange atomic.Pointer[func(pod string)]
}

func newTREInflightTracker() *treInflightTracker {
	return &treInflightTracker{counts: map[string]*treInflightCounts{}}
}

// treInflightTicket is one routed request; Release is idempotent and nil-safe.
type treInflightTicket struct {
	tracker        *treInflightTracker
	pod            string
	nonContinuable bool
	released       atomic.Bool
}

func (t *treInflightTracker) acquire(pod string, nonContinuable bool) *treInflightTicket {
	t.mu.Lock()
	c := t.counts[pod]
	if c == nil {
		c = &treInflightCounts{}
		t.counts[pod] = c
	}
	c.Total++
	if nonContinuable {
		c.NonContinuable++
	}
	t.mu.Unlock()
	t.changed(pod)
	return &treInflightTicket{tracker: t, pod: pod, nonContinuable: nonContinuable}
}

// Release decrements the pod's counters exactly once.
func (k *treInflightTicket) Release() {
	if k == nil || k.tracker == nil || !k.released.CompareAndSwap(false, true) {
		return
	}
	t := k.tracker
	t.mu.Lock()
	if c := t.counts[k.pod]; c != nil {
		if c.Total > 0 {
			c.Total--
		}
		if k.nonContinuable && c.NonContinuable > 0 {
			c.NonContinuable--
		}
	}
	t.mu.Unlock()
	t.changed(k.pod)
}

func (t *treInflightTracker) changed(pod string) {
	if fn := t.onChange.Load(); fn != nil {
		(*fn)(pod)
	}
}

func (t *treInflightTracker) snapshot(pod string) treInflightCounts {
	t.mu.Lock()
	defer t.mu.Unlock()
	if c := t.counts[pod]; c != nil {
		return *c
	}
	return treInflightCounts{}
}

// pods returns every pod with a counter entry and drops zero entries for pods not in keep.
func (t *treInflightTracker) podsAndPrune(keep map[string]struct{}) []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]string, 0, len(t.counts))
	for pod, c := range t.counts {
		if _, ok := keep[pod]; !ok && c.Total == 0 && c.NonContinuable == 0 {
			delete(t.counts, pod)
			continue
		}
		out = append(out, pod)
	}
	return out
}

// ---------------------------------------------------------------------------------------
// route table (D3)

type treRouteState struct {
	name      string
	namespace string
	gen       int64
	routable  bool
	managed   bool // carries the routable label or the route-gen annotation
}

func treRouteStateFromPod(pod *v1.Pod) treRouteState {
	st := treRouteState{name: pod.Name, namespace: pod.Namespace}
	if v, ok := pod.Labels[utils.TRERoutableLabel]; ok {
		st.managed = true
		st.routable = v == "true"
	}
	if v, ok := pod.Annotations[TRERouteGenAnnotation]; ok {
		st.managed = true
		if g, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
			st.gen = g
		} else {
			klog.Warningf("pod %s/%s has unparseable %s=%q; acking gen 0", pod.Namespace, pod.Name, TRERouteGenAnnotation, v)
		}
	}
	return st
}

type treGatewayState struct {
	// enabled: the route table is authoritative for routing commits (coordination on).
	enabled atomic.Bool
	// barrier: commits hold RLock across check+increment; route-table updates hold Lock.
	barrier sync.RWMutex
	routes  map[string]treRouteState // key namespace/name, guarded by barrier

	inflight *treInflightTracker
	writer   atomic.Pointer[treRedisWriter]
}

func newTREGatewayState() *treGatewayState {
	return &treGatewayState{routes: map[string]treRouteState{}, inflight: newTREInflightTracker()}
}

// treGW is process-wide like the other TRE switches: one ext_proc server per process.
var treGW = newTREGatewayState()

// observePod is the cache.TREPodObserver: it runs after the cache reflects the pod.
func (g *treGatewayState) observePod(oldPod, newPod *v1.Pod) {
	if newPod == nil {
		if oldPod == nil {
			return
		}
		key := utils.GeneratePodKey(oldPod.Namespace, oldPod.Name)
		g.barrier.Lock()
		delete(g.routes, key)
		g.barrier.Unlock()
		return
	}
	st := treRouteStateFromPod(newPod)
	key := utils.GeneratePodKey(newPod.Namespace, newPod.Name)
	g.barrier.Lock()
	prev, existed := g.routes[key]
	g.routes[key] = st
	g.barrier.Unlock()
	if st.managed && (!existed || prev.gen != st.gen || prev.routable != st.routable || !prev.managed) {
		if w := g.writer.Load(); w != nil {
			w.enqueueAck(key)
		}
	}
}

// seed fills the route table from the cache (entries set by the observer meanwhile win:
// they are at least as new as the listing).
func (g *treGatewayState) seed(lister cache.PodLister) []string {
	if lister == nil {
		return nil
	}
	g.barrier.Lock()
	defer g.barrier.Unlock()
	var keys []string
	for _, pod := range lister.ListPods() {
		if pod == nil {
			continue
		}
		key := utils.GeneratePodKey(pod.Namespace, pod.Name)
		if _, ok := g.routes[key]; !ok {
			g.routes[key] = treRouteStateFromPod(pod)
		}
		keys = append(keys, key)
	}
	return keys
}

// routableInTable is the commit-time check (caller holds barrier.RLock).
func (g *treGatewayState) routableInTableLocked(pod *v1.Pod) bool {
	st, ok := g.routes[utils.GeneratePodKey(pod.Namespace, pod.Name)]
	return ok && st.routable
}

// filterCandidates drops pods whose route-table state is not routable (coordination on).
func (g *treGatewayState) filterCandidates(pods []*v1.Pod) []*v1.Pod {
	if !g.enabled.Load() {
		return pods
	}
	g.barrier.RLock()
	defer g.barrier.RUnlock()
	out := pods[:0:0]
	for _, p := range pods {
		if g.routableInTableLocked(p) {
			out = append(out, p)
		}
	}
	return out
}

// commit counts the request against pod, provided that (with coordination on) the pod is
// still routable in the route table. The check and the increment are atomic with respect
// to route-table updates, which is what makes the seen-gen ack truthful.
func (g *treGatewayState) commit(pod *v1.Pod, nonContinuable bool) (*treInflightTicket, bool) {
	if pod == nil {
		return nil, true
	}
	g.barrier.RLock()
	defer g.barrier.RUnlock()
	if g.enabled.Load() && !g.routableInTableLocked(pod) {
		return nil, false
	}
	return g.inflight.acquire(pod.Name, nonContinuable), true
}

func (g *treGatewayState) routeSnapshot(key string) (treRouteState, bool) {
	g.barrier.RLock()
	defer g.barrier.RUnlock()
	st, ok := g.routes[key]
	return st, ok
}

func (g *treGatewayState) routeKeysAndPods() (keys []string, pods map[string]struct{}) {
	g.barrier.RLock()
	defer g.barrier.RUnlock()
	pods = make(map[string]struct{}, len(g.routes))
	for k, st := range g.routes {
		if st.managed {
			keys = append(keys, k)
		}
		pods[st.name] = struct{}{}
	}
	return keys, pods
}

// ---------------------------------------------------------------------------------------
// Redis writer

type treRedisWriter struct {
	client *redis.Client
	cfg    treGatewayConfig
	state  *treGatewayState

	mu            sync.Mutex
	pendingAcks   map[string]struct{} // route keys
	dirtyInflight map[string]struct{} // pod names
	kick          chan struct{}

	stopOnce sync.Once
	stop     chan struct{}
	wg       sync.WaitGroup
}

func newTRERedisWriter(client *redis.Client, cfg treGatewayConfig, state *treGatewayState) *treRedisWriter {
	return &treRedisWriter{
		client:        client,
		cfg:           cfg,
		state:         state,
		pendingAcks:   map[string]struct{}{},
		dirtyInflight: map[string]struct{}{},
		kick:          make(chan struct{}, 1),
		stop:          make(chan struct{}),
	}
}

func (w *treRedisWriter) signal() {
	select {
	case w.kick <- struct{}{}:
	default:
	}
}

func (w *treRedisWriter) enqueueAck(routeKey string) {
	w.mu.Lock()
	w.pendingAcks[routeKey] = struct{}{}
	w.mu.Unlock()
	w.signal()
}

func (w *treRedisWriter) markInflight(pod string) {
	w.mu.Lock()
	w.dirtyInflight[pod] = struct{}{}
	w.mu.Unlock()
	w.signal()
}

// markAll schedules a full refresh: acks for every managed pod and inflight for every pod
// in the route table or with live counters.
func (w *treRedisWriter) markAll() {
	keys, pods := w.state.routeKeysAndPods()
	inflightPods := w.state.inflight.podsAndPrune(pods)
	w.mu.Lock()
	for _, k := range keys {
		w.pendingAcks[k] = struct{}{}
	}
	for p := range pods {
		w.dirtyInflight[p] = struct{}{}
	}
	for _, p := range inflightPods {
		w.dirtyInflight[p] = struct{}{}
	}
	w.mu.Unlock()
}

func treInflightKey(pod string) string { return TREGatewayInflightKeyPrefix + pod }
func treSeenKey(pod string) string     { return TREGatewaySeenKeyPrefix + pod }

type treSeenValue struct {
	Gen      int64 `json:"gen"`
	Routable bool  `json:"routable"`
	TS       int64 `json:"ts"`
}

type treInflightValue struct {
	Total          int64 `json:"total"`
	NonContinuable int64 `json:"non_continuable"`
	TS             int64 `json:"ts"`
}

// flush writes everything pending in one pipeline: inflight values first, then acks.
// On failure all of it is re-queued.
func (w *treRedisWriter) flush(ctx context.Context) error {
	w.mu.Lock()
	acks, dirty := w.pendingAcks, w.dirtyInflight
	w.pendingAcks, w.dirtyInflight = map[string]struct{}{}, map[string]struct{}{}
	w.mu.Unlock()
	if len(acks) == 0 && len(dirty) == 0 {
		return nil
	}

	// Snapshot route states first; the table is only ever changed under the write lock,
	// so every commit that read an older state has already incremented inflight.
	type ackItem struct {
		st treRouteState
	}
	ackItems := make([]ackItem, 0, len(acks))
	for key := range acks {
		st, ok := w.state.routeSnapshot(key)
		if !ok || !st.managed {
			continue // pod gone from the cache; its seen hash expires by TTL
		}
		ackItems = append(ackItems, ackItem{st: st})
		dirty[st.name] = struct{}{}
	}

	now := time.Now().UnixMilli()
	id := w.cfg.instanceID
	ttl := w.cfg.keyTTL
	_, err := w.client.Pipelined(ctx, func(p redis.Pipeliner) error {
		for pod := range dirty {
			c := w.state.inflight.snapshot(pod)
			b, _ := json.Marshal(treInflightValue{Total: c.Total, NonContinuable: c.NonContinuable, TS: now})
			p.HSet(ctx, treInflightKey(pod), id, string(b))
			p.Expire(ctx, treInflightKey(pod), ttl)
		}
		for _, a := range ackItems {
			b, _ := json.Marshal(treSeenValue{Gen: a.st.gen, Routable: a.st.routable, TS: now})
			p.HSet(ctx, treSeenKey(a.st.name), id, string(b))
			p.Expire(ctx, treSeenKey(a.st.name), ttl)
		}
		return nil
	})
	if err != nil {
		w.mu.Lock()
		for k := range acks {
			w.pendingAcks[k] = struct{}{}
		}
		for p := range dirty {
			w.dirtyInflight[p] = struct{}{}
		}
		w.mu.Unlock()
		return err
	}
	return nil
}

// resetOwnInflight removes this instance's fields from every inflight hash: counts of a
// previous process with the same instance id are void (its ext_proc streams are gone).
func (w *treRedisWriter) resetOwnInflight(ctx context.Context) error {
	var cursor uint64
	for {
		keys, next, err := w.client.Scan(ctx, cursor, TREGatewayInflightKeyPrefix+"*", 200).Result()
		if err != nil {
			return err
		}
		if len(keys) > 0 {
			if _, err := w.client.Pipelined(ctx, func(p redis.Pipeliner) error {
				for _, k := range keys {
					p.HDel(ctx, k, w.cfg.instanceID)
				}
				return nil
			}); err != nil {
				return err
			}
		}
		if next == 0 {
			return nil
		}
		cursor = next
	}
}

func (w *treRedisWriter) heartbeat(ctx context.Context) error {
	now := time.Now()
	_, err := w.client.Pipelined(ctx, func(p redis.Pipeliner) error {
		p.ZAdd(ctx, TREGatewayInstancesKey, redis.Z{Score: float64(now.UnixMilli()), Member: w.cfg.instanceID})
		cutoff := now.Add(-w.cfg.instanceRetention).UnixMilli()
		p.ZRemRangeByScore(ctx, TREGatewayInstancesKey, "-inf", fmt.Sprintf("(%d", cutoff))
		return nil
	})
	return err
}

func (w *treRedisWriter) writeLoop() {
	defer w.wg.Done()
	refresh := time.NewTicker(w.cfg.refresh)
	defer refresh.Stop()
	var retry <-chan time.Time
	for {
		select {
		case <-w.stop:
			return
		case <-w.kick:
		case <-retry:
			retry = nil
		case <-refresh.C:
			w.markAll()
		}
		ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
		err := w.flush(ctx)
		cancel()
		if err != nil {
			klog.ErrorS(err, "TRE gateway: writing inflight/seen to redis failed; will retry")
			retry = time.After(treGatewayFlushRetryBackoff)
		}
	}
}

func (w *treRedisWriter) heartbeatLoop() {
	defer w.wg.Done()
	t := time.NewTicker(w.cfg.heartbeat)
	defer t.Stop()
	for {
		ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
		if err := w.heartbeat(ctx); err != nil {
			klog.ErrorS(err, "TRE gateway: heartbeat failed")
		}
		cancel()
		select {
		case <-w.stop:
			return
		case <-t.C:
		}
	}
}

func (w *treRedisWriter) shutdown() {
	w.stopOnce.Do(func() {
		close(w.stop)
		w.wg.Wait()
		ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
		defer cancel()
		if err := w.client.ZRem(ctx, TREGatewayInstancesKey, w.cfg.instanceID).Err(); err != nil {
			klog.ErrorS(err, "TRE gateway: removing instance heartbeat failed")
		}
		if err := w.resetOwnInflight(ctx); err != nil {
			klog.ErrorS(err, "TRE gateway: clearing own inflight fields failed")
		}
	})
}

// ---------------------------------------------------------------------------------------
// lifecycle

// startTREGatewayCoordination wires the cache observer, seeds the route table, resets this
// instance's inflight fields, performs a first full refresh and then starts the heartbeat
// and the writer. Call before serving ext_proc traffic.
func startTREGatewayCoordination(client *redis.Client, lister cache.PodLister, cfg treGatewayConfig) (*treRedisWriter, error) {
	if client == nil {
		return nil, errors.New("no redis client")
	}
	g := treGW
	w := newTRERedisWriter(client, cfg, g)
	g.writer.Store(w)
	onChange := func(pod string) { w.markInflight(pod) }
	g.inflight.onChange.Store(&onChange)
	cache.SetTREPodObserver(g.observePod)
	g.seed(lister)

	// From here on commits check the route table. Taking the write lock once drains any
	// commit that started before the switch without the check.
	g.enabled.Store(true)
	g.barrier.Lock()
	g.barrier.Unlock() //nolint:staticcheck // empty critical section is the drain

	ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
	defer cancel()
	if err := w.resetOwnInflight(ctx); err != nil {
		klog.ErrorS(err, "TRE gateway: resetting own inflight fields failed")
	}
	w.markAll()
	if err := w.flush(ctx); err != nil {
		klog.ErrorS(err, "TRE gateway: initial seen/inflight refresh failed; the writer retries")
		w.signal()
	}
	w.wg.Add(2)
	go w.writeLoop()
	go w.heartbeatLoop()
	klog.InfoS("TRE gateway coordination started", "instance", cfg.instanceID,
		"heartbeat", cfg.heartbeat, "refresh", cfg.refresh, "keyTTL", cfg.keyTTL)
	return w, nil
}

// StartTRECoordination enables D3/D4 coordination when wanted (TRE_GW_COORDINATION, by
// default on iff TRE_ROUTABLE_LABEL_FILTER=true). Without the routable-label gate an ack
// would not describe routing, so coordination then refuses to start.
func (s *Server) StartTRECoordination() {
	if !treCoordinationWanted() {
		klog.InfoS("TRE gateway coordination disabled")
		return
	}
	if !utils.TRERoutableLabelFilterEnabled() {
		klog.ErrorS(nil, "TRE gateway coordination requires "+utils.TRERoutableLabelFilterEnv+"=true; not starting")
		return
	}
	lister, _ := s.cache.(cache.PodLister)
	w, err := startTREGatewayCoordination(s.redisClient, lister, loadTREGatewayConfig())
	if err != nil {
		klog.ErrorS(err, "TRE gateway coordination not started")
		return
	}
	s.treWriter = w
}

func (s *Server) stopTRECoordination() {
	if s.treWriter != nil {
		s.treWriter.shutdown()
	}
}
