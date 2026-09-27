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
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/go-redis/v9"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	"github.com/vllm-project/aibrix/pkg/cache"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P3-GW-007..013): gateway side of "transparent sleep"
// (plan 2026-09-27, decisions D3/D4/D5/D6/D10).
//
// Contract with the service-manager (SM), all keys in the TRE Redis:
//
//   tre:v2:gw:instances        ZSET  member=<instance id>  score=heartbeat ms (Redis TIME)
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
//     sees an inflight value at least as new as the ack. Per pod (same UID) the route
//     table never goes back to a lower route-gen, so a stale informer event cannot undo
//     a newer state.
//   - inflight: in-memory counts are authoritative; Redis is a write-through mirror by a
//     single writer goroutine (coalesced, never reordered). Each routed request is released
//     exactly once when its ext_proc stream ends (response end_of_stream, error, client
//     disconnect, stream close, shutdown) - never on the first usage-bearing chunk.
//   - pod removal: this instance's fields in seen/inflight of that pod are deleted, and
//     requests still counted against the old pod object no longer count (a same-name
//     replacement starts from zero).
//   - startup is fail-closed: coordination that is wanted but cannot start (label gate
//     off, no Redis / API client, Redis or apiserver unreachable within
//     TRE_GW_STARTUP_TIMEOUT) is an error and the process exits. The first heartbeat is
//     written only after a fresh (quorum) pod LIST, the reset of this instance's stale
//     inflight fields and a full ack/inflight refresh, and before any request is served.
//   - shutdown: new commits are refused (503 + Retry-After) before the instance leaves
//     the live set and clears its inflight fields.

const (
	TREGatewayInstancesKey      = "tre:v2:gw:instances"
	TREGatewaySeenKeyPrefix     = "tre:v2:gw:seen:"
	TREGatewayInflightKeyPrefix = "tre:v2:gw:inflight:"
	// TRERouteGenAnnotation is incremented by the SM in the same patch that changes the
	// routable label.
	TRERouteGenAnnotation = "tre.aibrix.io/route-gen"
	// HeaderTREExcludePod lists pod names (comma separated; the header may repeat) that
	// must not serve this request (the sidecar's retry of a request a sleeping pod refused).
	HeaderTREExcludePod = "x-tre-exclude-pod"

	envTREGatewayCoordination      = "TRE_GW_COORDINATION"
	envTREGatewayInstanceID        = "TRE_GW_INSTANCE_ID"
	envTREGatewayHeartbeatInterval = "TRE_GW_HEARTBEAT_INTERVAL"
	envTREGatewayInstanceRetention = "TRE_GW_INSTANCE_RETENTION"
	envTREGatewayKeyTTL            = "TRE_GW_KEY_TTL"
	envTREGatewayRefreshInterval   = "TRE_GW_REFRESH_INTERVAL"
	envTREGatewayRetryAfterSeconds = "TRE_GW_RETRY_AFTER_SECONDS"
	envTREGatewayStartupTimeout    = "TRE_GW_STARTUP_TIMEOUT"
	envTREGatewayRedisRetryMax     = "TRE_GW_REDIS_RETRY_MAX"
	envTREDefaultRoutingStrategy   = "TRE_DEFAULT_ROUTING_STRATEGY"

	defaultTREGatewayHeartbeatInterval = 2 * time.Second
	defaultTREGatewayInstanceRetention = 10 * time.Minute
	defaultTREGatewayKeyTTL            = 300 * time.Second
	defaultTREGatewayRefreshInterval   = 30 * time.Second
	defaultTREGatewayRetryAfterSeconds = 1
	defaultTREGatewayStartupTimeout    = 60 * time.Second
	defaultTREGatewayRedisRetryBase    = 250 * time.Millisecond
	defaultTREGatewayRedisRetryMax     = 10 * time.Second

	treGatewayRedisTimeout = 5 * time.Second
	// treCommitAttempts bounds re-routing when the chosen pod turned unroutable between
	// candidate listing and commit (a hide raced the request).
	treCommitAttempts = 3
	// treFreshListPageSize: page size of the startup quorum LIST.
	treFreshListPageSize = 500
)

var (
	errTREAllCandidatesExcluded = errors.New("all routable pods are excluded by " + HeaderTREExcludePod)
	errTRECommitRace            = errors.New("routable pods changed during routing; retry")
	errTRENotRoutable           = errors.New("target pod is not routable in the acked route table")
	errTRENoTarget              = errors.New("router selected no target pod; TRE coordination requires pod-level routing")
	errTREShuttingDown          = errors.New("gateway instance is shutting down; retry")
	errTREUnsupportedRouter     = errors.New("routing strategy is not supported while TRE gateway coordination is on")
)

var treRedisErrors = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "tre_gateway_redis_errors_total",
	Help: "Failed Redis operations of the TRE gateway coordination (op=flush|heartbeat|startup).",
}, []string{"op"})

// ---------------------------------------------------------------------------------------
// configuration

type treGatewayConfig struct {
	instanceID        string
	heartbeat         time.Duration
	instanceRetention time.Duration
	keyTTL            time.Duration
	refresh           time.Duration
	// startupTimeout bounds the retries of every startup step (fail-closed after it).
	startupTimeout time.Duration
	// retryBase/retryMax: exponential backoff of failed Redis writes.
	retryBase time.Duration
	retryMax  time.Duration
	// freshPods lists pods with a quorum read from the apiserver (nil: skip; tests).
	freshPods func(ctx context.Context) ([]*v1.Pod, error)
}

func loadTREGatewayConfig() treGatewayConfig {
	return treGatewayConfig{
		instanceID:        treGatewayInstanceID(),
		heartbeat:         envDuration(envTREGatewayHeartbeatInterval, defaultTREGatewayHeartbeatInterval),
		instanceRetention: envDuration(envTREGatewayInstanceRetention, defaultTREGatewayInstanceRetention),
		keyTTL:            envDuration(envTREGatewayKeyTTL, defaultTREGatewayKeyTTL),
		refresh:           envDuration(envTREGatewayRefreshInterval, defaultTREGatewayRefreshInterval),
		startupTimeout:    envDuration(envTREGatewayStartupTimeout, defaultTREGatewayStartupTimeout),
		retryBase:         defaultTREGatewayRedisRetryBase,
		retryMax:          envDuration(envTREGatewayRedisRetryMax, defaultTREGatewayRedisRetryMax),
	}
}

func (c treGatewayConfig) withDefaults() treGatewayConfig {
	if c.startupTimeout <= 0 {
		c.startupTimeout = defaultTREGatewayStartupTimeout
	}
	if c.retryBase <= 0 {
		c.retryBase = defaultTREGatewayRedisRetryBase
	}
	if c.retryMax < c.retryBase {
		c.retryMax = c.retryBase
		if defaultTREGatewayRedisRetryMax > c.retryMax {
			c.retryMax = defaultTREGatewayRedisRetryMax
		}
	}
	return c
}

// treBackoff is the delay before retry number n (n >= 1): base, 2*base, ... capped at max.
func treBackoff(n int, base, max time.Duration) time.Duration {
	d := base
	for i := 1; i < n && d < max; i++ {
		d *= 2
	}
	if d > max {
		d = max
	}
	return d
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

// treDefaultRoutingStrategy is an optional last-resort strategy for requests that name
// none (no routing-strategy header, no config profile, no ROUTING_ALGORITHM). Off by
// default: deployments should set the upstream ROUTING_ALGORITHM, which older plugin
// images honour as well. Independently of it, with coordination on a request that
// resolves to no strategy is still routed per pod (see handleRequestBody).
var treDefaultRoutingStrategy = loadTREDefaultRoutingStrategy()

func loadTREDefaultRoutingStrategy() string {
	v := strings.TrimSpace(os.Getenv(envTREDefaultRoutingStrategy))
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

// mergeTREExcludeHeader appends one more x-tre-exclude-pod value (the header may appear
// several times; every occurrence counts).
func mergeTREExcludeHeader(prev, next string) string {
	next = strings.TrimSpace(next)
	switch {
	case strings.TrimSpace(prev) == "":
		return next
	case next == "":
		return prev
	}
	return prev + "," + next
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
	FunctionCall   json.RawMessage `json:"function_call"`
	// structured output / guided decoding: the grammar state at the seam is not
	// reconstructible from the emitted tokens by a plain continuation request.
	ResponseFormat    json.RawMessage `json:"response_format"`
	GuidedJSON        json.RawMessage `json:"guided_json"`
	GuidedRegex       json.RawMessage `json:"guided_regex"`
	GuidedChoice      json.RawMessage `json:"guided_choice"`
	GuidedGrammar     json.RawMessage `json:"guided_grammar"`
	GuidedJSONObject  json.RawMessage `json:"guided_json_object"`
	StructuralTag     json.RawMessage `json:"structural_tag"`
	StructuredOutputs json.RawMessage `json:"structured_outputs"`
}

// treNonContinuable reports whether a request cannot be continued after an abort and must
// therefore be drained: n>1, best_of>1, any logprobs output, echo, beam search, tool or
// function calling (streaming or not), structured output / guided decoding, and every
// non-generation endpoint (only completions and chat completions have a continuation
// path). Unparseable fields count as non-continuable.
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
	if treToolCallsPossible(f) {
		return true
	}
	return treStructuredOutput(f)
}

// treToolCallsPossible: tools/functions offered and not switched off with "none", or a
// tool/function choice other than none/auto (which forces a call).
func treToolCallsPossible(f treSamplingFields) bool {
	offered := rawNonEmptyArray(f.Tools) || rawNonEmptyArray(f.Functions)
	choice := f.ToolChoice
	if !rawPresent(choice) {
		choice = f.FunctionCall
	}
	if rawIs(choice, `"none"`) {
		return false
	}
	if offered {
		return true
	}
	return rawPresent(choice) && !rawIs(choice, `"auto"`)
}

func treStructuredOutput(f treSamplingFields) bool {
	if rawPresent(f.ResponseFormat) {
		var rf struct {
			Type string `json:"type"`
		}
		if err := sonic.Unmarshal(f.ResponseFormat, &rf); err != nil || (rf.Type != "" && rf.Type != "text") {
			return true
		}
	}
	for _, r := range []json.RawMessage{f.GuidedJSON, f.GuidedRegex, f.GuidedChoice, f.GuidedGrammar, f.StructuralTag, f.StructuredOutputs} {
		if rawPresent(r) {
			return true
		}
	}
	return rawPresent(f.GuidedJSONObject) && !rawIs(f.GuidedJSONObject, "false")
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

// treInflightTracker counts routed requests per pod name. An entry exists only while it
// is non-zero, so pods that come and go leave nothing behind (with or without
// coordination). Dropping a pod's entry (pod deleted) detaches the tickets still counted
// against it: releasing them later does not touch a same-name successor.
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
	entry          *treInflightCounts
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
	return &treInflightTicket{tracker: t, pod: pod, entry: c, nonContinuable: nonContinuable}
}

// Release decrements the pod's counters exactly once.
func (k *treInflightTicket) Release() {
	if k == nil || k.tracker == nil || !k.released.CompareAndSwap(false, true) {
		return
	}
	t := k.tracker
	t.mu.Lock()
	c := t.counts[k.pod]
	if c == nil || c != k.entry {
		t.mu.Unlock() // the pod was removed meanwhile; this request no longer counts
		return
	}
	if c.Total > 0 {
		c.Total--
	}
	if k.nonContinuable && c.NonContinuable > 0 {
		c.NonContinuable--
	}
	if c.Total == 0 && c.NonContinuable == 0 {
		delete(t.counts, k.pod)
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

func (t *treInflightTracker) has(pod string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.counts[pod] != nil
}

// drop forgets a removed pod's counters (outstanding tickets become no-ops).
func (t *treInflightTracker) drop(pod string) {
	t.mu.Lock()
	delete(t.counts, pod)
	t.mu.Unlock()
}

func (t *treInflightTracker) size() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.counts)
}

// pods returns every pod with live counters.
func (t *treInflightTracker) pods() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]string, 0, len(t.counts))
	for pod := range t.counts {
		out = append(out, pod)
	}
	return out
}

// ---------------------------------------------------------------------------------------
// route table (D3)

type treRouteState struct {
	name      string
	namespace string
	uid       k8stypes.UID
	created   time.Time
	gen       int64
	routable  bool
	managed   bool // carries the routable label or the route-gen annotation
}

func treRouteStateFromPod(pod *v1.Pod) treRouteState {
	st := treRouteState{name: pod.Name, namespace: pod.Namespace, uid: pod.UID, created: pod.CreationTimestamp.Time}
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

// treRouteSupersedes reports whether st may replace prev: for the same pod object the
// route-gen never goes back (a stale informer or listing cannot undo a newer state); a
// different object with the same name replaces an older one (by creation time).
func treRouteSupersedes(prev, st treRouteState) bool {
	if prev.uid != "" && st.uid != "" && prev.uid != st.uid {
		return !st.created.Before(prev.created)
	}
	return st.gen >= prev.gen
}

type treGatewayState struct {
	// enabled: the route table is authoritative for routing commits (coordination on).
	enabled atomic.Bool
	// closing: graceful shutdown started; commits are refused.
	closing atomic.Bool
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
		if oldPod != nil {
			g.removePod(oldPod)
		}
		return
	}
	st := treRouteStateFromPod(newPod)
	key := utils.GeneratePodKey(newPod.Namespace, newPod.Name)
	g.barrier.Lock()
	prev, existed := g.routes[key]
	if existed && !treRouteSupersedes(prev, st) {
		g.barrier.Unlock()
		klog.V(4).InfoS("TRE gateway: ignoring stale pod state", "pod", key, "gen", st.gen, "tableGen", prev.gen)
		return
	}
	g.routes[key] = st
	g.barrier.Unlock()

	replaced := existed && prev.uid != "" && st.uid != "" && prev.uid != st.uid
	if replaced {
		// Same name, new object (the delete was not observed): the old object's requests
		// and acks do not describe this pod.
		g.forgetPod(st.name)
	}
	if st.managed && (!existed || replaced || prev.gen != st.gen || prev.routable != st.routable || !prev.managed) {
		if w := g.writer.Load(); w != nil {
			w.enqueueAck(key)
		}
	}
}

// removePod drops a deleted pod from the route table and forgets its counters and this
// instance's Redis fields.
func (g *treGatewayState) removePod(pod *v1.Pod) {
	key := utils.GeneratePodKey(pod.Namespace, pod.Name)
	g.barrier.Lock()
	prev, ok := g.routes[key]
	if ok && pod.UID != "" && prev.uid != "" && prev.uid != pod.UID {
		g.barrier.Unlock() // removal of an older object; the table already holds a newer one
		return
	}
	delete(g.routes, key)
	g.barrier.Unlock()
	g.forgetPod(pod.Name)
}

func (g *treGatewayState) forgetPod(name string) {
	g.inflight.drop(name)
	if w := g.writer.Load(); w != nil {
		w.enqueueForget(name)
	}
}

// seed merges pods (a cache listing or a fresh apiserver LIST) into the route table under
// the supersede rule, and returns the keys whose state changed.
func (g *treGatewayState) seed(pods []*v1.Pod) []string {
	g.barrier.Lock()
	defer g.barrier.Unlock()
	var keys []string
	for _, pod := range pods {
		if pod == nil {
			continue
		}
		key := utils.GeneratePodKey(pod.Namespace, pod.Name)
		st := treRouteStateFromPod(pod)
		if prev, ok := g.routes[key]; ok && (!treRouteSupersedes(prev, st) || prev == st) {
			continue
		}
		g.routes[key] = st
		keys = append(keys, key)
	}
	return keys
}

// routableInTableLocked is the commit-time check (caller holds barrier.RLock).
func (g *treGatewayState) routableInTableLocked(pod *v1.Pod) bool {
	st, ok := g.routes[utils.GeneratePodKey(pod.Namespace, pod.Name)]
	if !ok || !st.routable {
		return false
	}
	return pod.UID == "" || st.uid == "" || pod.UID == st.uid
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

// commit counts the request against pod. With coordination on it is fail-closed: the pod
// must still be routable in the route table (checked atomically with the increment with
// respect to route-table updates, which is what makes the seen-gen ack truthful), a nil
// target is refused, and nothing is committed once shutdown has begun.
func (g *treGatewayState) commit(pod *v1.Pod, nonContinuable bool) (*treInflightTicket, error) {
	g.barrier.RLock()
	defer g.barrier.RUnlock()
	if g.enabled.Load() {
		switch {
		case g.closing.Load():
			return nil, errTREShuttingDown
		case pod == nil:
			return nil, errTRENoTarget
		case !g.routableInTableLocked(pod):
			return nil, errTRENotRoutable
		}
	} else if pod == nil {
		return nil, nil
	}
	return g.inflight.acquire(pod.Name, nonContinuable), nil
}

// beginShutdown refuses new commits; commits already past the check are drained by the
// write lock, so after it returns no new request is counted.
func (g *treGatewayState) beginShutdown() {
	g.barrier.Lock()
	g.closing.Store(true)
	g.barrier.Unlock()
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
	forget        map[string]struct{} // pod names whose own fields must be deleted
	kick          chan struct{}

	// clockOffsetMs = Redis TIME - local clock, measured at every heartbeat; all "ts"
	// values use the Redis clock so readers on other hosts can compare them.
	clockOffsetMs atomic.Int64

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
		forget:        map[string]struct{}{},
		kick:          make(chan struct{}, 1),
		stop:          make(chan struct{}),
	}
}

func (w *treRedisWriter) nowMs() int64 { return time.Now().UnixMilli() + w.clockOffsetMs.Load() }

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

func (w *treRedisWriter) enqueueForget(pod string) {
	w.mu.Lock()
	w.forget[pod] = struct{}{}
	w.mu.Unlock()
	w.signal()
}

// markAll schedules a full refresh: acks for every managed pod and inflight for every pod
// in the route table or with live counters.
func (w *treRedisWriter) markAll() {
	keys, pods := w.state.routeKeysAndPods()
	inflightPods := w.state.inflight.pods()
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

// flush writes everything pending in one pipeline: removals of deleted pods first, then
// inflight values, then acks. On failure all of it is re-queued.
func (w *treRedisWriter) flush(ctx context.Context) error {
	w.mu.Lock()
	acks, dirty, forget := w.pendingAcks, w.dirtyInflight, w.forget
	w.pendingAcks, w.dirtyInflight, w.forget = map[string]struct{}{}, map[string]struct{}{}, map[string]struct{}{}
	w.mu.Unlock()
	if len(acks) == 0 && len(dirty) == 0 && len(forget) == 0 {
		return nil
	}

	// Snapshot route states first; the table is only ever changed under the write lock,
	// so every commit that read an older state has already incremented inflight.
	ackStates := make([]treRouteState, 0, len(acks))
	for key := range acks {
		st, ok := w.state.routeSnapshot(key)
		if !ok || !st.managed {
			continue // pod gone from the table
		}
		ackStates = append(ackStates, st)
		dirty[st.name] = struct{}{}
	}
	_, known := w.state.routeKeysAndPods()

	now := w.nowMs()
	id := w.cfg.instanceID
	ttl := w.cfg.keyTTL
	_, err := w.client.Pipelined(ctx, func(p redis.Pipeliner) error {
		for pod := range forget {
			p.HDel(ctx, treSeenKey(pod), id)
			p.HDel(ctx, treInflightKey(pod), id)
		}
		for pod := range dirty {
			if _, ok := known[pod]; !ok && !w.state.inflight.has(pod) {
				continue // removed pod: do not recreate the field just deleted
			}
			c := w.state.inflight.snapshot(pod)
			b, _ := json.Marshal(treInflightValue{Total: c.Total, NonContinuable: c.NonContinuable, TS: now})
			p.HSet(ctx, treInflightKey(pod), id, string(b))
			p.Expire(ctx, treInflightKey(pod), ttl)
		}
		for _, st := range ackStates {
			b, _ := json.Marshal(treSeenValue{Gen: st.gen, Routable: st.routable, TS: now})
			p.HSet(ctx, treSeenKey(st.name), id, string(b))
			p.Expire(ctx, treSeenKey(st.name), ttl)
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
		for p := range forget {
			w.forget[p] = struct{}{}
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

// heartbeat stamps this instance with the Redis server clock (TIME), so liveness does not
// depend on the gateway host's clock, and prunes long-dead members.
func (w *treRedisWriter) heartbeat(ctx context.Context) error {
	before := time.Now()
	rt, err := w.client.Time(ctx).Result()
	if err != nil {
		return err
	}
	after := time.Now()
	w.clockOffsetMs.Store(rt.UnixMilli() - before.Add(after.Sub(before)/2).UnixMilli())
	nowMs := rt.UnixMilli()
	_, err = w.client.Pipelined(ctx, func(p redis.Pipeliner) error {
		p.ZAdd(ctx, TREGatewayInstancesKey, redis.Z{Score: float64(nowMs), Member: w.cfg.instanceID})
		cutoff := nowMs - w.cfg.instanceRetention.Milliseconds()
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
	failures := 0
	for {
		select {
		case <-w.stop:
			return
		case <-w.kick:
			if retry != nil {
				continue // backing off: the change stays pending for the retry
			}
		case <-retry:
			retry = nil
		case <-refresh.C:
			w.markAll()
			if retry != nil {
				continue
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
		err := w.flush(ctx)
		cancel()
		if err != nil {
			failures++
			treRedisErrors.WithLabelValues("flush").Inc()
			d := treBackoff(failures, w.cfg.retryBase, w.cfg.retryMax)
			if failures == 1 || failures%10 == 0 || d == w.cfg.retryMax && failures%3 == 0 {
				klog.ErrorS(err, "TRE gateway: writing inflight/seen to redis failed; backing off",
					"consecutiveFailures", failures, "retryIn", d)
			}
			retry = time.After(d)
			continue
		}
		if failures > 0 {
			klog.InfoS("TRE gateway: redis writes recovered", "afterFailures", failures)
			failures = 0
		}
	}
}

func (w *treRedisWriter) heartbeatLoop() {
	defer w.wg.Done()
	t := time.NewTicker(w.cfg.heartbeat)
	defer t.Stop()
	failures := 0
	for {
		select {
		case <-w.stop:
			return
		case <-t.C:
		}
		ctx, cancel := context.WithTimeout(context.Background(), treGatewayRedisTimeout)
		err := w.heartbeat(ctx)
		cancel()
		if err != nil {
			failures++
			treRedisErrors.WithLabelValues("heartbeat").Inc()
			if failures == 1 || failures%10 == 0 {
				klog.ErrorS(err, "TRE gateway: heartbeat failed", "consecutiveFailures", failures)
			}
			continue
		}
		if failures > 0 {
			klog.InfoS("TRE gateway: heartbeat recovered", "afterFailures", failures)
			failures = 0
		}
	}
}

func (w *treRedisWriter) shutdown() {
	w.stopOnce.Do(func() {
		// No request can be committed from here on (callers begin shutdown first), so the
		// cleared state below cannot be undercut by a late increment.
		w.state.beginShutdown()
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

// treRetryUntil runs op until it succeeds or ctx expires, with exponential backoff.
func treRetryUntil(ctx context.Context, cfg treGatewayConfig, what string, op func(context.Context) error) error {
	for attempt := 1; ; attempt++ {
		actx, cancel := context.WithTimeout(ctx, treGatewayRedisTimeout)
		err := op(actx)
		cancel()
		if err == nil {
			return nil
		}
		treRedisErrors.WithLabelValues("startup").Inc()
		d := treBackoff(attempt, cfg.retryBase, cfg.retryMax)
		klog.ErrorS(err, "TRE gateway: startup step failed; retrying", "step", what, "attempt", attempt, "retryIn", d)
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s: %w (gave up after %d attempts)", what, err, attempt)
		case <-time.After(d):
		}
	}
}

// treFreshPodLister lists TRE-managed pods with a quorum read (ResourceVersion ""), i.e.
// from etcd rather than the apiserver watch cache, so a restarted instance cannot ack a
// state older than what its previous incarnation acked.
func treFreshPodLister(client kubernetes.Interface) func(ctx context.Context) ([]*v1.Pod, error) {
	return func(ctx context.Context) ([]*v1.Pod, error) {
		opts := metav1.ListOptions{LabelSelector: utils.TRERoutableLabel, ResourceVersion: "", Limit: treFreshListPageSize}
		var out []*v1.Pod
		for {
			list, err := client.CoreV1().Pods(metav1.NamespaceAll).List(ctx, opts)
			if err != nil {
				return nil, err
			}
			for i := range list.Items {
				out = append(out, &list.Items[i])
			}
			if list.Continue == "" {
				return out, nil
			}
			opts.Continue = list.Continue
		}
	}
}

// startTREGatewayCoordination wires the cache observer, seeds the route table (cache
// listing, then a fresh apiserver LIST), resets this instance's inflight fields, performs
// a first full refresh and a first heartbeat, then starts the heartbeat and writer loops.
// Every step is retried up to cfg.startupTimeout; on failure everything is unwound and an
// error returned (the caller must not serve). Call before serving ext_proc traffic.
func startTREGatewayCoordination(client *redis.Client, lister cache.PodLister, cfg treGatewayConfig) (w *treRedisWriter, err error) {
	if client == nil {
		return nil, errors.New("no redis client")
	}
	cfg = cfg.withDefaults()
	g := treGW
	w = newTRERedisWriter(client, cfg, g)
	g.writer.Store(w)
	onChange := func(pod string) { w.markInflight(pod) }
	g.inflight.onChange.Store(&onChange)
	cache.SetTREPodObserver(g.observePod)
	defer func() {
		if err != nil {
			g.enabled.Store(false)
			cache.SetTREPodObserver(nil)
			g.inflight.onChange.Store(nil)
			g.writer.Store(nil)
			w = nil
		}
	}()
	if lister != nil {
		g.seed(lister.ListPods())
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.startupTimeout)
	defer cancel()
	if cfg.freshPods != nil {
		if err = treRetryUntil(ctx, cfg, "fresh pod list", func(c context.Context) error {
			pods, lerr := cfg.freshPods(c)
			if lerr != nil {
				return lerr
			}
			g.seed(pods)
			return nil
		}); err != nil {
			return nil, err
		}
	}

	// From here on commits check the route table. Taking the write lock once drains any
	// commit that started before the switch without the check.
	g.enabled.Store(true)
	g.barrier.Lock()
	g.barrier.Unlock() //nolint:staticcheck // empty critical section is the drain

	if err = treRetryUntil(ctx, cfg, "reset own inflight fields", w.resetOwnInflight); err != nil {
		return nil, err
	}
	w.markAll()
	if err = treRetryUntil(ctx, cfg, "initial seen/inflight refresh", w.flush); err != nil {
		return nil, err
	}
	// Live only now: the SM waits for this instance's acks from its first heartbeat on.
	if err = treRetryUntil(ctx, cfg, "first heartbeat", w.heartbeat); err != nil {
		return nil, err
	}
	w.wg.Add(2)
	go w.writeLoop()
	go w.heartbeatLoop()
	w.signal() // anything that changed during startup
	klog.InfoS("TRE gateway coordination started", "instance", cfg.instanceID,
		"heartbeat", cfg.heartbeat, "refresh", cfg.refresh, "keyTTL", cfg.keyTTL)
	return w, nil
}

// StartTRECoordination enables D3/D4 coordination when wanted (TRE_GW_COORDINATION, by
// default on iff TRE_ROUTABLE_LABEL_FILTER=true). It is fail-closed: when coordination is
// wanted but cannot run, it returns an error and the process must not serve (an instance
// that routes without heartbeat/acks would be invisible to the service-manager).
func (s *Server) StartTRECoordination() error {
	if !treCoordinationWanted() {
		klog.InfoS("TRE gateway coordination disabled")
		return nil
	}
	if !utils.TRERoutableLabelFilterEnabled() {
		return fmt.Errorf("%s=true requires %s=true (acks would not describe routing)", envTREGatewayCoordination, utils.TRERoutableLabelFilterEnv)
	}
	if s.redisClient == nil {
		return errors.New("TRE gateway coordination requires Redis (REDIS_HOST)")
	}
	if s.client == nil {
		return errors.New("TRE gateway coordination requires the Kubernetes API (not available in standalone mode)")
	}
	lister, _ := s.cache.(cache.PodLister)
	cfg := loadTREGatewayConfig()
	cfg.freshPods = treFreshPodLister(s.client)
	w, err := startTREGatewayCoordination(s.redisClient, lister, cfg)
	if err != nil {
		return fmt.Errorf("TRE gateway coordination not started: %w", err)
	}
	s.treWriter = w
	return nil
}

func (s *Server) stopTRECoordination() {
	if s.treWriter != nil {
		s.treWriter.shutdown()
	}
}

// treCheckRouterSupported: with coordination on, only routers that pick the target pod
// synchronously from the candidate list this request filtered (routable, exclusions) are
// allowed. Queue-based routers (slo*) route from another request's candidate list, and PD
// routing sends a prefill request to a second pod that would not be counted in inflight.
func treCheckRouterSupported(algorithm types.RoutingAlgorithm, router types.Router) error {
	if !treGW.enabled.Load() {
		return nil
	}
	if algorithm == routing.RouterPD {
		return fmt.Errorf("%w: %s (prefill pod is not tracked)", errTREUnsupportedRouter, algorithm)
	}
	if _, ok := router.(types.QueueRouter); ok {
		return fmt.Errorf("%w: %s (queue router)", errTREUnsupportedRouter, algorithm)
	}
	return nil
}
