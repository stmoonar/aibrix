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
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/aibrix/pkg/cache"
	routingalgorithms "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P3-GW-012/013) tests for the 2026-09-27 review findings.

func treUIDPod(name, ip, routable string, gen int, uid string, created time.Time) *v1.Pod {
	p := treGenPod(name, ip, routable, gen)
	p.UID = k8stypes.UID(uid)
	p.CreationTimestamp = metav1.NewTime(created)
	return p
}

func treRespBody(body string, eos bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
		ResponseBody: &extProcPb.HttpBody{Body: []byte(body), EndOfStream: eos},
	}}
}

func treResp200() *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: ":status", RawValue: []byte("200")}}}},
	}}
}

func treSSE(text, finish string, completion int) string {
	fr := "null"
	if finish != "" {
		fr = strconv.Quote(finish)
	}
	return fmt.Sprintf(`data: {"id":"c","model":"m","choices":[{"index":0,"text":%q,"finish_reason":%s}],"usage":{"prompt_tokens":3,"completion_tokens":%d,"total_tokens":%d}}`+"\n\n",
		text, fr, completion, 3+completion)
}

// P1-1: with continuous_usage_stats every chunk has usage; the inflight slot must be held
// until the response body's end_of_stream, and the ext_proc stream must not end early.
func TestTREStream_ContinuousUsageHoldsInflightUntilEndOfStream(t *testing.T) {
	for _, coordinated := range []bool{true, false} {
		t.Run(fmt.Sprintf("coordination=%v", coordinated), func(t *testing.T) {
			withTREGates(t, true, false)
			g := resetTREGateway(t)
			algo := registerTRENamedRouter(t, fmt.Sprintf("tre-test-cont-usage-%v", coordinated), &treNamedRouter{order: []string{"p"}})
			store := treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
			if coordinated {
				_, client := newTREMiniRedis(t)
				_, err := startTREGatewayCoordination(client, store.(cache.PodLister), testTREConfig())
				require.NoError(t, err)
			}
			s := newProcessTestServer(openShutdownCh(), nil)
			s.cache = store

			srv := &mockProcessServer{ctx: context.Background()}
			srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
			srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi","max_tokens":4,"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}`), nil).Once()
			srv.On("Recv").Return(treResp200(), nil).Once()
			chunks := []*extProcPb.ProcessingRequest{
				treRespBody(treSSE("a", "", 1), false),
				treRespBody(treSSE("b", "", 2), false),
				treRespBody(treSSE("c", "", 3), false),
				treRespBody(treSSE("d", "length", 4), false),
				treRespBody(`data: {"id":"c","model":"m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`+"\n\n", false),
				treRespBody("data: [DONE]\n\n", true),
			}
			for _, c := range chunks {
				srv.On("Recv").Return(c, nil).Once()
			}
			var during []int64
			srv.On("Send", mock.Anything).Run(func(args mock.Arguments) {
				if args.Get(0).(*extProcPb.ProcessingResponse).GetResponseBody() != nil {
					during = append(during, g.inflight.snapshot("p").Total)
				}
			}).Return(nil)

			require.NoError(t, s.Process(srv))

			processed := len(chunks) // coordinated: every chunk up to end_of_stream
			if !coordinated {
				// Upstream behaviour (stream ends at the final usage) is kept; only the
				// accounting fix applies: intermediate usage no longer completes it.
				processed = 4
			}
			srv.AssertNumberOfCalls(t, "Recv", 3+processed)
			require.Len(t, during, processed)
			for i, n := range during {
				assert.Equal(t, int64(1), n, "chunk %d: request still in flight", i)
			}
			assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"), "released at the end")
		})
	}
}

func TestTREFinalStreamUsage(t *testing.T) {
	assert.False(t, treFinalStreamUsage([]byte(`{"choices":[{"text":"a","finish_reason":null}],"usage":{}}`)))
	assert.False(t, treFinalStreamUsage([]byte(`{"choices":[{"delta":{"content":"a"}}],"usage":{}}`)))
	assert.True(t, treFinalStreamUsage([]byte(`{"choices":[{"text":"a","finish_reason":"stop"}],"usage":{}}`)))
	assert.True(t, treFinalStreamUsage([]byte(`{"choices":[],"usage":{}}`)))
	assert.True(t, treFinalStreamUsage([]byte(`{"usage":{}}`)))
	assert.True(t, treFinalStreamUsage([]byte(`{"type":"response.completed","response":{"usage":{}}}`)))
}

// P3: the exclude header may repeat; every occurrence is honoured.
func TestTREExclude_RepeatedHeadersAreMerged(t *testing.T) {
	withTREGates(t, true, false)
	resetTREGateway(t)
	_, _, _, rc := (&Server{}).HandleRequestHeaders(context.Background(), "req",
		treHeadersReq(PathCompletions, HeaderTREExcludePod, "a", "X-TRE-Exclude-Pod", "b, c", HeaderTREExcludePod, ""))
	assert.Equal(t, "a,b, c", rc.ReqHeaders[HeaderTREExcludePod])
	assert.Equal(t, map[string]struct{}{"a": {}, "b": {}, "c": {}}, treExcludedPods(rc.ReqHeaders))
	assert.Equal(t, "x", mergeTREExcludeHeader("", " x "))
	assert.Equal(t, "x", mergeTREExcludeHeader("x", " "))
}

// P3: pod deletion removes this instance's seen/inflight fields and the in-memory
// counters; a same-name replacement starts clean, and stale tickets of the old pod or
// stale events cannot touch it.
func TestTREPodDelete_ForgetsStateAndSameNameRecreateStartsClean(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	ctx := context.Background()
	t0 := time.Now().Add(-time.Hour)
	old := treUIDPod("p", "10.0.0.1", "true", 5, "uid-1", t0)
	w, err := startTREGatewayCoordination(client, treStaticLister{old}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()
	require.NoError(t, client.HSet(ctx, treSeenKey("p"), "gw-other", `{"gen":5,"routable":true,"ts":1}`).Err())
	require.NoError(t, client.HSet(ctx, treInflightKey("p"), "gw-other", `{"total":3,"non_continuable":0,"ts":1}`).Err())

	stale, err := g.commit(old, true)
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		v, ok := readInflight(t, client, "p", "gw-test-0")
		return ok && v.Total == 1
	}, 2*time.Second, 5*time.Millisecond)

	g.observePod(old, nil) // deleted
	require.Eventually(t, func() bool {
		_, seenOK := readSeen(t, client, "p", "gw-test-0")
		_, inOK := readInflight(t, client, "p", "gw-test-0")
		return !seenOK && !inOK
	}, 2*time.Second, 5*time.Millisecond)
	assert.Equal(t, 0, g.inflight.size(), "no in-memory leftovers")
	_, ok := readSeen(t, client, "p", "gw-other")
	assert.True(t, ok, "other instances' fields are untouched")
	_, err = g.commit(old, false)
	assert.ErrorIs(t, err, errTRENotRoutable, "a deleted pod is not routable")

	stale.Release() // the old pod's request ends late
	time.Sleep(50 * time.Millisecond)
	_, inOK := readInflight(t, client, "p", "gw-test-0")
	assert.False(t, inOK, "a late release does not resurrect the field")

	fresh := treUIDPod("p", "10.0.0.9", "true", 1, "uid-2", t0.Add(time.Minute))
	g.observePod(nil, fresh)
	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 1 && v.Routable
	}, 2*time.Second, 5*time.Millisecond)
	k, err := g.commit(fresh, false)
	require.NoError(t, err)
	stale.Release()
	assert.Equal(t, treInflightCounts{Total: 1}, g.inflight.snapshot("p"))
	_, err = g.commit(old, false)
	assert.ErrorIs(t, err, errTRENotRoutable, "the old object (other UID) is not the routable one")

	// A late event of the old object (older creation) does not replace the new one, and a
	// lower route-gen of the same object is ignored.
	g.observePod(nil, treUIDPod("p", "10.0.0.1", "true", 9, "uid-1", t0))
	g.observePod(nil, treUIDPod("p", "10.0.0.9", "false", 0, "uid-2", t0.Add(time.Minute)))
	st, _ := g.routeSnapshot("default/p")
	assert.Equal(t, k8stypes.UID("uid-2"), st.uid)
	assert.Equal(t, int64(1), st.gen)
	assert.True(t, st.routable)

	k.Release()
	assert.Equal(t, 0, g.inflight.size())
	require.Eventually(t, func() bool {
		v, ok := readInflight(t, client, "p", "gw-test-0")
		return ok && v.Total == 0
	}, 2*time.Second, 5*time.Millisecond)
}

// P3: without coordination the tracker leaves nothing behind either.
func TestTREInflight_NoLeakWithoutCoordination(t *testing.T) {
	g := resetTREGateway(t)
	for i := 0; i < 100; i++ {
		k, err := g.commit(treTestPod(fmt.Sprintf("p-%d", i), "10.0.0.1", "true"), i%2 == 0)
		require.NoError(t, err)
		k.Release()
	}
	assert.Equal(t, 0, g.inflight.size())
	k, err := g.commit(nil, false)
	assert.NoError(t, err)
	assert.Nil(t, k, "without coordination a request without a target is simply not counted")
}

// P1-2: with coordination on a commit without a target pod is refused.
func TestTRECommit_NilTargetRejectedWhenCoordinated(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	w, err := startTREGatewayCoordination(client, treStaticLister{}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()
	k, err := g.commit(nil, false)
	assert.Nil(t, k)
	assert.ErrorIs(t, err, errTRENoTarget)
}

// Concurrent commits/releases, route changes, deletes and flushes: race-free, counters
// never go negative, a hidden pod is never committed, and Redis converges to memory.
func TestTRECoordination_ConcurrentCommitObserveFlushInvariants(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	cfg := testTREConfig()
	cfg.refresh = 7 * time.Millisecond
	const nPods = 4
	pods := make([]*v1.Pod, nPods)
	for i := range pods {
		pods[i] = treGenPod(fmt.Sprintf("p%d", i), fmt.Sprintf("10.0.0.%d", i+1), "true", 1)
	}
	w, err := startTREGatewayCoordination(client, treStaticLister(pods), cfg)
	require.NoError(t, err)
	defer w.shutdown()

	var gens [nPods]atomic.Int64
	for i := range gens {
		gens[i].Store(1)
	}
	var violations atomic.Int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	// committers
	for c := 0; c < 16; c++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed))
			var held []*treInflightTicket
			for {
				select {
				case <-stop:
					for _, k := range held {
						k.Release()
					}
					return
				default:
				}
				i := r.Intn(nPods)
				k, err := g.commit(pods[i], r.Intn(3) == 0)
				if err == nil {
					held = append(held, k)
				} else if !errors.Is(err, errTRENotRoutable) {
					violations.Add(1)
				}
				if len(held) > 0 && r.Intn(2) == 0 {
					j := r.Intn(len(held))
					held[j].Release()
					held = append(held[:j], held[j+1:]...)
				}
			}
		}(int64(c))
	}
	// route changer: hide/unhide with increasing gens; after a hide returns, no commit
	// on that pod may succeed until it is unhidden.
	wg.Add(1)
	go func() {
		defer wg.Done()
		r := rand.New(rand.NewSource(99))
		for n := 0; n < 300; n++ {
			i := r.Intn(nPods)
			gen := gens[i].Add(1)
			routable := "true"
			if r.Intn(2) == 0 {
				routable = "false"
			}
			g.observePod(nil, treGenPod(pods[i].Name, pods[i].Status.PodIP, routable, int(gen)))
			if routable == "false" {
				if k, err := g.commit(pods[i], false); err == nil {
					k.Release()
					violations.Add(1)
				}
			}
			if n%50 == 49 { // occasionally the pod is deleted and comes back
				g.observePod(pods[i], nil)
				g.observePod(nil, treGenPod(pods[i].Name, pods[i].Status.PodIP, "true", int(gens[i].Add(1))))
			}
		}
	}()
	time.Sleep(300 * time.Millisecond)
	close(stop)
	wg.Wait()

	assert.Zero(t, violations.Load())
	assert.Equal(t, 0, g.inflight.size(), "every ticket released: no counters left")
	for i := range pods {
		i := i
		st, ok := g.routeSnapshot("default/" + pods[i].Name)
		require.True(t, ok)
		require.Eventually(t, func() bool {
			v, ok := readSeen(t, client, pods[i].Name, "gw-test-0")
			in, inOK := readInflight(t, client, pods[i].Name, "gw-test-0")
			return ok && v.Gen == st.gen && v.Routable == st.routable && inOK && in.Total == 0 && in.NonContinuable == 0
		}, 3*time.Second, 5*time.Millisecond, "redis converges for %s", pods[i].Name)
	}
}

// P1-2: coordination that is wanted but cannot run is an error (the process exits).
func TestTREStartup_FailClosed(t *testing.T) {
	t.Run("coordination explicitly on without the label gate", func(t *testing.T) {
		withTREGates(t, false, false)
		t.Setenv(envTREGatewayCoordination, "true")
		resetTREGateway(t)
		_, client := newTREMiniRedis(t)
		err := (&Server{redisClient: client, client: fake.NewSimpleClientset()}).StartTRECoordination()
		assert.ErrorContains(t, err, utils.TRERoutableLabelFilterEnv)
	})
	t.Run("default on (label gate) without redis", func(t *testing.T) {
		withTREGates(t, true, false)
		t.Setenv(envTREGatewayCoordination, "")
		resetTREGateway(t)
		err := (&Server{client: fake.NewSimpleClientset()}).StartTRECoordination()
		assert.ErrorContains(t, err, "Redis")
	})
	t.Run("without the kubernetes API", func(t *testing.T) {
		withTREGates(t, true, false)
		resetTREGateway(t)
		_, client := newTREMiniRedis(t)
		err := (&Server{redisClient: client}).StartTRECoordination()
		assert.ErrorContains(t, err, "Kubernetes")
	})
	t.Run("explicitly off", func(t *testing.T) {
		withTREGates(t, true, false)
		t.Setenv(envTREGatewayCoordination, "false")
		g := resetTREGateway(t)
		s := &Server{}
		assert.NoError(t, s.StartTRECoordination())
		assert.Nil(t, s.treWriter)
		assert.False(t, g.enabled.Load())
	})
	t.Run("redis unreachable beyond the startup timeout", func(t *testing.T) {
		g := resetTREGateway(t)
		mr, client := newTREMiniRedis(t)
		mr.Close()
		cfg := testTREConfig()
		cfg.startupTimeout = 200 * time.Millisecond
		cfg.retryBase = 10 * time.Millisecond
		cfg.retryMax = 50 * time.Millisecond
		start := time.Now()
		w, err := startTREGatewayCoordination(client, treStaticLister{treGenPod("p", "10.0.0.1", "true", 1)}, cfg)
		assert.Error(t, err)
		assert.Nil(t, w)
		assert.Less(t, time.Since(start), 10*time.Second, "bounded")
		assert.False(t, g.enabled.Load())
		assert.Nil(t, g.writer.Load())
		assert.Nil(t, cache.SetTREPodObserver(nil), "observer unwound")
	})
	t.Run("apiserver list keeps failing", func(t *testing.T) {
		g := resetTREGateway(t)
		mr, client := newTREMiniRedis(t)
		cfg := testTREConfig()
		cfg.startupTimeout = 150 * time.Millisecond
		cfg.retryBase = 10 * time.Millisecond
		cfg.freshPods = func(context.Context) ([]*v1.Pod, error) { return nil, errors.New("forbidden") }
		_, err := startTREGatewayCoordination(client, treStaticLister{}, cfg)
		assert.ErrorContains(t, err, "forbidden")
		assert.False(t, g.enabled.Load())
		assert.False(t, mr.Exists(TREGatewayInstancesKey), "never became live")
	})
	t.Run("transient redis outage within the timeout is retried", func(t *testing.T) {
		resetTREGateway(t)
		mr, client := newTREMiniRedis(t)
		mr.SetError("ERR simulated outage")
		time.AfterFunc(100*time.Millisecond, func() { mr.SetError("") })
		cfg := testTREConfig()
		cfg.startupTimeout = 5 * time.Second
		cfg.retryBase = 10 * time.Millisecond
		w, err := startTREGatewayCoordination(client, treStaticLister{treGenPod("p", "10.0.0.1", "true", 1)}, cfg)
		require.NoError(t, err)
		defer w.shutdown()
		_, ok := readSeen(t, client, "p", "gw-test-0")
		assert.True(t, ok)
		assert.True(t, mr.Exists(TREGatewayInstancesKey), "live only after the first successful heartbeat")
	})
}

// P1-2: heartbeat score and "ts" values come from the Redis clock, not the local one.
func TestTREHeartbeat_UsesRedisTime(t *testing.T) {
	resetTREGateway(t)
	mr, client := newTREMiniRedis(t)
	redisNow := time.Now().Add(-37 * time.Minute) // a skewed host clock
	mr.SetTime(redisNow)
	w, err := startTREGatewayCoordination(client, treStaticLister{treGenPod("p", "10.0.0.1", "true", 1)}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()
	score, err := client.ZScore(context.Background(), TREGatewayInstancesKey, "gw-test-0").Result()
	require.NoError(t, err)
	assert.InDelta(t, float64(redisNow.UnixMilli()), score, 1000)
	w.markAll()
	require.NoError(t, w.flush(context.Background()))
	v, ok := readSeen(t, client, "p", "gw-test-0")
	require.True(t, ok)
	assert.InDelta(t, float64(redisNow.UnixMilli()), float64(v.TS), 5000, "ts uses the Redis clock")
}

// P1-2 fresh LIST: the quorum listing supersedes a stale cache, and a stale informer
// event arriving later cannot bring the old state back.
func TestTREStartup_FreshListSupersedesStaleCache(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	created := time.Now().Add(-time.Hour)
	staleCache := treUIDPod("p", "10.0.0.1", "true", 3, "u1", created)
	fresh := treUIDPod("p", "10.0.0.1", "false", 4, "u1", created)
	cfg := testTREConfig()
	cfg.freshPods = func(context.Context) ([]*v1.Pod, error) { return []*v1.Pod{fresh}, nil }
	w, err := startTREGatewayCoordination(client, treStaticLister{staleCache}, cfg)
	require.NoError(t, err)
	defer w.shutdown()
	v, ok := readSeen(t, client, "p", "gw-test-0")
	require.True(t, ok)
	assert.Equal(t, int64(4), v.Gen)
	assert.False(t, v.Routable)
	g.observePod(nil, staleCache) // the informer replays the old state
	_, err = g.commit(staleCache, false)
	assert.ErrorIs(t, err, errTRENotRoutable)

	// The production lister issues a quorum read (ResourceVersion "") with the label.
	cs := fake.NewSimpleClientset(fresh)
	pods, err := treFreshPodLister(cs)(context.Background())
	require.NoError(t, err)
	require.Len(t, pods, 1)
	var sawList bool
	for _, a := range cs.Actions() {
		if a.GetVerb() == "list" {
			sawList = true
		}
	}
	assert.True(t, sawList)
}

// P3: persistent Redis errors back off exponentially (bounded) instead of retrying at a
// fixed short interval, are counted, and the writer recovers.
func TestTREWriter_RedisErrorsBackOff(t *testing.T) {
	assert.Equal(t, 250*time.Millisecond, treBackoff(1, 250*time.Millisecond, 10*time.Second))
	assert.Equal(t, 500*time.Millisecond, treBackoff(2, 250*time.Millisecond, 10*time.Second))
	assert.Equal(t, 8*time.Second, treBackoff(6, 250*time.Millisecond, 10*time.Second))
	assert.Equal(t, 10*time.Second, treBackoff(7, 250*time.Millisecond, 10*time.Second))
	assert.Equal(t, 10*time.Second, treBackoff(1000, 250*time.Millisecond, 10*time.Second))

	g := resetTREGateway(t)
	mr, client := newTREMiniRedis(t)
	cfg := testTREConfig()
	cfg.heartbeat = time.Hour
	cfg.retryBase = 5 * time.Millisecond
	cfg.retryMax = 80 * time.Millisecond
	pod := treGenPod("p", "10.0.0.1", "true", 1)
	w, err := startTREGatewayCoordination(client, treStaticLister{pod}, cfg)
	require.NoError(t, err)
	defer w.shutdown()

	before := testutil.ToFloat64(treRedisErrors.WithLabelValues("flush"))
	mr.SetError("ERR simulated outage")
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) { // a steady stream of changes keeps kicking the writer
		k, _ := g.commit(pod, false)
		k.Release()
		time.Sleep(time.Millisecond)
	}
	failures := testutil.ToFloat64(treRedisErrors.WithLabelValues("flush")) - before
	assert.GreaterOrEqual(t, failures, 2.0)
	assert.LessOrEqual(t, failures, 14.0, "5ms fixed retries would be ~100; backoff 5,10,20,40,80,80...")
	mr.SetError("")
	g.observePod(pod, treGenPod("p", "10.0.0.1", "false", 2))
	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 2
	}, 3*time.Second, 5*time.Millisecond)
}

// P3: graceful shutdown refuses new commits (503 + Retry-After) before leaving the live
// set and clearing inflight.
func TestTREShutdown_RefusesCommitsBeforeClearing(t *testing.T) {
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	store := treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
	w, err := startTREGatewayCoordination(client, store.(cache.PodLister), testTREConfig())
	require.NoError(t, err)
	algo := registerTRENamedRouter(t, "tre-test-shutdown-refuse", &treNamedRouter{order: []string{"p"}})
	s := &Server{cache: store, requestCountTracker: map[string]int{}, treWriter: w}

	s.Shutdown()
	assert.True(t, g.closing.Load())
	_, err = client.ZScore(context.Background(), TREGatewayInstancesKey, "gw-test-0").Result()
	assert.ErrorIs(t, err, redis.Nil)

	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(algo)
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req", treBodyReq(`{"model":"m","prompt":"x"}`), utils.User{})
	assert.Nil(t, ticket)
	require.NotNil(t, resp.GetImmediateResponse())
	assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, resp.GetImmediateResponse().GetStatus().GetCode())
	assert.Equal(t, strconv.Itoa(treRetryAfterSeconds), treResponseHeader(resp, "Retry-After"))
	assert.Contains(t, string(resp.GetImmediateResponse().GetBody()), "shutting down")
	assert.Equal(t, 0, g.inflight.size())
}

// P2-1: with coordination on, a request that resolves to no strategy is routed per pod
// (counted, ack-bound) instead of going down the Service path.
func TestTRERouterNotSet_ForcedToPodRoutingWhenCoordinated(t *testing.T) {
	if defaultRoutingStrategyEnabled && defaultRoutingStrategy != "" {
		t.Skip("ROUTING_ALGORITHM is set in this environment and takes precedence")
	}
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	store := treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
	w, err := startTREGatewayCoordination(client, store.(cache.PodLister), testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()
	routingalgorithms.Init()

	s := &Server{cache: store, requestCountTracker: map[string]int{}}
	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathChatCompletions
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req",
		treBodyReq(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), utils.User{})
	require.Nil(t, resp.GetImmediateResponse())
	target, ok := headerValue(resp, HeaderTargetPod)
	require.True(t, ok, "pod-level routing, not the Service path")
	assert.Equal(t, "10.0.0.1:8000", target)
	strategy, _ := headerValue(resp, HeaderRoutingStrategy)
	assert.Equal(t, string(routingalgorithms.RouterRandom), strategy)
	assert.Equal(t, treInflightCounts{Total: 1}, g.inflight.snapshot("p"))
	ticket.Release()
}

type treFakeQueueRouter struct{}

func (treFakeQueueRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	ctx.SetTargetPod(pods.All()[0])
	return ctx.TargetAddress(), nil
}
func (treFakeQueueRouter) Len() int { return 0 }

// P3: queue routers and PD routing are refused (400) while coordination is on.
func TestTRERouters_QueueAndPDRefusedWhenCoordinated(t *testing.T) {
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	store := treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
	w, err := startTREGatewayCoordination(client, store.(cache.PodLister), testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	queueAlgo := types.RoutingAlgorithm("tre-test-queue-router")
	routingalgorithms.Register(queueAlgo, func() (types.Router, error) { return treFakeQueueRouter{}, nil })
	routingalgorithms.Init()

	s := &Server{cache: store, requestCountTracker: map[string]int{}}
	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(queueAlgo)
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req", treBodyReq(`{"model":"m","prompt":"x"}`), utils.User{})
	assert.Nil(t, ticket)
	require.NotNil(t, resp.GetImmediateResponse())
	assert.Equal(t, envoyTypePb.StatusCode_BadRequest, resp.GetImmediateResponse().GetStatus().GetCode())
	assert.Contains(t, string(resp.GetImmediateResponse().GetBody()), "queue router")
	assert.Equal(t, 0, g.inflight.size())

	pods := &utils.PodArray{Pods: []*v1.Pod{treGenPod("p", "10.0.0.1", "true", 1)}}
	pdCtx := types.NewRoutingContext(context.Background(), routingalgorithms.RouterPD, "m", "", "req", "")
	_, err = (&Server{}).selectTargetPod(context.Background(), pdCtx, pods, "")
	assert.ErrorIs(t, err, errTREUnsupportedRouter)

	// Without coordination the same routers are untouched.
	g.enabled.Store(false)
	assert.NoError(t, treCheckRouterSupported(queueAlgo, treFakeQueueRouter{}))
	assert.NoError(t, treCheckRouterSupported(routingalgorithms.RouterPD, nil))
	g.enabled.Store(true)
}
