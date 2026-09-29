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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	miniredis "github.com/alicebob/miniredis/v2"
	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"

	"github.com/vllm-project/aibrix/pkg/cache"
	routingalgorithms "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P3-GW-007..011) tests. Like tre_routing_test.go they flip process-wide state
// (treGW, the cache observer, gates) and restore it, so none of them runs in parallel.

// resetTREGateway gives the test a fresh process-wide TRE gateway state.
func resetTREGateway(t *testing.T) *treGatewayState {
	t.Helper()
	prev := treGW
	treGW = newTREGatewayState()
	prevObs := cache.SetTREPodObserver(nil)
	t.Cleanup(func() {
		if w := treGW.writer.Load(); w != nil {
			w.shutdown()
		}
		treGW = prev
		cache.SetTREPodObserver(prevObs)
	})
	return treGW
}

func treGenPod(name, ip, routable string, gen int) *v1.Pod {
	p := treTestPod(name, ip, routable)
	p.Annotations = map[string]string{TRERouteGenAnnotation: strconv.Itoa(gen)}
	return p
}

type treStaticLister []*v1.Pod

func (l treStaticLister) ListPods() []*v1.Pod { return l }

func newTREMiniRedis(t *testing.T) (*miniredis.Miniredis, *redis.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return mr, client
}

func testTREConfig() treGatewayConfig {
	return treGatewayConfig{
		instanceID:        "gw-test-0",
		heartbeat:         20 * time.Millisecond,
		instanceRetention: time.Minute,
		keyTTL:            300 * time.Second,
		refresh:           time.Hour, // tests drive refreshes explicitly
	}
}

func readSeen(t *testing.T, c *redis.Client, pod, id string) (treSeenValue, bool) {
	t.Helper()
	raw, err := c.HGet(context.Background(), treSeenKey(pod), id).Result()
	if errors.Is(err, redis.Nil) {
		return treSeenValue{}, false
	}
	require.NoError(t, err)
	var v treSeenValue
	require.NoError(t, json.Unmarshal([]byte(raw), &v))
	return v, true
}

func readInflight(t *testing.T, c *redis.Client, pod, id string) (treInflightValue, bool) {
	t.Helper()
	raw, err := c.HGet(context.Background(), treInflightKey(pod), id).Result()
	if errors.Is(err, redis.Nil) {
		return treInflightValue{}, false
	}
	require.NoError(t, err)
	var v treInflightValue
	require.NoError(t, json.Unmarshal([]byte(raw), &v))
	return v, true
}

// ---------------------------------------------------------------------------------------
// item 1: heartbeat

func TestTREHeartbeat_RegistersRefreshesPrunesAndLeavesOnShutdown(t *testing.T) {
	g := resetTREGateway(t)
	mr, client := newTREMiniRedis(t)
	ctx := context.Background()
	// A long-dead instance must be pruned by the retention window.
	require.NoError(t, client.ZAdd(ctx, TREGatewayInstancesKey, redis.Z{Score: 1, Member: "gw-dead"}).Err())

	w, err := startTREGatewayCoordination(client, treStaticLister{}, testTREConfig())
	require.NoError(t, err)
	require.Same(t, w, g.writer.Load())

	var first float64
	require.Eventually(t, func() bool {
		s, err := client.ZScore(ctx, TREGatewayInstancesKey, "gw-test-0").Result()
		first = s
		return err == nil
	}, 2*time.Second, 5*time.Millisecond)
	assert.InDelta(t, float64(time.Now().UnixMilli()), first, 5000)
	require.Eventually(t, func() bool {
		s, _ := client.ZScore(ctx, TREGatewayInstancesKey, "gw-test-0").Result()
		return s > first
	}, 2*time.Second, 5*time.Millisecond, "heartbeat must be refreshed periodically")
	_, err = client.ZScore(ctx, TREGatewayInstancesKey, "gw-dead").Result()
	assert.ErrorIs(t, err, redis.Nil, "stale instances are pruned")

	w.shutdown()
	_, err = client.ZScore(ctx, TREGatewayInstancesKey, "gw-test-0").Result()
	assert.ErrorIs(t, err, redis.Nil, "graceful shutdown leaves the live set")
	_ = mr
}

func TestTREInstanceIDFallsBackToPodName(t *testing.T) {
	t.Setenv(envTREGatewayInstanceID, "")
	t.Setenv("POD_NAME", "tre-gateway-plugins-abc")
	assert.Equal(t, "tre-gateway-plugins-abc", treGatewayInstanceID())
	t.Setenv(envTREGatewayInstanceID, "explicit")
	assert.Equal(t, "explicit", treGatewayInstanceID())
}

// ---------------------------------------------------------------------------------------
// item 2: seen-gen ack after the routing state changed

func TestTREAck_StartupAcksCurrentStateBeforeHeartbeat(t *testing.T) {
	resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	lister := treStaticLister{
		treGenPod("p-awake", "10.0.0.1", "true", 7),
		treGenPod("p-asleep", "10.0.0.2", "false", 3),
		treTestPod("p-unmanaged", "10.0.0.3", ""), // no label/annotation: never acked
	}
	w, err := startTREGatewayCoordination(client, lister, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	v, ok := readSeen(t, client, "p-awake", "gw-test-0")
	require.True(t, ok, "startup acks synchronously, before the heartbeat makes the instance live")
	assert.Equal(t, int64(7), v.Gen)
	assert.True(t, v.Routable)
	v, ok = readSeen(t, client, "p-asleep", "gw-test-0")
	require.True(t, ok)
	assert.Equal(t, treSeenValue{Gen: 3, Routable: false, TS: v.TS}, v)
	_, ok = readSeen(t, client, "p-unmanaged", "gw-test-0")
	assert.False(t, ok)
	ttl := client.TTL(context.Background(), treSeenKey("p-awake")).Val()
	assert.Greater(t, ttl, 250*time.Second)
	assert.LessOrEqual(t, ttl, 300*time.Second)
}

// The ack for gen G must not be written while a routing commit that saw the previous
// state is still in progress, and it must be accompanied by an inflight value that
// already counts that request.
func TestTREAck_WaitsForInProgressCommitAndCarriesItsInflight(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	pod := treGenPod("p", "10.0.0.1", "true", 1)
	w, err := startTREGatewayCoordination(client, treStaticLister{pod}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	// A request is mid-commit: it holds the read side of the barrier (it has checked the
	// route table, not yet incremented).
	g.barrier.RLock()
	observed := make(chan struct{})
	go func() {
		g.observePod(pod, treGenPod("p", "10.0.0.1", "false", 2))
		close(observed)
	}()
	time.Sleep(50 * time.Millisecond)
	v, _ := readSeen(t, client, "p", "gw-test-0")
	assert.Equal(t, int64(1), v.Gen, "gen 2 must not be acked while an old-state commit is in progress")
	select {
	case <-observed:
		t.Fatal("route table updated while a commit held the barrier")
	default:
	}
	ticket := g.inflight.acquire("p", true) // the in-progress commit completes
	g.barrier.RUnlock()
	<-observed

	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 2 && !v.Routable
	}, 2*time.Second, 5*time.Millisecond)
	in, ok := readInflight(t, client, "p", "gw-test-0")
	require.True(t, ok)
	assert.Equal(t, int64(1), in.Total, "the ack's pipeline carries the old-state request")
	assert.Equal(t, int64(1), in.NonContinuable)

	// After the ack no request can be committed to p.
	_, cerr := g.commit(pod, false)
	assert.ErrorIs(t, cerr, errTRENotRoutable)

	ticket.Release()
	require.Eventually(t, func() bool {
		in, ok := readInflight(t, client, "p", "gw-test-0")
		return ok && in.Total == 0 && in.NonContinuable == 0
	}, 2*time.Second, 5*time.Millisecond)
}

// End to end with the real cache: the informer's update handler notifies only after the
// Store serves the new pod, and the resulting ack reflects it.
func TestTREAck_RealCacheUpdateThenAck(t *testing.T) {
	withTREGates(t, true, false)
	resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	old := treGenPod("p", "10.0.0.1", "true", 4)
	old.Labels["model.aibrix.ai/name"] = "m"
	store := cache.InitWithPods(cache.InitForTest(), []*v1.Pod{old}, "m")
	w, err := startTREGatewayCoordination(client, store, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	updated := old.DeepCopy()
	updated.Labels[utils.TRERoutableLabel] = "false"
	updated.Annotations[TRERouteGenAnnotation] = "5"
	cache.UpdatePodForTest(store, old, updated)

	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 5 && !v.Routable
	}, 2*time.Second, 5*time.Millisecond)
	cur, err := store.GetPod("p", "default")
	require.NoError(t, err)
	assert.Equal(t, "false", cur.Labels[utils.TRERoutableLabel])
}

func TestTREAck_PeriodicRefreshReacksAndRenewsTTL(t *testing.T) {
	resetTREGateway(t)
	mr, client := newTREMiniRedis(t)
	cfg := testTREConfig()
	cfg.refresh = 30 * time.Millisecond
	w, err := startTREGatewayCoordination(client, treStaticLister{treGenPod("p", "10.0.0.1", "true", 9)}, cfg)
	require.NoError(t, err)
	defer w.shutdown()

	// Simulate the hash expiring (or being wiped): the refresh must restore it.
	mr.Del(treSeenKey("p"))
	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 9
	}, 2*time.Second, 5*time.Millisecond)
	_, ok := readInflight(t, client, "p", "gw-test-0")
	assert.True(t, ok, "a live instance writes an inflight field for every pod it knows")
}

func TestTREAck_RedisFailureIsRetried(t *testing.T) {
	g := resetTREGateway(t)
	mr, client := newTREMiniRedis(t)
	pod := treGenPod("p", "10.0.0.1", "true", 1)
	w, err := startTREGatewayCoordination(client, treStaticLister{pod}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	mr.SetError("LOADING simulated outage")
	g.observePod(pod, treGenPod("p", "10.0.0.1", "false", 2))
	time.Sleep(100 * time.Millisecond)
	mr.SetError("")
	require.Eventually(t, func() bool {
		v, ok := readSeen(t, client, "p", "gw-test-0")
		return ok && v.Gen == 2
	}, 3*time.Second, 10*time.Millisecond)
}

func TestTREStartup_ResetsOwnInflightFieldsOnly(t *testing.T) {
	resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	ctx := context.Background()
	require.NoError(t, client.HSet(ctx, treInflightKey("gone"), "gw-test-0", `{"total":5,"non_continuable":1,"ts":1}`).Err())
	require.NoError(t, client.HSet(ctx, treInflightKey("gone"), "gw-other", `{"total":2,"non_continuable":0,"ts":1}`).Err())

	w, err := startTREGatewayCoordination(client, treStaticLister{}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	_, ok := readInflight(t, client, "gone", "gw-test-0")
	assert.False(t, ok, "counts of a previous process with the same id are void")
	_, ok = readInflight(t, client, "gone", "gw-other")
	assert.True(t, ok, "other instances' fields are untouched")
}

// ---------------------------------------------------------------------------------------
// item 3: inflight accounting through Process

// treNamedRouter picks the pod whose name is next in order and sets it as target.
type treNamedRouter struct {
	mu     sync.Mutex
	order  []string
	onPick func(name string)
}

func (r *treNamedRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	r.mu.Lock()
	name := r.order[0]
	if len(r.order) > 1 {
		r.order = r.order[1:]
	}
	r.mu.Unlock()
	for _, p := range pods.All() {
		if p.Name == name {
			if r.onPick != nil {
				r.onPick(name)
			}
			ctx.SetTargetPod(p)
			return ctx.TargetAddress(), nil
		}
	}
	return "", errors.New("pod " + name + " not among candidates")
}

func (r *treNamedRouter) Name() string { return "tre-named" }

func registerTRENamedRouter(t *testing.T, name string, r *treNamedRouter) types.RoutingAlgorithm {
	t.Helper()
	algo := types.RoutingAlgorithm(name)
	routingalgorithms.Register(algo, func() (types.Router, error) { return r, nil })
	routingalgorithms.Init()
	return algo
}

func treHeadersReq(path string, kv ...string) *extProcPb.ProcessingRequest {
	hs := []*configPb.HeaderValue{{Key: ":path", RawValue: []byte(path)}}
	for i := 0; i+1 < len(kv); i += 2 {
		hs = append(hs, &configPb.HeaderValue{Key: kv[i], RawValue: []byte(kv[i+1])})
	}
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: hs}},
	}}
}

func treBodyReq(body string) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestBody{
		RequestBody: &extProcPb.HttpBody{Body: []byte(body), EndOfStream: true},
	}}
}

func treProcessStore(pods ...*v1.Pod) cache.Cache {
	for _, p := range pods {
		p.Labels["model.aibrix.ai/name"] = "m"
	}
	return cache.InitWithPods(cache.InitForTest(), pods, "m")
}

// Every way an ext_proc stream can end after routing releases the inflight slot once.
func TestTREInflight_ReleasedOnEveryStreamEnd(t *testing.T) {
	withTREGates(t, true, false)

	endings := map[string]func(srv *mockProcessServer){
		"client disconnect (EOF)": func(srv *mockProcessServer) {
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Once()
		},
		"stream cancelled": func(srv *mockProcessServer) {
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Canceled, "gone")).Once()
		},
		"grpc error": func(srv *mockProcessServer) {
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Internal, "boom")).Once()
		},
		"non-grpc error": func(srv *mockProcessServer) {
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), errors.New("transport")).Once()
		},
		"upstream error response": func(srv *mockProcessServer) {
			srv.On("Recv").Return(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
					{Key: ":status", RawValue: []byte("503")}}}},
			}}, nil).Once()
			srv.On("Recv").Return(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{Body: []byte(`{"error":{"type":"EngineSleeping"}}`), EndOfStream: true},
			}}, nil).Once()
			srv.On("Recv").Return((*extProcPb.ProcessingRequest)(nil), status.Error(codes.Canceled, "done")).Once()
		},
		"completed": func(srv *mockProcessServer) {
			srv.On("Recv").Return(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
					{Key: ":status", RawValue: []byte("200")}}}},
			}}, nil).Once()
			srv.On("Recv").Return(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body:        []byte(`{"model":"m","choices":[{"text":"x"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
					EndOfStream: true,
				},
			}}, nil).Once()
		},
	}
	for name, ending := range endings {
		t.Run(name, func(t *testing.T) {
			g := resetTREGateway(t)
			algo := registerTRENamedRouter(t, "tre-test-inflight-"+name, &treNamedRouter{order: []string{"p"}})
			s := newProcessTestServer(openShutdownCh(), nil)
			s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))

			srv := &mockProcessServer{ctx: context.Background()}
			srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
			srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi","max_tokens":4,"n":2}`), nil).Once()
			ending(srv)
			var during []treInflightCounts
			srv.On("Send", mock.Anything).Run(func(args mock.Arguments) {
				if args.Get(0).(*extProcPb.ProcessingResponse).GetRequestBody() != nil {
					during = append(during, g.inflight.snapshot("p"))
				}
			}).Return(nil)

			_ = s.Process(srv)

			require.Len(t, during, 1)
			assert.Equal(t, treInflightCounts{Total: 1, NonContinuable: 1}, during[0], "n=2 is non-continuable")
			assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"), "no leak")
		})
	}
}

func TestTREInflight_ReleasedWhenSendFailsAndOnShutdown(t *testing.T) {
	withTREGates(t, true, false)

	t.Run("send fails", func(t *testing.T) {
		g := resetTREGateway(t)
		algo := registerTRENamedRouter(t, "tre-test-inflight-sendfail", &treNamedRouter{order: []string{"p"}})
		s := newProcessTestServer(openShutdownCh(), nil)
		s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
		srv := &mockProcessServer{ctx: context.Background()}
		srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
		srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi"}`), nil).Once()
		srv.On("Send", mock.MatchedBy(func(r *extProcPb.ProcessingResponse) bool { return r.GetRequestBody() == nil })).Return(nil)
		srv.On("Send", mock.MatchedBy(func(r *extProcPb.ProcessingResponse) bool { return r.GetRequestBody() != nil })).
			Return(status.Error(codes.Canceled, "client went away"))
		assert.Error(t, s.Process(srv))
		assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"))
	})

	t.Run("shutdown while waiting for the response", func(t *testing.T) {
		g := resetTREGateway(t)
		algo := registerTRENamedRouter(t, "tre-test-inflight-shutdown", &treNamedRouter{order: []string{"p"}})
		shutdown := make(chan struct{})
		s := newProcessTestServer(shutdown, nil)
		s.cache = treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
		srv := &mockProcessServer{ctx: context.Background()}
		srv.On("Recv").Return(treHeadersReq(PathCompletions, HeaderRoutingStrategy, string(algo)), nil).Once()
		srv.On("Recv").Return(treBodyReq(`{"model":"m","prompt":"hi"}`), nil).Once()
		block := make(chan struct{})
		defer close(block)
		srv.On("Recv").Run(func(mock.Arguments) { <-block }).Return((*extProcPb.ProcessingRequest)(nil), io.EOF).Maybe()
		srv.On("Send", mock.Anything).Run(func(args mock.Arguments) {
			if args.Get(0).(*extProcPb.ProcessingResponse).GetRequestBody() != nil {
				close(shutdown) // shutdown arrives while the stream waits for the upstream
			}
		}).Return(nil)
		err := s.Process(srv)
		st, _ := status.FromError(err)
		assert.Equal(t, codes.Unavailable, st.Code())
		assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"))
	})
}

func TestTREInflight_TicketReleaseIsIdempotentAndNilSafe(t *testing.T) {
	tr := newTREInflightTracker()
	a := tr.acquire("p", true)
	b := tr.acquire("p", false)
	assert.Equal(t, treInflightCounts{Total: 2, NonContinuable: 1}, tr.snapshot("p"))
	a.Release()
	a.Release()
	assert.Equal(t, treInflightCounts{Total: 1, NonContinuable: 0}, tr.snapshot("p"))
	b.Release()
	var nilTicket *treInflightTicket
	nilTicket.Release()
	assert.Equal(t, treInflightCounts{}, tr.snapshot("p"))
}

func TestTREInflight_ConcurrentAcquireRelease(t *testing.T) {
	tr := newTREInflightTracker()
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				k := tr.acquire("p", (i+j)%3 == 0)
				k.Release()
			}
		}(i)
	}
	wg.Wait()
	assert.Equal(t, treInflightCounts{}, tr.snapshot("p"))
}

func TestTREInflight_MirroredToRedis(t *testing.T) {
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	w, err := startTREGatewayCoordination(client, treStaticLister{treGenPod("p", "10.0.0.1", "true", 1)}, testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	k1, err := g.commit(treGenPod("p", "10.0.0.1", "true", 1), false)
	require.NoError(t, err)
	k2, err := g.commit(treGenPod("p", "10.0.0.1", "true", 1), true)
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		v, ok := readInflight(t, client, "p", "gw-test-0")
		return ok && v.Total == 2 && v.NonContinuable == 1
	}, 2*time.Second, 5*time.Millisecond)
	k1.Release()
	k2.Release()
	require.Eventually(t, func() bool {
		v, ok := readInflight(t, client, "p", "gw-test-0")
		return ok && v.Total == 0 && v.NonContinuable == 0 && v.TS > 0
	}, 2*time.Second, 5*time.Millisecond)
}

// A hide that lands between candidate listing and commit makes the request re-route; the
// hidden pod is never counted (and never targeted).
func TestTRECommit_RaceWithHideReroutes(t *testing.T) {
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	_, client := newTREMiniRedis(t)
	pa, pb := treGenPod("pa", "10.0.0.1", "true", 1), treGenPod("pb", "10.0.0.2", "true", 1)
	store := treProcessStore(pa, pb)
	w, err := startTREGatewayCoordination(client, store.(cache.PodLister), testTREConfig())
	require.NoError(t, err)
	defer w.shutdown()

	router := &treNamedRouter{order: []string{"pa", "pb"}}
	router.onPick = func(name string) {
		if name == "pa" { // SM hides pa while this request is being routed
			g.observePod(pa, treGenPod("pa", "10.0.0.1", "false", 2))
		}
	}
	algo := registerTRENamedRouter(t, "tre-test-commit-race", router)
	s := &Server{cache: store, requestCountTracker: map[string]int{}}
	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(algo)
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req", treBodyReq(`{"model":"m","prompt":"x"}`), utils.User{})
	require.Nil(t, resp.GetImmediateResponse())
	target, _ := headerValue(resp, HeaderTargetPod)
	assert.Equal(t, "10.0.0.2:8000", target)
	assert.Equal(t, treInflightCounts{Total: 1}, g.inflight.snapshot("pb"))
	assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("pa"))
	ticket.Release()
}

// ---------------------------------------------------------------------------------------
// item 3b: non_continuable classification

// treNonContinuableContract is shared with the reissue sidecar's test_non_continuable_contract
// (tre/reissue/tests/test_reissue_sidecar.py): one list of cases, two implementations. The
// path is relative to this package directory (go test runs in it).
var treNonContinuableContract = filepath.Join("..", "..", "..", "tre", "reissue", "contract", "non_continuable_cases.json")

type treNCContractCase struct {
	Name    string          `json:"name"`
	Path    string          `json:"path"`
	Body    json.RawMessage `json:"body"`
	BodyRaw *string         `json:"body_raw"`
	Want    bool            `json:"want"`
	Reason  *string         `json:"reason"`
}

func TestTRENonContinuableContract(t *testing.T) {
	data, err := os.ReadFile(treNonContinuableContract)
	require.NoError(t, err)
	var contract struct {
		Description string              `json:"description"`
		Cases       []treNCContractCase `json:"cases"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	require.NoError(t, dec.Decode(&contract))
	require.GreaterOrEqual(t, len(contract.Cases), 40)

	reasons := map[string]bool{}
	for _, c := range contract.Cases {
		t.Run(c.Name, func(t *testing.T) {
			require.True(t, (c.BodyRaw == nil) != (c.Body == nil), "exactly one of body / body_raw")
			body := []byte(c.Body)
			if c.BodyRaw != nil {
				body = []byte(*c.BodyRaw)
			}
			want := ""
			if c.Reason != nil {
				want = *c.Reason
				reasons[want] = true
			}
			assert.Equal(t, c.Want, want != "", "want and reason disagree")
			assert.Equal(t, want, treNonContinuableReason(c.Path, body))
			assert.Equal(t, c.Want, treNonContinuable(c.Path, body))
		})
	}
	all := []string{treNCEndpoint, treNCBody, treNCN, treNCLogprobs, treNCEcho, treNCBeamSearch,
		treNCTools, treNCStructuredOutput, treNCPromptForm}
	for _, r := range all {
		assert.True(t, reasons[r], "no contract case for reason %q", r)
	}
	assert.Len(t, reasons, len(all), "contract uses a reason the gateway does not know")
}

// ---------------------------------------------------------------------------------------
// item 4: x-tre-exclude-pod

func TestTREExclude_ParsesHeader(t *testing.T) {
	assert.Nil(t, parseTREExcludePods(""))
	assert.Nil(t, parseTREExcludePods(" , "))
	assert.Equal(t, map[string]struct{}{"a": {}, "b": {}}, parseTREExcludePods(" a,b ,,"))
}

func TestTREExclude_SelectTargetPodNeverPicksExcluded(t *testing.T) {
	withTREGates(t, true, false)
	resetTREGateway(t)
	algo := types.RoutingAlgorithm("tre-test-exclude-candidates")
	router := new(mockRouter)
	routingalgorithms.Register(algo, func() (types.Router, error) { return router, nil })
	routingalgorithms.Init()
	var seen []string
	router.On("Route", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
		for _, p := range args.Get(1).(types.PodList).All() {
			seen = append(seen, p.Name)
		}
	}).Return("10.0.0.3:8000", nil).Once()

	pods := &utils.PodArray{Pods: []*v1.Pod{
		treTestPod("a", "10.0.0.1", "true"),
		treTestPod("b", "10.0.0.2", "true"),
		treTestPod("c", "10.0.0.3", "true"),
	}}
	ctx := types.NewRoutingContext(context.Background(), algo, "m", "", "req", "")
	ctx.ReqHeaders[HeaderTREExcludePod] = "a"
	_, err := (&Server{}).selectTargetPod(context.Background(), ctx, pods, "")
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{"b", "c"}, seen)

	// Only one candidate left: the short-circuit must still honour the exclusion.
	ctx = types.NewRoutingContext(context.Background(), algo, "m", "", "req", "")
	ctx.ReqHeaders[HeaderTREExcludePod] = "a,b"
	addr, err := (&Server{}).selectTargetPod(context.Background(), ctx, pods, "")
	require.NoError(t, err)
	assert.Equal(t, "10.0.0.3:8000", addr)

	ctx = types.NewRoutingContext(context.Background(), algo, "m", "", "req", "")
	ctx.ReqHeaders[HeaderTREExcludePod] = "a,b,c"
	_, err = (&Server{}).selectTargetPod(context.Background(), ctx, pods, "")
	assert.ErrorIs(t, err, errTREAllCandidatesExcluded)
}

func treResponseHeader(resp *extProcPb.ProcessingResponse, key string) string {
	for _, h := range resp.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if h.GetHeader().GetKey() == key {
			return string(h.GetHeader().GetRawValue())
		}
	}
	return ""
}

func TestTREExclude_AllExcludedIs503WithRetryAfterAndNoInflight(t *testing.T) {
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	algo := registerTRENamedRouter(t, "tre-test-exclude-all", &treNamedRouter{order: []string{"p"}})
	store := treProcessStore(treGenPod("p", "10.0.0.1", "true", 1))
	s := &Server{cache: store, requestCountTracker: map[string]int{}}

	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(algo)
	rc.ReqHeaders[HeaderTREExcludePod] = "p"
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req", treBodyReq(`{"model":"m","prompt":"x"}`), utils.User{})
	assert.Nil(t, ticket)
	require.NotNil(t, resp.GetImmediateResponse())
	assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, resp.GetImmediateResponse().GetStatus().GetCode())
	assert.Equal(t, strconv.Itoa(treRetryAfterSeconds), treResponseHeader(resp, "Retry-After"))
	assert.Contains(t, string(resp.GetImmediateResponse().GetBody()), HeaderTREExcludePod)
	assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"))
}

func TestTREExclude_HeaderCapturedAndStrippedUpstream(t *testing.T) {
	withTREGates(t, true, false)
	resetTREGateway(t)
	s := &Server{}
	_, _, _, rc := s.HandleRequestHeaders(context.Background(), "req",
		treHeadersReq(PathCompletions, HeaderTREExcludePod, "p-old"))
	assert.Equal(t, "p-old", rc.ReqHeaders[HeaderTREExcludePod])

	algo := registerTRENamedRouter(t, "tre-test-exclude-strip", &treNamedRouter{order: []string{"p-new"}})
	s = &Server{cache: treProcessStore(treGenPod("p-old", "10.0.0.1", "true", 1), treGenPod("p-new", "10.0.0.2", "true", 1)),
		requestCountTracker: map[string]int{}}
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(algo)
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req", treBodyReq(`{"model":"m","prompt":"x"}`), utils.User{})
	defer ticket.Release()
	require.Nil(t, resp.GetImmediateResponse())
	assert.Equal(t, []string{HeaderTREExcludePod}, resp.GetRequestBody().GetResponse().GetHeaderMutation().GetRemoveHeaders())
	target, _ := headerValue(resp, HeaderTargetPod)
	assert.Equal(t, "10.0.0.2:8000", target)
}

// ---------------------------------------------------------------------------------------
// item 5: token-id prompts

func TestTREParseCompletionPrompt(t *testing.T) {
	cases := []struct {
		body    string
		msg     string
		ids     []int
		wantErr bool
	}{
		{`{"model":"m"}`, "", nil, false},
		{`{"model":"m","prompt":null}`, "", nil, false},
		{`{"model":"m","prompt":"hello"}`, "hello", nil, false},
		{`{"model":"m","prompt":["a","b"]}`, "a b", nil, false},
		{`{"model":"m","prompt":[1,2,30]}`, "1 2 30", []int{1, 2, 30}, false},
		{`{"model":"m","prompt":[[1,2],[3]]}`, "1 2 3", []int{1, 2, 3}, false},
		{`{"model":"m","prompt":[]}`, "", nil, true},
		{`{"model":"m","prompt":[[]]}`, "", nil, true},
		{`{"model":"m","prompt":[1,"a"]}`, "", nil, true},
		{`{"model":"m","prompt":[1.5]}`, "", nil, true},
		{`{"model":"m","prompt":{"x":1}}`, "", nil, true},
		{`{"model":"m","prompt":7}`, "", nil, true},
	}
	for _, c := range cases {
		model, msg, _, ids, errRes := validateCompletionRequestWithTokens("req", []byte(c.body))
		if c.wantErr {
			if assert.NotNil(t, errRes, c.body) {
				assert.Equal(t, envoyTypePb.StatusCode_BadRequest, errRes.GetImmediateResponse().GetStatus().GetCode(), c.body)
			}
			continue
		}
		assert.Nil(t, errRes, c.body)
		assert.Equal(t, "m", model, c.body)
		assert.Equal(t, c.msg, msg, c.body)
		assert.Equal(t, c.ids, ids, c.body)
	}
	// The legacy entry point (used by validateRequestBody) accepts arrays too.
	_, msg, _, errRes := validateCompletionRequest("req", []byte(`{"model":"m","prompt":[5,6]}`))
	assert.Nil(t, errRes)
	assert.Equal(t, "5 6", msg)
}

func TestTRETokenIDPromptRoutesAndCountsIDs(t *testing.T) {
	withTREGates(t, true, false)
	resetTREGateway(t)
	algo := registerTRENamedRouter(t, "tre-test-token-ids", &treNamedRouter{order: []string{"p"}})
	s := &Server{cache: treProcessStore(treGenPod("p", "10.0.0.1", "true", 1)), requestCountTracker: map[string]int{}}
	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathCompletions
	rc.ReqHeaders[HeaderRoutingStrategy] = string(algo)
	resp, model, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req",
		treBodyReq(`{"model":"m","prompt":[101,102,103,104],"max_tokens":8}`), utils.User{})
	defer ticket.Release()
	require.Nil(t, resp.GetImmediateResponse(), "token-id prompts are no longer a 400")
	assert.Equal(t, "m", model)
	n, err := rc.PromptLength()
	require.NoError(t, err)
	assert.Equal(t, 4, n, "token count is len(ids)")
}

// ---------------------------------------------------------------------------------------
// item 6: default routing strategy (D10)

func TestTREDefaultRoutingStrategy(t *testing.T) {
	prev := setTREDefaultRoutingStrategy("least-gpu-cache")
	t.Cleanup(func() { setTREDefaultRoutingStrategy(prev) })

	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	got, ok := deriveRoutingStrategyFromContext(rc)
	if !defaultRoutingStrategyEnabled || defaultRoutingStrategy == "" {
		assert.True(t, ok)
		assert.Equal(t, "least-gpu-cache", got, "no header, no profile, no ROUTING_ALGORITHM")
	}
	rc.ReqHeaders[HeaderRoutingStrategy] = "random"
	got, ok = deriveRoutingStrategyFromContext(rc)
	assert.True(t, ok)
	assert.Equal(t, "random", got, "an explicit header still wins")

	setTREDefaultRoutingStrategy("")
	delete(rc.ReqHeaders, HeaderRoutingStrategy)
	got, _ = deriveRoutingStrategyFromContext(rc)
	assert.Equal(t, defaultRoutingStrategy, got, "disabled: upstream behaviour")
}

func TestTREDefaultRoutingStrategyEnv(t *testing.T) {
	t.Setenv(envTREDefaultRoutingStrategy, "")
	assert.Equal(t, "", loadTREDefaultRoutingStrategy(), "off by default")
	t.Setenv(envTREDefaultRoutingStrategy, "none")
	assert.Equal(t, "", loadTREDefaultRoutingStrategy())
	t.Setenv(envTREDefaultRoutingStrategy, "random")
	assert.Equal(t, "random", loadTREDefaultRoutingStrategy())
}

// A request without a routing-strategy header is routed to a pod by ext_proc (target-pod
// set, inflight counted) instead of being handed to the HTTPRoute/Service path.
func TestTREDefaultRoutingStrategy_HeaderlessRequestGetsPodRouting(t *testing.T) {
	withTREGates(t, true, false)
	g := resetTREGateway(t)
	algo := registerTRENamedRouter(t, "tre-test-default-strategy", &treNamedRouter{order: []string{"p"}})
	prev := setTREDefaultRoutingStrategy(string(algo))
	t.Cleanup(func() { setTREDefaultRoutingStrategy(prev) })
	if defaultRoutingStrategyEnabled && defaultRoutingStrategy != "" {
		t.Skip("ROUTING_ALGORITHM is set in this environment and takes precedence")
	}

	s := &Server{cache: treProcessStore(treGenPod("p", "10.0.0.1", "true", 1)), requestCountTracker: map[string]int{}}
	rc := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	rc.ReqPath = PathChatCompletions
	resp, _, _, _, ticket := s.handleRequestBody(context.Background(), rc, "req",
		treBodyReq(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), utils.User{})
	require.Nil(t, resp.GetImmediateResponse())
	target, ok := headerValue(resp, HeaderTargetPod)
	assert.True(t, ok)
	assert.Equal(t, "10.0.0.1:8000", target)
	strategy, _ := headerValue(resp, HeaderRoutingStrategy)
	assert.Equal(t, string(algo), strategy)
	assert.Equal(t, treInflightCounts{Total: 1}, g.inflight.snapshot("p"))
	ticket.Release()
	assert.Equal(t, treInflightCounts{}, g.inflight.snapshot("p"))
}
