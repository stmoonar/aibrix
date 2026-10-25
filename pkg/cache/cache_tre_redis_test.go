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

package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	miniredis "github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
)

func TestWriteTREPodMetricsToRedisV2WritesSortedSets(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "v2")
	store, client := newTREMetricStoreForTest(t)

	err := treWrite(context.Background(), store, treRoundAt(12345))
	require.NoError(t, err)

	podKey := utils.GeneratePodKey("default", "pod-a")
	histEntries, err := client.ZRangeByScore(context.Background(), "tre:v2:hist:"+podKey, &redis.ZRangeBy{Min: "12345", Max: "12345"}).Result()
	require.NoError(t, err)
	require.Len(t, histEntries, 1)
	require.Contains(t, histEntries[0], "model_histogram_metrics")

	instEntries, err := client.ZRangeByScore(context.Background(), "tre:v2:inst:"+podKey, &redis.ZRangeBy{Min: "12345", Max: "12345"}).Result()
	require.NoError(t, err)
	require.Len(t, instEntries, 1)
	require.Contains(t, instEntries[0], "model_metrics")

	pods, err := client.SMembers(context.Background(), "tre:v2:pods:dsqwen-7b").Result()
	require.NoError(t, err)
	require.ElementsMatch(t, []string{podKey}, pods)

	legacyKeys, err := client.Keys(context.Background(), "aibrix:pod_*_metrics_*").Result()
	require.NoError(t, err)
	require.Empty(t, legacyKeys)
}

func TestWriteTREPodMetricsToRedisDefaultsToDualSchema(t *testing.T) {
	store, client := newTREMetricStoreForTest(t)

	err := treWrite(context.Background(), store, treRoundAt(70000))
	require.NoError(t, err)

	podKey := utils.GeneratePodKey("default", "pod-a")
	v2Entries, err := client.ZRangeByScore(context.Background(), "tre:v2:hist:"+podKey, &redis.ZRangeBy{Min: "70000", Max: "70000"}).Result()
	require.NoError(t, err)
	require.Len(t, v2Entries, 1)

	legacyKeys, err := client.Keys(context.Background(), "aibrix:pod_histogram_metrics_"+podKey+"_*").Result()
	require.NoError(t, err)
	require.Len(t, legacyKeys, 1)

	raw, err := client.Get(context.Background(), legacyKeys[0]).Bytes()
	require.NoError(t, err)
	var doc map[string]any
	require.NoError(t, json.Unmarshal(raw, &doc))
	require.Equal(t, float64(70000), doc["timestamp"])
}

func TestWriteTREPodMetricsToRedisStampsWrittenMSAndKeepsOneDocPerBoundary(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "v2")
	store, client := newTREMetricStoreForTest(t)
	clock := int64(1_000_000)
	saved := treWallClockMS
	treWallClockMS = func() int64 { return clock }
	t.Cleanup(func() { treWallClockMS = saved })

	require.NoError(t, treWrite(context.Background(), store, treRoundAt(990_000)))
	clock = 1_000_250 // the same boundary written again (retry / overlap)
	require.NoError(t, treWrite(context.Background(), store, treRoundAt(990_000)))

	podKey := utils.GeneratePodKey("default", "pod-a")
	for _, prefix := range []string{"tre:v2:inst:", "tre:v2:hist:"} {
		entries, err := client.ZRangeByScore(context.Background(), prefix+podKey, &redis.ZRangeBy{Min: "990000", Max: "990000"}).Result()
		require.NoError(t, err)
		require.Len(t, entries, 1, prefix)
		var doc map[string]any
		require.NoError(t, json.Unmarshal([]byte(entries[0]), &doc))
		require.Equal(t, float64(990_000), doc["timestamp"])
		require.Equal(t, float64(1_000_250), doc["written_ms"])
	}
}

// A failed scrape keeps the pod's old metrics, so scraped_ms must not advance: the
// controller tells frozen metrics from fresh ones by it.
func TestTREPodDocScrapedMSAdvancesOnlyOnSuccessfulScrape(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "v2")
	ctx := context.Background()

	// Engine stub: serves vLLM gauges until failing is set, then answers 500.
	var failing atomic.Bool
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if failing.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, `# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="dsqwen-7b"} 2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="dsqwen-7b"} 3
`)
	}))
	t.Cleanup(engine.Close)
	addr := engine.Listener.Addr().(*net.TCPAddr)

	// Keep the scrape's Prometheus re-export off the process-global registry.
	savedGauge := metrics.SetGaugeMetricFnForTest
	savedCounter := metrics.IncrementCounterMetricFnForTest
	metrics.SetGaugeMetricFnForTest = func(string, string, float64, []string, ...string) {}
	metrics.IncrementCounterMetricFnForTest = func(string, string, float64, []string, ...string) {}
	t.Cleanup(func() {
		metrics.SetGaugeMetricFnForTest = savedGauge
		metrics.IncrementCounterMetricFnForTest = savedCounter
	})

	clock := int64(1_000_000)
	savedClock := treWallClockMS
	treWallClockMS = func() int64 { return clock }
	t.Cleanup(func() { treWallClockMS = savedClock })

	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	pod := &v1.Pod{}
	pod.Name = "pod-a"
	pod.Namespace = "default"
	pod.Labels = map[string]string{MetricPortLabel: strconv.Itoa(addr.Port)}
	pod.Status.PodIP = addr.IP.String()
	pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
	store := InitWithPods(NewForTest(), []*v1.Pod{pod}, "dsqwen-7b")
	store.redisClient = client
	// Seed a histogram so a histogram doc is written too (the stub serves gauges only).
	store = InitWithPodsModelMetrics(store, map[string]map[string]metrics.MetricValue{
		"pod-a": {
			metrics.TimeToFirstTokenSeconds: &metrics.HistogramMetricValue{
				Sum:     1.5,
				Count:   2,
				Buckets: map[string]float64{"0.5": 1, "+Inf": 2},
			},
		},
	})
	// No retries: a failed scrape returns at once.
	store.engineMetricsFetcher = metrics.NewEngineMetricsFetcherWithConfig(metrics.EngineMetricsFetcherConfig{Timeout: 2 * time.Second})

	podKey := utils.GeneratePodKey("default", "pod-a")
	metaPod, ok := store.metaPods.Load(podKey)
	require.True(t, ok)
	waitingKey := store.getPodModelMetricName("dsqwen-7b", metrics.NumRequestsWaiting)

	readDoc := func(prefix string, roundT int64) map[string]any {
		t.Helper()
		boundary := strconv.FormatInt(roundT, 10)
		entries, err := client.ZRangeByScore(ctx, prefix+podKey, &redis.ZRangeBy{Min: boundary, Max: boundary}).Result()
		require.NoError(t, err)
		require.Len(t, entries, 1, prefix)
		var doc map[string]any
		require.NoError(t, json.Unmarshal([]byte(entries[0]), &doc))
		return doc
	}
	waitingIn := func(doc map[string]any) any {
		t.Helper()
		modelMetrics, ok := doc["model_metrics"].(map[string]any)
		require.True(t, ok)
		return modelMetrics[waitingKey]
	}

	// Never scraped: the field is omitted.
	require.NoError(t, treWrite(ctx, store, treRoundAt(990_000)))
	require.NotContains(t, readDoc(treV2HistogramKeyPrefix, 990_000), "scraped_ms")

	// Successful scrape: both docs carry its time.
	clock = 1_000_100
	store.refreshPodMetrics(metaPod)
	clock = 1_000_200
	require.NoError(t, treWrite(ctx, store, treRoundAt(1_000_000)))
	inst := readDoc(treV2InstantKeyPrefix, 1_000_000)
	require.Equal(t, float64(1_000_100), inst["scraped_ms"])
	require.Equal(t, float64(3), waitingIn(inst))
	require.Equal(t, float64(1_000_100), readDoc(treV2HistogramKeyPrefix, 1_000_000)["scraped_ms"])

	// Failed scrape: old metrics are still written with a fresh written_ms, but
	// scraped_ms stays at the last success.
	failing.Store(true)
	clock = 1_005_100
	store.refreshPodMetrics(metaPod)
	clock = 1_005_200
	require.NoError(t, treWrite(ctx, store, treRoundAt(1_005_000)))
	inst = readDoc(treV2InstantKeyPrefix, 1_005_000)
	require.Equal(t, float64(1_005_200), inst["written_ms"])
	require.Equal(t, float64(1_000_100), inst["scraped_ms"])
	require.Equal(t, float64(3), waitingIn(inst))
	require.Equal(t, float64(1_000_100), readDoc(treV2HistogramKeyPrefix, 1_005_000)["scraped_ms"])
}

func newTREMetricStoreForTest(t *testing.T) (*Store, *redis.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	pod := &v1.Pod{}
	pod.Name = "pod-a"
	pod.Namespace = "default"
	pod.Status.PodIP = "10.0.0.1"
	pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}

	store := InitWithPods(NewForTest(), []*v1.Pod{pod}, "dsqwen-7b")
	store.redisClient = client
	store = InitWithPodsModelMetrics(store, map[string]map[string]metrics.MetricValue{
		"pod-a": {
			metrics.NumRequestsWaiting: &metrics.SimpleMetricValue{Value: 3},
			metrics.NumRequestsRunning: &metrics.SimpleMetricValue{Value: 2},
			metrics.TimeToFirstTokenSeconds: &metrics.HistogramMetricValue{
				Sum:   1.5,
				Count: 2,
				Buckets: map[string]float64{
					"0.5": 1,
					"1.0": 2,
				},
			},
		},
	})

	return store, client
}

func TestTREMetricSchemaModeRejectsUnknownValue(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "invalid")
	_, _, err := treMetricSchemaMode()
	require.Error(t, err)
	require.True(t, strings.Contains(err.Error(), "TRE_REDIS_SCHEMA"))
}

// treRoundAt is a round on an evidence boundary: instant and histogram docs at stamp.
func treRoundAt(stamp int64) treRound {
	return treRound{instantStamp: stamp, histStamp: stamp}
}

// Instant docs every tick, histogram docs on the first round of each evidence boundary
// (stamped with the boundary, however late the round) - including the boundary in
// progress when the writer (re)starts.
func TestTRERoundPlannerInstantEveryTickHistOnEvidenceBoundary(t *testing.T) {
	p := &treRoundPlanner{instantMS: 500, evidenceMS: 10_000}
	var hist []int64
	instant := 0
	for roundT := int64(1_003_500); roundT <= 1_030_000; roundT += 500 {
		r := p.next(roundT)
		p.commit(r)
		require.Equal(t, roundT, r.instantStamp)
		instant++
		if r.histStamp != 0 {
			hist = append(hist, r.histStamp)
		}
	}
	require.Equal(t, 54, instant)
	// First boundary after the (re)start: written, its scraped_ms says when.
	require.Equal(t, []int64{1_000_000, 1_010_000, 1_020_000, 1_030_000}, hist)

	// The round at 1_040_000 came 1.5 s late: it still writes boundary 1_040_000.
	r := p.next(1_041_500)
	require.Equal(t, treRound{instantStamp: 1_041_500, histStamp: 1_040_000}, r)
	p.commit(r)
	require.Equal(t, int64(0), p.next(1_042_000).histStamp)
}

// Review round 1 / round 3 (L2): after a failed MULTI/EXEC the boundary is not lost -
// the next round (runTRERound, the loop's own commit decision) writes its histogram
// doc, stamped with the boundary; the declared-cadence hash is in the same MULTI.
func TestTREBoundaryRetriedAfterExecFailure(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "v2")
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store, _ := newTREMetricStoreForTest(t)
	store.redisClient = client
	store.treCadence = treWriteCadence{instant: 500 * time.Millisecond, evidence: 2 * time.Second}
	ctx := context.Background()
	p := &treRoundPlanner{instantMS: 500, evidenceMS: 2_000, lastHistT: 1_008_000}

	mr.SetError("LOADING redis is loading")
	_, err := runTRERound(ctx, store, p, 1_010_040)
	require.ErrorIs(t, err, errTREExec)
	require.Equal(t, int64(1_008_000), p.lastHistT)
	mr.SetError("")
	r, err := runTRERound(ctx, store, p, 1_010_530)
	require.NoError(t, err)
	require.Equal(t, treRound{instantStamp: 1_010_500, histStamp: 1_010_000}, r)

	podKey := utils.GeneratePodKey("default", "pod-a")
	entries, err := client.ZRangeByScore(ctx, treV2HistogramKeyPrefix+podKey, &redis.ZRangeBy{Min: "-inf", Max: "+inf"}).Result()
	require.NoError(t, err)
	require.Len(t, entries, 1)
	var doc map[string]any
	require.NoError(t, json.Unmarshal([]byte(entries[0]), &doc))
	require.Equal(t, float64(1_010_000), doc["timestamp"])
	require.Equal(t, int64(1_010_000), p.lastHistT)

	cadence, err := client.HGetAll(ctx, treV2CadenceKey).Result()
	require.NoError(t, err)
	require.Equal(t, "500", cadence["instant_ms"])
	require.Equal(t, "2000", cadence["evidence_ms"])
	require.NotEmpty(t, cadence["written_ms"])
	// Round 5: the round's histogram boundary, written with its docs (the failed
	// round never appears).
	require.Equal(t, "1010000", cadence["last_hist_ms"])
}

// Review round 3 (L3): a histogram doc that cannot be built (a NaN sum: json.Marshal
// refuses it) does not move lastHistT; once it can be built the same boundary is
// written by the next round.
func TestTREBoundaryRetriedAfterHistogramBuildError(t *testing.T) {
	t.Setenv("TRE_REDIS_SCHEMA", "v2")
	store, client := newTREMetricStoreForTest(t)
	store.treCadence = treWriteCadence{instant: 500 * time.Millisecond, evidence: 2 * time.Second}
	ctx := context.Background()
	p := &treRoundPlanner{instantMS: 500, evidenceMS: 2_000, lastHistT: 1_008_000}
	podKey := utils.GeneratePodKey("default", "pod-a")
	metaPod, ok := store.metaPods.Load(podKey)
	require.True(t, ok)
	ttftKey := store.getPodModelMetricName("dsqwen-7b", metrics.TimeToFirstTokenSeconds)
	good, ok := metaPod.ModelMetrics.Load(ttftKey)
	require.True(t, ok)
	metaPod.ModelMetrics.Store(ttftKey, &metrics.HistogramMetricValue{Sum: math.NaN(), Count: 2, Buckets: map[string]float64{"+Inf": 2}})

	_, err := runTRERound(ctx, store, p, 1_010_000)
	require.Error(t, err)
	require.Equal(t, int64(1_008_000), p.lastHistT)
	inst, err := client.ZCard(ctx, treV2InstantKeyPrefix+podKey).Result()
	require.NoError(t, err)
	require.Equal(t, int64(1), inst) // the instant doc of the round is written
	// Round 5 (L2): the boundary is not declared written while a doc of it is missing.
	declared, err := client.HGet(ctx, treV2CadenceKey, "last_hist_ms").Result()
	require.ErrorIs(t, err, redis.Nil, declared)

	metaPod.ModelMetrics.Store(ttftKey, good)
	r, err := runTRERound(ctx, store, p, 1_010_500)
	require.NoError(t, err)
	require.Equal(t, int64(1_010_000), r.histStamp)
	require.Equal(t, int64(1_010_000), p.lastHistT)
	declared, err = client.HGet(ctx, treV2CadenceKey, "last_hist_ms").Result()
	require.NoError(t, err)
	require.Equal(t, "1010000", declared)
}

// Review round 3 (L4): of two overlapping scrapes of one pod, the older fetch that
// finishes last is dropped; the stamp is taken when the values are stored and is read
// with them.
func TestTREScrapeStampBelongsToTheStoredValues(t *testing.T) {
	clock := int64(1_000_000)
	saved := treWallClockMS
	treWallClockMS = func() int64 { return clock }
	t.Cleanup(func() { treWallClockMS = saved })
	pod := &Pod{}
	value := "none"

	clock = 1_000_400 // the newer fetch (started at 1_000_200) finishes first
	require.True(t, pod.storeScrape(1_000_200, func() { value = "new" }))
	clock = 1_000_900 // the older fetch (started at 1_000_100) finishes last: dropped
	require.False(t, pod.storeScrape(1_000_100, func() { value = "old" }))
	pod.readScrape(func(scrapedMS int64, ok bool) {
		require.True(t, ok)
		require.Equal(t, int64(1_000_400), scrapedMS)
		require.Equal(t, "new", value)
	})
}

func treWrite(ctx context.Context, store *Store, r treRound) error {
	_, err := store.writeTREPodMetricsToRedis(ctx, r)
	return err
}

// An instant-only round writes the instant docs of every pod in one pipeline and no
// histogram doc or v1 legacy key; the legacy keys stay on the 10 s grid at any evidence
// period (their 2 h TTL would otherwise multiply the key count).
func TestWriteTREPodMetricsInstantRoundSkipsHistogramAndLegacy(t *testing.T) {
	store, client := newTREMetricStoreForTest(t)
	ctx := context.Background()
	require.NoError(t, treWrite(ctx, store, treRound{instantStamp: 1_000_500}))

	podKey := utils.GeneratePodKey("default", "pod-a")
	inst, err := client.ZRangeByScore(ctx, treV2InstantKeyPrefix+podKey, &redis.ZRangeBy{Min: "-inf", Max: "+inf"}).Result()
	require.NoError(t, err)
	require.Len(t, inst, 1)
	hist, err := client.ZCard(ctx, treV2HistogramKeyPrefix+podKey).Result()
	require.NoError(t, err)
	require.Zero(t, hist)
	legacy, err := client.Keys(ctx, "aibrix:pod_*_metrics_*").Result()
	require.NoError(t, err)
	require.Empty(t, legacy)

	// A 2 s evidence boundary off the 10 s grid: histogram doc, still no legacy key.
	require.NoError(t, treWrite(ctx, store, treRoundAt(1_012_000)))
	hist, err = client.ZCard(ctx, treV2HistogramKeyPrefix+podKey).Result()
	require.NoError(t, err)
	require.Equal(t, int64(1), hist)
	legacy, err = client.Keys(ctx, "aibrix:pod_*_metrics_*").Result()
	require.NoError(t, err)
	require.Empty(t, legacy)

	require.NoError(t, treWrite(ctx, store, treRoundAt(1_010_000)))
	legacy, err = client.Keys(ctx, "aibrix:pod_*_metrics_*").Result()
	require.NoError(t, err)
	require.Len(t, legacy, 2)
}
