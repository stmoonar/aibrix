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
	"os"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/utils"
	"k8s.io/klog/v2"
)

const (
	treRedisSchemaEnv       = "TRE_REDIS_SCHEMA"
	treV2RetentionMS        = int64(30 * 60 * 1000)
	treV2FallbackTTL        = 2 * time.Hour
	treV1PodMetricsTTL      = treV2FallbackTTL
	treV2HistogramKeyPrefix = "tre:v2:hist:"
	treV2InstantKeyPrefix   = "tre:v2:inst:"
	treV2PodsKeyPrefix      = "tre:v2:pods:"
	treV1HistogramKeyPrefix = "aibrix:pod_histogram_metrics_"
	treV1InstantKeyPrefix   = "aibrix:pod_instant_metrics_"

	// TRE (2026-10-25): two write periods, both independent of the upstream request
	// trace interval (RequestTraceWriteInterval, unchanged).
	//  - instant sample period: tre:v2:inst docs (queue gauges, KV-cache fill). The
	//    controller checks per-pod completeness and freshness on this grid
	//    (controller env of the same name).
	//  - evidence period: tre:v2:hist docs (token / latency histograms) and the v1 legacy
	//    keys. The controller's phase-aligned window grid, O1 grid and SafeScale evidence
	//    step (controller TRE_EVIDENCE_PERIOD_MS). Must be a multiple of the instant period.
	treInstantIntervalEnv     = "TRE_INSTANT_SAMPLE_INTERVAL_MS"
	treEvidencePeriodEnv      = "TRE_EVIDENCE_PERIOD_MS"
	defaultTREInstantInterval = 500 * time.Millisecond
	defaultTREEvidencePeriod  = 10 * time.Second
)

// treWriteCadence is the pair of TRE doc write periods (see the constants above).
type treWriteCadence struct {
	instant  time.Duration
	evidence time.Duration
}

// loadTREWriteCadence reads the two periods from the environment. An invalid pair
// (non-positive, or an evidence period that is not a multiple of the instant period)
// falls back to the defaults with an error log; the controller's startup cadence check
// then reports any mismatch with its own settings.
func loadTREWriteCadence() treWriteCadence {
	cadence := treWriteCadence{
		instant:  time.Duration(utils.LoadEnvInt(treInstantIntervalEnv, int(defaultTREInstantInterval/time.Millisecond))) * time.Millisecond,
		evidence: time.Duration(utils.LoadEnvInt(treEvidencePeriodEnv, int(defaultTREEvidencePeriod/time.Millisecond))) * time.Millisecond,
	}
	if cadence.instant <= 0 || cadence.evidence <= 0 || cadence.evidence%cadence.instant != 0 {
		klog.Errorf("TRE write cadence: %s=%s must divide %s=%s; using %s / %s",
			treInstantIntervalEnv, cadence.instant, treEvidencePeriodEnv, cadence.evidence,
			defaultTREInstantInterval, defaultTREEvidencePeriod)
		return treWriteCadence{instant: defaultTREInstantInterval, evidence: defaultTREEvidencePeriod}
	}
	return cadence
}

// treRound is what one writer tick writes: instant docs stamped instantStamps (one, or
// two on a catch-up) and, when histStamp != 0, the histogram docs stamped histStamp.
type treRound struct {
	instantStamps []int64
	histStamp     int64
}

// treRoundPlanner turns writer ticks (roundT on the instant grid) into rounds. A
// histogram doc is written on every evidence boundary B. If the tick at B was late
// by up to one instant period (rounded to B + instant), B is caught up in that tick:
// histogram AND instant docs stamped B, so the controller's window ending at B still
// has its end docs (sampled at most one instant period late; their scraped_ms tells).
// A boundary missed by more is lost (the controller then sees the pods' window end
// missing and holds). The boundary in progress when the writer starts is not caught up.
type treRoundPlanner struct {
	instantMS  int64
	evidenceMS int64
	lastHistT  int64
	started    bool
}

func (p *treRoundPlanner) next(roundT int64) treRound {
	round := treRound{instantStamps: []int64{roundT}}
	boundary := roundT - floorMod(roundT, p.evidenceMS)
	if boundary > p.lastHistT {
		if roundT == boundary || (p.started && roundT-boundary <= p.instantMS) {
			round.histStamp = boundary
			if roundT != boundary {
				round.instantStamps = []int64{boundary, roundT}
			}
		}
		p.lastHistT = boundary
	}
	p.started = true
	return round
}

func floorMod(a, b int64) int64 {
	if b <= 0 {
		return 0
	}
	m := a % b
	if m < 0 {
		m += b
	}
	return m
}

var treInstantMetricNames = map[string]struct{}{
	metrics.NumRequestsRunning:         {},
	metrics.NumRequestsWaiting:         {},
	metrics.NumRequestsSwapped:         {},
	metrics.GPUCacheUsagePerc:          {},
	metrics.KVCacheHitRate:             {},
	metrics.RealtimeNumRequestsRunning: {},
	metrics.RealtimeNormalizedPendings: {},
}

var treHistogramMetricNames = map[string]struct{}{
	metrics.TimeToFirstTokenSeconds:   {},
	metrics.TimePerOutputTokenSeconds: {},
	metrics.E2ERequestLatencySeconds:  {},
	metrics.RequestPromptTokens:       {},
	metrics.RequestGenerationTokens:   {},
}

// TRE-PATCH(P2-GW-003): write TRE pod metrics using v1/v2/dual Redis schemas.
//
// TRE (2026-10-25): every pod of a round goes into ONE MULTI/EXEC pipeline (one round
// trip, and a reader sees a round for all pods or for none). v1 under CPU throttling
// lost ticks with one serial SET per pod; the controller then averaged over too few
// samples. The v1 legacy keys (one SET per doc, 2 h TTL) stay on the 10 s grid of
// RequestTraceWriteInterval whatever the TRE periods, so their key count is unchanged.
func (c *Store) writeTREPodMetricsToRedis(ctx context.Context, round treRound) error {
	if c.redisClient == nil {
		return nil
	}
	writeV1, writeV2, err := treMetricSchemaMode()
	if err != nil {
		return err
	}

	legacyRound := writeV1 && round.histStamp != 0 && floorMod(round.histStamp, int64(RequestTraceWriteInterval/time.Millisecond)) == 0

	var firstErr error
	pipe := c.redisClient.TxPipeline()
	models := make(map[string]struct{})
	c.metaPods.Range(func(podKey string, metaPod *Pod) bool {
		if !utils.FilterReadyPod(metaPod.Pod) {
			return true
		}

		// TRE O1 (2026-10-01): written_ms is the gateway wall clock at the write, so
		// the controller can measure the gateway-vs-controller clock offset directly
		// (timestamp is the boundary, written up to one period later).
		base := map[string]any{
			"written_ms":    treWallClockMS(),
			"pod_name":      metaPod.Name,
			"pod_namespace": metaPod.Namespace,
			"pod_ip":        metaPod.Status.PodIP,
		}
		// TRE (2026-10-04): scraped_ms is the gateway wall clock of the pod's last
		// successful engine scrape (omitted if none yet). A failed scrape keeps the old
		// metrics but does not advance it, so the controller can tell stale metrics from
		// fresh ones. Read before the metrics: they are at least as new as scraped_ms.
		if scrapedMS, ok := metaPod.lastScrapeSuccess(); ok {
			base["scraped_ms"] = scrapedMS
		}
		instantMetrics, instantModels := collectTREInstantMetrics(metaPod)
		var histogramMetrics map[string]any
		var histogramModels map[string]struct{}
		if round.histStamp != 0 {
			histogramMetrics, histogramModels = collectTREHistogramMetrics(metaPod)
		}

		if len(instantMetrics) > 0 {
			for _, stamp := range round.instantStamps {
				doc := cloneMetricEnvelope(base)
				doc["timestamp"] = stamp
				doc["model_metrics"] = instantMetrics
				legacy := legacyRound && stamp == round.histStamp
				if err := queueTREMetricDocument(ctx, pipe, podKey, stamp, doc, false, legacy, writeV2); err != nil && firstErr == nil {
					firstErr = err
				}
			}
		}
		if len(histogramMetrics) > 0 {
			doc := cloneMetricEnvelope(base)
			doc["timestamp"] = round.histStamp
			doc["model_histogram_metrics"] = histogramMetrics
			if err := queueTREMetricDocument(ctx, pipe, podKey, round.histStamp, doc, true, legacyRound, writeV2); err != nil && firstErr == nil {
				firstErr = err
			}
		}

		if writeV2 {
			for model := range unionStringSets(instantModels, histogramModels) {
				pipe.SAdd(ctx, treV2PodsKeyPrefix+model, podKey)
				models[model] = struct{}{}
			}
		}
		return true
	})
	for model := range models {
		pipe.Expire(ctx, treV2PodsKeyPrefix+model, treV2FallbackTTL)
	}
	if pipe.Len() == 0 {
		return firstErr
	}
	if _, err := pipe.Exec(ctx); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// treWallClockMS is the gateway wall clock in ms (a variable so tests can pin it).
var treWallClockMS = func() int64 { return time.Now().UnixMilli() }

func treMetricSchemaMode() (bool, bool, error) {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(treRedisSchemaEnv))) {
	case "", "dual":
		return true, true, nil
	case "v1":
		return true, false, nil
	case "v2":
		return false, true, nil
	default:
		return false, false, fmt.Errorf("%s must be one of v1, v2, or dual", treRedisSchemaEnv)
	}
}

// queueTREMetricDocument adds one doc's writes to the round's pipeline.
func queueTREMetricDocument(ctx context.Context, pipe redis.Pipeliner, podKey string, stamp int64, doc map[string]any, histogram bool, writeV1 bool, writeV2 bool) error {
	value, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if writeV1 {
		prefix := treV1InstantKeyPrefix
		if histogram {
			prefix = treV1HistogramKeyPrefix
		}
		pipe.Set(ctx, fmt.Sprintf("%s%s_%d", prefix, podKey, stamp), value, treV1PodMetricsTTL)
	}
	if writeV2 {
		prefix := treV2InstantKeyPrefix
		if histogram {
			prefix = treV2HistogramKeyPrefix
		}
		key := prefix + podKey
		// One doc per boundary: written_ms makes every write unique, so a rewrite of
		// the same boundary replaces the earlier doc instead of adding a second one
		// (the controller counts a window's instant docs).
		boundary := fmt.Sprintf("%d", stamp)
		pipe.ZRemRangeByScore(ctx, key, boundary, boundary)
		pipe.ZAdd(ctx, key, redis.Z{Score: float64(stamp), Member: string(value)})
		pipe.ZRemRangeByScore(ctx, key, "-inf", fmt.Sprintf("%d", stamp-treV2RetentionMS))
		pipe.Expire(ctx, key, treV2FallbackTTL)
	}
	return nil
}

func collectTREInstantMetrics(metaPod *Pod) (map[string]any, map[string]struct{}) {
	values := make(map[string]any)
	models := make(map[string]struct{})
	metaPod.ModelMetrics.Range(func(key string, metricValue metrics.MetricValue) bool {
		model, metricName, ok := splitPodModelMetricKey(key)
		if !ok {
			return true
		}
		if _, wanted := treInstantMetricNames[metricName]; !wanted {
			return true
		}
		simpleValue, ok := metricValue.(*metrics.SimpleMetricValue)
		if !ok {
			return true
		}
		values[key] = simpleValue.GetSimpleValue()
		models[model] = struct{}{}
		return true
	})
	return values, models
}

func collectTREHistogramMetrics(metaPod *Pod) (map[string]any, map[string]struct{}) {
	values := make(map[string]any)
	models := make(map[string]struct{})
	metaPod.ModelMetrics.Range(func(key string, metricValue metrics.MetricValue) bool {
		model, metricName, ok := splitPodModelMetricKey(key)
		if !ok {
			return true
		}
		if _, wanted := treHistogramMetricNames[metricName]; !wanted {
			return true
		}
		histValue, ok := metricValue.(*metrics.HistogramMetricValue)
		if !ok {
			return true
		}
		values[key] = map[string]any{
			"sum":     histValue.GetSum(),
			"count":   histValue.GetCount(),
			"buckets": histValue.Buckets,
		}
		models[model] = struct{}{}
		return true
	})
	return values, models
}

func splitPodModelMetricKey(key string) (string, string, bool) {
	model, metricName, ok := strings.Cut(key, "/")
	if !ok || model == "" || metricName == "" {
		return "", "", false
	}
	return model, metricName, true
}

func cloneMetricEnvelope(in map[string]any) map[string]any {
	out := make(map[string]any, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}

func unionStringSets(a map[string]struct{}, b map[string]struct{}) map[string]struct{} {
	out := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		out[k] = struct{}{}
	}
	for k := range b {
		out[k] = struct{}{}
	}
	return out
}

// nearestBoundaryMS returns the wall-clock multiple of intervalMS nearest to
// nowMS. The ticker runs on the monotonic clock, aligned once at start, so a
// tick lands just after a boundary; if the wall clock later steps backward a
// little, a floor would stamp the tick with the previous boundary (written
// twice, the current one skipped until a restart). Rounding to the nearest
// boundary keeps every tick on its own boundary while |step| < intervalMS/2.
func nearestBoundaryMS(nowMS, intervalMS int64) int64 {
	if intervalMS <= 0 {
		return nowMS
	}
	shifted := nowMS + intervalMS/2
	rem := shifted % intervalMS
	if rem < 0 {
		rem += intervalMS
	}
	return shifted - rem
}

// initTREPodMetricsTraceCache writes the TRE instant docs once per instant sample
// period and the histogram docs on every evidence boundary (treRoundPlanner). Like the
// upstream trace writer (initTraceCache), it first waits for the next wall-clock
// boundary, so the doc for boundary T is written at about T instead of
// T + (pod start time mod interval). Without this the phase the controller learns
// depended on when the gateway started.
func initTREPodMetricsTraceCache(store *Store, stopCh <-chan struct{}) {
	cadence := loadTREWriteCadence()
	instantMS := int64(cadence.instant / time.Millisecond)
	planner := &treRoundPlanner{instantMS: instantMS, evidenceMS: int64(cadence.evidence / time.Millisecond)}
	// Start within the first quarter of an instant period after a boundary, else wait.
	maxOffset := cadence.instant / 4
	go func() {
		if delay := tickerAlignmentDelay(time.Now(), cadence.instant, maxOffset); delay > 0 {
			alignTimer := time.NewTimer(delay)
			select {
			case <-alignTimer.C:
			case <-stopCh:
				alignTimer.Stop()
				return
			}
		}
		ticker := time.NewTicker(cadence.instant)
		defer ticker.Stop()
		klog.Infof("TRE pod metrics ticker start at %s (instant %s, evidence %s)", time.Now(), cadence.instant, cadence.evidence)
		for {
			select {
			case <-ticker.C:
				nowMS := time.Now().UnixMilli()
				roundT := nearestBoundaryMS(nowMS, instantMS)
				if err := store.writeTREPodMetricsToRedis(context.Background(), planner.next(roundT)); err != nil {
					klog.ErrorS(err, "error storing TRE pod metrics to redis")
				}
			case <-stopCh:
				return
			}
		}
	}()
}
