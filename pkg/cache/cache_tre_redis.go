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
	"errors"
	"fmt"
	"os"
	"runtime/debug"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/utils"
	"k8s.io/klog/v2"
)

const (
	treRedisSchemaEnv = "TRE_REDIS_SCHEMA"
	// Retention per doc kind (review round 3, M3): histogram docs keep 30 min (the
	// controller's 90 s token-baseline lookback, SafeScale evidence, calibration
	// backfills); instant docs - 60 per pod per 30 s window at 0.5 s - keep 5 min (one
	// window plus margin; calibration captures spool them during a cell).
	treV2HistRetentionMS    = int64(30 * 60 * 1000)
	treV2InstantRetentionMS = int64(5 * 60 * 1000)
	treV2FallbackTTL        = 2 * time.Hour
	treV1PodMetricsTTL      = treV2FallbackTTL
	treV2HistogramKeyPrefix = "tre:v2:hist:"
	treV2InstantKeyPrefix   = "tre:v2:inst:"
	treV2PodsKeyPrefix      = "tre:v2:pods:"
	treV2CadenceKey         = "tre:v2:gateway:cadence"
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
	defaultTREEvidencePeriod  = 2 * time.Second
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

// treRound is what one writer tick writes: the instant docs stamped instantStamp and,
// when histStamp != 0, the histogram docs stamped histStamp.
type treRound struct {
	instantStamp int64
	histStamp    int64
}

// errTREExec marks a failed MULTI/EXEC of a round: nothing of the round was written.
var errTREExec = errors.New("TRE round MULTI/EXEC failed")

// treRoundPlanner turns writer ticks (roundT on the instant grid) into rounds
// (2026-10-25, review round 1).
//   - Every tick writes the instant docs stamped roundT.
//   - The histogram docs of evidence boundary B = floor(roundT / evidence) * evidence
//     are written on the first round with B > lastHistT, stamped B; their scraped_ms
//     says when the counters were observed (the controller divides the token delta by
//     that span). A late tick is no special case, and the boundary in progress when the
//     writer (re)starts is written too.
//   - lastHistT moves only after the round's histogram docs were all built and its
//     EXEC succeeded (commit(), via runTRERound), so a failed EXEC or a histogram doc
//     that could not be built retries B on the next round.
type treRoundPlanner struct {
	instantMS  int64
	evidenceMS int64
	lastHistT  int64
}

func (p *treRoundPlanner) next(roundT int64) treRound {
	round := treRound{instantStamp: roundT}
	if boundary := roundT - floorMod(roundT, p.evidenceMS); boundary > p.lastHistT {
		round.histStamp = boundary
	}
	return round
}

// commit records a round whose histogram docs were written.
func (p *treRoundPlanner) commit(round treRound) {
	if round.histStamp > p.lastHistT {
		p.lastHistT = round.histStamp
	}
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
//
// It returns histWritten = the round's histogram docs were all built and EXEC
// succeeded (review round 3, L3): only then may the planner move lastHistT. A schema
// error, a per-pod build error (e.g. a NaN json.Marshal refuses) or a failed EXEC
// leaves the boundary to be retried on the next round.
func (c *Store) writeTREPodMetricsToRedis(ctx context.Context, round treRound) (histWritten bool, err error) {
	if c.redisClient == nil {
		return false, nil
	}
	writeV1, writeV2, err := treMetricSchemaMode()
	if err != nil {
		return false, err
	}

	legacyRound := writeV1 && round.histStamp != 0 && floorMod(round.histStamp, int64(RequestTraceWriteInterval/time.Millisecond)) == 0

	var firstErr error
	histOK := round.histStamp != 0
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
		// TRE (2026-10-04): scraped_ms is the gateway wall clock at which the pod's
		// cached engine metrics were stored by their scrape (omitted if none yet). A
		// failed scrape keeps the old metrics and does not advance it, so the controller
		// can tell stale metrics from fresh ones. Read with the metrics under the pod's
		// scrape lock (review round 3, L4): the stamp always belongs to these values.
		var instantMetrics, histogramMetrics map[string]any
		var instantModels, histogramModels map[string]struct{}
		metaPod.readScrape(func(scrapedMS int64, ok bool) {
			if ok {
				base["scraped_ms"] = scrapedMS
			}
			instantMetrics, instantModels = collectTREInstantMetrics(metaPod)
			if round.histStamp != 0 {
				histogramMetrics, histogramModels = collectTREHistogramMetrics(metaPod)
			}
		})

		if len(instantMetrics) > 0 {
			doc := cloneMetricEnvelope(base)
			doc["timestamp"] = round.instantStamp
			doc["model_metrics"] = instantMetrics
			legacy := legacyRound && round.instantStamp == round.histStamp
			if err := queueTREMetricDocument(ctx, pipe, podKey, round.instantStamp, doc, false, legacy, writeV2); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		if len(histogramMetrics) > 0 {
			doc := cloneMetricEnvelope(base)
			doc["timestamp"] = round.histStamp
			doc["model_histogram_metrics"] = histogramMetrics
			if err := queueTREMetricDocument(ctx, pipe, podKey, round.histStamp, doc, true, legacyRound, writeV2); err != nil {
				histOK = false
				if firstErr == nil {
					firstErr = err
				}
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
	if writeV2 && pipe.Len() > 0 && c.treCadence.instant > 0 {
		// Declared cadence (review round 1, M4/L6): the controller compares it with its
		// own periods on every snapshot read (no inference from doc spacing).
		// last_round_ms (round 4) = this round's instant stamp, in the round's MULTI:
		// the controller's sampler takes a window ending at E as written once
		// last_round_ms >= E.
		pipe.HSet(ctx, treV2CadenceKey,
			"instant_ms", c.treCadence.instant.Milliseconds(),
			"evidence_ms", c.treCadence.evidence.Milliseconds(),
			"written_ms", treWallClockMS(),
			"last_round_ms", round.instantStamp,
			"version", treGatewayVersion())
		pipe.Expire(ctx, treV2CadenceKey, treV2FallbackTTL)
	}
	if pipe.Len() == 0 {
		return false, firstErr
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return false, fmt.Errorf("%w: %v", errTREExec, err)
	}
	return histOK, firstErr
}

// runTRERound is one writer tick at gateway wall clock nowMS (review round 3, L2: the
// loop and the tests run this same commit decision): the round of the nearest instant
// boundary is written, and the planner commits its histogram boundary only when every
// histogram doc of it was written.
func runTRERound(ctx context.Context, store *Store, planner *treRoundPlanner, nowMS int64) (treRound, error) {
	round := planner.next(nearestBoundaryMS(nowMS, planner.instantMS))
	histWritten, err := store.writeTREPodMetricsToRedis(ctx, round)
	if histWritten {
		planner.commit(round)
	}
	return round, err
}

// treGatewayVersion is the build of this gateway (the main module's VCS revision when
// the binary carries one, else its module version), for the declared-cadence hash.
func treGatewayVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			return setting.Value
		}
	}
	return info.Main.Version
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
		retention := treV2InstantRetentionMS
		if histogram {
			retention = treV2HistRetentionMS
		}
		pipe.ZRemRangeByScore(ctx, key, "-inf", fmt.Sprintf("%d", stamp-retention))
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
	store.treCadence = cadence
	planner := &treRoundPlanner{
		instantMS:  int64(cadence.instant / time.Millisecond),
		evidenceMS: int64(cadence.evidence / time.Millisecond),
	}
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
				if _, err := runTRERound(context.Background(), store, planner, treWallClockMS()); err != nil {
					klog.ErrorS(err, "error storing TRE pod metrics to redis")
				}
			case <-stopCh:
				return
			}
		}
	}()
}
