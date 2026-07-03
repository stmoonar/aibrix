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
)

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
func (c *Store) writeTREPodMetricsToRedis(ctx context.Context, roundT int64) error {
	if c.redisClient == nil {
		return nil
	}
	writeV1, writeV2, err := treMetricSchemaMode()
	if err != nil {
		return err
	}

	var firstErr error
	c.metaPods.Range(func(podKey string, metaPod *Pod) bool {
		if !utils.FilterReadyPod(metaPod.Pod) {
			return true
		}

		base := map[string]any{
			"timestamp":     roundT,
			"pod_name":      metaPod.Name,
			"pod_namespace": metaPod.Namespace,
			"pod_ip":        metaPod.Status.PodIP,
		}
		instantMetrics, instantModels := collectTREInstantMetrics(metaPod)
		histogramMetrics, histogramModels := collectTREHistogramMetrics(metaPod)

		if len(instantMetrics) > 0 {
			doc := cloneMetricEnvelope(base)
			doc["model_metrics"] = instantMetrics
			if err := c.writeTREMetricDocument(ctx, podKey, roundT, doc, false, writeV1, writeV2); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		if len(histogramMetrics) > 0 {
			doc := cloneMetricEnvelope(base)
			doc["model_histogram_metrics"] = histogramMetrics
			if err := c.writeTREMetricDocument(ctx, podKey, roundT, doc, true, writeV1, writeV2); err != nil && firstErr == nil {
				firstErr = err
			}
		}

		if writeV2 {
			for model := range unionStringSets(instantModels, histogramModels) {
				if err := c.redisClient.SAdd(ctx, treV2PodsKeyPrefix+model, podKey).Err(); err != nil && firstErr == nil {
					firstErr = err
				}
				if err := c.redisClient.Expire(ctx, treV2PodsKeyPrefix+model, treV2FallbackTTL).Err(); err != nil && firstErr == nil {
					firstErr = err
				}
			}
		}
		return true
	})
	return firstErr
}

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

func (c *Store) writeTREMetricDocument(ctx context.Context, podKey string, roundT int64, doc map[string]any, histogram bool, writeV1 bool, writeV2 bool) error {
	value, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if writeV1 {
		prefix := treV1InstantKeyPrefix
		if histogram {
			prefix = treV1HistogramKeyPrefix
		}
		if err := c.redisClient.Set(ctx, fmt.Sprintf("%s%s_%d", prefix, podKey, roundT), value, treV1PodMetricsTTL).Err(); err != nil {
			return err
		}
	}
	if writeV2 {
		prefix := treV2InstantKeyPrefix
		if histogram {
			prefix = treV2HistogramKeyPrefix
		}
		key := prefix + podKey
		if err := c.redisClient.ZAdd(ctx, key, redis.Z{Score: float64(roundT), Member: string(value)}).Err(); err != nil {
			return err
		}
		cutoff := roundT - treV2RetentionMS
		if err := c.redisClient.ZRemRangeByScore(ctx, key, "-inf", fmt.Sprintf("%d", cutoff)).Err(); err != nil {
			return err
		}
		if err := c.redisClient.Expire(ctx, key, treV2FallbackTTL).Err(); err != nil {
			return err
		}
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

func initTREPodMetricsTraceCache(store *Store, stopCh <-chan struct{}) {
	ticker := time.NewTicker(RequestTraceWriteInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				nowMS := time.Now().UnixMilli()
				intervalMS := int64(RequestTraceWriteInterval / time.Millisecond)
				roundT := nowMS - nowMS%intervalMS
				if err := store.writeTREPodMetricsToRedis(context.Background(), roundT); err != nil {
					klog.ErrorS(err, "error storing TRE pod metrics to redis")
				}
			case <-stopCh:
				return
			}
		}
	}()
}
