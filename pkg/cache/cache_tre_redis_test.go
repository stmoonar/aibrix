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
	"strings"
	"testing"

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

	err := store.writeTREPodMetricsToRedis(context.Background(), 12345)
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

	err := store.writeTREPodMetricsToRedis(context.Background(), 67890)
	require.NoError(t, err)

	podKey := utils.GeneratePodKey("default", "pod-a")
	v2Entries, err := client.ZRangeByScore(context.Background(), "tre:v2:hist:"+podKey, &redis.ZRangeBy{Min: "67890", Max: "67890"}).Result()
	require.NoError(t, err)
	require.Len(t, v2Entries, 1)

	legacyKeys, err := client.Keys(context.Background(), "aibrix:pod_histogram_metrics_"+podKey+"_*").Result()
	require.NoError(t, err)
	require.Len(t, legacyKeys, 1)

	raw, err := client.Get(context.Background(), legacyKeys[0]).Bytes()
	require.NoError(t, err)
	var doc map[string]any
	require.NoError(t, json.Unmarshal(raw, &doc))
	require.Equal(t, float64(67890), doc["timestamp"])
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
