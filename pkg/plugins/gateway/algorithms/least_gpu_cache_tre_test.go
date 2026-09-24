/*
Copyright 2025 The Aibrix Team.

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

package routingalgorithms

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
)

// TRE-PATCH(P2-GW-004): least-gpu-cache under the routable-label gate. Route() itself is
// unchanged from v1 (min gpu_cache_usage_perc, uniform tie-break, pods without the metric
// skipped); its only candidate filter is the random fallback when no pod has the metric,
// which goes through utils.SelectRandomPod and therefore the gate. Not parallel.

func treLGCPod(name, ip, routable string) *v1.Pod {
	return newPod(name, ip, true, map[string]string{
		"model.aibrix.ai/port": "8000",
		utils.TRERoutableLabel: routable,
	})
}

func TestLeastGpuCache_FallbackHonoursTRERoutableGate(t *testing.T) {
	prev := utils.SetTRERoutableLabelFilter(true)
	defer utils.SetTRERoutableLabelFilter(prev)

	pods := []*v1.Pod{
		treLGCPod("sleeping-a", "1.1.1.1", "false"),
		treLGCPod("sleeping-b", "2.2.2.2", "false"),
		treLGCPod("awake", "3.3.3.3", "true"),
	}
	c := cache.NewWithPodsModelMetricsForTest(pods, "m1", map[string]map[string]metrics.MetricValue{})
	r := leastGpuCacheRouter{cache: c}
	for i := 0; i < 50; i++ {
		ctx := types.NewRoutingContext(context.Background(), RouterLeastGpuCache, "m1", "", "req", "")
		target, err := r.Route(ctx, podsFromCache(c))
		assert.NoError(t, err)
		assert.Equal(t, "3.3.3.3:8000", target)
	}
}

// v1 parity: ties are broken uniformly at random among the minimum pods.
func TestLeastGpuCache_TieBreakCoversAllMinimumPods(t *testing.T) {
	pods := []*v1.Pod{
		treLGCPod("a", "1.1.1.1", "true"),
		treLGCPod("b", "2.2.2.2", "true"),
		treLGCPod("c", "3.3.3.3", "true"),
	}
	c := cache.NewWithPodsModelMetricsForTest(pods, "m1", map[string]map[string]metrics.MetricValue{
		"a": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.25}},
		"b": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.25}},
		"c": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.9}},
	})
	r := leastGpuCacheRouter{cache: c}
	seen := map[string]int{}
	for i := 0; i < 400; i++ {
		ctx := types.NewRoutingContext(context.Background(), RouterLeastGpuCache, "m1", "", "req", "")
		target, err := r.Route(ctx, podsFromCache(c))
		assert.NoError(t, err)
		seen[target]++
	}
	assert.Zero(t, seen["3.3.3.3:8000"])
	assert.Greater(t, seen["1.1.1.1:8000"], 100)
	assert.Greater(t, seen["2.2.2.2:8000"], 100)
}
