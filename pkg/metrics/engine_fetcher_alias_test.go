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

package metrics

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// vLLM 0.30 exports only the renamed families (no gpu_cache_usage_perc, no
// time_per_output_token_seconds).
const mockVllm030Metrics = `# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.4
# HELP vllm:inter_token_latency_seconds Histogram of inter-token latency in seconds.
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{engine="0",le="0.05",model_name="m"} 3.0
vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 4.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="m"} 0.2
vllm:inter_token_latency_seconds_count{engine="0",model_name="m"} 4.0
`

// An engine exporting both names (vLLM 0.10.x): the newer name wins, whichever one the
// metric definition maps (on a real engine both carry the same value).
const mockVllmBothCacheMetrics = `# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="m"} 0.7
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="m"} 0.1
`

func TestEngineMetricsFetcher_RenamedVllmMetricsAreRead(t *testing.T) {
	server := setupMockServer(mockVllm030Metrics, 200, 0)
	defer server.Close()
	endpoint := strings.TrimPrefix(server.URL, "http://")

	result, err := NewEngineMetricsFetcher().FetchAllTypedMetrics(
		context.Background(), endpoint, "vllm", "pod-030",
		[]string{GPUCacheUsagePerc, TimePerOutputTokenSeconds})
	require.NoError(t, err)

	cache, ok := result.ModelMetrics["m/"+GPUCacheUsagePerc]
	require.True(t, ok, "gpu_cache_usage_perc read from vllm:kv_cache_usage_perc; errors: %v", result.Errors)
	assert.Equal(t, 0.4, cache.GetSimpleValue())

	tpot, ok := result.ModelMetrics["m/"+TimePerOutputTokenSeconds]
	require.True(t, ok, "time_per_output_token_seconds read from vllm:inter_token_latency_seconds; errors: %v", result.Errors)
	hist := tpot.GetHistogramValue()
	require.NotNil(t, hist)
	assert.Equal(t, 4.0, hist.Count)
	assert.Equal(t, 0.2, hist.Sum)
}

func TestEngineMetricsFetcher_NewestNamePreferred(t *testing.T) {
	server := setupMockServer(mockVllmBothCacheMetrics, 200, 0)
	defer server.Close()
	endpoint := strings.TrimPrefix(server.URL, "http://")

	result, err := NewEngineMetricsFetcher().FetchAllTypedMetrics(
		context.Background(), endpoint, "vllm", "pod-old", []string{GPUCacheUsagePerc, KVCacheUsagePerc})
	require.NoError(t, err)
	for _, name := range []string{GPUCacheUsagePerc, KVCacheUsagePerc} {
		cache, ok := result.ModelMetrics["m/"+name]
		require.True(t, ok, name)
		assert.Equal(t, 0.1, cache.GetSimpleValue(), name)
	}
}

func TestEngineMetricCandidates(t *testing.T) {
	assert.Equal(t, []string{"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"},
		EngineMetricCandidates("vllm:gpu_cache_usage_perc"))
	assert.Equal(t, []string{"vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"},
		EngineMetricCandidates("vllm:inter_token_latency_seconds"))
	assert.Equal(t, []string{"vllm:e2e_request_latency_seconds"},
		EngineMetricCandidates("vllm:e2e_request_latency_seconds"))
}

func TestLookupMetricFamily_MissingEverywhere(t *testing.T) {
	_, name, ok := lookupMetricFamily(nil, "vllm:gpu_cache_usage_perc")
	assert.False(t, ok)
	assert.Equal(t, "vllm:gpu_cache_usage_perc", name)
}
