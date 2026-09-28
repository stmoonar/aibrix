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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TRE-PATCH(P2-GW-006): real /metrics bodies of a vLLM 0.30 fork pod (live capture) and of
// vLLM 0.10.1 (stock stat logger rendered offline); shared with the TRE Python guard test
// tre/deploy/tests/test_vllm_metric_names.py.
var vllmSampleFiles = map[string]string{
	"0.30.0": "vllm-0.30.0.prom",
	"0.10.1": "vllm-0.10.1.prom",
}

const vllmSampleModel = "dsqwen-7b"

// Metrics the TRE gateway path reads from vLLM: routing (least-gpu-cache / least-kv-cache,
// prefix cache), the TRE Redis instant + histogram docs (queue, KV fill, TTFT / TPOT / e2e,
// token histograms).
var treVllmMetrics = []string{
	NumRequestsRunning,
	NumRequestsWaiting,
	GPUCacheUsagePerc,
	KVCacheUsagePerc,
	PrefixCacheQueriesTotal,
	PrefixCacheHitTotal,
	TimeToFirstTokenSeconds,
	TimePerOutputTokenSeconds,
	InterTokenLatencySeconds,
	E2ERequestLatencySeconds,
	RequestPromptTokens,
	RequestGenerationTokens,
}

func readVllmSample(t *testing.T, version string) string {
	t.Helper()
	path := filepath.Join("..", "..", "tre", "deploy", "tests", "fixtures", "vllm_metrics", vllmSampleFiles[version])
	body, err := os.ReadFile(path)
	require.NoError(t, err)
	return string(body)
}

func TestEngineMetricsFetcher_RealVllmSamples(t *testing.T) {
	for version := range vllmSampleFiles {
		t.Run(version, func(t *testing.T) {
			server := setupMockServer(readVllmSample(t, version), 200, 0)
			defer server.Close()
			endpoint := strings.TrimPrefix(server.URL, "http://")

			result, err := NewEngineMetricsFetcher().FetchAllTypedMetrics(
				context.Background(), endpoint, "vllm", "sample-"+version, treVllmMetrics)
			require.NoError(t, err)

			for _, name := range treVllmMetrics {
				def, ok := Metrics[name]
				require.True(t, ok, name)
				var value MetricValue
				if def.MetricScope == PodMetricScope {
					value, ok = result.Metrics[name]
				} else {
					value, ok = result.ModelMetrics[vllmSampleModel+"/"+name]
				}
				assert.True(t, ok, "vLLM %s: %s (%s) not read; errors: %v",
					version, name, def.EngineMetricsNameMapping["vllm"], result.Errors)
				if ok && def.MetricType.Raw == Histogram {
					assert.NotNil(t, value.GetHistogramValue(), "%s %s histogram", version, name)
				}
			}
		})
	}
}

// Every name of an equivalence group is a real vLLM family in at least one sample, and
// every group resolves on every sample (the point of the table).
func TestEngineMetricEquivalents_ResolveOnRealSamples(t *testing.T) {
	families := map[string]map[string]bool{}
	for version := range vllmSampleFiles {
		families[version] = map[string]bool{}
		for _, line := range strings.Split(readVllmSample(t, version), "\n") {
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			name := strings.SplitN(strings.SplitN(line, "{", 2)[0], " ", 2)[0]
			families[version][strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(name, "_bucket"), "_sum"), "_count")] = true
			families[version][name] = true
		}
	}
	for _, group := range engineMetricEquivalents {
		for _, name := range group {
			seen := false
			for version := range vllmSampleFiles {
				seen = seen || families[version][name]
			}
			assert.True(t, seen, "%s is exported by no sample", name)
		}
		for version := range vllmSampleFiles {
			found := false
			for _, name := range group {
				found = found || families[version][name]
			}
			assert.True(t, found, "vLLM %s exports no name of %v", version, group)
		}
	}
}
