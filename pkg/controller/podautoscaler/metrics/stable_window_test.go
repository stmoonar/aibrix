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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
)

func windowKey(pa string) types.MetricKey {
	return types.MetricKey{Namespace: "default", Name: "m", MetricName: "kv_cache_usage_perc",
		PaNamespace: "default", PaName: pa}
}

func TestEnsureStableWindow_PerPA(t *testing.T) {
	c := NewMetricsClient(time.Second)
	c.EnsureStableWindow(windowKey("a"), 20*time.Second)
	c.EnsureStableWindow(windowKey("b"), 0) // default
	assert.Equal(t, 20*time.Second, c.StableWindowDuration(windowKey("a")))
	assert.Equal(t, 180*time.Second, c.StableWindowDuration(windowKey("b")))
	// Unconfigured key reports the default; auto-init via UpdateMetrics also uses it.
	assert.Equal(t, 180*time.Second, c.StableWindowDuration(windowKey("c")))
	require.NoError(t, c.UpdateMetrics(time.Now(), windowKey("c"), 1))
	assert.Equal(t, 180*time.Second, c.StableWindowDuration(windowKey("c")))
	// Panic window is untouched.
	assert.Equal(t, panicWindowDuration, c.panicWindows[windowKey("a").String()].Duration())
}

// With a 20 s window, a sample 25 s old no longer contributes to the stable average.
func TestEnsureStableWindow_AgesOutSamples(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		window time.Duration
		want   float64
	}{
		{20 * time.Second, 0.2},
		{0, 0.5}, // default 180 s keeps both samples
	} {
		c := NewMetricsClient(time.Second)
		k := windowKey("a")
		c.EnsureStableWindow(k, tc.window)
		require.NoError(t, c.UpdateMetrics(t0, k, 0.8))
		require.NoError(t, c.UpdateMetrics(t0.Add(25*time.Second), k, 0.2))
		stable, _, err := c.GetMetricValue(k, t0.Add(25*time.Second))
		require.NoError(t, err)
		assert.InDelta(t, tc.want, stable, 1e-9, "window %v", tc.window)
	}
}

func TestEnsureStableWindow_Resize(t *testing.T) {
	c := NewMetricsClient(time.Second)
	k := windowKey("a")
	c.EnsureStableWindow(k, 180*time.Second)
	require.NoError(t, c.UpdateMetrics(time.Now(), k, 0.5))
	c.EnsureStableWindow(k, 180*time.Second) // same length: samples kept
	assert.Equal(t, 1, c.stableWindows[k.String()].Size())
	c.EnsureStableWindow(k, 20*time.Second) // new length: rebuilt
	assert.Equal(t, 20*time.Second, c.StableWindowDuration(k))
	assert.Equal(t, 0, c.stableWindows[k.String()].Size())
}
