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

package podautoscaler

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	scalingctx "github.com/vllm-project/aibrix/pkg/controller/podautoscaler/context"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

// tre/deploy/baselines/apa holds the APA baseline CRs of this fork (skipped elsewhere).
const apaBaselineDir = "../../../tre/deploy/baselines/apa"

// The committed APA baseline CRs must be parsed to the intended values, and those values
// must appear in the "Effective autoscaling config" log line.
func TestAPABaselineCRsEffectiveConfig(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join(apaBaselineDir, "*-apa.yaml"))
	if len(files) == 0 {
		t.Skip("no APA baseline CRs in this tree")
	}
	require.Len(t, files, 3)

	var buf bytes.Buffer
	klog.LogToStderr(false)
	klog.SetOutput(&buf)
	defer func() {
		klog.Flush()
		klog.SetOutput(os.Stderr)
		klog.LogToStderr(true)
	}()

	scheme := runtime.NewScheme()
	_ = autoscalingv1alpha1.AddToScheme(scheme)
	autoScaler := NewDefaultAutoScaler(&mockMetricFetcherFactory{
		mockMetricFetcher: mockMetricFetcher{metricsValue: 0.5},
	}, fake.NewClientBuilder().WithScheme(scheme).Build())

	for _, f := range files {
		raw, err := os.ReadFile(f)
		require.NoError(t, err)
		var pa autoscalingv1alpha1.PodAutoscaler
		require.NoError(t, yaml.Unmarshal(raw, &pa), f)
		require.Equal(t, autoscalingv1alpha1.APA, pa.Spec.ScalingStrategy, f)

		sc := scalingctx.NewBaseScalingContext()
		require.NoError(t, sc.UpdateByPaTypes(&pa), f)
		sc.SetMinReplicas(*pa.Spec.MinReplicas)
		sc.SetMaxReplicas(pa.Spec.MaxReplicas)

		// v1-aligned values (see tre/deploy/RELEASE-20261001-apa-window.md).
		assert.Equal(t, 20*time.Second, sc.GetStableWindow(), f)
		assert.Equal(t, 0.2, sc.GetUpFluctuationTolerance(), f)
		assert.Equal(t, 0.8, sc.GetDownFluctuationTolerance(), f)
		assert.Equal(t, 2.0, sc.GetMaxScaleUpRate(), f)
		assert.Equal(t, 2.0, sc.GetMaxScaleDownRate(), f)
		assert.Equal(t, time.Duration(0), sc.GetScaleUpCooldownWindow(), f)
		assert.Equal(t, time.Duration(0), sc.GetScaleDownCooldownWindow(), f)
		assert.Equal(t, int32(1), sc.GetMinReplicas(), f)
		assert.Equal(t, int32(4), sc.GetMaxReplicas(), f)

		buf.Reset()
		_, err = autoScaler.ComputeDesiredReplicas(context.TODO(), ReplicaComputeRequest{
			PodAutoscaler:   pa,
			ScalingContext:  sc,
			CurrentReplicas: 1,
			Pods:            []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-1"}}},
			Timestamp:       time.Now(),
		})
		require.NoError(t, err, f)
		klog.Flush()

		var line string
		for _, l := range strings.Split(buf.String(), "\n") {
			if strings.Contains(l, "Effective autoscaling config") {
				line = l
			}
		}
		require.NotEmpty(t, line, "no Effective autoscaling config log for %s", f)
		for _, want := range []string{
			`stableWindow="20s"`, "upTolerance=0.2", "downTolerance=0.8",
			"maxScaleUpRate=2", "maxScaleDownRate=2",
			`scaleUpCooldown="0s"`, `scaleDownCooldown="0s"`,
			"minReplicas=1", "maxReplicas=4",
		} {
			assert.Contains(t, line, want, f)
		}
		assert.NotContains(t, buf.String(), "Ignoring unrecognized autoscaling annotation", f)
	}
}
