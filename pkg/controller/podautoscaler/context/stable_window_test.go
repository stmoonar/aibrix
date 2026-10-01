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

package context

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func paWithAnnotations(annotations map[string]string) *autoscalingv1alpha1.PodAutoscaler {
	return &autoscalingv1alpha1.PodAutoscaler{
		ObjectMeta: v1.ObjectMeta{Name: "m-apa", Namespace: "default", Annotations: annotations},
		Spec:       autoscalingv1alpha1.PodAutoscalerSpec{ScalingStrategy: autoscalingv1alpha1.APA},
	}
}

func TestStableWindow_Annotation(t *testing.T) {
	ctx := NewBaseScalingContext()
	require.NoError(t, ctx.UpdateByPaTypes(paWithAnnotations(map[string]string{
		types.APAWindowLabel: "20s",
	})))
	assert.Equal(t, 20*time.Second, ctx.GetStableWindow())
}

func TestStableWindow_Default(t *testing.T) {
	ctx := NewBaseScalingContext()
	require.NoError(t, ctx.UpdateByPaTypes(paWithAnnotations(nil)))
	assert.Equal(t, 180*time.Second, ctx.GetStableWindow())
	assert.Equal(t, types.DefaultStableWindowDuration, ctx.GetStableWindow())
}

func TestStableWindow_Invalid(t *testing.T) {
	for _, v := range []string{"abc", "0s", "-5s", "500ms"} {
		ctx := NewBaseScalingContext()
		err := ctx.UpdateByPaTypes(paWithAnnotations(map[string]string{types.APAWindowLabel: v}))
		assert.Error(t, err, v)
	}
}

// The annotation set used by the TRE APA baseline CRs: every key must take effect by name.
func TestAPABaselineAnnotationsTakeEffect(t *testing.T) {
	ctx := NewBaseScalingContext()
	require.NoError(t, ctx.UpdateByPaTypes(paWithAnnotations(map[string]string{
		"autoscaling.aibrix.ai/scale-up-tolerance":   "0.1",
		"autoscaling.aibrix.ai/scale-down-tolerance": "0.2",
		"autoscaling.aibrix.ai/max-scale-up-rate":    "2",
		"apa.autoscaling.aibrix.ai/window":           "20s",
	})))
	assert.Equal(t, 0.1, ctx.GetUpFluctuationTolerance())
	assert.Equal(t, 0.2, ctx.GetDownFluctuationTolerance())
	assert.Equal(t, 2.0, ctx.GetMaxScaleUpRate())
	assert.Equal(t, 20*time.Second, ctx.GetStableWindow())
}

// Keys that do not match a parser leave the defaults in place (and are logged).
func TestLegacyToleranceKeysAreIgnored(t *testing.T) {
	ctx := NewBaseScalingContext()
	require.NoError(t, ctx.UpdateByPaTypes(paWithAnnotations(map[string]string{
		"autoscaling.aibrix.ai/up-fluctuation-tolerance":   "0.5",
		"autoscaling.aibrix.ai/down-fluctuation-tolerance": "0.5",
	})))
	assert.Equal(t, 0.1, ctx.GetUpFluctuationTolerance())
	assert.Equal(t, 0.1, ctx.GetDownFluctuationTolerance())
}

func TestIsAutoscalingAnnotation(t *testing.T) {
	assert.True(t, isAutoscalingAnnotation("autoscaling.aibrix.ai/up-fluctuation-tolerance"))
	assert.True(t, isAutoscalingAnnotation("apa.autoscaling.aibrix.ai/foo"))
	assert.True(t, isAutoscalingAnnotation("kpa.autoscaling.aibrix.ai/stable-window"))
	assert.False(t, isAutoscalingAnnotation("autoscaling.aibrix.ai/storm-service-mode"))
	assert.False(t, isAutoscalingAnnotation("kubectl.kubernetes.io/last-applied-configuration"))
}

func TestStableWindow_MinimumAccepted(t *testing.T) {
	ctx := NewBaseScalingContext()
	require.NoError(t, ctx.UpdateByPaTypes(paWithAnnotations(map[string]string{types.APAWindowLabel: "1s"})))
	assert.Equal(t, time.Second, ctx.GetStableWindow())
}

// The window annotation is APA-only: a KPA/HPA PodAutoscaler keeps the default.
func TestStableWindow_NonAPAIgnored(t *testing.T) {
	for _, s := range []autoscalingv1alpha1.ScalingStrategyType{autoscalingv1alpha1.KPA, autoscalingv1alpha1.HPA} {
		pa := paWithAnnotations(map[string]string{types.APAWindowLabel: "20s"})
		pa.Spec.ScalingStrategy = s
		ctx := NewBaseScalingContext()
		require.NoError(t, ctx.UpdateByPaTypes(pa))
		assert.Equal(t, types.DefaultStableWindowDuration, ctx.GetStableWindow(), string(s))
	}
}

func TestWarnOnce(t *testing.T) {
	pa := paWithAnnotations(nil)
	pa.Name = "warn-once"
	key := "warn-once|autoscaling.aibrix.ai/x=1"
	_, before := warnedAnnotations.Load("default/" + key)
	assert.False(t, before)
	warnOnce(pa, "autoscaling.aibrix.ai/x", "msg", "1")
	_, after := warnedAnnotations.Load("default/" + key)
	assert.True(t, after)
}
