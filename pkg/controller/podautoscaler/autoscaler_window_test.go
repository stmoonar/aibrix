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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	scalingctx "github.com/vllm-project/aibrix/pkg/controller/podautoscaler/context"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// The pipeline must size each PodAutoscaler's stable window from its own annotation.
func TestComputeDesiredReplicas_StableWindowPerPA(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = autoscalingv1alpha1.AddToScheme(scheme)
	autoScaler := NewDefaultAutoScaler(&mockMetricFetcherFactory{
		mockMetricFetcher: mockMetricFetcher{metricsValue: 0.5},
	}, fake.NewClientBuilder().WithScheme(scheme).Build())

	for _, tc := range []struct {
		name        string
		annotations map[string]string
		want        time.Duration
	}{
		{"windowed-apa", map[string]string{types.APAWindowLabel: "20s"}, 20 * time.Second},
		{"default-apa", nil, 180 * time.Second},
	} {
		pa := autoscalingv1alpha1.PodAutoscaler{
			ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: tc.name, Annotations: tc.annotations},
			Spec: autoscalingv1alpha1.PodAutoscalerSpec{
				MetricsSources: []autoscalingv1alpha1.MetricSource{{
					MetricSourceType: autoscalingv1alpha1.POD,
					TargetMetric:     "kv_cache_usage_perc",
					TargetValue:      "0.5",
				}},
				ScaleTargetRef:  corev1.ObjectReference{Kind: "Deployment", Name: "m"},
				ScalingStrategy: "APA",
			},
		}
		sc := scalingctx.NewBaseScalingContext()
		sc.MaxReplicas = 4
		require.NoError(t, sc.UpdateByPaTypes(&pa))

		_, err := autoScaler.ComputeDesiredReplicas(context.TODO(), ReplicaComputeRequest{
			PodAutoscaler:   pa,
			ScalingContext:  sc,
			CurrentReplicas: 1,
			Pods:            []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-1"}}},
			Timestamp:       time.Now(),
		})
		require.NoError(t, err)

		key := types.MetricKey{Namespace: "default", Name: "m", MetricName: "kv_cache_usage_perc",
			PaNamespace: "default", PaName: tc.name}
		assert.Equal(t, tc.want, autoScaler.metricsClient.StableWindowDuration(key), tc.name)
	}
}
