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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	orchestrationv1alpha1 "github.com/vllm-project/aibrix/api/orchestration/v1alpha1"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGetCurrentReplicasFromScale(t *testing.T) {
	expectedReplicas := int32(2)

	scaleStormService := &unstructured.Unstructured{}
	scaleStormService.SetAPIVersion("orchestration.aibrix.ai/v1alpha1")
	scaleStormService.SetKind("StormService")

	table := []struct {
		name  string
		pa    *autoscalingv1alpha1.PodAutoscaler
		ss    *orchestrationv1alpha1.StormService
		scale *unstructured.Unstructured
	}{
		{
			name: "llm_model_with_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "Deployment",
						Name: "test-llm",
					},
				},
			},
			ss: &orchestrationv1alpha1.StormService{},
			scale: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": map[string]interface{}{
						"replicas": int64(expectedReplicas),
					},
				},
			},
		},
		{
			name: "storm_service_with_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Annotations: map[string]string{
						AutoscalingStormServiceModeAnnotationKey: "replica",
					},
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{},
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "StormService",
						Name: "test-storm",
					},
				},
			},
			ss: &orchestrationv1alpha1.StormService{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-storm",
					Namespace: "default",
				},
				Spec: orchestrationv1alpha1.StormServiceSpec{
					Replicas: &expectedReplicas,
				},
			},
			scale: scaleStormService,
		},
		{
			name: "storm_service_with_status_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{
						RoleName: "prefill",
					},
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "StormService",
						Name: "test-storm",
					},
				},
			},
			ss: &orchestrationv1alpha1.StormService{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-storm",
					Namespace: "default",
				},
				Spec: orchestrationv1alpha1.StormServiceSpec{},
				Status: orchestrationv1alpha1.StormServiceStatus{
					RoleStatuses: []orchestrationv1alpha1.RoleStatus{
						{
							Name:     "prefill",
							Replicas: expectedReplicas,
						},
					},
				},
			},
			scale: scaleStormService,
		},
		{
			name: "storm_service_with_template_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{
						RoleName: "prefill",
					},
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "StormService",
						Name: "test-storm",
					},
				},
			},
			ss: &orchestrationv1alpha1.StormService{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-storm",
					Namespace: "default",
				},
				Spec: orchestrationv1alpha1.StormServiceSpec{
					Template: orchestrationv1alpha1.RoleSetTemplateSpec{
						Spec: &orchestrationv1alpha1.RoleSetSpec{
							Roles: []orchestrationv1alpha1.RoleSpec{
								{
									Name:     "prefill",
									Replicas: &expectedReplicas,
								},
							},
						},
					},
				},
			},
			scale: scaleStormService,
		},
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = autoscalingv1alpha1.AddToScheme(scheme)
			_ = orchestrationv1alpha1.AddToScheme(scheme)

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(tt.pa, tt.ss).
				Build()

			workloadScale := NewWorkloadScale(fakeClient, nil)

			currentReplicas, err := workloadScale.GetCurrentReplicasFromScale(context.TODO(), tt.pa, tt.scale)

			assert.NoError(t, err)
			assert.Equal(t, expectedReplicas, currentReplicas)
		})
	}
}

func TestAPASleepModeGetsCurrentReplicasFromServiceManage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/models_replicas", r.URL.Path)
		assert.Equal(t, "test-llm", r.URL.Query().Get("models"))
		_, _ = w.Write([]byte(`{"test-llm":3}`))
	}))
	defer server.Close()

	pa := &autoscalingv1alpha1.PodAutoscaler{
		ObjectMeta: v1.ObjectMeta{Namespace: "default"},
		Spec: autoscalingv1alpha1.PodAutoscalerSpec{
			ScalingStrategy: autoscalingv1alpha1.APA,
			ScaleTargetRef: corev1.ObjectReference{
				Kind: "Deployment",
				Name: "test-llm",
			},
		},
	}
	scale := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"replicas": int64(99),
			},
		},
	}
	workloadScale := &workloadScale{
		serviceManageURL:        server.URL,
		serviceManageHTTPClient: server.Client(),
		sleepModeEnabled:        true,
	}

	currentReplicas, err := workloadScale.GetCurrentReplicasFromScale(context.TODO(), pa, scale)

	assert.NoError(t, err)
	assert.Equal(t, int32(3), currentReplicas)
}

// TestAPASleepModeSetDesiredReplicasPostsAnAbsolutePlan: APA sends an ABSOLUTE target through
// POST /v2/plans (wake at_least / sleep at_most on the "apa" path), never a delta; nothing when the
// target is already met; a refused wake (nothing changed) is a warning like v1's 200 actual 0;
// writer_busy and a failed item are errors. The Deployment's spec.replicas is never touched.
func TestAPASleepModeSetDesiredReplicasPostsAnAbsolutePlan(t *testing.T) {
	refusedWake := `{"plan_id":"pl-1","status":"failed","error":"partial","reason":"partial","sleeps":[],` +
		`"wakes":[{"model":"test-llm","status":"refused","unfilled":3,"error":{"error":"gpu_busy","reason":"slot_occupied"}}]}`
	failedWake := `{"plan_id":"pl-2","status":"failed","error":"partial","reason":"partial","sleeps":[],` +
		`"wakes":[{"model":"test-llm","status":"failed","unfilled":3,"error":{"error":"wake_failed"}}]}`
	cases := []struct {
		name     string
		current  int32
		desired  int32
		status   int
		response string
		wantPlan string // "" = no POST /v2/plans
		wantErr  bool
	}{
		{name: "scale up: wake at_least desired", current: 2, desired: 5, status: http.StatusOK,
			response: `{"plan_id":"pl-0","status":"done","sleeps":[],"wakes":[{"model":"test-llm","status":"done","awake_before":2,"awake_after":5}]}`,
			wantPlan: `{"sleeps":[],"wakes":[{"model":"test-llm","at_least":5,"reason":"apa"}]}`},
		{name: "scale down to zero: sleep at_most on the apa path", current: 3, desired: 0, status: http.StatusOK,
			response: `{"plan_id":"pl-0","status":"done","sleeps":[{"model":"test-llm","status":"clamped","taken":2,"clamped_by_floor":true}],"wakes":[]}`,
			wantPlan: `{"sleeps":[{"model":"test-llm","at_most":0,"sleep_path":"apa","reason":"apa"}],"wakes":[]}`},
		{name: "target already met: no plan", current: 2, desired: 2},
		{name: "refused wake, nothing changed: warning", current: 2, desired: 5, status: http.StatusConflict,
			response: refusedWake, wantPlan: `{"sleeps":[],"wakes":[{"model":"test-llm","at_least":5,"reason":"apa"}]}`},
		{name: "failed wake: error", current: 2, desired: 5, status: http.StatusConflict,
			response: failedWake, wantPlan: `{"sleeps":[],"wakes":[{"model":"test-llm","at_least":5,"reason":"apa"}]}`,
			wantErr: true},
		{name: "writer busy: error", current: 3, desired: 1, status: http.StatusConflict,
			response: `{"detail":"writer lock busy","error":"writer_busy","retry_after_s":1.0}`,
			wantPlan: `{"sleeps":[{"model":"test-llm","at_most":1,"sleep_path":"apa","reason":"apa"}],"wakes":[]}`,
			wantErr:  true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var plans []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, http.MethodPost, r.Method)
				switch r.URL.Path {
				case "/models_replicas":
					assert.Equal(t, "test-llm", r.URL.Query().Get("models"))
					_, _ = fmt.Fprintf(w, `{"test-llm":%d}`, tc.current)
				case "/v2/plans":
					assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
					body, _ := io.ReadAll(r.Body)
					plans = append(plans, string(body))
					w.WriteHeader(tc.status)
					_, _ = w.Write([]byte(tc.response))
				default:
					t.Fatalf("unexpected service_manage path: %s", r.URL.Path)
				}
			}))
			defer server.Close()

			pa := &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{Namespace: "default"},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					ScalingStrategy: autoscalingv1alpha1.APA,
					ScaleTargetRef: corev1.ObjectReference{
						Kind:       "Deployment",
						Name:       "test-llm",
						APIVersion: "apps/v1",
					},
				},
			}
			deployment := &appsv1.Deployment{
				ObjectMeta: v1.ObjectMeta{Name: "test-llm", Namespace: "default"},
				Spec:       appsv1.DeploymentSpec{Replicas: ptr.To[int32](1)},
			}
			scheme := runtime.NewScheme()
			_ = autoscalingv1alpha1.AddToScheme(scheme)
			_ = appsv1.AddToScheme(scheme)
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pa, deployment).Build()
			workloadScale := &workloadScale{
				client:                  fakeClient,
				serviceManageURL:        server.URL,
				serviceManageHTTPClient: server.Client(),
				sleepModeEnabled:        true,
			}

			err := workloadScale.SetDesiredReplicas(context.TODO(), pa, tc.desired)

			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			if tc.wantPlan == "" {
				assert.Empty(t, plans)
			} else if assert.Len(t, plans, 1) {
				assert.JSONEq(t, tc.wantPlan, plans[0])
			}
			updated := &appsv1.Deployment{}
			assert.NoError(t, fakeClient.Get(context.TODO(), client.ObjectKey{Namespace: "default", Name: "test-llm"}, updated))
			assert.Equal(t, int32(1), *updated.Spec.Replicas)
		})
	}
}

// TestAPASleepModeServiceManageCallOutlivesTheReconcileDeadline: Reconcile bounds its context at
// DefaultReconcileTimeoutDuration (10 s), below a plan's worst case; the SM calls use their own
// deadline (the client timeout), so an SM slower than the reconcile context still completes. A
// canceled parent (manager shutdown) still ends the call.
func TestAPASleepModeServiceManageCallOutlivesTheReconcileDeadline(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models_replicas":
			_, _ = w.Write([]byte(`{"test-llm":1}`))
		case "/v2/plans":
			select {
			case <-release:
			case <-r.Context().Done():
				return
			}
			_, _ = w.Write([]byte(`{"plan_id":"pl-0","status":"done","sleeps":[],` +
				`"wakes":[{"model":"test-llm","status":"done","awake_before":1,"awake_after":2}]}`))
		}
	}))
	defer server.Close()
	pa := &autoscalingv1alpha1.PodAutoscaler{
		ObjectMeta: v1.ObjectMeta{Namespace: "default"},
		Spec: autoscalingv1alpha1.PodAutoscalerSpec{
			ScalingStrategy: autoscalingv1alpha1.APA,
			ScaleTargetRef:  corev1.ObjectReference{Kind: "Deployment", Name: "test-llm"},
		},
	}
	client := server.Client()
	client.Timeout = 30 * time.Second
	ws := &workloadScale{serviceManageURL: server.URL, serviceManageHTTPClient: client, sleepModeEnabled: true}

	// The reconcile context expires while the SM is still executing the plan.
	reconcileCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	go func() {
		<-reconcileCtx.Done()
		close(release) // the SM answers only after the reconcile deadline
	}()
	assert.NoError(t, ws.SetDesiredReplicas(reconcileCtx, pa, 2))

	// Shutdown: a canceled parent ends the call.
	shutdownCtx, stop := context.WithCancel(context.Background())
	stopped := make(chan error, 1)
	blocked := make(chan struct{})
	server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models_replicas" {
			_, _ = w.Write([]byte(`{"test-llm":1}`))
			return
		}
		_, _ = io.ReadAll(r.Body) // the server notices a closed connection only once the body is read
		close(blocked)
		<-r.Context().Done()
	})
	go func() { stopped <- ws.SetDesiredReplicas(shutdownCtx, pa, 2) }()
	<-blocked
	stop()
	select {
	case err := <-stopped:
		assert.Error(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("a canceled parent context did not end the service-manager call")
	}
}

func TestAPASleepModeConfigServiceManageTimeout(t *testing.T) {
	t.Setenv("APA_SCALE_SLEEP_MODE", "1")
	t.Setenv("SERVICE_MANAGE_URL", "http://sm.test:8000")

	t.Setenv("SERVICE_MANAGE_TIMEOUT_S", "")
	config, err := newServiceManageConfigFromEnv()
	assert.NoError(t, err)
	assert.Equal(t, 360*time.Second, config.timeout) // the SM's api_call_timeout_s

	t.Setenv("SERVICE_MANAGE_TIMEOUT_S", "400.5")
	config, err = newServiceManageConfigFromEnv()
	assert.NoError(t, err)
	assert.Equal(t, 400500*time.Millisecond, config.timeout)

	for _, bad := range []string{"0", "-1", "soon"} {
		t.Setenv("SERVICE_MANAGE_TIMEOUT_S", bad)
		_, err = newServiceManageConfigFromEnv()
		assert.ErrorContains(t, err, "SERVICE_MANAGE_TIMEOUT_S")
	}
}

func TestAPASleepModeConfigRequiresServiceManageURL(t *testing.T) {
	t.Setenv("APA_SCALE_SLEEP_MODE", "1")
	t.Setenv("SERVICE_MANAGE_URL", "")

	_, err := newServiceManageConfigFromEnv()

	assert.ErrorContains(t, err, "SERVICE_MANAGE_URL")
}

func TestGetPodSelectorFromScale(t *testing.T) {
	t.Run("llm_model", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = autoscalingv1alpha1.AddToScheme(scheme)
		_ = orchestrationv1alpha1.AddToScheme(scheme)

		pa := &autoscalingv1alpha1.PodAutoscaler{
			ObjectMeta: v1.ObjectMeta{
				Namespace: "default",
			},
			Spec: autoscalingv1alpha1.PodAutoscalerSpec{
				ScaleTargetRef: corev1.ObjectReference{
					Kind: "Deployment",
					Name: "test-llm",
				},
			},
		}
		scale := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"spec": map[string]interface{}{
					"selector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"model.aibrix.ai/name": "deepseek-llm-7b-chat",
						},
					},
				},
			},
		}

		fakeClient := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(pa).
			Build()

		workloadScale := NewWorkloadScale(fakeClient, nil)

		labelsSelector, err := workloadScale.GetPodSelectorFromScale(context.TODO(), pa, scale)

		assert.NoError(t, err)
		assert.NotNil(t, labelsSelector)
		requirements, _ := labelsSelector.Requirements()
		assert.Len(t, requirements, 1)
		assert.Equal(t, "model.aibrix.ai/name", requirements[0].Key())
		assert.Len(t, requirements[0].ValuesUnsorted(), 1)
		assert.Equal(t, "deepseek-llm-7b-chat", requirements[0].ValuesUnsorted()[0])
	})

	t.Run("storm_service", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = autoscalingv1alpha1.AddToScheme(scheme)
		_ = orchestrationv1alpha1.AddToScheme(scheme)

		pa := &autoscalingv1alpha1.PodAutoscaler{
			ObjectMeta: v1.ObjectMeta{
				Namespace: "default",
			},
			Spec: autoscalingv1alpha1.PodAutoscalerSpec{
				ScaleTargetRef: corev1.ObjectReference{
					Kind: "StormService",
					Name: "test-storm",
				},
			},
		}
		ss := &orchestrationv1alpha1.StormService{
			ObjectMeta: v1.ObjectMeta{
				Name:      "test-storm",
				Namespace: "default",
			},
			Spec: orchestrationv1alpha1.StormServiceSpec{},
		}

		scale := &unstructured.Unstructured{}
		scale.SetAPIVersion("orchestration.aibrix.ai/v1alpha1")
		scale.SetKind("StormService")

		fakeClient := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(pa, ss).
			Build()

		workloadScale := NewWorkloadScale(fakeClient, nil)

		labelsSelector, err := workloadScale.GetPodSelectorFromScale(context.TODO(), pa, scale)

		assert.NoError(t, err)
		assert.NotNil(t, labelsSelector)
		requirements, _ := labelsSelector.Requirements()
		assert.Len(t, requirements, 1)
		assert.Equal(t, "storm-service-name", requirements[0].Key())
		assert.Len(t, requirements[0].ValuesUnsorted(), 1)
		assert.Equal(t, "test-storm", requirements[0].ValuesUnsorted()[0])
	})

	t.Run("storm_service_with_role", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = autoscalingv1alpha1.AddToScheme(scheme)
		_ = orchestrationv1alpha1.AddToScheme(scheme)

		pa := &autoscalingv1alpha1.PodAutoscaler{
			ObjectMeta: v1.ObjectMeta{
				Namespace: "default",
			},
			Spec: autoscalingv1alpha1.PodAutoscalerSpec{
				SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{
					RoleName: "prefill",
				},
				ScaleTargetRef: corev1.ObjectReference{
					Kind: "StormService",
					Name: "test-storm",
				},
			},
		}
		ss := &orchestrationv1alpha1.StormService{
			ObjectMeta: v1.ObjectMeta{
				Name:      "test-storm",
				Namespace: "default",
			},
			Spec: orchestrationv1alpha1.StormServiceSpec{},
		}

		scale := &unstructured.Unstructured{}
		scale.SetAPIVersion("orchestration.aibrix.ai/v1alpha1")
		scale.SetKind("StormService")

		fakeClient := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(pa, ss).
			Build()

		workloadScale := NewWorkloadScale(fakeClient, nil)

		labelsSelector, err := workloadScale.GetPodSelectorFromScale(context.TODO(), pa, scale)

		assert.NoError(t, err)
		assert.NotNil(t, labelsSelector)
		requirements, _ := labelsSelector.Requirements()
		assert.Len(t, requirements, 2)
		assert.Equal(t, "role-name", requirements[0].Key())
		assert.Len(t, requirements[0].ValuesUnsorted(), 1)
		assert.Equal(t, "prefill", requirements[0].ValuesUnsorted()[0])
		assert.Equal(t, "storm-service-name", requirements[1].Key())
		assert.Len(t, requirements[1].ValuesUnsorted(), 1)
		assert.Equal(t, "test-storm", requirements[1].ValuesUnsorted()[0])
	})
}

func TestSetDesiredReplicas(t *testing.T) {
	currentReplicas := int32(1)
	expectedReplicas := int32(2)

	table := []struct {
		name           string
		pa             *autoscalingv1alpha1.PodAutoscaler
		deployment     *appsv1.Deployment
		ss             *orchestrationv1alpha1.StormService
		assertReplicas func(t *testing.T, fakeClient client.Client)
	}{
		{
			name: "llm_model_with_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					ScaleTargetRef: corev1.ObjectReference{
						Kind:       "Deployment",
						Name:       "test-llm",
						APIVersion: "apps/v1",
					},
				},
			},
			deployment: &appsv1.Deployment{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-llm",
					Namespace: "default",
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: &currentReplicas,
				},
			},
			ss: &orchestrationv1alpha1.StormService{},
			assertReplicas: func(t *testing.T, fakeClient client.Client) {
				deployment := &appsv1.Deployment{}
				err := fakeClient.Get(context.TODO(), client.ObjectKey{Namespace: "default", Name: "test-llm"}, deployment)
				assert.NoError(t, err)
				assert.Equal(t, expectedReplicas, *deployment.Spec.Replicas)
			},
		},
		{
			name: "storm_service_with_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Annotations: map[string]string{
						AutoscalingStormServiceModeAnnotationKey: "replica",
					},
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{
						RoleName: "",
					},
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "StormService",
						Name: "test-storm",
					},
				},
			},
			deployment: &appsv1.Deployment{},
			ss: &orchestrationv1alpha1.StormService{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-storm",
					Namespace: "default",
				},
				Spec: orchestrationv1alpha1.StormServiceSpec{
					Replicas: &currentReplicas,
				},
			},
			assertReplicas: func(t *testing.T, fakeClient client.Client) {
				ss := &orchestrationv1alpha1.StormService{}
				err := fakeClient.Get(context.TODO(), client.ObjectKey{Namespace: "default", Name: "test-storm"}, ss)
				assert.NoError(t, err)
				assert.Equal(t, expectedReplicas, *ss.Spec.Replicas)
			},
		},
		{
			name: "storm_service_with_template_spec_replicas",
			pa: &autoscalingv1alpha1.PodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
				},
				Spec: autoscalingv1alpha1.PodAutoscalerSpec{
					SubTargetSelector: &autoscalingv1alpha1.SubTargetSelector{
						RoleName: "prefill",
					},
					ScaleTargetRef: corev1.ObjectReference{
						Kind: "StormService",
						Name: "test-storm",
					},
				},
			},
			deployment: &appsv1.Deployment{},
			ss: &orchestrationv1alpha1.StormService{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-storm",
					Namespace: "default",
				},
				Spec: orchestrationv1alpha1.StormServiceSpec{
					Template: orchestrationv1alpha1.RoleSetTemplateSpec{
						Spec: &orchestrationv1alpha1.RoleSetSpec{
							Roles: []orchestrationv1alpha1.RoleSpec{
								{
									Name:     "prefill",
									Replicas: &currentReplicas,
								},
							},
						},
					},
				},
			},
			assertReplicas: func(t *testing.T, fakeClient client.Client) {
				ss := &orchestrationv1alpha1.StormService{}
				err := fakeClient.Get(context.TODO(), client.ObjectKey{Namespace: "default", Name: "test-storm"}, ss)
				assert.NoError(t, err)
				assert.Equal(t, expectedReplicas, *ss.Spec.Template.Spec.Roles[0].Replicas)
			},
		},
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = autoscalingv1alpha1.AddToScheme(scheme)
			_ = orchestrationv1alpha1.AddToScheme(scheme)
			_ = appsv1.AddToScheme(scheme)

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(tt.pa, tt.deployment, tt.ss).
				Build()

			workloadScale := NewWorkloadScale(fakeClient, nil)

			err := workloadScale.SetDesiredReplicas(context.TODO(), tt.pa, expectedReplicas)

			assert.NoError(t, err)
			tt.assertReplicas(t, fakeClient)
		})
	}
}
