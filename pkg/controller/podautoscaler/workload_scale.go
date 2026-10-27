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

package podautoscaler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	orchestrationv1alpha1 "github.com/vllm-project/aibrix/api/orchestration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/vllm-project/aibrix/pkg/controller/constants"
)

const (
	AutoscalingStormServiceModeAnnotationKey = "autoscaling.aibrix.ai/storm-service-mode"
	serviceManageURLEnv                      = "SERVICE_MANAGE_URL"
	apaScaleSleepModeEnv                     = "APA_SCALE_SLEEP_MODE"
	// serviceManageTimeoutEnv: client timeout (seconds) of every service-manager call. A plan
	// may wait for the writer lock (service_manager.writer_lock_wait_s) and then hold it, so the
	// default matches the SM's service_manager.api_call_timeout_s.
	serviceManageTimeoutEnv     = "SERVICE_MANAGE_TIMEOUT_S"
	defaultServiceManageTimeout = 360 * time.Second
	// apaSleepPath is the service-manager sleep path of an APA scale-down: the one its v1
	// /scale_service used (no drain, like every sleep of the service-manager).
	apaSleepPath = "apa"
)

// WorkloadScale provides scaling operations for different workload types.
// It provides the mechanism to get/set replica counts on workload resources,
// while AutoScaler provides the intelligence to compute desired replica counts.
// The interface is stateless - all methods take PodAutoscaler as a parameter.
type WorkloadScale interface {
	// Validate checks if the target is valid and scalable
	Validate(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler) error

	// GetCurrentReplicasFromScale extracts the current replica count from a scale object
	GetCurrentReplicasFromScale(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, scale *unstructured.Unstructured) (int32, error)

	// SetDesiredReplicas updates the replica count
	SetDesiredReplicas(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, replicas int32) error

	// GetPodSelectorFromScale extracts the label selector from an existing scale object
	// For role-level scaling, it adds the role label requirement
	// This avoids re-fetching the scale object when the controller already has it
	GetPodSelectorFromScale(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, scale *unstructured.Unstructured) (labels.Selector, error)
}

// workloadScale is a stateless implementation of WorkloadScale
type workloadScale struct {
	client                  client.Client
	restMapper              meta.RESTMapper
	serviceManageURL        string
	serviceManageHTTPClient *http.Client
	sleepModeEnabled        bool
}

type serviceManageConfig struct {
	url              string
	sleepModeEnabled bool
	timeout          time.Duration
}

// NewWorkloadScale creates a stateless WorkloadScale implementation
func NewWorkloadScale(
	client client.Client,
	restMapper meta.RESTMapper,
) WorkloadScale {
	return &workloadScale{
		client:                  client,
		restMapper:              restMapper,
		serviceManageHTTPClient: &http.Client{Timeout: defaultServiceManageTimeout},
	}
}

// TRE-PATCH(P2-APA-001): wire APA sleep mode through service-manager without a hard-coded URL.
func NewWorkloadScaleFromEnv(
	client client.Client,
	restMapper meta.RESTMapper,
) (WorkloadScale, error) {
	config, err := newServiceManageConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return &workloadScale{
		client:                  client,
		restMapper:              restMapper,
		serviceManageURL:        config.url,
		serviceManageHTTPClient: &http.Client{Timeout: config.timeout},
		sleepModeEnabled:        config.sleepModeEnabled,
	}, nil
}

func newServiceManageConfigFromEnv() (serviceManageConfig, error) {
	sleepModeEnabled := os.Getenv(apaScaleSleepModeEnv) != "0"
	serviceManageURL := strings.TrimSpace(os.Getenv(serviceManageURLEnv))
	if sleepModeEnabled && serviceManageURL == "" {
		return serviceManageConfig{}, fmt.Errorf("%s must be set when %s is enabled", serviceManageURLEnv, apaScaleSleepModeEnv)
	}
	timeout := defaultServiceManageTimeout
	if raw := strings.TrimSpace(os.Getenv(serviceManageTimeoutEnv)); raw != "" {
		seconds, err := strconv.ParseFloat(raw, 64)
		if err != nil || seconds <= 0 {
			return serviceManageConfig{}, fmt.Errorf("%s must be a positive number of seconds, got %q", serviceManageTimeoutEnv, raw)
		}
		timeout = time.Duration(seconds * float64(time.Second))
	}
	return serviceManageConfig{
		url:              serviceManageURL,
		sleepModeEnabled: sleepModeEnabled,
		timeout:          timeout,
	}, nil
}

func (s *workloadScale) Validate(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler) error {
	// Role-level scaling validation
	if pa.Spec.SubTargetSelector != nil {
		return s.validateRoleScaling(ctx, pa)
	}

	// For generic scaling, just verify the resource exists
	// We don't need to validate /scale subresource anymore
	return nil
}

func (s *workloadScale) validateRoleScaling(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler) error {
	ref := pa.Spec.ScaleTargetRef
	if ref.Kind != "StormService" {
		return fmt.Errorf("subTargetSelector only supported for StormService, got %q", ref.Kind)
	}

	if pa.Spec.SubTargetSelector.RoleName == "" {
		return fmt.Errorf("subTargetSelector.roleName must be set")
	}

	ns := ref.Namespace
	if ns == "" {
		ns = pa.Namespace
	}

	ss := &orchestrationv1alpha1.StormService{}
	if err := s.client.Get(ctx, client.ObjectKey{Namespace: ns, Name: ref.Name}, ss); err != nil {
		return fmt.Errorf("failed to get StormService: %w", err)
	}

	if ss.Spec.Template.Spec == nil {
		return fmt.Errorf("StormService template.spec is nil")
	}

	// Check if role exists
	for _, role := range ss.Spec.Template.Spec.Roles {
		if role.Name == pa.Spec.SubTargetSelector.RoleName {
			return nil
		}
	}

	return fmt.Errorf("role %q not found in StormService %s", pa.Spec.SubTargetSelector.RoleName, ref.Name)
}

func (s *workloadScale) GetCurrentReplicasFromScale(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, scale *unstructured.Unstructured) (int32, error) {
	// TRE-PATCH(P2-APA-001): APA sleep mode counts wake replicas from service-manager, not Kubernetes spec.replicas.
	if s.shouldUseAPASleepMode(pa) {
		return s.getWakeReplicasFromServiceManage(ctx, pa.Spec.ScaleTargetRef.Name)
	}

	// Role-level scaling
	if s.isStormServiceWorkload(scale) && pa.Spec.SubTargetSelector != nil {
		return s.getCurrentReplicasForRole(ctx, pa)
	}

	// Generic scaling - extract from spec.replicas
	currentReplicasInt64, found, err := unstructured.NestedInt64(scale.Object, "spec", "replicas")
	if !found {
		return 0, fmt.Errorf("the 'replicas' field was not found in the scale object")
	}
	if err != nil {
		return 0, fmt.Errorf("failed to get 'replicas' from scale: %w", err)
	}
	return int32(currentReplicasInt64), nil
}

func (s *workloadScale) getCurrentReplicasForRole(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler) (int32, error) {
	ref := pa.Spec.ScaleTargetRef
	ns := ref.Namespace
	if ns == "" {
		ns = pa.Namespace
	}

	ss := &orchestrationv1alpha1.StormService{}
	if err := s.client.Get(ctx, client.ObjectKey{Namespace: ns, Name: ref.Name}, ss); err != nil {
		return 0, err
	}

	// replica mode, return the replicas directly
	// we can not easily use `*ss.Spec.Replicas > 1` as condition since 1 could be pool or replica both case under autoscaling scenarios
	if pa.Annotations[AutoscalingStormServiceModeAnnotationKey] == "replica" {
		return *ss.Spec.Replicas, nil
	}

	roleName := pa.Spec.SubTargetSelector.RoleName
	for _, roleStatus := range ss.Status.RoleStatuses {
		if roleStatus.Name == roleName {
			return roleStatus.Replicas, nil
		}
	}

	// Fallback to spec
	if ss.Spec.Template.Spec == nil {
		return 0, fmt.Errorf("stormservice %s/%s template.spec is nil", ns, ref.Name)
	}

	for _, role := range ss.Spec.Template.Spec.Roles {
		if role.Name == roleName {
			if role.Replicas != nil {
				return *role.Replicas, nil
			}
			return 0, nil
		}
	}

	return 0, fmt.Errorf("role %s not found", roleName)
}

func (s *workloadScale) SetDesiredReplicas(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, replicas int32) error {
	// TRE-PATCH(P2-APA-002): APA sleep mode applies an absolute wake/sleep target through service-manager.
	if s.shouldUseAPASleepMode(pa) {
		return s.setWakeReplicasViaServiceManage(ctx, pa.Spec.ScaleTargetRef.Name, replicas)
	}

	// Role-level scaling
	if pa.Spec.SubTargetSelector != nil {
		return s.setDesiredReplicasForRole(ctx, pa, replicas)
	}

	// Generic scaling - use RetryOnConflict to handle concurrent updates
	ref := pa.Spec.ScaleTargetRef
	ns := ref.Namespace
	if ns == "" {
		ns = pa.Namespace
	}

	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Parse API version
		gv, err := schema.ParseGroupVersion(ref.APIVersion)
		if err != nil {
			return fmt.Errorf("invalid apiVersion %q: %w", ref.APIVersion, err)
		}

		// Create unstructured object for the resource
		scale := &unstructured.Unstructured{}
		scale.SetGroupVersionKind(schema.GroupVersionKind{
			Group:   gv.Group,
			Version: gv.Version,
			Kind:    ref.Kind,
		})

		// Get current resource
		if err := s.client.Get(ctx, client.ObjectKey{Namespace: ns, Name: ref.Name}, scale); err != nil {
			return err
		}

		// Update replicas field
		if err := unstructured.SetNestedField(scale.Object, int64(replicas), "spec", "replicas"); err != nil {
			return fmt.Errorf("failed to set replicas field: %w", err)
		}

		// Update the resource
		// Note: we have choice to use scale api, but it requires /scale RBAC, to simplify the scenario, let's use current way
		if err := s.client.Update(ctx, scale); err != nil {
			return err
		}

		klog.InfoS("Scaled resource", "kind", ref.Kind, "name", ref.Name, "ns", ns, "replicas", replicas)
		return nil
	})
}

func (s *workloadScale) setDesiredReplicasForRole(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, replicas int32) error {
	ref := pa.Spec.ScaleTargetRef
	ns := ref.Namespace
	if ns == "" {
		ns = pa.Namespace
	}
	roleName := pa.Spec.SubTargetSelector.RoleName

	// Use RetryOnConflict to handle concurrent updates
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		cur := &orchestrationv1alpha1.StormService{}
		if err := s.client.Get(ctx, client.ObjectKey{Namespace: ns, Name: ref.Name}, cur); err != nil {
			return err
		}
		upd := cur.DeepCopy()
		// TODO: tricky part. it's hard to know replica=1 is pooling or replica mode, we can use autoscaling to limit it.
		// we can extract the method and fallback to ss object annotation to check as well.
		if pa.Annotations[AutoscalingStormServiceModeAnnotationKey] == "replica" {
			upd.Spec.Replicas = ptr.To(replicas)
			return s.client.Patch(ctx, upd, client.MergeFrom(cur))
		}
		// handle pool mode
		if upd.Spec.Template.Spec == nil {
			return fmt.Errorf("template.spec is nil")
		}
		found := false
		for i := range upd.Spec.Template.Spec.Roles {
			if upd.Spec.Template.Spec.Roles[i].Name == roleName {
				upd.Spec.Template.Spec.Roles[i].Replicas = ptr.To(replicas)
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("role %q not found", roleName)
		}
		return s.client.Patch(ctx, upd, client.MergeFrom(cur))
	})
}

func (s *workloadScale) shouldUseAPASleepMode(pa *autoscalingv1alpha1.PodAutoscaler) bool {
	return s.sleepModeEnabled && pa.Spec.ScalingStrategy == autoscalingv1alpha1.APA
}

func (s *workloadScale) serviceManageClient() *http.Client {
	if s.serviceManageHTTPClient != nil {
		return s.serviceManageHTTPClient
	}
	return &http.Client{Timeout: defaultServiceManageTimeout}
}

// serviceManageContext is the context of one service-manager call. The reconcile context carries
// DefaultReconcileTimeoutDuration (10 s, podautoscaler_controller.go Reconcile), far below a plan's
// worst case (writer lock wait + one hold, service_manager.api_call_timeout_s): a call bound to it
// would be cut off mid-plan while the SM completes the plan anyway. So the call gets its OWN
// deadline - the client timeout (SERVICE_MANAGE_TIMEOUT_S) - and keeps the parent's values; it still
// ends at once when the parent is CANCELED (the manager shutting down), but not when the parent
// only reached its reconcile deadline.
//
// Consequence (kept on purpose, APA baseline behaviour): the PodAutoscaler controller runs ONE
// reconcile worker (MaxConcurrentReconciles is not raised), so a slow SM call delays the reconciles
// of the other PodAutoscalers until it returns. With absolute targets nothing stacks up meanwhile:
// the next reconcile re-reads the awake count and sends the target it wants then.
func (s *workloadScale) serviceManageContext(parent context.Context) (context.Context, context.CancelFunc) {
	timeout := s.serviceManageClient().Timeout
	if timeout <= 0 {
		timeout = defaultServiceManageTimeout
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), timeout)
	stop := context.AfterFunc(parent, func() {
		if errors.Is(parent.Err(), context.Canceled) {
			cancel()
		}
	})
	return ctx, func() {
		stop()
		cancel()
	}
}

func (s *workloadScale) serviceManageEndpoint(path string, values url.Values) (string, error) {
	if strings.TrimSpace(s.serviceManageURL) == "" {
		return "", fmt.Errorf("%s is required for APA sleep mode", serviceManageURLEnv)
	}
	endpoint := strings.TrimRight(s.serviceManageURL, "/") + path
	if len(values) > 0 {
		endpoint += "?" + values.Encode()
	}
	return endpoint, nil
}

func (s *workloadScale) getWakeReplicasFromServiceManage(ctx context.Context, modelName string) (int32, error) {
	reqURL, err := s.serviceManageEndpoint("/models_replicas", url.Values{"models": []string{modelName}})
	if err != nil {
		return 0, err
	}
	ctx, cancel := s.serviceManageContext(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to build service_manage /models_replicas request: %w", err)
	}
	resp, err := s.serviceManageClient().Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to call service_manage /models_replicas: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read service_manage response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("service_manage /models_replicas returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]int32
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("failed to parse service_manage response: %w", err)
	}
	replicas, ok := result[modelName]
	if !ok {
		return 0, fmt.Errorf("model %s not found in service_manage response", modelName)
	}
	return replicas, nil
}

// TRE-PATCH(P2-APA-002): an ABSOLUTE target through POST /v2/plans instead of a v1 /scale_service
// delta. The service-manager resolves "at least / at most desiredReplicas" under its writer lock, so
// two reconciles that computed different deltas from one awake count can no longer stack (the
// lost-update race of the whole-lock design, tre/docs/design/20261002-sm-wholelock.md section 7).
// One model per call (APA scales per PodAutoscaler): desired > current -> one wake item
// at_least=desired; desired < current -> one sleep item at_most=desired on the "apa" sleep path.
func (s *workloadScale) setWakeReplicasViaServiceManage(ctx context.Context, modelName string, desiredReplicas int32) error {
	currentReplicas, err := s.getWakeReplicasFromServiceManage(ctx, modelName)
	if err != nil {
		return err
	}
	if desiredReplicas == currentReplicas {
		return nil
	}
	plan := servicePlanRequest{Sleeps: []servicePlanSleepItem{}, Wakes: []servicePlanWakeItem{}}
	if desiredReplicas > currentReplicas {
		plan.Wakes = append(plan.Wakes, servicePlanWakeItem{Model: modelName, AtLeast: desiredReplicas, Reason: "apa"})
	} else {
		plan.Sleeps = append(plan.Sleeps, servicePlanSleepItem{
			Model: modelName, AtMost: desiredReplicas, SleepPath: apaSleepPath, Reason: "apa",
		})
	}
	return s.postServicePlan(ctx, modelName, currentReplicas, desiredReplicas, plan)
}

// servicePlanRequest is the body of the service-manager's POST /v2/plans (one item per model).
type servicePlanRequest struct {
	Sleeps []servicePlanSleepItem `json:"sleeps"`
	Wakes  []servicePlanWakeItem  `json:"wakes"`
}

type servicePlanSleepItem struct {
	Model     string `json:"model"`
	AtMost    int32  `json:"at_most"`
	SleepPath string `json:"sleep_path"`
	Reason    string `json:"reason,omitempty"`
}

type servicePlanWakeItem struct {
	Model   string `json:"model"`
	AtLeast int32  `json:"at_least"`
	Reason  string `json:"reason,omitempty"`
}

// servicePlanItemResult is one item of the plan response (only the fields APA logs or decides on).
type servicePlanItemResult struct {
	Model          string                  `json:"model"`
	Status         string                  `json:"status"`
	AwakeBefore    *int32                  `json:"awake_before,omitempty"`
	AwakeAfter     *int32                  `json:"awake_after,omitempty"`
	Taken          int32                   `json:"taken"`
	Unfilled       int32                   `json:"unfilled"`
	ClampedByFloor bool                    `json:"clamped_by_floor"`
	Error          *servicePlanErrorDetail `json:"error,omitempty"`
}

type servicePlanErrorDetail struct {
	Error  string `json:"error"`
	Reason string `json:"reason"`
	Detail string `json:"detail"`
}

// servicePlanResponse is the plan response (200), or the 409 body: "partial" carries the whole
// response; "writer_busy" / "routable_unknown" carry no items (nothing was done).
type servicePlanResponse struct {
	PlanID string                  `json:"plan_id"`
	Status string                  `json:"status"`
	Error  string                  `json:"error"`
	Sleeps []servicePlanItemResult `json:"sleeps"`
	Wakes  []servicePlanItemResult `json:"wakes"`
}

func (s *workloadScale) postServicePlan(ctx context.Context, modelName string, current, desired int32, plan servicePlanRequest) error {
	reqURL, err := s.serviceManageEndpoint("/v2/plans", nil)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(plan)
	if err != nil {
		return fmt.Errorf("failed to encode service_manage plan: %w", err)
	}
	ctx, cancel := s.serviceManageContext(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to build service_manage /v2/plans request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.serviceManageClient().Do(req)
	if err != nil {
		return fmt.Errorf("failed to call service_manage /v2/plans: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read service_manage plan response: %w", err)
	}
	var planResp servicePlanResponse
	parseErr := json.Unmarshal(body, &planResp)
	var item *servicePlanItemResult
	if parseErr == nil {
		for _, items := range [][]servicePlanItemResult{planResp.Wakes, planResp.Sleeps} {
			for i := range items {
				if items[i].Model == modelName {
					item = &items[i]
				}
			}
		}
	}
	switch {
	case resp.StatusCode == http.StatusOK && parseErr == nil && item != nil:
		// done / noop / clamped (floor) / partial (some wakes placed): what happened stays; the
		// next reconcile re-reads the awake count.
		if item.Status != "done" && item.Status != "noop" {
			klog.Warningf("service_manage plan %s for %s: %d -> %d is %s (unfilled %d, clamped_by_floor %t)",
				planResp.PlanID, modelName, current, desired, item.Status, item.Unfilled, item.ClampedByFloor)
		}
		klog.InfoS("service_manage plan completed", "model", modelName, "plan", planResp.PlanID,
			"current", current, "desired", desired, "status", item.Status, "taken", item.Taken, "unfilled", item.Unfilled)
		return nil
	case resp.StatusCode == http.StatusConflict && parseErr == nil && planResp.Error == "partial" &&
		item != nil && item.Status == "refused":
		// Nothing changed: the wake could not be placed (or a sleep was refused). The v1
		// /scale_service answered this case 200 with actual 0; keep that: warn, no error, so the
		// controller does not requeue with backoff against a state that has not changed.
		reason := ""
		if item.Error != nil {
			reason = item.Error.Reason
		}
		klog.Warningf("service_manage plan %s for %s: %d -> %d refused (%s), nothing changed",
			planResp.PlanID, modelName, current, desired, reason)
		return nil
	default:
		// writer_busy, routable_unknown, a failed wake / sleep, a bad request, or an answer this
		// client cannot read: an error, as for any service-manager error before.
		return fmt.Errorf("service_manage /v2/plans returned status %d: %s", resp.StatusCode, string(body))
	}
}

func (s *workloadScale) getPodSelectorForRole(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler) (labels.Selector, error) {
	ref := pa.Spec.ScaleTargetRef
	ns := ref.Namespace
	if ns == "" {
		ns = pa.Namespace
	}

	ss := &orchestrationv1alpha1.StormService{}
	if err := s.client.Get(ctx, client.ObjectKey{Namespace: ns, Name: ref.Name}, ss); err != nil {
		klog.ErrorS(err, "Failed to get StormService", "namespace", ns, "name", ref.Name)
		return nil, err
	}
	// Note: it's possible user just configure StormService without role name, in that case, we just aggregate all the pods.
	labelSelector := labels.SelectorFromSet(labels.Set{
		constants.StormServiceNameLabelKey: ss.Name,
	})

	roleName := ""
	if pa.Spec.SubTargetSelector != nil && pa.Spec.SubTargetSelector.RoleName != "" {
		req, err := labels.NewRequirement(constants.RoleNameLabelKey, selection.Equals, []string{pa.Spec.SubTargetSelector.RoleName})
		if err != nil {
			return nil, err
		}
		labelSelector = labelSelector.Add(*req)
		roleName = pa.Spec.SubTargetSelector.RoleName
	}

	// ss.Spec.Template.Spec may be nil in test cases
	if ss.Spec.Template.Spec != nil {
		for _, role := range ss.Spec.Template.Spec.Roles {
			// skip if roleName specified but not equal
			if roleName != "" && role.Name != roleName {
				continue
			}

			// add new selector for podGroup index 0, based on #1704
			if role.PodGroupSize != nil && *role.PodGroupSize > 1 {
				req, err := labels.NewRequirement(constants.PodGroupIndexLabelKey, selection.Equals, []string{"0"})
				if err != nil {
					return nil, err
				}
				labelSelector = labelSelector.Add(*req)
			}
		}
	}
	return labelSelector, nil
}

func (s *workloadScale) GetPodSelectorFromScale(ctx context.Context, pa *autoscalingv1alpha1.PodAutoscaler, scale *unstructured.Unstructured) (labels.Selector, error) {
	// For role-level scaling, get StormService and add role requirement
	isStormService := s.isStormServiceWorkload(scale)
	if isStormService {
		return s.getPodSelectorForRole(ctx, pa)
	}

	// For generic scaling, extract from scale object
	// Try status.selector first (string format used by /scale subresource)
	statusSelector, found, err := unstructured.NestedString(scale.Object, "status", "selector")
	if err == nil && found && strings.TrimSpace(statusSelector) != "" {
		return labels.Parse(statusSelector)
	}

	// Try spec.selector (LabelSelector format)
	selectorMap, found, err := unstructured.NestedMap(scale.Object, "spec", "selector")
	if err != nil {
		return nil, fmt.Errorf("failed to get 'spec.selector' from scale: %w", err)
	}
	if !found {
		// No selector found, return error
		return nil, fmt.Errorf("scale object %q is missing spec.selector", scale.GetName())
	}

	// Convert selectorMap to a *metav1.LabelSelector object
	selector := &metav1.LabelSelector{}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(selectorMap, selector)
	if err != nil {
		return nil, fmt.Errorf("failed to convert 'spec.selector' to LabelSelector: %w", err)
	}

	labelsSelector, err := metav1.LabelSelectorAsSelector(selector)
	if err != nil {
		return nil, fmt.Errorf("failed to convert LabelSelector to labels.Selector: %w", err)
	}

	return labelsSelector, nil
}

func (s *workloadScale) isStormServiceWorkload(scale *unstructured.Unstructured) bool {
	isStormService := false
	if scale.GetAPIVersion() == "orchestration.aibrix.ai/v1alpha1" && scale.GetKind() == "StormService" {
		isStormService = true
	}
	return isStormService
}

// isStormServiceWorkload checks if the given scale object is a StormService
