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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
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
	defaultServiceManageTimeout              = 10 * time.Second
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
		serviceManageHTTPClient: &http.Client{Timeout: defaultServiceManageTimeout},
		sleepModeEnabled:        config.sleepModeEnabled,
	}, nil
}

func newServiceManageConfigFromEnv() (serviceManageConfig, error) {
	sleepModeEnabled := os.Getenv(apaScaleSleepModeEnv) != "0"
	serviceManageURL := strings.TrimSpace(os.Getenv(serviceManageURLEnv))
	if sleepModeEnabled && serviceManageURL == "" {
		return serviceManageConfig{}, fmt.Errorf("%s must be set when %s is enabled", serviceManageURLEnv, apaScaleSleepModeEnv)
	}
	return serviceManageConfig{
		url:              serviceManageURL,
		sleepModeEnabled: sleepModeEnabled,
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
	// TRE-PATCH(P2-APA-001): APA sleep mode applies wake/sleep deltas through service-manager.
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

func (s *workloadScale) setWakeReplicasViaServiceManage(ctx context.Context, modelName string, desiredReplicas int32) error {
	currentReplicas, err := s.getWakeReplicasFromServiceManage(ctx, modelName)
	if err != nil {
		return err
	}
	delta := desiredReplicas - currentReplicas
	if delta == 0 {
		return nil
	}
	scaleType := "up"
	scaleValue := delta
	if delta < 0 {
		scaleType = "down"
		scaleValue = -delta
	}
	return s.scaleViaServiceManage(ctx, modelName, scaleType, scaleValue)
}

func (s *workloadScale) scaleViaServiceManage(ctx context.Context, modelName, scaleType string, scaleValue int32) error {
	reqURL, err := s.serviceManageEndpoint("/scale_service", url.Values{
		"model_name":  []string{modelName},
		"scale_type":  []string{scaleType},
		"scale_value": []string{fmt.Sprintf("%d", scaleValue)},
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, nil)
	if err != nil {
		return fmt.Errorf("failed to build service_manage /scale_service request: %w", err)
	}
	resp, err := s.serviceManageClient().Do(req)
	if err != nil {
		return fmt.Errorf("failed to call service_manage /scale_service: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read service_manage scale response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("service_manage /scale_service returned status %d: %s", resp.StatusCode, string(body))
	}

	var scaleResp struct {
		Actual    int32 `json:"actual"`
		Requested int32 `json:"requested"`
	}
	if err := json.Unmarshal(body, &scaleResp); err != nil {
		return fmt.Errorf("failed to parse service_manage scale response: %w", err)
	}
	if scaleResp.Actual < scaleResp.Requested {
		klog.Warningf("service_manage scale_%s for %s: requested %d but only %d succeeded", scaleType, modelName, scaleResp.Requested, scaleResp.Actual)
	}
	klog.InfoS("service_manage scale completed", "model", modelName, "scaleType", scaleType, "requested", scaleResp.Requested, "actual", scaleResp.Actual)
	return nil
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
