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

package gateway

import (
	"context"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/metrics"
	routingalgorithms "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P2-GW-004/005) tests. None of these call t.Parallel: they flip process-wide
// switches and restore them on exit, which is only safe while no parallel test runs.

func treTestPod(name, ip, routable string) *v1.Pod {
	labels := map[string]string{"model.aibrix.ai/port": "8000"}
	if routable != "" {
		labels[utils.TRERoutableLabel] = routable
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default", Labels: labels},
		Status: v1.PodStatus{
			PodIP:      ip,
			Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
		},
	}
}

func withTREGates(t *testing.T, labelFilter, modelHeader bool) {
	t.Helper()
	prevFilter := utils.SetTRERoutableLabelFilter(labelFilter)
	prevHeader := setTRERouteModelHeader(modelHeader)
	t.Cleanup(func() {
		utils.SetTRERoutableLabelFilter(prevFilter)
		setTRERouteModelHeader(prevHeader)
	})
}

// The candidate list handed to the router must exclude sleeping (routable=false) and
// hidden/unlabelled pods when the gate is on.
func TestTRESelectTargetPod_RouterOnlySeesRoutablePods(t *testing.T) {
	withTREGates(t, true, false)

	algo := types.RoutingAlgorithm("tre-test-routable-candidates")
	router := new(mockRouter)
	routingalgorithms.Register(algo, func() (types.Router, error) { return router, nil })
	routingalgorithms.Init()

	var seen []string
	router.On("Route", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
		for _, p := range args.Get(1).(types.PodList).All() {
			seen = append(seen, p.Name)
		}
	}).Return("10.0.0.3:8000", nil).Once()

	pods := &utils.PodArray{Pods: []*v1.Pod{
		treTestPod("sleeping", "10.0.0.1", "false"),
		treTestPod("hidden", "10.0.0.2", ""),
		treTestPod("awake-a", "10.0.0.3", "true"),
		treTestPod("awake-b", "10.0.0.4", "true"),
	}}
	server := &Server{}
	ctx := types.NewRoutingContext(context.Background(), algo, "m", "", "req", "")
	addr, err := server.selectTargetPod(context.Background(), ctx, pods, "")

	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.3:8000", addr)
	assert.ElementsMatch(t, []string{"awake-a", "awake-b"}, seen)
	router.AssertExpectations(t)
}

// With a single routable pod the router is skipped entirely; it must be that pod, never a
// sleeping one (which would otherwise make the candidate list length 2).
func TestTRESelectTargetPod_SingleRoutablePodShortCircuit(t *testing.T) {
	withTREGates(t, true, false)

	algo := types.RoutingAlgorithm("tre-test-single-routable")
	router := new(mockRouter)
	routingalgorithms.Register(algo, func() (types.Router, error) { return router, nil })
	routingalgorithms.Init()

	pods := &utils.PodArray{Pods: []*v1.Pod{
		treTestPod("sleeping", "10.0.0.1", "false"),
		treTestPod("awake", "10.0.0.2", "true"),
	}}
	server := &Server{}
	ctx := types.NewRoutingContext(context.Background(), algo, "m", "", "req", "")
	addr, err := server.selectTargetPod(context.Background(), ctx, pods, "")

	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.2:8000", addr)
	router.AssertNotCalled(t, "Route", mock.Anything, mock.Anything)
}

// End to end through the real least-gpu-cache router: a sleeping pod reports ~0 KV-cache
// usage, so without the gate it is the minimum and wins; with the gate it is never picked.
func TestTRESelectTargetPod_LeastGpuCacheSkipsIdleSleepingPod(t *testing.T) {
	pods := []*v1.Pod{
		treTestPod("sleeping", "10.0.0.1", "false"),
		treTestPod("awake-busy", "10.0.0.2", "true"),
		treTestPod("awake-light", "10.0.0.3", "true"),
	}
	store := cache.InitWithPodsModelMetrics(cache.InitWithPods(cache.InitForTest(), pods, "m"),
		map[string]map[string]metrics.MetricValue{
			"sleeping":    {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.0}},
			"awake-busy":  {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.7}},
			"awake-light": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.2}},
		})
	routingalgorithms.Init()
	podList, err := store.ListPodsByModel("m")
	assert.NoError(t, err)
	server := &Server{}

	route := func() string {
		ctx := types.NewRoutingContext(context.Background(), routingalgorithms.RouterLeastGpuCache, "m", "", "req", "")
		addr, err := server.selectTargetPod(context.Background(), ctx, podList, "")
		assert.NoError(t, err)
		return addr
	}

	withTREGates(t, false, false)
	assert.Equal(t, "10.0.0.1:8000", route(), "gate off: idle sleeping pod is the least-cache pod")

	utils.SetTRERoutableLabelFilter(true)
	for i := 0; i < 50; i++ {
		assert.Equal(t, "10.0.0.3:8000", route())
	}
}

// validateModelAvailability counts only routable pods: a model whose pods are all asleep
// is rejected with 503 instead of being routed to a sleeping pod.
func TestTREValidateModelAvailability_AllSleepingIs503(t *testing.T) {
	withTREGates(t, true, false)
	t.Setenv("HOT_SWITCH", "0")

	mc := &MockCache{}
	mc.On("HasModel", "m").Return(true)
	mc.On("ListPodsByModel", "m").Return(&utils.PodArray{Pods: []*v1.Pod{
		treTestPod("sleeping-a", "10.0.0.1", "false"),
		treTestPod("sleeping-b", "10.0.0.2", ""),
	}}, nil)
	server := &Server{cache: mc}

	pods, errRes := server.validateModelAvailability("req", "m")
	assert.Nil(t, pods)
	if assert.NotNil(t, errRes) {
		assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, errRes.GetImmediateResponse().GetStatus().GetCode())
	}

	utils.SetTRERoutableLabelFilter(false)
	pods, errRes = server.validateModelAvailability("req", "m")
	assert.Nil(t, errRes, "gate off keeps the upstream readiness-only behaviour")
	assert.NotNil(t, pods)
}

func runTREHandleRequestBody(t *testing.T, algo types.RoutingAlgorithm) *extProcPb.ProcessingResponse {
	t.Helper()
	router := new(mockRouter)
	routingalgorithms.Register(algo, func() (types.Router, error) { return router, nil })
	routingalgorithms.Init()
	router.On("Route", mock.Anything, mock.Anything).Return("10.0.0.3:8000", nil)

	mc := &MockCache{Cache: cache.NewForTest()}
	mc.On("HasModel", "dsqwen-7b").Return(true)
	mc.On("ListPodsByModel", "dsqwen-7b").Return(&utils.PodArray{Pods: []*v1.Pod{
		treTestPod("awake-a", "10.0.0.3", "true"),
		treTestPod("awake-b", "10.0.0.4", "true"),
	}}, nil)
	mc.On("AddRequestCount", mock.Anything, mock.Anything, "dsqwen-7b").Return(int64(1))

	server := &Server{cache: mc, requestCountTracker: map[string]int{}}
	req := &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestBody{
		RequestBody: &extProcPb.HttpBody{Body: []byte(`{"model":"dsqwen-7b","prompt":"hi","max_tokens":4}`)},
	}}
	ctx := types.NewRoutingContext(context.Background(), "", "", "", "req", "")
	ctx.ReqPath = PathCompletions
	ctx.ReqHeaders[HeaderRoutingStrategy] = string(algo)

	resp, model, _, _ := server.HandleRequestBody(context.Background(), ctx, "req", req, utils.User{})
	assert.Equal(t, "dsqwen-7b", model)
	assert.Nil(t, resp.GetImmediateResponse())
	return resp
}

func headerValue(resp *extProcPb.ProcessingResponse, key string) (string, bool) {
	for _, h := range resp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == key {
			return string(h.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

// The explicit-routing path stamps the body model onto the `model` header only when
// TRE_ROUTE_MODEL_HEADER is on, alongside the usual target-pod header.
func TestTRERouteModelHeader(t *testing.T) {
	withTREGates(t, true, true)
	resp := runTREHandleRequestBody(t, "tre-test-model-header-on")
	got, ok := headerValue(resp, HeaderModel)
	assert.True(t, ok)
	assert.Equal(t, "dsqwen-7b", got)
	target, ok := headerValue(resp, HeaderTargetPod)
	assert.True(t, ok)
	assert.Equal(t, "10.0.0.3:8000", target)

	setTRERouteModelHeader(false)
	resp = runTREHandleRequestBody(t, "tre-test-model-header-off")
	_, ok = headerValue(resp, HeaderModel)
	assert.False(t, ok, "default (upstream) behaviour sets no model header on the routing path")
}
