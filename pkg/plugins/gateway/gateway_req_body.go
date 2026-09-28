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
	"errors"
	"fmt"
	"strconv"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"github.com/vllm-project/aibrix/pkg/constants"
	"github.com/vllm-project/aibrix/pkg/metrics"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// HandleRequestBody routes one request body. It is the stream-less entry point (tests,
// benchmarks): the inflight slot taken on the target pod is released before returning,
// because without an ext_proc stream there is no request end to wait for. Process uses
// handleRequestBody and holds the slot until the stream ends.
func (s *Server) HandleRequestBody(ctx context.Context, routingCtx *types.RoutingContext, requestID string, req *extProcPb.ProcessingRequest, user utils.User) (*extProcPb.ProcessingResponse, string, bool, int64) {
	resp, model, stream, term, ticket := s.handleRequestBody(ctx, routingCtx, requestID, req, user)
	ticket.Release()
	return resp, model, stream, term
}

// handleRequestBody is HandleRequestBody plus the inflight ticket (TRE-PATCH P3-GW-009) of
// the routed request; the caller must Release it when the request ends. The ticket is nil
// whenever no pod was committed (errors, HTTPRoute path).
//
//nolint:gocyclo
func (s *Server) handleRequestBody(ctx context.Context, routingCtx *types.RoutingContext, requestID string, req *extProcPb.ProcessingRequest, user utils.User) (*extProcPb.ProcessingResponse, string, bool, int64, *treInflightTicket) {
	var term int64 // Identify the trace window
	var ticket *treInflightTicket

	requestPath := routingCtx.ReqPath

	body := req.Request.(*extProcPb.ProcessingRequest_RequestBody)

	ctx, span := tracer.Start(ctx, "HandleRequestBody")
	defer span.End()

	var model, message string
	var stream bool
	var routingAlgorithm types.RoutingAlgorithm
	var errRes *extProcPb.ProcessingResponse

	// Check if this is a multipart request (audio endpoints)
	contentType := routingCtx.ReqHeaders[contentTypeKey]
	if isAudioRequest(requestPath) && isMultipartRequest(contentType) {
		// Parse multipart form data for audio endpoints
		model, stream, errRes = parseMultipartFormData(requestID, contentType, body.RequestBody.GetBody())
		if errRes != nil {
			return errRes, model, stream, term, nil
		}
		message = "" // Audio requests don't have a text message for token counting
	} else {
		// Use existing JSON validation for other endpoints
		var promptTokenIDs []int
		model, message, stream, promptTokenIDs, errRes = validateRequestBodyWithTokens(requestID, requestPath, body.RequestBody.GetBody(), user)
		if errRes != nil {
			return errRes, model, stream, term, nil
		}
		// TRE-PATCH(P3-GW-010, D6): a token-id prompt counts as its ids, not as text.
		if promptTokenIDs != nil {
			routingCtx.SetPromptTokens(promptTokenIDs)
		}
	}

	routingCtx.Model = model
	routingCtx.Message = message
	routingCtx.ReqBody = body.RequestBody.GetBody()

	// early reject if model doesn't exist or no pods are ready
	var podsArr types.PodList
	podsArr, errRes = s.validateModelAvailability(requestID, model)
	if errRes != nil {
		return errRes, model, stream, term, nil
	}

	// Read engine label from pods and assign to routing context
	if pods := podsArr.All(); len(pods) > 0 {
		routingCtx.Engine = pods[0].Labels[constants.ModelLabelEngine]
		if routingCtx.Engine == "" {
			routingCtx.Engine = pods[0].Annotations[constants.ModelLabelEngine]
		}
	}

	// Resolve model config profile from annotation and apply overrides
	applyConfigProfile(routingCtx, podsArr.All())

	// Derive and validate routing strategy (headers -> profile -> env); return 400 on invalid
	if strategy, enabled := deriveRoutingStrategyFromContext(routingCtx); enabled {
		var ok bool
		if routingAlgorithm, ok = routing.Validate(strategy); !ok {
			klog.ErrorS(nil, "incorrect routing strategy", "requestID", requestID, "routing-strategy", strategy)
			return buildErrorResponse(envoyTypePb.StatusCode_BadRequest, fmt.Sprintf("incorrect routing strategy %s", strategy), "", "", HeaderErrorRouting, "true"), model, stream, term, nil
		}
		routingCtx.Algorithm = routingAlgorithm
	}

	// TRE-PATCH(P3-GW-010/013, D5/D10): exclusion and coordination need pod-level routing.
	// If no strategy resolved (no header, profile, ROUTING_ALGORITHM or
	// TRE_DEFAULT_ROUTING_STRATEGY), use the random router rather than the Service path:
	// the Service path could hand the request back to an excluded pod, and a request it
	// carries is neither counted in inflight nor bound to the acked route table.
	excludedPods := treExcludedPods(routingCtx.ReqHeaders)
	if routingAlgorithm == routing.RouterNotSet && (len(excludedPods) > 0 || treGW.enabled.Load()) {
		routingAlgorithm = routing.RouterRandom
		routingCtx.Algorithm = routingAlgorithm
	}

	// Pre-allocate for the routing path (4 headers: strategy, target-pod, content-length, X-Request-Id).
	headers := make([]*configPb.HeaderValueOption, 0, 4)

	// Path rewriting for image/video generation based on engine type
	// xdit engine uses /generate and /generatevideo endpoints
	// vllm/vllm-omni uses OpenAI-compatible /v1/images/generations
	if rewritePath := getEngineBasedPathRewrite(requestPath, podsArr.All()); rewritePath != "" {
		headers = buildEnvoyProxyHeaders(headers, ":path", rewritePath)
	}

	if errRes = s.enforceModelRPS(ctx, model, routingCtx); errRes != nil {
		return errRes, model, stream, term, nil
	}
	needsRollback := true
	defer func() {
		if needsRollback {
			s.decrModelRPS(ctx, model, routingCtx)
		}
	}()

	if routingAlgorithm == routing.RouterNotSet {
		if err := s.validateHTTPRouteStatus(ctx, model); err != nil {
			return buildErrorResponse(envoyTypePb.StatusCode_ServiceUnavailable, err.Error(), ErrorCodeServiceUnavailable, "", HeaderErrorRouting, "true"), model, stream, term, nil
		}
		headers = buildEnvoyProxyHeaders(headers, HeaderModel, model)
		klog.InfoS("request_start", "request_id", requestID, "request_path", requestPath, "model", model, "stream", stream)
	} else {
		externalFilter := routingCtx.ReqHeaders[HeaderExternalFilter]
		nonContinuable := treNonContinuable(requestPath, routingCtx.ReqBody)
		var targetPodIP string
		var err error
		// TRE-PATCH(P3-GW-009, D3/D4): commit the chosen pod (inflight +1) atomically with
		// a re-check against the acked route table; if a hide won the race, route again.
		for attempt := 1; ; attempt++ {
			targetPodIP, err = s.selectTargetPod(ctx, routingCtx, podsArr, externalFilter)
			if targetPodIP == "" || err != nil {
				break
			}
			var target *v1.Pod
			if routingCtx.HasRouted() {
				target = routingCtx.TargetPod()
			}
			var cerr error
			if ticket, cerr = treGW.commit(target, nonContinuable); cerr == nil {
				break
			}
			if !errors.Is(cerr, errTRENotRoutable) {
				targetPodIP, err = "", cerr // shutting down / no target: fail closed
				break
			}
			if attempt >= treCommitAttempts {
				targetPodIP, err = "", errTRECommitRace
				break
			}
			klog.V(4).InfoS("target pod became unroutable before commit; re-routing", "requestID", requestID, "pod", target.Name, "attempt", attempt)
			routingCtx.ResetTargetPod()
			if podsArr, err = s.cache.ListPodsByModel(model); err != nil {
				targetPodIP = ""
				break
			}
		}
		if targetPodIP == "" || err != nil {
			klog.ErrorS(err, "failed to select target pod", "requestID", requestID, "routingStrategy", routingAlgorithm, "model", model, "routingDuration", routingCtx.GetRoutingDelay())
			if errors.Is(err, errTREUnsupportedRouter) {
				return buildErrorResponse(envoyTypePb.StatusCode_BadRequest, err.Error(), "", "", HeaderErrorRouting, "true"), model, stream, term, nil
			}
			msg := "error on selecting target pod"
			if errors.Is(err, errTREAllCandidatesExcluded) || errors.Is(err, errTRECommitRace) ||
				errors.Is(err, errTREShuttingDown) || errors.Is(err, errTRENoTarget) {
				msg = err.Error()
			}
			// Retry-After: the sidecar (and well-behaved clients) retry shortly; a hidden or
			// excluded pod set is transient during a sleep/wake transition.
			return buildErrorResponse(envoyTypePb.StatusCode_ServiceUnavailable, msg, ErrorCodeServiceUnavailable, "", HeaderErrorRouting, "true",
				"Retry-After", strconv.Itoa(treRetryAfterSeconds)), model, stream, term, nil
		}
		headers = buildEnvoyProxyHeaders(headers,
			HeaderRoutingStrategy, string(routingAlgorithm),
			HeaderTargetPod, targetPodIP,
			"content-length", strconv.Itoa(len(routingCtx.ReqBody)),
			"X-Request-Id", routingCtx.RequestID)
		if treRouteModelHeader.Load() {
			headers = buildEnvoyProxyHeaders(headers, HeaderModel, model)
		}

		var targetPodName, targetNamespace string
		var request_count float64
		if routingCtx.HasRouted() && routingCtx.TargetPod() != nil {
			targetPodName = routingCtx.TargetPod().Name
			targetNamespace = routingCtx.TargetPod().Namespace
			request_count = getRunningRequestsByPod(s, targetPodName, targetNamespace)
		}

		routingDelay := routingCtx.GetRoutingDelay()
		if routingAlgorithm == routing.RouterPD && !routingCtx.PrefillStartTime.IsZero() {
			routingDelay = routingCtx.PrefillStartTime.Sub(routingCtx.RequestTime)
		}
		klog.InfoS("request_start", "request_id", requestID, "request_path", requestPath, "model", model, "stream", stream, "routing_strategy", routingAlgorithm,
			"target_pod", targetPodName, "target_pod_ip", targetPodIP, "outstanding_requests", request_count, "routing_time_taken", routingDelay)
	}

	needsRollback = false
	routingCtx.RequestEndTime = time.Now()
	term = s.cache.AddRequestCount(routingCtx, requestID, model)

	var removeHeaders []string
	if _, ok := routingCtx.ReqHeaders[HeaderTREExcludePod]; ok {
		removeHeaders = []string{HeaderTREExcludePod} // gateway-internal, not for the engine
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders:    headers,
						RemoveHeaders: removeHeaders,
					},
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_Body{
							Body: routingCtx.ReqBody,
						},
					},
				},
			},
		},
	}, model, stream, term, ticket
}

// getEngineBasedPathRewrite returns the rewritten path for image/video generation endpoints
// based on the engine type specified in the pod labels/annotations.
// Returns empty string if no rewrite is needed (e.g., for vllm/vllm-omni which uses OpenAI-compatible paths).
func getEngineBasedPathRewrite(requestPath string, pods []*v1.Pod) string {
	if len(pods) == 0 {
		return ""
	}

	// Get engine type from the first pod (all pods for a model should have the same engine)
	pod := pods[0]
	engine := pod.Labels[constants.ModelLabelEngine]
	if engine == "" {
		engine = pod.Annotations[constants.ModelLabelEngine]
	}

	// Only xdit engine needs path rewriting to its native endpoints
	if engine == EngineXdit {
		switch requestPath {
		case PathImagesGenerations:
			return PathXditGenerate
		case PathVideoGenerations:
			return PathXditGenerateVideo
		}
	}

	// vllm, vllm-omni, sglang, and other engines use OpenAI-compatible paths
	return ""
}

// validateModelAvailability checks that the model exists in cache and has routable pods.
// Returns the pod list and nil on success, or nil and an error response on failure.
func (s *Server) validateModelAvailability(requestID, model string) (types.PodList, *extProcPb.ProcessingResponse) {
	if !s.cache.HasModel(model) {
		klog.ErrorS(nil, "model doesn't exist in cache, probably wrong model name", "requestID", requestID, "model", model)
		return nil, generateErrorResponse(envoyTypePb.StatusCode_BadRequest,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte(model)}}},
			fmt.Sprintf("model %s does not exist", model), ErrorCodeModelNotFound, "model")
	}

	podsArr, err := s.cache.ListPodsByModel(model)
	routablePods := 0
	if podsArr != nil {
		routablePods = utils.CountRoutablePods(podsArr.All())
	}
	if err != nil || podsArr == nil || routablePods == 0 {
		// TRE-PATCH(P2-GW-002): new gateway rejects before queue routing, so trigger wake-up here.
		if err == nil && podsArr != nil && routablePods == 0 {
			routing.SubmitWakeUpIfEnabled(model, 0)
		}
		klog.ErrorS(err, "no ready pod available", "requestID", requestID, "model", model)
		return nil, generateErrorResponse(envoyTypePb.StatusCode_ServiceUnavailable,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte("true")}}},
			fmt.Sprintf("error on getting pods for model %s", model), ErrorCodeServiceUnavailable, "")
	}

	return podsArr, nil
}

// Helper to fetch running requests on a pod with safe zero fallback.
func getRunningRequestsByPod(s *Server, podName, namespace string) float64 {
	mv, err := s.cache.GetMetricValueByPod(podName, namespace, metrics.RealtimeNumRequestsRunning)
	if err != nil || mv == nil {
		return 0
	}
	return mv.GetSimpleValue()
}
