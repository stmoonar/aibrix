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
	"sync/atomic"

	"github.com/vllm-project/aibrix/pkg/utils"
)

// TRE-PATCH(P2-GW-005): stamp the body model onto the `model` request header on the
// explicit-routing (routing-strategy) path.
//
// The tre-v2 gateway sends ext_proc-routed traffic to one ORIGINAL_DST cluster PER MODEL
// (route match: routing-strategy + model header) so the per-model circuit breakers and
// the controller's per-model Envoy stats keep working. A v1-style client (OpenAI SDK with
// only the routing-strategy header) carries the model in the body alone; setting the
// header here, together with the ClearRouteCache already returned at the request-header
// phase, lets Envoy re-pick the per-model route when the router filter runs. It also
// makes the header authoritative (body model wins over a stale client header). Off
// unless TRE_ROUTE_MODEL_HEADER=true (set only on the tre-v2 gateway-plugins Deployment).
const treRouteModelHeaderEnv = "TRE_ROUTE_MODEL_HEADER"

var treRouteModelHeader atomic.Bool

func init() {
	treRouteModelHeader.Store(utils.LoadEnvBool(treRouteModelHeaderEnv, false))
}

// setTRERouteModelHeader overrides the switch (tests) and returns the previous value.
func setTRERouteModelHeader(enabled bool) bool {
	return treRouteModelHeader.Swap(enabled)
}
