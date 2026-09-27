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
	"os"
	"testing"
)

// TestMain pins the optional TRE default routing strategy off (its code default), even if
// TRE_DEFAULT_ROUTING_STRATEGY is set in the environment, so the upstream tests keep
// exercising the "no strategy -> HTTPRoute" path they were written for. TRE tests that
// need it set it explicitly via setTREDefaultRoutingStrategy.
func TestMain(m *testing.M) {
	setTREDefaultRoutingStrategy("")
	os.Exit(m.Run())
}
