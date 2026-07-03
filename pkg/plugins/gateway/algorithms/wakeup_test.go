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

package routingalgorithms

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCallWakeUpServiceRequiresConfiguredURL(t *testing.T) {
	t.Setenv("SERVEMENT_URL", "")

	success, delayed, err := callWakeUpService("dsqwen-7b", 0, 5)

	if err == nil {
		t.Fatal("expected SERVEMENT_URL validation error")
	}
	if !strings.Contains(err.Error(), "SERVEMENT_URL") {
		t.Fatalf("expected SERVEMENT_URL in error, got %v", err)
	}
	if success || delayed {
		t.Fatalf("expected false response flags on config error, got success=%v delayed=%v", success, delayed)
	}
}

func TestCallWakeUpServicePostsToConfiguredURL(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/wake_up" {
			t.Fatalf("path = %s, want /wake_up", r.URL.Path)
		}
		query := r.URL.Query()
		if query.Get("model_name") != "dsqwen-7b" {
			t.Fatalf("model_name = %s", query.Get("model_name"))
		}
		if query.Get("kind") != "0" {
			t.Fatalf("kind = %s, want 0", query.Get("kind"))
		}
		if query.Get("queue_len") != "7" {
			t.Fatalf("queue_len = %s, want 7", query.Get("queue_len"))
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"success":true,"delayed":false}`))
	}))
	defer server.Close()
	t.Setenv("SERVEMENT_URL", server.URL)

	success, delayed, err := callWakeUpService("dsqwen-7b", 4, 7)

	if err != nil {
		t.Fatalf("callWakeUpService returned error: %v", err)
	}
	if !called {
		t.Fatal("expected configured server to be called")
	}
	if !success || delayed {
		t.Fatalf("unexpected response flags: success=%v delayed=%v", success, delayed)
	}
}
