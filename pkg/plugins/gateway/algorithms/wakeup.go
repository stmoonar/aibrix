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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

const (
	wakeUpServiceURLEnv = "SERVEMENT_URL"
	wakeUpPollInterval  = 500 * time.Millisecond
)

// TRE-PATCH(P2-GW-001): gateway wake-up dispatcher for sleeping warm-pool pods.
// GetLRUDataReader is set by gateway code when kind-specific wake-up requests need LRU data.
var GetLRUDataReader func() (io.Reader, string)

var wakeUpKind = func() int {
	if v, err := strconv.Atoi(os.Getenv("QZ_KIND")); err == nil {
		return v
	}
	return 0
}()

// WakeUpServiceResponse represents the service-manager /wake_up response.
type WakeUpServiceResponse struct {
	Success      bool            `json:"success"`
	Delayed      bool            `json:"delayed"`
	StrategyType string          `json:"strategy_type"`
	Strategy     *WakeUpStrategy `json:"strategy,omitempty"`
	TotalCost    float64         `json:"total_cost"`
	WakeUpTime   float64         `json:"wake_up_time"`
}

// WakeUpStrategy represents the optional strategy details in a /wake_up response.
type WakeUpStrategy struct {
	ServesToSleep  []string `json:"serves_to_sleep"`
	ServesToWakeup []string `json:"serves_to_wakeup"`
}

// WakeUpDispatcher serializes wake-up calls across models and deduplicates pending work.
type WakeUpDispatcher struct {
	ch       chan string
	pending  sync.Map
	queueLen sync.Map
}

var globalDispatcher = &WakeUpDispatcher{ch: make(chan string, 64)}

func init() {
	go globalDispatcher.run()
}

// SubmitWakeUpIfEnabled submits a gateway wake-up request when HOT_SWITCH is enabled.
func SubmitWakeUpIfEnabled(model string, queueLen int) bool {
	if os.Getenv("HOT_SWITCH") != "1" {
		return false
	}
	globalDispatcher.Submit(model, queueLen)
	return true
}

// Submit enqueues a model wake-up request. It is non-blocking and deduplicates by model.
func (d *WakeUpDispatcher) Submit(model string, queueLen int) {
	d.queueLen.Store(model, queueLen)
	if _, loaded := d.pending.LoadOrStore(model, struct{}{}); loaded {
		return
	}
	select {
	case d.ch <- model:
	default:
		d.pending.Delete(model)
		klog.Warningf("wakeup dispatcher queue full, dropping %s", model)
	}
}

func (d *WakeUpDispatcher) run() {
	for model := range d.ch {
		d.pending.Delete(model)
		queueLen := 0
		if v, ok := d.queueLen.Load(model); ok {
			if ql, ok := v.(int); ok {
				queueLen = ql
			}
		}

		success, delayed, err := callWakeUpService(model, wakeUpKind, queueLen)
		if err != nil {
			klog.ErrorS(err, "dispatcher: callWakeUpService failed", "model", model)
		} else if delayed {
			klog.V(4).InfoS("dispatcher: wake_up delayed", "model", model, "queueLen", queueLen)
		} else if !success {
			klog.InfoS("dispatcher: wake_up returned success=false", "model", model, "queueLen", queueLen)
		}

		if delay := getWakeUpDelayTime(); delay > 0 {
			time.Sleep(delay)
		}
	}
}

func getWakeUpDelayTime() time.Duration {
	v := os.Getenv("delay_time1")
	if v != "" {
		ms, err := strconv.ParseInt(v, 10, 64)
		if err == nil && ms >= 0 {
			return time.Duration(ms) * time.Millisecond
		}
	}
	return time.Second
}

func callWakeUpService(model string, kind int, queueLen int) (bool, bool, error) {
	endpoint, err := wakeUpServiceEndpoint(model, kind, queueLen)
	if err != nil {
		return false, false, err
	}

	var body io.Reader
	if kind != 0 && kind != 4 && GetLRUDataReader != nil {
		body, _ = GetLRUDataReader()
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, body)
	if err != nil {
		return false, false, fmt.Errorf("failed to create wake_up request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false, false, fmt.Errorf("failed to execute wake_up request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, false, fmt.Errorf("failed to read wake_up response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return false, false, fmt.Errorf("wake_up request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	var wakeUpResp WakeUpServiceResponse
	if err := json.Unmarshal(respBody, &wakeUpResp); err != nil {
		return false, false, fmt.Errorf("failed to unmarshal wake_up response: %w", err)
	}
	return wakeUpResp.Success, wakeUpResp.Delayed, nil
}

func wakeUpServiceEndpoint(model string, kind int, queueLen int) (string, error) {
	base := strings.TrimSpace(os.Getenv(wakeUpServiceURLEnv))
	if base == "" {
		return "", fmt.Errorf("%s must be set for gateway wake_up requests", wakeUpServiceURLEnv)
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("invalid %s: %w", wakeUpServiceURLEnv, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("%s must include scheme and host", wakeUpServiceURLEnv)
	}

	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/wake_up"
	query := parsed.Query()
	query.Set("model_name", model)
	if kind == 4 {
		kind = 0
	}
	query.Set("kind", strconv.Itoa(kind))
	query.Set("queue_len", strconv.Itoa(queueLen))
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
