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
package cache

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/utils"
	atomic_ext "go.uber.org/atomic"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

type Pod struct {
	*v1.Pod
	Models       *utils.Registry[string]                    // Model/adapter names that the pod is running
	Metrics      utils.SyncMap[string, metrics.MetricValue] // Pod metrics (metric_name -> value)
	ModelMetrics utils.SyncMap[string, metrics.MetricValue] // Pod-model metrics (model_name/metric_name -> value)

	// Realtime statstistic
	runningRequests        int32 // Realtime running requests counter.
	completedRequests      int64 // Monotonically increasing count of finished requests (gateway-tracked).
	pendingLoadUtilization atomic_ext.Float64

	// Log frenquency control
	lastTraceLogTimestamp int64

	// TRE (review round 3, L4): scrapeMu makes a scrape's stored metric values and its
	// stamp one unit. lastFetchStartMS (the fetch start of the stored scrape) orders
	// overlapping scrapes of one pod: an older fetch that finishes last is dropped, so
	// old values never pair with a newer stamp. lastScrapeSuccessMS is the gateway
	// wall clock (treWallClockMS) at which the stored values were written; 0 = never.
	scrapeMu            sync.Mutex
	lastFetchStartMS    int64
	lastScrapeSuccessMS int64
}

// storeScrape stores a successful scrape that started at fetchStartMS: store() writes
// its values into the pod cache and the stamp is taken at that moment, both under the
// pod's scrape lock. A scrape that started before the stored one is dropped (false).
func (pod *Pod) storeScrape(fetchStartMS int64, store func()) bool {
	pod.scrapeMu.Lock()
	defer pod.scrapeMu.Unlock()
	if fetchStartMS < pod.lastFetchStartMS {
		return false
	}
	store()
	pod.lastFetchStartMS = fetchStartMS
	if now := treWallClockMS(); now > pod.lastScrapeSuccessMS {
		pod.lastScrapeSuccessMS = now
	}
	return true
}

// readScrape calls read with the stored scrape's stamp (ok = false if the pod has never
// been scraped successfully) under the scrape lock, so the values read inside belong
// to that stamp.
func (pod *Pod) readScrape(read func(scrapedMS int64, ok bool)) {
	pod.scrapeMu.Lock()
	defer pod.scrapeMu.Unlock()
	read(pod.lastScrapeSuccessMS, pod.lastScrapeSuccessMS > 0)
}

func (pod *Pod) CanLogPodTrace(level klog.Level) bool {
	// Skip log throttling for greater log level.
	if klog.V(level).Enabled() {
		return true
	}

	lastTs := atomic.LoadInt64(&pod.lastTraceLogTimestamp)
	ts := time.Now().UnixNano()
	if ts-lastTs < int64(traceLogInterval) {
		return false
	}

	// Ensure only one attempt pass.
	return atomic.CompareAndSwapInt64(&pod.lastTraceLogTimestamp, lastTs, ts)
}
