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
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
)

// TRE-PATCH(P3-GW-008): pod observer for the gateway's route-generation acks.
//
// The TRE gateway acknowledges every routable-label / route-gen change it has applied
// (Redis tre:v2:gw:seen:<pod>). An ack is only truthful once the change is visible to
// routing, i.e. AFTER the Store has swapped in the new pod object. The informer handlers
// therefore call the observer after they release c.mu, and only when the pod was actually
// (re)stored or removed. The observer runs on the informer goroutine and must not block.

// TREPodObserver receives a pod after the cache reflects it. newPod == nil means the pod
// was removed from the cache (oldPod carries at least name and namespace).
type TREPodObserver func(oldPod, newPod *v1.Pod)

var trePodObserver atomic.Pointer[TREPodObserver]

// SetTREPodObserver installs fn (nil clears it) and returns the previous observer.
func SetTREPodObserver(fn TREPodObserver) TREPodObserver {
	var prev *TREPodObserver
	if fn == nil {
		prev = trePodObserver.Swap(nil)
	} else {
		prev = trePodObserver.Swap(&fn)
	}
	if prev == nil {
		return nil
	}
	return *prev
}

func notifyTREPodObserver(oldPod, newPod *v1.Pod) {
	if fn := trePodObserver.Load(); fn != nil {
		(*fn)(oldPod, newPod)
	}
}

// PodLister is implemented by Store; it lets the gateway seed its route state from the
// cache contents without widening the Cache interface.
type PodLister interface {
	ListPods() []*v1.Pod
}

// UpdatePodForTest feeds a pod informer update event to the store (tests only, like
// InitWithPods).
func UpdatePodForTest(s *Store, oldPod, newPod *v1.Pod) {
	s.updatePod(oldPod, newPod)
}
