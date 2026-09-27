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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type treObservation struct {
	oldPod, newPod *v1.Pod
	cached         *v1.Pod // what the store served when the observer ran
	lockFree       bool    // c.mu was not held when the observer ran
}

func observeTREPods(t *testing.T, s *Store) *[]treObservation {
	t.Helper()
	var got []treObservation
	prev := SetTREPodObserver(func(oldPod, newPod *v1.Pod) {
		obs := treObservation{oldPod: oldPod, newPod: newPod}
		name, ns := "", ""
		if newPod != nil {
			name, ns = newPod.Name, newPod.Namespace
		} else if oldPod != nil {
			name, ns = oldPod.Name, oldPod.Namespace
		}
		obs.cached, _ = s.GetPod(name, ns)
		if s.mu.TryLock() {
			obs.lockFree = true
			s.mu.Unlock()
		}
		got = append(got, obs)
	})
	t.Cleanup(func() { SetTREPodObserver(prev) })
	return &got
}

func treObserverPod(routable, gen string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "p", Namespace: "default",
			Labels:      map[string]string{modelIdentifier: "m", "tre.aibrix.io/routable": routable},
			Annotations: map[string]string{"tre.aibrix.io/route-gen": gen},
		},
		Status: v1.PodStatus{PodIP: "10.0.0.1"},
	}
}

// TRE-PATCH(P3-GW-008): the observer must only see a pod once the Store serves it, and
// must run outside the store lock.
func TestTREPodObserver_RunsAfterStoreReflectsChange(t *testing.T) {
	s := NewForTest()
	got := observeTREPods(t, s)

	p1 := treObserverPod("true", "1")
	s.addPod(p1)
	p2 := treObserverPod("false", "2")
	s.updatePod(p1, p2)
	s.deletePod(p2)

	require.Len(t, *got, 3)
	add, upd, del := (*got)[0], (*got)[1], (*got)[2]
	assert.Same(t, p1, add.newPod)
	assert.Same(t, p1, add.cached, "add: observer sees the stored pod")
	assert.Same(t, p2, upd.newPod)
	assert.Same(t, p2, upd.cached, "update: observer runs after the new object is stored")
	assert.Equal(t, "false", upd.cached.Labels["tre.aibrix.io/routable"])
	assert.Nil(t, del.newPod)
	assert.Nil(t, del.cached, "delete: observer runs after removal")
	for i, o := range *got {
		assert.True(t, o.lockFree, "observer %d ran under the store lock", i)
	}
}

func TestTREPodObserver_NotCalledForIgnoredPods(t *testing.T) {
	s := NewForTest()
	got := observeTREPods(t, s)
	s.addPod(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "no-model", Namespace: "default"}})
	assert.Empty(t, *got)
}
