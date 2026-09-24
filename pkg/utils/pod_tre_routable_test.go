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

package utils

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TRE-PATCH(P2-GW-004) tests. Not parallel: they flip the process-wide gate.

func treRoutablePod(name, routable string, ready bool) *v1.Pod {
	labels := map[string]string{}
	if routable != "" {
		labels[TRERoutableLabel] = routable
	}
	status := v1.ConditionTrue
	if !ready {
		status = v1.ConditionFalse
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
		Status: v1.PodStatus{
			PodIP:      "10.0.0.1",
			Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: status}},
		},
	}
}

func setGate(t *testing.T, enabled bool) {
	t.Helper()
	prev := SetTRERoutableLabelFilter(enabled)
	t.Cleanup(func() { SetTRERoutableLabelFilter(prev) })
}

func treFixture() []*v1.Pod {
	terminating := treRoutablePod("terminating-routable", "true", true)
	terminating.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	return []*v1.Pod{
		treRoutablePod("awake", "true", true),
		treRoutablePod("sleeping", "false", true),
		treRoutablePod("unlabelled", "", true),
		treRoutablePod("weird-value", "True ", true),
		treRoutablePod("unready-routable", "true", false),
		terminating,
	}
}

func names(pods []*v1.Pod) []string {
	out := make([]string, 0, len(pods))
	for _, p := range pods {
		out = append(out, p.Name)
	}
	return out
}

func TestTRERoutableGateOffKeepsUpstreamBehaviour(t *testing.T) {
	setGate(t, false)
	pods := treFixture()
	assert.False(t, TRERoutableLabelFilterEnabled())
	assert.ElementsMatch(t, []string{"awake", "sleeping", "unlabelled", "weird-value"}, names(FilterRoutablePods(pods)))
	assert.Equal(t, 4, CountRoutablePods(pods))
}

func TestTRERoutableGateOnKeepsOnlyRoutableTrue(t *testing.T) {
	setGate(t, true)
	pods := treFixture()
	assert.True(t, TRERoutableLabelFilterEnabled())
	assert.Equal(t, []string{"awake"}, names(FilterRoutablePods(pods)))
	assert.Equal(t, 1, CountRoutablePods(pods))
	assert.Equal(t, []string{"awake"}, names(FilterRoutablePodsInPlace(treFixture())))
	for i := 0; i < 20; i++ {
		pod, err := SelectRandomPod(pods, func(n int) int { return i % n })
		assert.NoError(t, err)
		assert.Equal(t, "awake", pod.Name)
	}
}

// Scraping and the TRE redis writer rely on FilterReadyPod: the gate must not hide
// sleeping pods from them.
func TestTRERoutableGateDoesNotNarrowFilterReadyPod(t *testing.T) {
	setGate(t, true)
	assert.True(t, FilterReadyPod(treRoutablePod("sleeping", "false", true)))
	assert.False(t, FilterRoutingCandidatePod(treRoutablePod("sleeping", "false", true)))
	assert.True(t, FilterRoutingCandidatePod(treRoutablePod("awake", "true", true)))
	assert.False(t, FilterRoutingCandidatePod(treRoutablePod("unready", "true", false)))
}

func TestTRERoutableGateNoCandidates(t *testing.T) {
	setGate(t, true)
	pods := []*v1.Pod{treRoutablePod("sleeping", "false", true)}
	assert.Equal(t, 0, CountRoutablePods(pods))
	_, err := SelectRandomPod(pods, func(n int) int { return 0 })
	assert.Error(t, err)
}
