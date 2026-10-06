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
	"time"
)

// The first tick after the alignment delay must land on a wall-clock
// boundary (within maxOffset), whatever the start phase is.
func TestTickerAlignmentDelayLandsOnBoundary(t *testing.T) {
	interval := RequestTraceWriteInterval
	maxOffset := MaxRequestTraceIntervalOffset
	base := time.Unix(1791295340, 0) // a multiple of 10 s
	for _, off := range []time.Duration{
		0, 300 * time.Millisecond, maxOffset, maxOffset + time.Millisecond,
		2 * time.Second, 8828 * time.Millisecond, interval - time.Nanosecond,
	} {
		now := base.Add(off)
		d := tickerAlignmentDelay(now, interval, maxOffset)
		if d < 0 || d >= interval {
			t.Fatalf("offset %v: delay %v out of [0, interval)", off, d)
		}
		firstTick := now.Add(d).Add(interval)
		phase := time.Duration(firstTick.UnixNano()) % interval
		if phase > maxOffset {
			t.Fatalf("offset %v: first tick phase %v > %v", off, phase, maxOffset)
		}
	}
}
