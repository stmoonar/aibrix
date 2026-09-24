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
	"bytes"
	"sync"
)

// TRE-PATCH(P2-GW-006): reassemble SSE lines split across ext_proc body chunks.
//
// With response body mode Streamed, Envoy forwards the upstream body in whatever pieces
// its socket reads produced, so one ext_proc message can end in the middle of an SSE
// `data:` line. vLLM puts `"usage":null` in every chunk, so the streaming usage scan
// (HandleResponseBody) runs on every message; a truncated `data:` line failed its JSON
// validity check and the plugin answered 500 mid-stream. v1's openai-go SSE decoder
// never saw such a line as a complete event, so v1 did not have this failure.
//
// Only complete lines ('\n'-terminated) are scanned; the unterminated tail is held per
// request and prepended to the next message, and flushed on end-of-stream. What Envoy
// forwards to the client is untouched - this only changes what the plugin parses.
var sseCarry sync.Map // requestID -> []byte

// sseCompleteLines returns the bytes of chunk (prefixed with any held tail) that end at
// the last newline, holding the rest for the next call. At end of stream everything is
// returned and nothing is held.
func sseCompleteLines(requestID string, chunk []byte, endOfStream bool) []byte {
	data := chunk
	if prev, ok := sseCarry.LoadAndDelete(requestID); ok {
		held := prev.([]byte)
		data = make([]byte, 0, len(held)+len(chunk))
		data = append(append(data, held...), chunk...)
	}
	if endOfStream {
		return data
	}
	idx := bytes.LastIndexByte(data, '\n')
	if idx < len(data)-1 {
		tail := make([]byte, len(data)-idx-1)
		copy(tail, data[idx+1:])
		sseCarry.Store(requestID, tail)
	}
	if idx < 0 {
		return nil
	}
	return data[:idx+1]
}

// clearSSECarry drops any held tail for a finished or aborted request.
func clearSSECarry(requestID string) {
	sseCarry.Delete(requestID)
}
