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
	"context"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// A vLLM /v1/completions stream as it comes off the wire: every event carries
// "usage":null, the last one before [DONE] carries the real usage.
const vllmStream = `data: {"id":"cmpl-1","object":"text_completion","model":"dsqwen-7b","choices":[{"index":0,"text":"a","finish_reason":null}],"usage":null}

data: {"id":"cmpl-1","object":"text_completion","model":"dsqwen-7b","choices":[{"index":0,"text":"b","finish_reason":"length"}],"usage":null}

data: {"id":"cmpl-1","object":"text_completion","model":"dsqwen-7b","choices":[],"usage":{"prompt_tokens":12,"total_tokens":15,"completion_tokens":3}}

data: [DONE]

`

func bodyMsg(b string, eos bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
		ResponseBody: &extProcPb.HttpBody{Body: []byte(b), EndOfStream: eos},
	}}
}

// Feed the stream to HandleResponseBody cut into pieces of every size from 1 byte up:
// no cut may produce a mid-stream error, and the usage must always be recovered.
func TestTRESSEStreamSplitAtEveryOffset(t *testing.T) {
	for size := 1; size <= len(vllmStream); size += 7 {
		reqID := "sse-split"
		mc := &MockCache{Cache: cache.NewForTest()}
		mc.On("DoneRequestTrace", mock.Anything, reqID, "dsqwen-7b", int64(12), int64(3), int64(0)).Once()
		server := &Server{cache: mc}
		ctx := types.NewRoutingContext(context.Background(), "least-gpu-cache", "dsqwen-7b", "", reqID, "")
		ctx.ReqPath = PathCompletions
		ctx.RequestTime = time.Now()

		complete := false
		for off := 0; off < len(vllmStream); off += size {
			end := off + size
			if end > len(vllmStream) {
				end = len(vllmStream)
			}
			resp, done := server.HandleResponseBody(context.Background(), ctx, reqID,
				bodyMsg(vllmStream[off:end], end == len(vllmStream)), utils.User{}, 0, "dsqwen-7b", true, 0, complete)
			if !assert.Nil(t, resp.GetImmediateResponse(), "size %d offset %d: mid-stream error", size, off) {
				return
			}
			complete = done
		}
		assert.True(t, complete, "size %d", size)
		mc.AssertExpectations(t)
		_, held := sseCarry.Load(reqID)
		assert.False(t, held, "size %d: tail must not outlive the stream", size)
		clearSSECarry(reqID)
	}
}

func TestTRESSECompleteLines(t *testing.T) {
	defer clearSSECarry("r")
	assert.Nil(t, sseCompleteLines("r", []byte("data: {\"a\""), false))
	assert.Equal(t, "data: {\"a\":1}\n", string(sseCompleteLines("r", []byte(":1}\ndata: {"), false)))
	assert.Equal(t, "data: {\"b\":2}\n\n", string(sseCompleteLines("r", []byte("\"b\":2}\n\n"), false)))
	_, held := sseCarry.Load("r")
	assert.False(t, held)
	assert.Nil(t, sseCompleteLines("r", []byte("partial"), false))
	assert.Equal(t, "partial-end", string(sseCompleteLines("r", []byte("-end"), true)))
	_, held = sseCarry.Load("r")
	assert.False(t, held)
}

// A genuinely malformed but complete line is still reported, as upstream does.
func TestTRESSEMalformedCompleteLineStillErrors(t *testing.T) {
	mc := &MockCache{Cache: cache.NewForTest()}
	mc.On("DoneRequestTrace", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Maybe()
	server := &Server{cache: mc}
	ctx := types.NewRoutingContext(context.Background(), "least-gpu-cache", "m", "", "bad", "")
	ctx.ReqPath = PathCompletions
	ctx.RequestTime = time.Now()
	resp, complete := server.HandleResponseBody(context.Background(), ctx, "bad",
		bodyMsg("data: {\"usage\": { broken\n\n", false), utils.User{}, 0, "m", true, 0, false)
	assert.True(t, complete)
	assert.NotNil(t, resp.GetImmediateResponse())
	clearSSECarry("bad")
}
