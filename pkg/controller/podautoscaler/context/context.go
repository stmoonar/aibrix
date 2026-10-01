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

package context

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
	"k8s.io/klog/v2"
)

// ScalingContext defines the generalized context that holds all necessary data for scaling calculations.
// This is the single source of truth for all scaling configuration, extracted per-PodAutoscaler.
type ScalingContext interface {
	GetTargetValueForMetric(metricName string) (float64, bool)
	GetUpFluctuationTolerance() float64
	GetDownFluctuationTolerance() float64
	GetMaxScaleUpRate() float64
	GetMaxScaleDownRate() float64
	GetCurrentUsePerPod() float64
	SetCurrentUsePerPod(float64)
	UpdateByPaTypes(pa *autoscalingv1alpha1.PodAutoscaler) error
	GetMinReplicas() int32
	GetMaxReplicas() int32
	GetStableValue() float64
	SetStableValue(float64)
	GetPanicValue() float64
	SetPanicValue(float64)
	SetActivationScale(value int32)
	GetActivationScale() int32
	GetPanicThreshold() float64
	GetInPanicMode() bool
	SetInPanicMode(bool)
	GetMaxPanicPods() int32
	SetMaxPanicPods(int32)
	GetScaleUpCooldownWindow() time.Duration
	GetScaleDownCooldownWindow() time.Duration
	GetScaleToZero() bool
	GetStableWindow() time.Duration
}

// baseScalingContext provides a base implementation of the ScalingContext interface.
type baseScalingContext struct {
	// Maximum rate at which to scale up
	MaxScaleUpRate float64
	// Maximum rate at which to scale down, a value of 2.5 means the count can reduce to at most 2.5 times less than the current value in one step.
	MaxScaleDownRate float64
	// The current use per pod.
	currentUsePerPod float64
	// The minimum number of replicas to which the target can be scaled down.
	MinReplicas int32
	// The maximum number of replicas to which the target can be scaled up
	MaxReplicas int32
	// Tolerance for fluctuations in metrics before scaling up
	UpFluctuationTolerance float64
	// Tolerance for fluctuations in metrics before scaling down
	DownFluctuationTolerance float64
	// Cooldown window for scale-up decisions (prevents rapid scale-ups)
	ScaleUpCooldownWindow time.Duration
	// Cooldown window for scale-down decisions (prevents rapid scale-downs)
	ScaleDownCooldownWindow time.Duration
	// Scale to zero flag
	ScaleToZero bool
	// Length of the stable metric window the recommendation is averaged over
	StableWindow time.Duration
	// Panic threshold for KPA
	PanicThreshold float64

	// Stable and Panic values
	StableValue float64
	PanicValue  float64

	// Panic mode state
	InPanicMode bool
	// MetricTargets used to store multiple metrics
	MetricTargets map[string]MetricTarget // key: metric name (e.g., "cpu", "gpu_cache_usage_perc")
}

type MetricTarget struct {
	// The value of scaling metric per pod that we target to maintain.
	TargetValue float64
	// The total value of scaling metric that a pod can maintain.
	TotalValue float64
	// The metric used for scaling, i.e. CPU, Memory, QPS.
	ScalingMetric string
	MetricType    autoscalingv1alpha1.MetricSourceType
}

var _ ScalingContext = (*baseScalingContext)(nil)

// NewBaseScalingContext creates a new instance of BaseScalingContext with default values.
func NewBaseScalingContext() *baseScalingContext {
	return &baseScalingContext{
		MaxScaleUpRate:           2,                 // Scale up rate of 200%, allowing rapid scaling
		MaxScaleDownRate:         2,                 // Scale down rate of 50%, for more gradual reduction
		UpFluctuationTolerance:   0.1,               // Default 10% tolerance for scale-up
		DownFluctuationTolerance: 0.1,               // Default 10% tolerance for scale-down
		ScaleUpCooldownWindow:    0 * time.Second,   // Default: no cooldown for scale-up
		ScaleDownCooldownWindow:  300 * time.Second, // Default: 5 minutes cooldown for scale-down
		ScaleToZero:              false,             // Default: do not scale to zero
		PanicThreshold:           2.0,               // Default panic threshold for KPA
		MetricTargets:            make(map[string]MetricTarget),
		StableWindow:             types.DefaultStableWindowDuration, // overridden by apa.autoscaling.aibrix.ai/window
	}
}

// annotationParser is a function type for parsing annotation values
type annotationParser func(b *baseScalingContext, value string) error

// annotationParsers maps annotation keys to their parsing functions
var annotationParsers = map[string]annotationParser{
	types.MaxScaleUpRateLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseFloat(value, 64)
		if err == nil {
			b.MaxScaleUpRate = v
		}
		return err
	},
	types.MaxScaleDownRateLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseFloat(value, 64)
		if err == nil {
			b.MaxScaleDownRate = v
		}
		return err
	},
	types.ScaleUpToleranceLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseFloat(value, 64)
		if err == nil {
			b.UpFluctuationTolerance = v
		}
		return err
	},
	types.ScaleDownToleranceLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseFloat(value, 64)
		if err == nil {
			b.DownFluctuationTolerance = v
		}
		return err
	},
	types.PanicThresholdLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseFloat(value, 64)
		if err == nil {
			b.PanicThreshold = v
		}
		return err
	},
	types.ScaleUpCooldownWindowLabel: func(b *baseScalingContext, value string) error {
		v, err := time.ParseDuration(value)
		if err == nil {
			b.ScaleUpCooldownWindow = v
		}
		return err
	},
	types.ScaleDownCooldownWindowLabel: func(b *baseScalingContext, value string) error {
		v, err := time.ParseDuration(value)
		if err == nil {
			b.ScaleDownCooldownWindow = v
		}
		return err
	},
	types.APAWindowLabel: func(b *baseScalingContext, value string) error {
		v, err := time.ParseDuration(value)
		if err == nil && v < types.MinStableWindowDuration {
			err = fmt.Errorf("%s must be at least %s, got %q", types.APAWindowLabel,
				types.MinStableWindowDuration, value)
		}
		if err == nil {
			b.StableWindow = v
		}
		return err
	},
	types.ScaleToZeroLabel: func(b *baseScalingContext, value string) error {
		v, err := strconv.ParseBool(value)
		if err == nil {
			b.ScaleToZero = v
		}
		return err
	},
}

// UpdateByPaTypes should be invoked in any scaling context that embeds BaseScalingContext.
func (b *baseScalingContext) UpdateByPaTypes(pa *autoscalingv1alpha1.PodAutoscaler) error {
	for _, ms := range pa.Spec.MetricsSources {
		targetValue, err := strconv.ParseFloat(ms.TargetValue, 64)
		if err != nil {
			klog.ErrorS(err, "Failed to parse target value", "metric", ms.TargetMetric, "value", ms.TargetValue)
			return fmt.Errorf("invalid targetValue for metric %q: %w", ms.TargetMetric, err)
		}
		if targetValue <= 0 {
			return fmt.Errorf("targetValue for metric %q must be greater than 0, got %f", ms.TargetMetric, targetValue)
		}

		b.MetricTargets[ms.TargetMetric] = MetricTarget{
			TargetValue:   targetValue,
			TotalValue:    100.0,
			ScalingMetric: ms.TargetMetric,
			MetricType:    ms.MetricSourceType,
		}
	}

	// Parse annotations using registered parsers
	for key, value := range pa.Annotations {
		if parser, ok := annotationParsers[key]; ok {
			if err := parser(b, value); err != nil {
				return err
			}
		} else if isAutoscalingAnnotation(key) {
			// A misspelled or renamed key is otherwise silently ignored and the default applies.
			warnOnce(pa, key, "Ignoring unrecognized autoscaling annotation; the default value applies", value)
		}
	}
	// The window annotation is an APA key; other strategies keep the default stable window.
	if _, ok := pa.Annotations[types.APAWindowLabel]; ok && pa.Spec.ScalingStrategy != autoscalingv1alpha1.APA {
		warnOnce(pa, types.APAWindowLabel, "Ignoring APA window annotation on a non-APA PodAutoscaler",
			pa.Annotations[types.APAWindowLabel])
		b.StableWindow = types.DefaultStableWindowDuration
	}
	return nil
}

// warnedAnnotations remembers (PodAutoscaler UID or namespace/name, annotation key, value)
// already reported, so each ignored annotation is logged once per PodAutoscaler instead of
// on every reconcile.
var warnedAnnotations sync.Map

func warnOnce(pa *autoscalingv1alpha1.PodAutoscaler, key, msg, value string) {
	id := string(pa.UID)
	if id == "" {
		id = pa.Namespace + "/" + pa.Name
	}
	if _, seen := warnedAnnotations.LoadOrStore(id+"|"+key+"="+value, struct{}{}); seen {
		return
	}
	klog.InfoS(msg, "podAutoscaler", klog.KObj(pa), "annotation", key, "value", value)
}

// nonScalingContextAnnotations are autoscaling-prefixed keys consumed outside the scaling context.
var nonScalingContextAnnotations = map[string]bool{
	types.AutoscalingLabelPrefix + "storm-service-mode": true,
}

func isAutoscalingAnnotation(key string) bool {
	if nonScalingContextAnnotations[key] {
		return false
	}
	for _, prefix := range []string{types.AutoscalingLabelPrefix, "apa." + types.AutoscalingLabelPrefix,
		"kpa." + types.AutoscalingLabelPrefix} {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}

func (b *baseScalingContext) SetCurrentUsePerPod(value float64) {
	b.currentUsePerPod = value
}

func (b *baseScalingContext) SetMinReplicas(minReplicas int32) {
	b.MinReplicas = minReplicas
}

func (b *baseScalingContext) SetMaxReplicas(maxReplicas int32) {
	b.MaxReplicas = maxReplicas
}

func (b *baseScalingContext) GetUpFluctuationTolerance() float64 {
	return b.UpFluctuationTolerance
}

func (b *baseScalingContext) GetDownFluctuationTolerance() float64 {
	return b.DownFluctuationTolerance
}

func (b *baseScalingContext) GetMaxScaleUpRate() float64 {
	return b.MaxScaleUpRate
}

func (b *baseScalingContext) GetMaxScaleDownRate() float64 {
	return b.MaxScaleDownRate
}

func (b *baseScalingContext) GetCurrentUsePerPod() float64 {
	return b.currentUsePerPod
}

func (b *baseScalingContext) GetTargetValueForMetric(metricName string) (float64, bool) {
	if target, ok := b.MetricTargets[metricName]; ok {
		return target.TargetValue, true
	}
	return 0, false
}

func (b *baseScalingContext) GetScalingTolerance() (up float64, down float64) {
	return b.MaxScaleUpRate, b.MaxScaleDownRate
}

func (b *baseScalingContext) GetMinReplicas() int32 {
	return b.MinReplicas
}

func (b *baseScalingContext) GetMaxReplicas() int32 {
	return b.MaxReplicas
}

func (b *baseScalingContext) GetStableValue() float64 {
	return b.StableValue
}

func (b *baseScalingContext) SetStableValue(value float64) {
	b.StableValue = value
}

func (b *baseScalingContext) GetPanicValue() float64 {
	return b.PanicValue
}

func (b *baseScalingContext) SetPanicValue(value float64) {
	b.PanicValue = value
}

func (b *baseScalingContext) GetActivationScale() int32 {
	return 1
}

func (b *baseScalingContext) SetActivationScale(value int32) {
	// No-op in base implementation
}

func (b *baseScalingContext) GetPanicThreshold() float64 {
	return b.PanicThreshold
}

func (b *baseScalingContext) GetInPanicMode() bool {
	return b.InPanicMode
}

func (b *baseScalingContext) SetInPanicMode(inPanic bool) {
	b.InPanicMode = inPanic
}

func (b *baseScalingContext) GetMaxPanicPods() int32 {
	return 0
}

func (b *baseScalingContext) SetMaxPanicPods(pods int32) {
	// No-op in base implementation
}

func (b *baseScalingContext) GetScaleUpCooldownWindow() time.Duration {
	return b.ScaleUpCooldownWindow
}

func (b *baseScalingContext) GetScaleDownCooldownWindow() time.Duration {
	return b.ScaleDownCooldownWindow
}

func (b *baseScalingContext) GetScaleToZero() bool {
	return b.ScaleToZero
}

func (b *baseScalingContext) GetStableWindow() time.Duration {
	return b.StableWindow
}
