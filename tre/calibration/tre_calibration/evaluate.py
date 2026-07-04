from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from tre_calibration.dataset import CalibrationWindow


@dataclass(frozen=True)
class ThresholdEvaluation:
    auroc: float
    spearman_health: float
    balanced_accuracy: float
    true_healthy: int
    false_healthy: int
    true_violation: int
    false_violation: int


def evaluate_threshold(windows: Iterable[CalibrationWindow], *, theta: float) -> ThresholdEvaluation:
    rows = [row for row in windows if math.isfinite(row.signal)]
    scores = [row.signal for row in rows]
    healthy_labels = [1 if row.slo_met else 0 for row in rows]
    health_scores = [
        row.health_score if row.health_score is not None else (1.0 if row.slo_met else 0.0)
        for row in rows
    ]

    true_healthy = false_healthy = true_violation = false_violation = 0
    for row in rows:
        pred_healthy = row.signal >= theta
        if row.slo_met and pred_healthy:
            true_healthy += 1
        elif row.slo_met:
            false_violation += 1
        elif pred_healthy:
            false_healthy += 1
        else:
            true_violation += 1

    healthy_recall = true_healthy / (true_healthy + false_violation) if true_healthy + false_violation else 0.0
    violation_recall = true_violation / (true_violation + false_healthy) if true_violation + false_healthy else 0.0

    return ThresholdEvaluation(
        auroc=_auc(scores, healthy_labels),
        spearman_health=_spearman(scores, health_scores),
        balanced_accuracy=0.5 * (healthy_recall + violation_recall),
        true_healthy=true_healthy,
        false_healthy=false_healthy,
        true_violation=true_violation,
        false_violation=false_violation,
    )


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return 0.5

    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(values_x, values_y) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0

    ranks_x = _rankdata([x for x, _ in pairs])
    ranks_y = _rankdata([y for _, y in pairs])
    return _pearson(ranks_x, ranks_y)


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    den_x = sum((x - mean_x) ** 2 for x in values_x)
    den_y = sum((y - mean_y) ** 2 for y in values_y)
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)
