from __future__ import annotations

from typing import Iterable

import numpy as np


def _serialize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.uint8)
    return array > 0


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = _serialize_mask(mask_a)
    mask_b = _serialize_mask(mask_b)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def _greedy_one_to_one_matches(
    iou_matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    if iou_matrix.size == 0:
        return matches

    ranked = sorted(
        [
            (float(iou_matrix[pred_idx, gt_idx]), pred_idx, gt_idx)
            for pred_idx in range(iou_matrix.shape[0])
            for gt_idx in range(iou_matrix.shape[1])
            if iou_matrix[pred_idx, gt_idx] >= threshold
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    used_pred = set()
    used_gt = set()

    for iou, pred_idx, gt_idx in ranked:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, float(iou)))

    return matches


def compute_instance_pq(
    predicted_instances: Iterable[np.ndarray],
    gt_instances: Iterable[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute TP, FP, FN, SQ, RQ and PQ for instance masks using 1:1 matching."""
    predicted_instances = [np.asarray(mask) for mask in predicted_instances]
    gt_instances = [np.asarray(mask) for mask in gt_instances]

    if not predicted_instances and not gt_instances:
        return {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "SQ": 0.0,
            "RQ": 0.0,
            "PQ": 0.0,
        }

    iou_matrix = np.zeros((len(predicted_instances), len(gt_instances)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(predicted_instances):
        for gt_idx, gt_mask in enumerate(gt_instances):
            iou_matrix[pred_idx, gt_idx] = _compute_iou(pred_mask, gt_mask)

    matches = _greedy_one_to_one_matches(iou_matrix, iou_threshold)
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}

    tp = len(matches)
    fp = len(predicted_instances) - tp
    fn = len(gt_instances) - tp

    if tp > 0:
        sq = float(np.mean([iou for _, _, iou in matches]))
    else:
        sq = 0.0

    denominator = tp + 0.5 * fp + 0.5 * fn
    rq = float(tp / denominator) if denominator > 0 else 0.0
    pq = sq * rq

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "SQ": float(sq),
        "RQ": float(rq),
        "PQ": float(pq),
    }
