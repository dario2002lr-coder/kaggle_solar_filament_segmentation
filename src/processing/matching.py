import numpy as np
import pandas as pd

from src.processing.masks import calculate_iou


def labeled_mask_to_instances(
    labeled_mask: np.ndarray,
    num_components: int,
) -> list[np.ndarray]:
    """
    Convert a labeled mask into individual binary instance masks.

    Parameters
    ----------
    labeled_mask:
        Labeled image where each connected component has a unique label.
    num_components:
        Number of connected components.

    Returns
    -------
    list[np.ndarray]
        One binary mask per connected component.
    """
    return [
        labeled_mask == component_id
        for component_id in range(1, num_components + 1)
    ]


def compare_instances(
    predicted_instances: list[np.ndarray],
    gt_instances: list[np.ndarray],
    match_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare predicted filament instances with ground-truth instances.

    For each predicted instance, the ground-truth instance with the
    highest IoU is selected as its best match. Ground-truth instances
    without a valid match are also included as false negatives.

    Parameters
    ----------
    predicted_instances:
        List of binary masks corresponding to predicted filaments.
    gt_instances:
        List of binary masks corresponding to ground-truth filaments.
    match_threshold:
        Minimum IoU required for a valid match.

    Returns
    -------
    pd.DataFrame
        Table containing prediction IDs, ground-truth IDs, IoU values,
        and match types.
    """
    records = []

    # Compare every prediction with every ground-truth instance.
    for pred_id, pred_mask in enumerate(
        predicted_instances,
        start=1,
    ):
        best_gt_id = None
        best_iou = 0.0

        for gt_id, gt_mask in enumerate(
            gt_instances,
            start=1,
        ):
            iou = calculate_iou(
                pred_mask,
                gt_mask,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= match_threshold:
            match_type = "TP"
        else:
            match_type = "FP"

        records.append(
            {
                "pred_id": pred_id,
                "gt_id": best_gt_id,
                "iou": best_iou,
                "match_type": match_type,
            }
        )

    # Ground-truth instances without a valid prediction.
    matched_gt_ids = {
        record["gt_id"]
        for record in records
        if record["match_type"] == "TP"
    }

    for gt_id in range(1, len(gt_instances) + 1):
        if gt_id not in matched_gt_ids:
            records.append(
                {
                    "pred_id": None,
                    "gt_id": gt_id,
                    "iou": 0.0,
                    "match_type": "FN",
                }
            )

    return pd.DataFrame(records)