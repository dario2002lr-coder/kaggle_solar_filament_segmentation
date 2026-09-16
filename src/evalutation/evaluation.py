import pandas as pd


def evaluate_min_area(
    predictions_df: pd.DataFrame,
    min_area: int,
) -> dict:
    """
    Evaluate detection and segmentation quality after filtering
    predictions below a minimum component area.
    """

    # Keep only predicted instances.
    predictions = predictions_df[
        predictions_df["pred_id"].notna()
    ]

    # Keep predictions whose area is above the threshold.
    kept = predictions[
        predictions["area"] >= min_area
    ]

    # True positives among the kept predictions.
    tp = (
        kept["match_type"] == "TP"
    ).sum()

    # False positives among the kept predictions.
    fp = (
        kept["match_type"] != "TP"
    ).sum()

    # True positives removed by the area filter.
    removed_tp = (
        (predictions["match_type"] == "TP")
        & (predictions["area"] < min_area)
    ).sum()

    # False negatives already present before filtering.
    base_fn = (
        predictions_df["match_type"] == "FN"
    ).sum()

    # Removing a true positive turns it into a false negative.
    fn = base_fn + removed_tp

    # Segmentation quality: mean IoU of retained true positives.
    tp_ious = kept.loc[
        kept["match_type"] == "TP",
        "iou",
    ]

    sq = tp_ious.mean() if len(tp_ious) > 0 else 0.0

    # Recognition quality.
    denominator = (
        tp
        + 0.5 * fp
        + 0.5 * fn
    )

    rq = tp / denominator if denominator > 0 else 0.0

    # Panoptic quality.
    pq = sq * rq

    return {
        "min_area": min_area,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SQ": sq,
        "RQ": rq,
        "PQ": pq,
    }


def find_best_min_area(
    predictions_df: pd.DataFrame,
    min_area_values: list[int],
) -> pd.DataFrame:
    """
    Evaluate multiple minimum-area thresholds and rank them by PQ.
    """

    results = [
        evaluate_min_area(
            predictions_df=predictions_df,
            min_area=min_area,
        )
        for min_area in min_area_values
    ]

    return (
        pd.DataFrame(results)
        .sort_values(
            "PQ",
            ascending=False,
        )
        .reset_index(drop=True)
    )