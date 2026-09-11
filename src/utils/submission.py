import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pathlib import Path


def masks_to_submission(
    labeled_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Convert labeled instance masks to Kaggle submission format."""
    records = []

    for filename, labeled_mask in labeled_masks.items():

        image_id = filename.rsplit(".", 1)[0]

        component_ids = np.unique(labeled_mask)
        component_ids = component_ids[component_ids != 0]

        for filament_number, component_id in enumerate(
            component_ids,
            start=1,
        ):
            component_mask = (
                labeled_mask == component_id
            ).astype(np.uint8)

            rle = mask_utils.encode(
                np.asfortranarray(component_mask)
            )

            rle_counts = rle["counts"].decode("utf-8")

            records.append(
                {
                    "filament_id": (
                        f"{image_id}_{filament_number}"
                    ),
                    "segmentation_rle": rle_counts,
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "filament_id",
            "segmentation_rle",
        ],
    )


def save_submission(
    submission: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a submission DataFrame as a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        output_path,
        index=False,
    )