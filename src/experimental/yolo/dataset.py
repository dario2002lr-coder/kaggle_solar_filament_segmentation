from __future__ import annotations

from pathlib import Path

import numpy as np

from src.processing.masks import build_masks
from src.splitting.split import split_masks
from src.utils.data import get_train_images_dir, load_annotations, remove_duplicate_annotations


def _polygon_to_yolo_segment(
    polygon: list[float],
    width: int,
    height: int,
) -> list[float]:
    """Convert a COCO polygon into normalized YOLO segmentation coordinates."""
    if not polygon:
        return []

    coords = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)

    normalized = np.stack(
        [coords[:, 0] / width, coords[:, 1] / height],
        axis=1,
    ).reshape(-1)

    return normalized.astype(float).tolist()


def prepare_yolo_segmentation_dataset(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    class_name: str = "filament",
) -> dict[str, Path]:
    """
    Build a YOLO segmentation dataset from the original COCO annotations.

    The generated dataset remains inside the experimental folder and does not
    mutate the original dataset or the U-Net pipeline.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    annotations = load_annotations(
        raw_data_dir
        / "train"
        / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )

    annotations_clean = remove_duplicate_annotations(annotations)

    masks_by_filename = build_masks(annotations_clean)

    train_masks, val_masks, test_masks = split_masks(
        masks=masks_by_filename,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=random_state,
    )

    split_map = {
        "train": train_masks,
        "val": val_masks,
        "test": test_masks,
    }

    images_dir = get_train_images_dir(raw_data_dir)

    image_lookup = {
        image["id"]: image
        for image in annotations_clean["images"]
    }

    annotations_by_image = {}
    for annotation in annotations_clean["annotations"]:
        image_id = annotation["image_id"]
        annotations_by_image.setdefault(image_id, []).append(annotation)

    dataset_root = output_dir / "yolo_segmentation"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Ultralytics dataset configuration.
    data_yaml = """train: {train_dir}
val: {val_dir}

nc: 1
names:
  0: {class_name}
"""

    for split_name, split_masks_dict in split_map.items():
        split_dir = dataset_root / split_name
        images_split_dir = split_dir / "images"
        labels_split_dir = split_dir / "labels"

        images_split_dir.mkdir(parents=True, exist_ok=True)
        labels_split_dir.mkdir(parents=True, exist_ok=True)

        for filename in split_masks_dict:
            source_image = images_dir / filename
            destination_image = images_split_dir / filename

            if not destination_image.exists():
                destination_image.write_bytes(source_image.read_bytes())

            image_id = next(
                image["id"]
                for image in annotations_clean["images"]
                if image["file_name"] == filename
            )

            width = image_lookup[image_id]["width"]
            height = image_lookup[image_id]["height"]

            labels = []

            for annotation in annotations_by_image.get(image_id, []):
                segmentation = annotation.get("segmentation", [])

                if not segmentation:
                    continue

                for polygon in segmentation:
                    if not polygon:
                        continue

                    points = _polygon_to_yolo_segment(
                        polygon=polygon,
                        width=width,
                        height=height,
                    )

                    if not points:
                        continue

                    line = ["0"] + [f"{value:.6f}" for value in points]
                    labels.append(" ".join(line))

            label_path = labels_split_dir / f"{Path(filename).stem}.txt"

            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )

    dataset_yaml_path = dataset_root / "dataset.yaml"

    dataset_yaml_path.write_text(
        data_yaml.format(
            train_dir=str((dataset_root / "train" / "images").as_posix()),
            val_dir=str((dataset_root / "val" / "images").as_posix()),
            class_name=class_name,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_root": dataset_root,
        "train": dataset_root / "train",
        "val": dataset_root / "val",
        "test": dataset_root / "test",
        "yaml": dataset_yaml_path,
        "class_name": class_name,
    }
