from pathlib import Path
import json


def get_train_images_dir(raw_data_dir: Path) -> Path:
    return raw_data_dir / "train" / "train_images"


def get_test_images_dir(raw_data_dir: Path) -> Path:
    return raw_data_dir / "test" / "test_images"


def get_annotations_path(raw_data_dir: Path) -> Path:
    return raw_data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def load_annotations(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_duplicate_annotations(annotations: dict) -> dict:
    """
    Remove duplicate image records and their associated annotations.

    Images are considered duplicates when they share the same filename.
    The first occurrence of each filename is kept.

    Parameters
    ----------
    annotations:
        Loaded annotation dictionary.

    Returns
    -------
    dict
        Annotation dictionary with duplicate images removed.
    """

    unique_images = {}
    duplicate_image_ids = set()

    for image in annotations["images"]:
        file_name = image["file_name"]

        if file_name in unique_images:
            duplicate_image_ids.add(image["id"])
        else:
            unique_images[file_name] = image

    unique_image_ids = {
        image["id"]
        for image in unique_images.values()
    }

    annotations_clean = annotations.copy()

    annotations_clean["images"] = list(unique_images.values())

    annotations_clean["annotations"] = [
        annotation
        for annotation in annotations["annotations"]
        if annotation["image_id"] in unique_image_ids
    ]

    return annotations_clean