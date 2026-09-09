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