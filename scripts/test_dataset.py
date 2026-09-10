import torch
from pathlib import Path

from src.datasets.segmentation import SegmentationDataset
from src.processing.masks import build_masks
from src.splitting.split import split_masks
from src.utils.config import load_config
from src.utils.data import (
    get_train_images_dir,
    load_annotations,
    remove_duplicate_annotations,
)


config = load_config()

raw_data_dir = Path(
    config["paths"]["raw_data_dir"]
)

annotations = load_annotations(
    raw_data_dir
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

annotations_clean = remove_duplicate_annotations(
    annotations
)

masks = build_masks(annotations_clean)

train_masks, val_masks, test_masks = split_masks(
    masks
)

train_images_dir = get_train_images_dir(
    raw_data_dir
)

dataset = SegmentationDataset(
    masks=train_masks,
    images_dir=train_images_dir,
)

image, mask = dataset[0]

print(f"Dataset size: {len(dataset)}")
print(f"Image shape:  {image.shape}")
print(f"Mask shape:   {mask.shape}")
print(f"Image dtype:  {image.dtype}")
print(f"Mask dtype:   {mask.dtype}")
print(f"Mask values:  {torch.unique(mask)}")