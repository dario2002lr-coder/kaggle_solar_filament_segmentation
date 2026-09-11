from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for binary image segmentation.

    Parameters
    ----------
    masks:
        Dictionary mapping image filenames to their individual
        binary masks.

    images_dir:
        Directory containing the corresponding images.
    """

    def __init__(
        self,
        masks: dict[str, list[np.ndarray]],
        images_dir: Path,
    ):
        self.masks = masks
        self.images_dir = Path(images_dir)
        self.filenames = list(masks.keys())

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image_path = self.images_dir / filename

        image = np.array(
            Image.open(image_path),
            dtype=np.float32,
        )

        # Normalize grayscale image to [0, 1]
        image /= 255.0

        # Combine all filament masks into one binary mask
        individual_masks = self.masks[filename]

        if individual_masks:
            mask = np.any(
                np.stack(individual_masks),
                axis=0,
            ).astype(np.float32)
        else:
            mask = np.zeros_like(
                image,
                dtype=np.float32,
            )

        # H x W -> 1 x H x W
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class InferenceDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
    ):
        self.images_dir = Path(images_dir)
        self.filenames = sorted(
            path.name
            for path in self.images_dir.glob("*.jpeg")
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = self.images_dir / filename

        image = np.array(
            Image.open(image_path),
            dtype=np.float32,
        )

        image /= 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        return image, filename