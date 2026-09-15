import torch
from torchvision.transforms import v2


def get_train_transforms(
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    rotation_p: float = 1.0,
    rotation_degrees: tuple[float, float] = (0, 360),
):
    """Return data augmentation transforms for training."""

    return v2.Compose([
        v2.RandomHorizontalFlip(
            p=horizontal_flip_p,
        ),
        v2.RandomVerticalFlip(
            p=vertical_flip_p,
        ),
        v2.RandomApply(
            [
                v2.RandomRotation(
                    degrees=rotation_degrees,
                    expand=False,
                )
            ],
            p=rotation_p,
        ),
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),
    ])