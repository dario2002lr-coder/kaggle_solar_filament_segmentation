from pathlib import Path

import torch
import torch.nn as nn


def load_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load model weights from a training checkpoint.

    Parameters
    ----------
    model:
        Model instance with the same architecture used during training.
    checkpoint_path:
        Path to the saved checkpoint.
    device:
        Device used for inference.

    Returns
    -------
    nn.Module
        Loaded model in evaluation mode.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model