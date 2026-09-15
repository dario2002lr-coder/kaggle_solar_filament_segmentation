import torch


def predict_masks(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Generate binary segmentation masks from input images.

    Parameters
    ----------
    model:
        Trained segmentation model.
    images:
        Batch of images with shape (B, C, H, W).
    device:
        Device used for inference.
    threshold:
        Probability threshold used to obtain binary masks.

    Returns
    -------
    torch.Tensor
        Binary predicted masks with shape (B, 1, H, W).
    """
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= threshold

    return predictions