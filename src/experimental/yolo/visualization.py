from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_instance_comparison(
    image: np.ndarray,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    filename: str,
    output_dir: str | Path,
    title: str = "Instance comparison",
) -> Path:
    """Create a visual comparison between ground-truth and predicted instances."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    gt_overlay = np.zeros_like(image, dtype=np.float32)
    for idx, mask in enumerate(gt_masks):
        gt_overlay[mask > 0] = idx + 1
    axes[1].imshow(gt_overlay, cmap="tab20")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    pred_overlay = np.zeros_like(image, dtype=np.float32)
    for idx, mask in enumerate(pred_masks):
        pred_overlay[mask > 0] = idx + 1
    axes[2].imshow(pred_overlay, cmap="tab20")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.suptitle(f"{title} - {filename}")
    out_path = output_dir / f"{Path(filename).stem}_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path
