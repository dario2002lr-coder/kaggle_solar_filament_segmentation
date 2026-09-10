from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import dice_score, iou_score


def _run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Run one complete training epoch.
    """
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, targets in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        total_dice += dice_score(
            logits=logits.detach(),
            targets=targets,
            threshold=threshold,
        )

        total_iou += iou_score(
            logits=logits.detach(),
            targets=targets,
            threshold=threshold,
        )

    num_batches = len(dataloader)

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


@torch.no_grad()
def _run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Run one complete validation epoch.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, targets in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = loss_fn(logits, targets)

        total_loss += loss.item()

        total_dice += dice_score(
            logits=logits,
            targets=targets,
            threshold=threshold,
        )

        total_iou += iou_score(
            logits=logits,
            targets=targets,
            threshold=threshold,
        )

    num_batches = len(dataloader)

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 7,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    threshold: float = 0.5,
    device: str | torch.device = "cuda",
    use_amp: bool = True,
    checkpoint_path: str | Path = (
        "artifacts/models/best_model.pth"
    ),
) -> dict[str, list[float]]:
    """
    Train a segmentation model with validation,
    learning-rate scheduling and early stopping.
    """
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    use_amp = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "learning_rate": [],
    }

    best_val_dice = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        train_loss, train_dice, train_iou = (
            _run_training_epoch(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                threshold=threshold,
            )
        )

        val_loss, val_dice, val_iou = (
            _run_validation_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                threshold=threshold,
            )
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)

        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_dice > best_val_dice:

            best_val_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                checkpoint_path,
            )

            print(
                f"  -> Best model saved "
                f"(Val Dice: {val_dice:.4f})"
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Val Dice: {best_val_dice:.4f}"
            )
            break

    return history