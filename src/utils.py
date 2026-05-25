"""Plotting, ONNX export, and artifact persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def plot_training_curves(
    history: dict[str, list[float]], output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history["train_loss"], label="Train")
    ax_loss.plot(history["val_loss"], label="Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(history["train_acc"], label="Train")
    ax_acc.plot(history["val_acc"], label="Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves → {path}")


def save_history_csv(
    history: dict[str, list[float]], output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    path = output_dir / "training_history.csv"
    df.to_csv(path)
    print(f"Training history → {path}")


def save_config(config: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config snapshot → {path}")


def export_onnx(
    model: nn.Module,
    output_dir: Path,
    filename: str = "model.onnx",
    input_shape: tuple[int, ...] = (1, 32, 32, 3),
) -> Path:
    """Export a trained model to ONNX with dynamic batch size.

    The model must accept NHWC uint8-compatible float input, which matches
    the Normalize/NormalizeCNN layers used in Lab 1 and Lab 2.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, requires_grad=False).to(device)

    path = output_dir / filename
    torch.onnx.export(
        model,
        dummy,
        str(path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX model → {path}")
    return path
