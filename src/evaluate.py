"""Inference, metrics computation, and report generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader. Returns (y_true_onehot, y_pred_logits)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device)).cpu().numpy()
            y_pred.append(outputs)
            y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_all(
    model: nn.Module,
    dataloaders: dict,
    device: torch.device,
    classes: list[int],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate model on train and test splits; print reports; save metrics.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "test"):
        y_true, y_pred = predict(model, dataloaders[split], device)
        y_true_idx = y_true.argmax(axis=-1)
        y_pred_idx = y_pred.argmax(axis=-1)

        report = classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=[str(c) for c in classes],
            output_dict=True,
            digits=4,
        )
        rows.append(
            {
                "split": split,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
            }
        )

        print(f"\n{'=' * 52}\n{split.upper()} SET\n{'=' * 52}")
        print(
            classification_report(
                y_true_idx,
                y_pred_idx,
                target_names=[str(c) for c in classes],
                digits=4,
            )
        )

    df = pd.DataFrame(rows)
    path = output_dir / "metrics.csv"
    df.to_csv(path, index=False)
    print(f"\nFinal metrics saved → {path}")
    return df
