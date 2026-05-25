"""Entry point for Lab 2 — CIFAR-100 CNN classification.

Usage:
    python lab2/scripts/run_lab2.py
    python lab2/scripts/run_lab2.py --config lab2/configs/config.yaml
    python lab2/scripts/run_lab2.py --pool avg
    python lab2/scripts/run_lab2.py --pool all   # runs all 3 pooling variants
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, load_cifar100, make_subset
from src.evaluate import evaluate_all
from src.models import Cifar100CNN
from src.train import train
from src.utils import export_onnx, plot_training_curves, save_config, save_history_csv

OUTPUTS = ROOT / "lab2" / "outputs"
DEFAULT_CONFIG = ROOT / "lab2" / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 2 — CIFAR-100 CNN")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--pool",
        choices=["avg", "max", "stride", "all"],
        default=None,
        help="Override pool_type from config. 'all' runs every variant.",
    )
    return p.parse_args()


def run_one(
    pool_type: str,
    loaders: dict,
    config: dict,
    device: torch.device,
) -> dict:
    """Train one pooling variant, save artifacts, return summary row."""
    print(f"\n{'=' * 60}")
    print(f"  Pool type: {pool_type.upper()}")
    print(f"{'=' * 60}\n")

    out_root = OUTPUTS / pool_type
    ckpt_dir = out_root / "checkpoints"
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    met_dir = out_root / "metrics"
    onnx_dir = out_root / "onnx"

    num_classes = len(config["classes"])
    model = Cifar100CNN(
        hidden_size=config["hidden_size"],
        num_classes=num_classes,
        pool_type=pool_type,
    ).to(device)
    print(model, "\n")

    run_config = {**config, "pool_type": pool_type}
    save_config(run_config, met_dir)

    t0 = time.perf_counter()
    history = train(model, loaders, config, device, ckpt_dir)
    elapsed = time.perf_counter() - t0

    plot_training_curves(history, fig_dir)
    save_history_csv(history, tab_dir)

    metrics_df = evaluate_all(model, loaders, device, config["classes"], met_dir)

    # reload best checkpoint for ONNX export
    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    export_onnx(model, onnx_dir, filename=f"cifar100_cnn_{pool_type}.onnx")

    test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]
    train_row = metrics_df[metrics_df["split"] == "train"].iloc[0]
    return {
        "pool_type": pool_type,
        "train_acc": f"{train_row['accuracy'] * 100:.2f}%",
        "test_acc": f"{test_row['accuracy'] * 100:.2f}%",
        "train_f1": f"{train_row['macro_f1']:.4f}",
        "test_f1": f"{test_row['macro_f1']:.4f}",
        "time_s": f"{elapsed:.0f}",
    }


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    pool_override = args.pool or config.get("pool_type", "avg")
    pool_types = list(Cifar100CNN.POOL_TYPES) if pool_override == "all" else [pool_override]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Classes : {config['classes']}")
    print(f"Pools   : {pool_types}")
    print(f"Config  : {config}\n")

    raw = load_cifar100(ROOT / config["data_dir"])
    train_X, train_y = make_subset(raw["train"], config["classes"])
    test_X, test_y = make_subset(raw["test"], config["classes"])

    num_classes = len(config["classes"])
    print(f"Train: {len(train_X)} samples | Test: {len(test_X)} samples | Classes: {num_classes}\n")

    loaders = build_dataloaders(
        train_X, train_y, test_X, test_y,
        num_classes=num_classes,
        batch_size=config["batch_size"],
    )

    rows = []
    for pool_type in pool_types:
        rows.append(run_one(pool_type, loaders, config, device))

    # comparison table when running all variants
    if len(rows) > 1:
        df = pd.DataFrame(rows)
        path = OUTPUTS / "pooling_comparison.csv"
        df.to_csv(path, index=False)
        print(f"\n{'=' * 60}")
        print("  POOLING COMPARISON")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))
        print(f"\nComparison saved → {path}")

    print(f"\nAll outputs saved → {OUTPUTS}")


if __name__ == "__main__":
    main()
