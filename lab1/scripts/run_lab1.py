"""Entry point for Lab 1, Part 3 — CIFAR-100 MLP classification.

Usage:
    python lab1/scripts/run_lab1.py
    python lab1/scripts/run_lab1.py --config lab1/configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, load_cifar100, make_subset
from src.evaluate import evaluate_all
from src.models import Cifar100MLP
from src.train import train
from src.utils import plot_training_curves, save_config, save_history_csv

OUTPUTS = ROOT / "lab1" / "outputs"
DEFAULT_CONFIG = ROOT / "lab1" / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 1 — CIFAR-100 MLP")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Classes : {config['classes']}")
    print(f"Config  : {config}\n")

    # data
    raw = load_cifar100(ROOT / config["data_dir"])
    train_X, train_y = make_subset(raw["train"], config["classes"])
    test_X, test_y = make_subset(raw["test"], config["classes"])

    num_classes = len(config["classes"])
    print(
        f"Train: {len(train_X)} samples  |  Test: {len(test_X)} samples  "
        f"|  Classes: {num_classes}\n"
    )

    loaders = build_dataloaders(
        train_X, train_y, test_X, test_y,
        num_classes=num_classes,
        batch_size=config["batch_size"],
    )

    # model
    model = Cifar100MLP(
        hidden_size=config["hidden_size"],
        num_classes=num_classes,
    ).to(device)
    print(model, "\n")

    # train
    save_config(config, OUTPUTS / "metrics")
    history = train(model, loaders, config, device, OUTPUTS / "checkpoints")

    # artifacts
    plot_training_curves(history, OUTPUTS / "figures")
    save_history_csv(history, OUTPUTS / "tables")

    # evaluate
    evaluate_all(model, loaders, device, config["classes"], OUTPUTS / "metrics")

    print(f"\nAll outputs saved → {OUTPUTS}")


if __name__ == "__main__":
    main()
