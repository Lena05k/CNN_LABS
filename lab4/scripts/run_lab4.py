"""Entry point for Lab 4 — Transfer Learning (fine-tuning CIFAR-100 pretrained models).

Usage:
    # Single run — frozen head, SGD (defaults from config):
    python lab4/scripts/run_lab4.py

    # Experiment A only — frozen vs full fine-tuning:
    python lab4/scripts/run_lab4.py --exp mode

    # Experiment B only — optimizer comparison (frozen head):
    python lab4/scripts/run_lab4.py --exp optimizer

    # All experiments (5 runs total):
    python lab4/scripts/run_lab4.py --all

    # Use mobilenetv2 (odd variant):
    python lab4/scripts/run_lab4.py --model mobilenetv2 --all

Outputs per variant:
    lab4/outputs/<exp>/<variant>/checkpoints/best_model.pth
    lab4/outputs/<exp>/<variant>/figures/training_curves.png
    lab4/outputs/<exp>/<variant>/tables/training_history.csv
    lab4/outputs/<exp>/<variant>/metrics/metrics.csv
    lab4/outputs/tensorboard/<exp>_<variant>/

Summary tables:
    lab4/outputs/mode_comparison.csv
    lab4/outputs/optimizer_comparison.csv
    lab4/outputs/onnx/cifar100_lab4_best.onnx
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

from src.data import build_dataloaders_aug, load_cifar100, make_subset
from src.evaluate import evaluate_all
from src.models import build_pretrained_model, freeze_backbone, unfreeze_all
from src.train import train_finetune
from src.utils import export_onnx, plot_training_curves, save_config, save_history_csv

OUTPUTS = ROOT / "lab4" / "outputs"
DEFAULT_CONFIG = ROOT / "lab4" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 4 — Transfer Learning")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], default=None,
                   help="Override model_name from config.")
    p.add_argument("--exp", choices=["mode", "optimizer"], default=None,
                   help="Run a single experiment group.")
    p.add_argument("--all", action="store_true",
                   help="Run both experiment groups.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    exp_name: str,
    variant_label: str,
    loaders: dict,
    config: dict,
    frozen: bool,
    optimizer_name: str,
    device: torch.device,
) -> dict:
    print(f"\n{'=' * 64}")
    print(f"  Experiment : {exp_name}")
    print(f"  Variant    : {variant_label}")
    print(f"  Model      : {config['model_name']}")
    print(f"  Frozen     : {frozen}")
    print(f"  Optimizer  : {optimizer_name}")
    print(f"{'=' * 64}\n")

    out_root = OUTPUTS / exp_name / variant_label
    ckpt_dir = out_root / "checkpoints"
    fig_dir  = out_root / "figures"
    tab_dir  = out_root / "tables"
    met_dir  = out_root / "metrics"
    tb_dir   = OUTPUTS / "tensorboard" / f"{exp_name}_{variant_label}"
    for d in (ckpt_dir, fig_dir, tab_dir, met_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_classes = len(config["classes"])

    # Build pretrained model (fresh for every run)
    model, head_pattern = build_pretrained_model(config["model_name"], num_classes)
    if frozen:
        freeze_backbone(model, head_pattern)
    else:
        unfreeze_all(model)
    model.to(device)

    run_cfg = {**config, "frozen": frozen, "optimizer": optimizer_name}
    epochs  = config["epochs"] if frozen else config.get("epochs_full", 100)
    run_cfg["epochs"] = epochs
    save_config(run_cfg, met_dir)

    t0 = time.perf_counter()
    history = train_finetune(
        model, loaders, run_cfg, device, ckpt_dir,
        optimizer_name=optimizer_name,
        tb_log_dir=tb_dir,
    )
    elapsed = time.perf_counter() - t0

    plot_training_curves(history, fig_dir)
    save_history_csv(history, tab_dir)

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    metrics_df = evaluate_all(model, loaders, device, config["classes"], met_dir)

    test_row  = metrics_df[metrics_df["split"] == "test"].iloc[0]
    train_row = metrics_df[metrics_df["split"] == "train"].iloc[0]
    return {
        "exp": exp_name,
        "variant": variant_label,
        "model": config["model_name"],
        "frozen": frozen,
        "optimizer": optimizer_name,
        "epochs": epochs,
        "train_acc": f"{train_row['accuracy'] * 100:.2f}%",
        "test_acc":  f"{test_row['accuracy']  * 100:.2f}%",
        "train_f1":  f"{train_row['macro_f1']:.4f}",
        "test_f1":   f"{test_row['macro_f1']:.4f}",
        "time_s":    f"{elapsed:.0f}",
        "_test_acc_raw": test_row["accuracy"],
        "_model": model,
    }


# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

def experiment_mode(loaders, config, device) -> list[dict]:
    """Experiment A: frozen head vs full fine-tuning (optimizer=SGD)."""
    rows = []
    for mode in config["modes"]:         # ["frozen", "full"]
        frozen = (mode == "frozen")
        rows.append(run_one("mode", mode, loaders, config, frozen, "sgd", device))
    return rows


def experiment_optimizer(loaders, config, device) -> list[dict]:
    """Experiment B: 3 optimizers with frozen head."""
    rows = []
    for opt in config["optimizer_variants"]:  # ["sgd", "adam", "adamw"]
        rows.append(run_one("optimizer", f"frozen_{opt}", loaders, config,
                            frozen=True, optimizer_name=opt, device=device))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_comparison(rows: list[dict], path: Path, title: str) -> None:
    cols = ["variant", "model", "frozen", "optimizer", "epochs",
            "train_acc", "test_acc", "train_f1", "test_f1", "time_s"]
    df = pd.DataFrame([{c: r[c] for c in cols if c in r} for r in rows])
    df.to_csv(path, index=False)
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")
    print(df.to_string(index=False))
    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.model:
        config["model_name"] = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {config['model_name']}")
    print(f"Classes: {config['classes']}\n")

    raw = load_cifar100(ROOT / config["data_dir"])
    train_X, train_y = make_subset(raw["train"], config["classes"])
    test_X,  test_y  = make_subset(raw["test"],  config["classes"])
    num_classes = len(config["classes"])
    print(f"Train: {len(train_X)} | Test: {len(test_X)} | Classes: {num_classes}\n")

    loaders = build_dataloaders_aug(
        train_X, train_y, test_X, test_y,
        num_classes=num_classes,
        batch_size=config["batch_size"],
        aug_variant=config.get("aug_variant", "light"),
        aug_prob=config.get("aug_prob", 0.5),
    )

    run_all = args.all or (args.exp is None)
    all_rows: list[dict] = []

    if run_all or args.exp == "mode":
        rows_a = experiment_mode(loaders, config, device)
        _save_comparison(rows_a, OUTPUTS / "mode_comparison.csv", "MODE COMPARISON")
        all_rows.extend(rows_a)

    if run_all or args.exp == "optimizer":
        rows_b = experiment_optimizer(loaders, config, device)
        _save_comparison(rows_b, OUTPUTS / "optimizer_comparison.csv", "OPTIMIZER COMPARISON")
        all_rows.extend(rows_b)

    if not all_rows:
        # Single default run
        row = run_one("single", "default", loaders, config,
                      frozen=True, optimizer_name="sgd", device=device)
        all_rows.append(row)

    # Export best model
    best_row = max(all_rows, key=lambda r: r["_test_acc_raw"])
    onnx_dir = OUTPUTS / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(best_row["_model"], onnx_dir, filename="cifar100_lab4_best.onnx")
    print(
        f"\nBest: {best_row['exp']}/{best_row['variant']} "
        f"test_acc={best_row['test_acc']}"
    )
    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
