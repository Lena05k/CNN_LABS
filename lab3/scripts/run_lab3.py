"""Entry point for Lab 3 — CIFAR-100 CNN with Dropout & Augmentation.

Usage:
    # single run with defaults from config:
    python lab3/scripts/run_lab3.py

    # specify a config file:
    python lab3/scripts/run_lab3.py --config lab3/configs/config.yaml

    # run ONE experiment group:
    python lab3/scripts/run_lab3.py --exp dropout
    python lab3/scripts/run_lab3.py --exp weight_decay
    python lab3/scripts/run_lab3.py --exp augmentation

    # run ALL three comparison experiments (9 training runs total):
    python lab3/scripts/run_lab3.py --all

The script produces, for every variant:
    lab3/outputs/<exp>/<variant>/checkpoints/best_model.pth
    lab3/outputs/<exp>/<variant>/figures/training_curves.png
    lab3/outputs/<exp>/<variant>/tables/training_history.csv
    lab3/outputs/<exp>/<variant>/metrics/metrics.csv
    lab3/outputs/tensorboard/<exp>_<variant>/   (TensorBoard logs)

Final comparison tables:
    lab3/outputs/dropout_comparison.csv
    lab3/outputs/weight_decay_comparison.csv
    lab3/outputs/augmentation_comparison.csv

Best overall model:
    lab3/outputs/onnx/cifar100_cnn_lab3_best.onnx
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
from src.models import Cifar100CNNReg
from src.train import train_reg
from src.utils import export_onnx, plot_training_curves, save_config, save_history_csv

OUTPUTS = ROOT / "lab3" / "outputs"
DEFAULT_CONFIG = ROOT / "lab3" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 3 — CIFAR-100 CNN Dropout & Aug")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--exp",
        choices=["dropout", "weight_decay", "augmentation"],
        default=None,
        help="Run a single experiment group.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all three experiment groups (9 runs).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def run_one(
    exp_name: str,
    variant_label: str,
    loaders: dict,
    config: dict,
    run_config: dict,
    device: torch.device,
) -> dict:
    """Train one variant, save all artifacts, return a summary row dict."""
    print(f"\n{'=' * 64}")
    print(f"  Experiment : {exp_name}")
    print(f"  Variant    : {variant_label}")
    print(f"  dropout_p  : {run_config.get('dropout_p')}")
    print(f"  weight_decay: {run_config.get('weight_decay')}")
    print(f"  aug        : {run_config.get('aug_variant')}")
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
    model = Cifar100CNNReg(
        hidden_size=config["hidden_size"],
        num_classes=num_classes,
        dropout_p=run_config["dropout_p"],
    ).to(device)
    print(model, "\n")

    save_config(run_config, met_dir)

    t0 = time.perf_counter()
    history = train_reg(model, loaders, run_config, device, ckpt_dir, tb_log_dir=tb_dir)
    elapsed = time.perf_counter() - t0

    plot_training_curves(history, fig_dir)
    save_history_csv(history, tab_dir)

    # reload best checkpoint before evaluation
    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    metrics_df = evaluate_all(model, loaders, device, config["classes"], met_dir)

    test_row  = metrics_df[metrics_df["split"] == "test"].iloc[0]
    train_row = metrics_df[metrics_df["split"] == "train"].iloc[0]
    return {
        "exp": exp_name,
        "variant": variant_label,
        "dropout_p": run_config["dropout_p"],
        "weight_decay": run_config["weight_decay"],
        "aug_variant": run_config["aug_variant"],
        "train_acc": f"{train_row['accuracy'] * 100:.2f}%",
        "test_acc": f"{test_row['accuracy'] * 100:.2f}%",
        "train_f1": f"{train_row['macro_f1']:.4f}",
        "test_f1": f"{test_row['macro_f1']:.4f}",
        "time_s": f"{elapsed:.0f}",
        # used for best-model selection (raw float)
        "_test_acc_raw": test_row["accuracy"],
        "_model": model,
        "_ckpt": best_ckpt,
    }


# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

def experiment_dropout(loaders, config, device) -> list[dict]:
    """Experiment A: compare 3 dropout rates (fixed aug=light, wd=1e-5)."""
    rows = []
    for p in config["dropout_variants"]:
        run_cfg = {
            **config,
            "dropout_p": p,
            "weight_decay": 1e-5,
            "aug_variant": "light",
        }
        label = f"dropout_{p}"
        rows.append(run_one("dropout", label, loaders[label], run_cfg, run_cfg, device))
    return rows


def experiment_weight_decay(loaders, config, device) -> list[dict]:
    """Experiment B: compare 3 weight_decay values (fixed aug=light, dropout=0.2)."""
    rows = []
    for wd in config["weight_decay_variants"]:
        run_cfg = {
            **config,
            "dropout_p": 0.2,
            "weight_decay": wd,
            "aug_variant": "light",
        }
        label = f"wd_{wd}"
        rows.append(run_one("weight_decay", label, loaders[label], run_cfg, run_cfg, device))
    return rows


def experiment_augmentation(loaders, config, device) -> list[dict]:
    """Experiment C: compare 3 augmentation intensities (fixed dropout=0.2, wd=1e-5)."""
    rows = []
    for aug in config["aug_variants"]:
        run_cfg = {
            **config,
            "dropout_p": 0.2,
            "weight_decay": 1e-5,
            "aug_variant": aug,
        }
        label = f"aug_{aug}"
        rows.append(run_one("augmentation", label, loaders[label], run_cfg, run_cfg, device))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loaders_for_exp(config: dict, raw_train, raw_test) -> dict:
    """Build a dict of dataloaders keyed by variant label for all experiments."""
    num_classes = len(config["classes"])
    train_X, train_y = make_subset(raw_train, config["classes"])
    test_X, test_y   = make_subset(raw_test, config["classes"])

    all_keys: dict[str, dict] = {}

    # Experiment A
    for p in config["dropout_variants"]:
        all_keys[f"dropout_{p}"] = {
            "aug_variant": "light", "aug_prob": config.get("aug_prob", 0.5)
        }
    # Experiment B
    for wd in config["weight_decay_variants"]:
        all_keys[f"wd_{wd}"] = {
            "aug_variant": "light", "aug_prob": config.get("aug_prob", 0.5)
        }
    # Experiment C
    for aug in config["aug_variants"]:
        all_keys[f"aug_{aug}"] = {
            "aug_variant": aug, "aug_prob": config.get("aug_prob", 0.5)
        }

    loaders: dict[str, dict] = {}
    for key, kw in all_keys.items():
        loaders[key] = build_dataloaders_aug(
            train_X, train_y, test_X, test_y,
            num_classes=num_classes,
            batch_size=config["batch_size"],
            aug_variant=kw["aug_variant"],
            aug_prob=kw["aug_prob"],
        )
    return loaders


def _save_comparison(rows: list[dict], path: Path, exp_name: str) -> None:
    display_cols = [
        "variant", "dropout_p", "weight_decay", "aug_variant",
        "train_acc", "test_acc", "train_f1", "test_f1", "time_s",
    ]
    df = pd.DataFrame(rows)[[c for c in display_cols if c in df.columns if c in pd.DataFrame(rows).columns]]
    # rebuild properly
    df = pd.DataFrame(
        [{c: r[c] for c in display_cols if c in r} for r in rows]
    )
    df.to_csv(path, index=False)
    print(f"\n{'=' * 64}")
    print(f"  {exp_name.upper()} COMPARISON")
    print(f"{'=' * 64}")
    print(df.to_string(index=False))
    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Classes : {config['classes']}")
    print(f"Epochs  : {config['epochs']}\n")

    # Determine which experiments to run
    run_all = args.all or (args.exp is None)
    exp_filter = args.exp  # None → run default single config or all

    raw = load_cifar100(ROOT / config["data_dir"])

    if run_all or exp_filter is not None:
        # Pre-build all needed dataloaders (all share the same raw data)
        print("Building dataloaders for all variant experiments …")
        all_loaders = _build_loaders_for_exp(
            config, raw["train"], raw["test"]
        )

        all_rows: list[dict] = []

        if run_all or exp_filter == "dropout":
            rows_a = experiment_dropout(all_loaders, config, device)
            _save_comparison(rows_a, OUTPUTS / "dropout_comparison.csv", "Dropout")
            all_rows.extend(rows_a)

        if run_all or exp_filter == "weight_decay":
            rows_b = experiment_weight_decay(all_loaders, config, device)
            _save_comparison(rows_b, OUTPUTS / "weight_decay_comparison.csv", "Weight Decay")
            all_rows.extend(rows_b)

        if run_all or exp_filter == "augmentation":
            rows_c = experiment_augmentation(all_loaders, config, device)
            _save_comparison(rows_c, OUTPUTS / "augmentation_comparison.csv", "Augmentation")
            all_rows.extend(rows_c)

        # Export best model to ONNX
        best_row = max(all_rows, key=lambda r: r["_test_acc_raw"])
        best_model = best_row["_model"]
        print(
            f"\nBest overall: {best_row['exp']}/{best_row['variant']} "
            f"test_acc={best_row['test_acc']}"
        )
        onnx_dir = OUTPUTS / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_onnx(best_model, onnx_dir, filename="cifar100_cnn_lab3_best.onnx")

    else:
        # Single default run
        num_classes = len(config["classes"])
        train_X, train_y = make_subset(raw["train"], config["classes"])
        test_X, test_y   = make_subset(raw["test"], config["classes"])
        print(
            f"Train: {len(train_X)} samples | "
            f"Test: {len(test_X)} samples | Classes: {num_classes}\n"
        )

        loaders = build_dataloaders_aug(
            train_X, train_y, test_X, test_y,
            num_classes=num_classes,
            batch_size=config["batch_size"],
            aug_variant=config.get("aug_variant", "light"),
            aug_prob=config.get("aug_prob", 0.5),
        )
        run_cfg = {**config}
        row = run_one(
            exp_name="single",
            variant_label="default",
            loaders=loaders,
            config=config,
            run_config=run_cfg,
            device=device,
        )
        onnx_dir = OUTPUTS / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_onnx(row["_model"], onnx_dir, filename="cifar100_cnn_lab3.onnx")

    print(f"\nAll outputs saved → {OUTPUTS}")


if __name__ == "__main__":
    main()
