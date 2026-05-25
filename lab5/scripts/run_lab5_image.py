"""Lab 5 Part 1 — Image Autoencoder for CIFAR-100 embedding visualization.

Usage:
    # Single run with defaults (hidden_size=512):
    python lab5/scripts/run_lab5_image.py

    # Compare 3 hidden sizes:
    python lab5/scripts/run_lab5_image.py --all

Outputs per hidden_size variant:
    lab5/outputs/image_ae/<hidden>/checkpoints/best_model.pth
    lab5/outputs/image_ae/<hidden>/figures/training_curves.png
    lab5/outputs/image_ae/<hidden>/figures/embedding_pca.png
    lab5/outputs/image_ae/<hidden>/figures/reconstructions.png
    lab5/outputs/image_ae/<hidden>/tables/training_history.csv
    lab5/outputs/image_ae/<hidden>/metrics/metrics.csv

Summary:
    lab5/outputs/image_ae/hidden_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders_aug, load_cifar100, make_subset
from src.models import Cifar100AE
from src.train import train_ae
from src.utils import plot_training_curves, save_config, save_history_csv

OUTPUTS = ROOT / "lab5" / "outputs" / "image_ae"
DEFAULT_CONFIG = ROOT / "lab5" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_reconstructions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
    n: int = 8,
) -> None:
    """Save a side-by-side grid of original vs reconstructed images."""
    model.eval()
    images, recons = [], []
    with torch.no_grad():
        for inputs, _ in loader:
            out, _, _ = model(inputs.to(device))
            images.append(inputs.numpy())
            recons.append(out.cpu().numpy())
            if len(images) * inputs.shape[0] >= n:
                break
    images = np.concatenate(images, axis=0)[:n]  # (n, 32, 32, 3)
    recons = np.concatenate(recons, axis=0)[:n]  # (n, 3072) → reshape

    # De-normalize recon to [0,1]
    recons = (recons - recons.min()) / (recons.max() - recons.min() + 1e-9)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(images[i].astype(np.uint8).reshape(32, 32, 3))
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].reshape(32, 32, 3))
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.tight_layout()
    path = out_dir / "reconstructions.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Reconstructions → {path}")


def plot_embedding_pca(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
    classes: list[int],
) -> None:
    """Extract embeddings, project to 2D via PCA, scatter-plot with class colour."""
    model.eval()
    embeddings, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            _, emb, _ = model(inputs.to(device))
            embeddings.append(emb.cpu().numpy())
            labels_all.append(labels.argmax(-1).numpy())
    embeddings = np.concatenate(embeddings)
    labels_all = np.concatenate(labels_all)

    proj = PCA(n_components=2).fit_transform(embeddings)
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, cls in enumerate(range(len(classes))):
        mask = labels_all == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   s=10, alpha=0.6, label=f"class {classes[cls]}",
                   color=colours[idx % len(colours)])
    ax.legend()
    ax.set_title("Embedding PCA (2D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    path = out_dir / "embedding_pca.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Embedding PCA → {path}")


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def run_one(
    hidden_size: int,
    loaders: dict,
    config: dict,
    device: torch.device,
) -> dict:
    label = f"hidden_{hidden_size}"
    print(f"\n{'=' * 60}\n  Hidden size: {hidden_size}  "
          f"(bottleneck = {hidden_size // 8} dims)\n{'=' * 60}\n")

    out_root = OUTPUTS / label
    ckpt_dir = out_root / "checkpoints"
    fig_dir  = out_root / "figures"
    tab_dir  = out_root / "tables"
    met_dir  = out_root / "metrics"
    tb_dir   = OUTPUTS / "tensorboard" / label
    for d in (ckpt_dir, fig_dir, tab_dir, met_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_cfg = {**config, "hidden_size": hidden_size}
    save_config(run_cfg, met_dir)

    model = Cifar100AE(hidden_size=hidden_size).to(device)
    print(model)

    history = train_ae(model, loaders, run_cfg, device, ckpt_dir, tb_log_dir=tb_dir)
    plot_training_curves(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"],
         "train_acc":  history["train_r2"],   "val_acc":  history["val_r2"]},
        fig_dir,
    )
    save_history_csv(history, tab_dir)

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # Visualisations (no-aug eval loader)
    eval_loaders = loaders   # test loader has no augmentation
    plot_reconstructions(model, eval_loaders["test"], device, fig_dir)
    plot_embedding_pca(model, eval_loaders["test"], device, fig_dir, config["classes"])

    # R² on test set
    import torch.nn as nn
    criterion = nn.MSELoss()
    r2_vals = []
    with torch.no_grad():
        for inputs, _ in eval_loaders["test"]:
            out, _, normed = model(inputs.to(device))
            ss_res = ((normed.cpu() - out.cpu()) ** 2).sum().item()
            ss_tot = ((normed.cpu() - normed.cpu().mean()) ** 2).sum().item()
            r2_vals.append(1 - ss_res / (ss_tot + 1e-9))
    val_r2 = float(np.mean(r2_vals))

    # Save metrics
    pd.DataFrame([{"hidden_size": hidden_size, "bottleneck": hidden_size // 8,
                   "val_r2": val_r2}]).to_csv(met_dir / "metrics.csv", index=False)
    print(f"  val R² = {val_r2:.4f}")

    return {
        "hidden_size": hidden_size,
        "bottleneck": hidden_size // 8,
        "val_r2": f"{val_r2:.4f}",
        "_val_r2_raw": val_r2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 5 Part 1 — Image AE")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--all", action="store_true",
                   help="Compare all hidden_size variants.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Classes : {config['classes']}\n")

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

    hidden_sizes = (
        config["hidden_variants"] if args.all else [config["hidden_size"]]
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = [run_one(h, loaders, config, device) for h in hidden_sizes]

    if len(rows) > 1:
        df = pd.DataFrame(rows)[["hidden_size", "bottleneck", "val_r2"]]
        path = OUTPUTS / "hidden_comparison.csv"
        df.to_csv(path, index=False)
        print(f"\n{'=' * 60}\n  HIDDEN SIZE COMPARISON\n{'=' * 60}")
        print(df.to_string(index=False))
        print(f"\nSaved → {path}")

    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
