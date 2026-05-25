"""Lab 8 Part 1 — Autoregression (AR) vs LSTM on a synthetic time series.

The synthetic series follows the notebook formula:
    y[i] = sin(i/50) - sin(i/200) + (2i/N)^2 + noise

Two experiments:
    A) AR (LinearRegression) — one-step and recursive multi-step prediction
    B) LSTM (bidirectional) — sliding-window training

Usage:
    python lab8/scripts/run_lab8_synthetic.py
    python lab8/scripts/run_lab8_synthetic.py --config lab8/configs/config.yaml

Outputs:
    lab8/outputs/figures/synthetic_series.png
    lab8/outputs/figures/ar_prediction.png
    lab8/outputs/figures/lstm_training.png
    lab8/outputs/figures/lstm_prediction.png
    lab8/outputs/checkpoints/lstm_synthetic.pth
    lab8/outputs/metrics/synthetic_metrics.csv
    lab8/outputs/tables/synthetic_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models import LSTMForecaster
from src.train import train_lstm

OUTPUTS = ROOT / "lab8" / "outputs"
DEFAULT_CONFIG = ROOT / "lab8" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 8 Part 1 — Synthetic TS: AR vs LSTM")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------

def make_synthetic_series(n: int = 1000, noise_std: float = 0.05, seed: int = 42) -> np.ndarray:
    """y[i] = sin(i/50) - sin(i/200) + (2i/N)^2 + noise"""
    rng = np.random.default_rng(seed)
    X = np.arange(n, dtype=np.float64)
    y = np.sin(X / 50) - np.sin(X / 200) + (2 * X / n) ** 2
    y += rng.normal(0, noise_std, n)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def sliding_windows(
    series: np.ndarray,
    seq_len: int,
    num_features: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - seq_len - num_features + 1):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len: i + seq_len + num_features])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-9))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


# ---------------------------------------------------------------------------
# AR (autoregression baseline)
# ---------------------------------------------------------------------------

def run_ar(
    series: np.ndarray,
    seq_len: int,
    horizon: int,
    train_frac: float,
    fig_dir: Path,
) -> dict:
    from sklearn.linear_model import LinearRegression

    X, y = sliding_windows(series, seq_len, 1)
    n_train = int(len(X) * train_frac)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train].ravel(), y[n_train:].ravel()

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    # One-step prediction
    y_pred_te = lr.predict(X_te)

    r2 = r2_score(y_te, y_pred_te)
    rms = rmse(y_te, y_pred_te)
    print(f"  AR one-step  →  R²={r2:.4f}  RMSE={rms:.4f}")

    # Recursive multi-step prediction
    window = series[-seq_len:].copy().tolist()
    recursive_pred = []
    for _ in range(horizon):
        x = np.array(window[-seq_len:], dtype=np.float32).reshape(1, -1)
        nxt = float(lr.predict(x)[0])
        recursive_pred.append(nxt)
        window.append(nxt)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # One-step
    axes[0].plot(y_te, label="true")
    axes[0].plot(y_pred_te, label=f"AR pred  R²={r2:.3f}")
    axes[0].set_title("AR — One-step prediction (test set)")
    axes[0].set_xlabel("Step")
    axes[0].legend()
    # Recursive
    axes[1].plot(np.arange(len(series)), series, label="series")
    axes[1].plot(
        np.arange(len(series), len(series) + horizon),
        recursive_pred,
        label=f"AR recursive ({horizon} steps)",
        color="orange",
    )
    axes[1].set_title("AR — Recursive forecast")
    axes[1].set_xlabel("Time step")
    axes[1].legend()
    plt.tight_layout()
    path = fig_dir / "ar_prediction.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"AR prediction → {path}")

    return {"ar_r2": r2, "ar_rmse": rms}


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def run_lstm(
    series: np.ndarray,
    config: dict,
    device: torch.device,
    ckpt_dir: Path,
    fig_dir: Path,
) -> dict:
    seq_len      = config.get("seq_len", 50)
    num_features = config.get("num_features", 1)
    train_frac   = config.get("train_frac", 0.8)
    batch_size   = config.get("batch_size", 64)

    X, y = sliding_windows(series, seq_len, num_features)
    n_train = int(len(X) * train_frac)

    X_tr = torch.tensor(X[:n_train])
    y_tr = torch.tensor(y[:n_train])
    X_te = torch.tensor(X[n_train:])
    y_te = torch.tensor(y[n_train:])

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(
        num_features=num_features,
        input_size=1,
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 2),
        bidirectional=config.get("bidirectional", True),
        dropout_p=config.get("dropout_p", 0.4),
    ).to(device)

    print(f"\n  LSTM params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_lstm(
        model, train_loader, val_loader, config, device,
        checkpoint_dir=ckpt_dir,
        tb_log_dir=None,
    )

    # Save last checkpoint
    torch.save(model.state_dict(), ckpt_dir / "lstm_synthetic.pth")

    # Evaluate on test set
    model.eval()
    preds_list = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            p, _ = model(x_b.to(device))
            preds_list.append(p.cpu().numpy())
    preds = np.concatenate(preds_list).ravel()
    y_true = y_te.numpy().ravel()

    r2  = r2_score(y_true, preds)
    rms = rmse(y_true, preds)
    print(f"\n  LSTM test  →  R²={r2:.4f}  RMSE={rms:.4f}")

    # Plot training curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_r2"], label="train R²")
    axes[1].plot(history["val_r2"],   label="val R²")
    axes[1].set_title("R² Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    path = fig_dir / "lstm_training.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"LSTM training curves → {path}")

    # Plot predictions
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="true", alpha=0.8)
    ax.plot(preds,  label=f"LSTM  R²={r2:.3f}", alpha=0.8)
    ax.set_title("LSTM — Test set prediction")
    ax.set_xlabel("Step")
    ax.legend()
    plt.tight_layout()
    path = fig_dir / "lstm_prediction.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"LSTM prediction → {path}")

    return {"lstm_r2": r2, "lstm_rmse": rms}, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for d in (OUTPUTS / "figures", OUTPUTS / "checkpoints",
              OUTPUTS / "metrics", OUTPUTS / "tables"):
        d.mkdir(parents=True, exist_ok=True)

    fig_dir  = OUTPUTS / "figures"
    ckpt_dir = OUTPUTS / "checkpoints"
    met_dir  = OUTPUTS / "metrics"
    tab_dir  = OUTPUTS / "tables"

    n      = config.get("synthetic_n",    1000)
    noise  = config.get("noise_std",      0.05)
    seed   = config.get("synthetic_seed", 42)
    series = make_synthetic_series(n, noise, seed)
    print(f"Synthetic series: n={n}  mean={series.mean():.3f}  std={series.std():.3f}")

    # Plot series
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(series)
    ax.set_title("Synthetic time series")
    ax.set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_series.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # AR
    print("\n--- AR Baseline ---")
    ar_metrics = run_ar(
        series,
        seq_len=config.get("seq_len", 50),
        horizon=config.get("forecast_horizon", 100),
        train_frac=config.get("train_frac", 0.8),
        fig_dir=fig_dir,
    )

    # LSTM
    print("\n--- LSTM ---")
    lstm_metrics, history = run_lstm(series, config, device, ckpt_dir, fig_dir)

    # Save metrics
    metrics = {**ar_metrics, **lstm_metrics}
    pd.DataFrame([metrics]).to_csv(met_dir / "synthetic_metrics.csv", index=False)
    print(f"\nMetrics → {met_dir / 'synthetic_metrics.csv'}")

    # Save training history
    pd.DataFrame(history).to_csv(tab_dir / "synthetic_history.csv", index=False)
    print(f"History → {tab_dir / 'synthetic_history.csv'}")

    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
