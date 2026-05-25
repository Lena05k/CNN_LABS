"""Lab 8 Part 2 — LSTM on real weather data (temperature forecasting).

Expected CSV format (one file per city):
    year,month,day,Temperature
    2000,1,1,−3.2
    ...

Place CSV files in lab8/data/<city>.csv
City names are listed in config.yaml under `weather_cities`.

Usage:
    python lab8/scripts/run_lab8_weather.py
    python lab8/scripts/run_lab8_weather.py --config lab8/configs/config.yaml
    python lab8/scripts/run_lab8_weather.py --city city1.csv city2.csv

Outputs:
    lab8/outputs/figures/weather_<city>_series.png
    lab8/outputs/figures/weather_<city>_prediction.png
    lab8/outputs/checkpoints/lstm_weather_<city>.pth
    lab8/outputs/metrics/weather_metrics.csv
    lab8/outputs/tables/weather_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models import LSTMForecaster
from src.train import train_lstm

OUTPUTS = ROOT / "lab8" / "outputs"
DATA_DIR = ROOT / "lab8" / "data"
DEFAULT_CONFIG = ROOT / "lab8" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 8 Part 2 — Weather LSTM")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--city", nargs="+", default=None,
                   help="CSV filenames (with or without .csv) in lab8/data/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-9))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


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


def load_weather(path: Path, col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if col not in df.columns:
        # Try case-insensitive match
        col_lower = {c.lower(): c for c in df.columns}
        col = col_lower.get(col.lower(), df.columns[-1])
        print(f"  Column '{col}' used for temperature.")
    series = df[col].dropna().to_numpy(dtype=np.float32)
    return series


def normalise(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu, sigma = series.mean(), series.std() + 1e-9
    return (series - mu) / sigma, float(mu), float(sigma)


# ---------------------------------------------------------------------------
# Train & evaluate one city
# ---------------------------------------------------------------------------

def run_city(
    city_name: str,
    series_raw: np.ndarray,
    config: dict,
    device: torch.device,
    ckpt_dir: Path,
    fig_dir: Path,
) -> dict:
    import matplotlib.pyplot as plt

    seq_len      = config.get("seq_len",      50)
    num_features = config.get("num_features",  1)
    train_frac   = config.get("train_frac",  0.8)
    batch_size   = config.get("batch_size",   64)

    series, mu, sigma = normalise(series_raw)
    print(f"\n  {city_name}: n={len(series)}  raw_mean={mu:.2f}  raw_std={sigma:.2f}")

    # Plot raw series
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(series_raw)
    ax.set_title(f"Temperature series — {city_name}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(fig_dir / f"weather_{city_name}_series.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

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

    history = train_lstm(
        model, train_loader, val_loader, config, device,
        checkpoint_dir=ckpt_dir,
        tb_log_dir=None,
    )
    torch.save(model.state_dict(), ckpt_dir / f"lstm_weather_{city_name}.pth")

    # Test evaluation
    model.eval()
    preds_list = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            p, _ = model(x_b.to(device))
            preds_list.append(p.cpu().numpy())

    preds_norm = np.concatenate(preds_list).ravel()
    y_true_norm = y_te.numpy().ravel()

    # De-normalise
    preds_raw  = preds_norm * sigma + mu
    y_true_raw = y_true_norm * sigma + mu

    r2  = r2_score(y_true_raw, preds_raw)
    rms = rmse(y_true_raw, preds_raw)
    print(f"  {city_name} → R²={r2:.4f}  RMSE={rms:.4f} °")

    # Prediction plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true_raw, label="true", alpha=0.8)
    ax.plot(preds_raw,  label=f"LSTM  R²={r2:.3f}", alpha=0.8)
    ax.set_title(f"LSTM — {city_name} test set")
    ax.set_xlabel("Day (test)")
    ax.set_ylabel("Temperature")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"weather_{city_name}_prediction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Prediction plot → {fig_dir / f'weather_{city_name}_prediction.png'}")

    return {
        "city": city_name,
        "n_total": len(series),
        "n_train": n_train,
        "r2": r2,
        "rmse": rms,
    }, history


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

    # Resolve city list
    if args.city:
        city_files = [Path(c) if Path(c).suffix else Path(f"{c}.csv") for c in args.city]
        city_files = [(DATA_DIR / c.name if not c.is_absolute() else c) for c in city_files]
    else:
        city_names = config.get("weather_cities", ["city1", "city2"])
        city_files = [DATA_DIR / f"{c}.csv" for c in city_names]

    # Check files
    missing = [p for p in city_files if not p.exists()]
    if missing:
        print("\n" + "=" * 60)
        print("  WEATHER CSV FILES NOT FOUND")
        print("=" * 60)
        for p in missing:
            print(f"  Missing: {p}")
        print("\nExpected CSV columns: year, month, day, Temperature")
        print(f"Place files in: {DATA_DIR}/")
        sys.exit(0)

    col = config.get("weather_col", "Temperature")
    all_metrics = []
    all_history_rows = []

    for csv_path in city_files:
        city_name = csv_path.stem
        series = load_weather(csv_path, col)
        if len(series) < config.get("seq_len", 50) * 3:
            print(f"  WARNING: {city_name} has only {len(series)} points — too short, skipping.")
            continue

        metrics, history = run_city(city_name, series, config, device, ckpt_dir, fig_dir)
        all_metrics.append(metrics)
        for ep, row in enumerate(zip(
            history["train_loss"], history["val_loss"],
            history["train_r2"],   history["val_r2"],
        )):
            all_history_rows.append({
                "city": city_name, "epoch": ep,
                "train_loss": row[0], "val_loss": row[1],
                "train_r2": row[2], "val_r2": row[3],
            })

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(met_dir / "weather_metrics.csv", index=False)
        print(f"\nMetrics → {met_dir / 'weather_metrics.csv'}")

    if all_history_rows:
        pd.DataFrame(all_history_rows).to_csv(tab_dir / "weather_history.csv", index=False)
        print(f"History → {tab_dir / 'weather_history.csv'}")

    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
