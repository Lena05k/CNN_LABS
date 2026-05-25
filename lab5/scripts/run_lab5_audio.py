"""Lab 5 Part 2 — Audio Denoising Autoencoder.

The script expects two WAV files (1 channel, 16 kHz):
    lab5/data/audio.wav  — clean audio signal
    lab5/data/noise.wav  — noise sample to add

If the files are not present, instructions are printed and the script exits.

Convert your files with ffmpeg (examples):
    ffmpeg -y -i my_song.mp4   -ac 1 -ar 16000 lab5/data/audio.wav
    ffmpeg -y -i pink_noise.mp3 -ac 1 -ar 16000 lab5/data/noise.wav

Usage:
    python lab5/scripts/run_lab5_audio.py
    python lab5/scripts/run_lab5_audio.py --config lab5/configs/config.yaml
    python lab5/scripts/run_lab5_audio.py --audio my_song.wav --noise pink.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "lab5" / "outputs" / "audio_ae"
DEFAULT_CONFIG = ROOT / "lab5" / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 5 Part 2 — Audio Denoising AE")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--audio",  default=None, help="Path to clean audio WAV.")
    p.add_argument("--noise",  default=None, help="Path to noise WAV.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy")
    if missing:
        print(f"Missing packages: {missing}. Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def _load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile
    fs, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    # Normalise to [-1, 1]
    max_val = np.abs(data).max()
    if max_val > 1.0:
        data = data / max_val
    return fs, data


def _build_stft_dataset(
    signal_noised: np.ndarray,
    signal_noise: np.ndarray,
    fs: int,
    nperseg: int = 512,
    batch_size: int = 128,
    train_frac: float = 0.9,
) -> tuple[DataLoader, DataLoader, np.ndarray, float, int]:
    """Compute STFT, build train/test DataLoaders and return normalisation."""
    from scipy import signal as sp

    _, _, Zxx_noised = sp.stft(signal_noised, fs=fs, nperseg=nperseg)
    _, _, Zxx_noise  = sp.stft(signal_noise,  fs=fs, nperseg=nperseg)

    # Stack real/imag → (time, 2, F)
    def stft_to_tensor(Z: np.ndarray) -> torch.Tensor:
        # Z: (F, T) complex
        X = np.stack([np.real(Z).T, np.imag(Z).T], axis=1)  # (T, 2, F)
        return torch.tensor(X, dtype=torch.float32)

    tx = stft_to_tensor(Zxx_noised)   # input  (noised signal STFT)
    ty = stft_to_tensor(Zxx_noise)    # target (noise STFT — model learns to predict noise)

    # Normalise by std of input
    norm_scale = float(tx.reshape(-1, 2).std().item())
    tx = tx / (norm_scale + 1e-9)
    ty = ty / (norm_scale + 1e-9)

    n_train = int(tx.shape[0] * train_frac)
    train_ds = TensorDataset(tx[:n_train], ty[:n_train])
    test_ds  = TensorDataset(tx[n_train:], ty[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    freq_bins = Zxx_noised.shape[0]
    return train_loader, test_loader, Zxx_noised, norm_scale, freq_bins


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-9))


def _eval_loader(net, loader, device) -> float:
    net.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for x, _ in loader:
            preds = net(x.to(device)).cpu().numpy()
            yt_all.append(x.numpy().reshape(x.shape[0], -1))
            yp_all.append(preds.reshape(preds.shape[0], -1))
    return _r2(np.concatenate(yt_all), np.concatenate(yp_all))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_deps()
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    audio_path = Path(args.audio or ROOT / config.get("audio_file", "lab5/data/audio.wav"))
    noise_path = Path(args.noise or ROOT / config.get("noise_file", "lab5/data/noise.wav"))

    # Check files exist
    missing_files = [p for p in (audio_path, noise_path) if not p.exists()]
    if missing_files:
        print("\n" + "=" * 60)
        print("  AUDIO FILES NOT FOUND")
        print("=" * 60)
        for p in missing_files:
            print(f"  Missing: {p}")
        print("\nConvert your audio files with ffmpeg:")
        print("  ffmpeg -y -i <your_song>  -ac 1 -ar 16000 lab5/data/audio.wav")
        print("  ffmpeg -y -i <noise_file> -ac 1 -ar 16000 lab5/data/noise.wav")
        print("\nThen re-run this script.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Audio : {audio_path}")
    print(f"Noise : {noise_path}\n")

    # Load audio
    fs_audio, data = _load_wav(audio_path)
    fs_noise, noise_raw = _load_wav(noise_path)
    print(f"Sample rate: {fs_audio} Hz | Duration: {len(data)/fs_audio:.1f}s")

    # Trim / tile noise to match signal length
    if len(noise_raw) < len(data):
        repeat = int(np.ceil(len(data) / len(noise_raw)))
        noise_raw = np.tile(noise_raw, repeat)
    noise_raw = noise_raw[: len(data)]

    # Scale noise
    scale = config.get("noise_scale", 0.1)
    noise_scaled = scale * (data.max() / (np.abs(noise_raw).max() + 1e-9)) * noise_raw
    data_noised = data + noise_scaled
    print(f"Noise scale: {scale}  |  Signal max: {data.max():.3f}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # STFT
    batch_size = config.get("audio_batch_size", 128)
    train_loader, test_loader, Zxx_noised, norm_scale, freq_bins = _build_stft_dataset(
        data_noised, noise_scaled, fs=fs_audio, batch_size=batch_size
    )
    print(f"Freq bins: {freq_bins}  |  Train batches: {len(train_loader)}\n")

    # Build model
    from src.models import DenoisingAE
    net = DenoisingAE().to(device)
    print(net)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.get("audio_lr", 1e-4)
    )
    epochs = config.get("audio_epochs", 15)

    train_r2_hist, val_r2_hist, loss_hist = [], [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            # Crop to target length (ConvTranspose1d may add 1 sample)
            min_len = min(pred.shape[-1], y.shape[-1])
            loss = criterion(pred[..., :min_len], y[..., :min_len])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        tr2 = _eval_loader(net, train_loader, device)
        vr2 = _eval_loader(net, test_loader,  device)
        train_r2_hist.append(tr2)
        val_r2_hist.append(vr2)
        loss_hist.append(avg_loss)
        print(f"Epoch {epoch+1:>2}/{epochs} | loss={avg_loss:.5f} | "
              f"train R²={tr2:.4f} | val R²={vr2:.4f}")

    # Save checkpoint
    ckpt_path = OUTPUTS / "checkpoints" / "denoising_ae.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt_path)
    print(f"\nCheckpoint → {ckpt_path}")

    # Plot training curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(loss_hist, label="train loss")
    ax[0].set_title("MSE Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(train_r2_hist, label="train R²")
    ax[1].plot(val_r2_hist,   label="val R²")
    ax[1].set_title("R² Score")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    fig_dir = OUTPUTS / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves → {fig_path}")

    # Denoise full signal
    from scipy import signal as sp
    net.eval()
    _, _, Zxx_noised_full = sp.stft(data_noised, fs=fs_audio, nperseg=512)

    def stft_to_tensor(Z):
        X = np.stack([np.real(Z).T, np.imag(Z).T], axis=1)
        return torch.tensor(X, dtype=torch.float32)

    tx_full = stft_to_tensor(Zxx_noised_full) / (norm_scale + 1e-9)
    pred_chunks = []
    bsz = batch_size
    with torch.no_grad():
        for i in range(0, len(tx_full), bsz):
            chunk = tx_full[i: i + bsz].to(device)
            pred_chunks.append(net(chunk).cpu().numpy())
    pred_arr = np.concatenate(pred_chunks, axis=0)  # (T, 2, F)
    # Back to complex STFT
    pred_arr *= norm_scale
    pred_complex = (pred_arr[:, 0, :] + 1j * pred_arr[:, 1, :]).T  # (F, T)

    _, denoised = sp.istft(Zxx_noised_full - pred_complex, fs_audio)
    denoised = denoised[: len(data)]

    r2_noised   = _r2(data, data_noised[:len(data)])
    r2_denoised = _r2(data, denoised)
    print(f"\nR² (noised vs clean)  : {r2_noised:.4f}")
    print(f"R² (denoised vs clean): {r2_denoised:.4f}")

    # Save denoised audio
    from scipy.io import wavfile
    out_wav = OUTPUTS / "denoised.wav"
    denoised_int = np.clip(denoised * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(out_wav), fs_audio, denoised_int)
    print(f"Denoised audio → {out_wav}")

    # Save metrics
    import pandas as pd
    pd.DataFrame([{"r2_noised": r2_noised, "r2_denoised": r2_denoised}]).to_csv(
        OUTPUTS / "metrics" / "metrics.csv", index=False
    )
    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
