"""Lab 7 — U-Net segmentation + Ensemble for forest damage detection.

Stack: TensorFlow / Keras  (same as the original notebook).
Install dependencies:
    pip install tensorflow>=2.10.0 scikit-image>=0.20.0 imgaug>=0.4.0 opencv-python>=4.7.0

Dataset:
    Place the multispectral tif file in lab7/data/forest.tif
    (256×256×27: ch0 = binary mask, ch1–13 = new Sentinel-2, ch14–26 = old Sentinel-2)

Usage:
    python lab7/scripts/run_lab7.py                   # train 3 U-Nets + 3 ensembles
    python lab7/scripts/run_lab7.py --config lab7/configs/config.yaml
    python lab7/scripts/run_lab7.py --data lab7/data/forest.tif
    python lab7/scripts/run_lab7.py --exp unet        # train 3 U-Nets only
    python lab7/scripts/run_lab7.py --exp ensemble    # ensembles only (loads checkpoints)

Outputs:
    lab7/outputs/figures/  — training curves, confusion matrices, sample predictions
    lab7/outputs/checkpoints/ — saved .keras models
    lab7/outputs/metrics/metrics.csv
    lab7/outputs/tables/training_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    required = {
        "tensorflow": "tensorflow>=2.10.0",
        "skimage": "scikit-image>=0.20.0",
        "cv2": "opencv-python>=4.7.0",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\nMissing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 7 — U-Net + Ensemble")
    p.add_argument("--config", default=str(ROOT / "lab7" / "configs" / "config.yaml"))
    p.add_argument("--data",   default=None, help="Override data_file from config.")
    p.add_argument(
        "--exp",
        choices=["unet", "ensemble", "all"],
        default="all",
        help="Which experiment to run.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature engineering  (mirrors notebook prepare_snaps)
# ---------------------------------------------------------------------------

def prepare_snaps(img: "np.ndarray") -> "np.ndarray":
    """Convert raw 256×256×27 tif into 256×256×16 feature map.

    Features (16 channels):
        ch0–11  : band differences (new − old) for channels 1–12
        ch12    : NDVI-like index on new bands  (ch8 − ch4) / (ch8 + ch4 + 1e-6)
        ch13    : NDVI-like index on old bands  (ch21 − ch17) / (ch21 + ch17 + 1e-6)
        ch14–15 : first 2 PCA-like raw new bands (ch1, ch2) for visual context
    """
    import numpy as np
    new = img[:, :, 1:14].astype(np.float32)      # (H, W, 13)
    old = img[:, :, 14:27].astype(np.float32)     # (H, W, 13)

    diff = new - old                               # (H, W, 13) band differences
    diff_12 = diff[:, :, :12]                     # first 12

    ndvi_new = (new[:, :, 7] - new[:, :, 3]) / (new[:, :, 7] + new[:, :, 3] + 1e-6)
    ndvi_old = (old[:, :, 7] - old[:, :, 3]) / (old[:, :, 7] + old[:, :, 3] + 1e-6)

    visual_new = new[:, :, :2]                    # 2 visual channels

    feats = np.concatenate([
        diff_12,
        ndvi_new[:, :, None],
        ndvi_old[:, :, None],
        visual_new,
    ], axis=-1)                                   # (H, W, 16)

    # Normalise per-channel to [0, 1]
    for c in range(feats.shape[-1]):
        mn, mx = feats[:, :, c].min(), feats[:, :, c].max()
        feats[:, :, c] = (feats[:, :, c] - mn) / (mx - mn + 1e-9)

    return feats


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _dice_loss(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow as tf
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def _bce_dice_loss(y_true, y_pred):
    import tensorflow as tf
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + _dice_loss(y_true, y_pred)


def _tversky_loss(alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
    def loss(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
        return 1.0 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    loss.__name__ = f"tversky_a{alpha}_b{beta}"
    return loss


def _dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow as tf
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(tf.round(y_pred), [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


# ---------------------------------------------------------------------------
# U-Net architecture  (TF/Keras)
# ---------------------------------------------------------------------------

def _build_unet(img_size: int = 256, in_channels: int = 16, filters: int = 32):
    """Build a 4-level U-Net for binary segmentation."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    def encoder_block(x, f):
        s = conv_block(x, f)
        p = layers.MaxPooling2D(2)(s)
        return s, p

    def decoder_block(x, skip, f):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
        return x

    inputs = tf.keras.Input(shape=(img_size, img_size, in_channels))

    # Encoder
    s1, p1 = encoder_block(inputs, filters)
    s2, p2 = encoder_block(p1,     filters * 2)
    s3, p3 = encoder_block(p2,     filters * 4)
    s4, p4 = encoder_block(p3,     filters * 8)

    # Bridge
    b = conv_block(p4, filters * 16)

    # Decoder
    d1 = decoder_block(b,  s4, filters * 8)
    d2 = decoder_block(d1, s3, filters * 4)
    d3 = decoder_block(d2, s2, filters * 2)
    d4 = decoder_block(d3, s1, filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return Model(inputs, outputs, name="unet")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _make_patches(
    feats: "np.ndarray",
    mask: "np.ndarray",
    patch_size: int = 256,
    stride: int = 256,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Split feature map and mask into (patch_size×patch_size) patches."""
    import numpy as np
    H, W = feats.shape[:2]
    patches_X, patches_y = [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patches_X.append(feats[r:r+patch_size, c:c+patch_size])
            patches_y.append(mask[r:r+patch_size, c:c+patch_size, None])
    return np.array(patches_X, dtype=np.float32), np.array(patches_y, dtype=np.float32)


def _augment_patches(
    X: "np.ndarray",
    y: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Flip + 90° rotations augmentation without imgaug dependency."""
    import numpy as np
    Xa, ya = [X], [y]
    # Horizontal flip
    Xa.append(X[:, :, ::-1, :])
    ya.append(y[:, :, ::-1, :])
    # Vertical flip
    Xa.append(X[:, ::-1, :, :])
    ya.append(y[:, ::-1, :, :])
    # 90° rotations
    for k in [1, 2, 3]:
        Xa.append(np.rot90(X, k=k, axes=(1, 2)))
        ya.append(np.rot90(y, k=k, axes=(1, 2)))
    return (
        np.concatenate(Xa, axis=0),
        np.concatenate(ya, axis=0),
    )


# ---------------------------------------------------------------------------
# Training one U-Net
# ---------------------------------------------------------------------------

def _train_unet(
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    config: dict,
    loss_name: str,
    ckpt_dir: Path,
) -> tuple:
    import tensorflow as tf

    img_size   = config["img_size"]
    in_channels = config["in_channels"]
    filters    = config.get("filters", 32)
    lr         = config.get("learning_rate", 1e-3)
    epochs     = config.get("epochs", 30)
    batch_size = config.get("batch_size", 8)

    model = _build_unet(img_size, in_channels, filters)

    if loss_name == "dice":
        loss_fn = _dice_loss
    elif loss_name == "bce":
        loss_fn = _bce_dice_loss
    elif loss_name == "tversky":
        alpha = config.get("tversky_alpha", 0.7)
        beta  = config.get("tversky_beta",  0.3)
        loss_fn = _tversky_loss(alpha, beta)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[_dice_coefficient],
    )

    ckpt_path = str(ckpt_dir / f"unet_{loss_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    print(f"\n{'='*60}")
    print(f" Training U-Net  loss={loss_name}  ({model.count_params():,} params)")
    print(f"{'='*60}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(
        ckpt_path,
        custom_objects={
            "_dice_loss": _dice_loss,
            "_bce_dice_loss": _bce_dice_loss,
            "_dice_coefficient": _dice_coefficient,
        },
    )
    return best_model, history


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------

def _ensemble_average(models: list, X: "np.ndarray") -> "np.ndarray":
    import numpy as np
    preds = np.array([m.predict(X, verbose=0) for m in models])
    return preds.mean(axis=0)


def _ensemble_weighted(models: list, X: "np.ndarray", weights: list[float]) -> "np.ndarray":
    import numpy as np
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    preds = np.array([m.predict(X, verbose=0) for m in models])
    return (preds * w[:, None, None, None, None]).sum(axis=0)


def _build_stacking_meta(num_models: int, img_size: int = 256):
    """Tiny Conv2D meta-learner that takes concatenated predictions as input."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = tf.keras.Input(shape=(img_size, img_size, num_models))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(1,  1, activation="sigmoid")(x)
    return Model(inp, x, name="meta_learner")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_history(histories: dict[str, object], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        axes[0].plot(hist.history["loss"],     label=f"{name} train")
        axes[0].plot(hist.history["val_loss"], label=f"{name} val", linestyle="--")
        axes[1].plot(hist.history["dice_coefficient"],     label=f"{name} train")
        axes[1].plot(hist.history["val_dice_coefficient"], label=f"{name} val", linestyle="--")
    for ax, title in zip(axes, ["Loss", "Dice Coefficient"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    plt.tight_layout()
    path = out_dir / "unet_training_history.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history → {path}")


def _plot_predictions(
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    preds_dict: dict[str, "np.ndarray"],
    out_dir: Path,
    n: int = 3,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(preds_dict)
    fig, axes = plt.subplots(n, 2 + n_models, figsize=(4 * (2 + n_models), 4 * n))
    axes = np.array(axes).reshape(n, 2 + n_models)

    for i in range(min(n, len(X_val))):
        axes[i, 0].imshow(X_val[i, :, :, 0], cmap="gray")
        axes[i, 0].set_title("Input (ch0)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y_val[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        for j, (name, pred) in enumerate(preds_dict.items()):
            axes[i, 2 + j].imshow(pred[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2 + j].set_title(name)
            axes[i, 2 + j].axis("off")

    plt.tight_layout()
    path = out_dir / "predictions.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Predictions plot → {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_dice(y_true: "np.ndarray", y_pred: "np.ndarray") -> float:
    import numpy as np
    p = (y_pred > 0.5).astype(np.float32).ravel()
    t = y_true.astype(np.float32).ravel()
    inter = (p * t).sum()
    return float(2.0 * inter / (p.sum() + t.sum() + 1e-6))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_deps()
    args = parse_args()

    import numpy as np
    import pandas as pd
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = Path(args.data or ROOT / config["data_file"])
    if not data_path.exists():
        print(f"\nData file not found: {data_path}")
        print("Place the multispectral tif in lab7/data/forest.tif")
        print("(256×256×27: ch0 = binary mask, ch1–13 = new Sentinel-2, ch14–26 = old)")
        sys.exit(0)

    # Output dirs
    OUTPUTS  = ROOT / "lab7" / "outputs"
    fig_dir  = OUTPUTS / "figures"
    ckpt_dir = OUTPUTS / "checkpoints"
    met_dir  = OUTPUTS / "metrics"
    tab_dir  = OUTPUTS / "tables"
    for d in (fig_dir, ckpt_dir, met_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load tif
    print(f"Loading {data_path} …")
    try:
        import skimage.io as skio
        img = skio.imread(str(data_path))          # (H, W, 27)
    except Exception:
        import cv2
        img = cv2.imread(str(data_path), cv2.IMREAD_UNCHANGED)

    print(f"Image shape: {img.shape}")
    mask = (img[:, :, 0] > 0).astype(np.uint8)    # binary mask from ch0

    feats = prepare_snaps(img)                      # (H, W, 16)
    print(f"Feature shape: {feats.shape}  |  Mask positive fraction: {mask.mean():.3f}")

    img_size = config["img_size"]
    X_all, y_all = _make_patches(feats, mask, patch_size=img_size, stride=img_size)
    print(f"Patches: {len(X_all)}")

    # Train / val split
    n_train = max(1, int(len(X_all) * config.get("train_frac", 0.8)))
    idx = np.random.permutation(len(X_all))
    X_train, y_train = X_all[idx[:n_train]], y_all[idx[:n_train]]
    X_val,   y_val   = X_all[idx[n_train:]], y_all[idx[n_train:]]

    # Augmentation
    if config.get("augmentation", True):
        X_train, y_train = _augment_patches(X_train, y_train)
        print(f"After augmentation — train patches: {len(X_train)}")

    run_unet     = args.exp in ("unet",     "all")
    run_ensemble = args.exp in ("ensemble", "all")

    loss_names: list[str] = config.get("losses", ["dice", "bce", "tversky"])
    trained_models = []
    histories      = {}
    val_dices      = []

    if run_unet:
        for loss_name in loss_names:
            model, history = _train_unet(
                X_train, y_train, X_val, y_val, config, loss_name, ckpt_dir
            )
            trained_models.append(model)
            histories[loss_name] = history

            y_pred = model.predict(X_val, verbose=0)
            d = _compute_dice(y_val, y_pred)
            val_dices.append(d)
            print(f"  {loss_name:12s} val Dice: {d:.4f}")

        _plot_history(histories, fig_dir)

        # Save per-model metrics
        rows = [
            {"loss": ln, "val_dice": d}
            for ln, d in zip(loss_names, val_dices)
        ]
        pd.DataFrame(rows).to_csv(met_dir / "unet_metrics.csv", index=False)

        # Save training history
        hist_rows = []
        for loss_name, hist in histories.items():
            for ep, (lo, vlo, di, vdi) in enumerate(zip(
                hist.history["loss"],
                hist.history["val_loss"],
                hist.history["dice_coefficient"],
                hist.history["val_dice_coefficient"],
            )):
                hist_rows.append({
                    "loss_fn": loss_name, "epoch": ep,
                    "loss": lo, "val_loss": vlo,
                    "dice": di, "val_dice": vdi,
                })
        pd.DataFrame(hist_rows).to_csv(tab_dir / "training_history.csv", index=False)
        print(f"Training history CSV → {tab_dir / 'training_history.csv'}")

    else:
        # Load checkpoints for ensemble-only mode
        import tensorflow as tf
        custom = {
            "_dice_loss": _dice_loss,
            "_bce_dice_loss": _bce_dice_loss,
            "_dice_coefficient": _dice_coefficient,
        }
        for loss_name in loss_names:
            ckpt_path = ckpt_dir / f"unet_{loss_name}.keras"
            if ckpt_path.exists():
                trained_models.append(tf.keras.models.load_model(str(ckpt_path), custom_objects=custom))
                y_pred = trained_models[-1].predict(X_val, verbose=0)
                val_dices.append(_compute_dice(y_val, y_pred))
            else:
                print(f"Checkpoint not found: {ckpt_path}  — skipping this model")

    # -------------------------------------------------------------------
    # Ensembles
    # -------------------------------------------------------------------
    if run_ensemble and len(trained_models) >= 2:
        import tensorflow as tf
        ensemble_types: list[str] = config.get("ensemble_types", ["average", "weighted", "stacking"])
        ens_preds: dict[str, np.ndarray] = {}
        ens_metrics = []

        for ens_type in ensemble_types:
            if ens_type == "average":
                pred = _ensemble_average(trained_models, X_val)
            elif ens_type == "weighted":
                pred = _ensemble_weighted(trained_models, X_val, val_dices if val_dices else [1.0] * len(trained_models))
            elif ens_type == "stacking":
                # Build stacked input: concat model predictions along channel axis
                stacked_inp = np.concatenate(
                    [m.predict(X_val, verbose=0) for m in trained_models], axis=-1
                )  # (N, H, W, n_models)

                meta = _build_stacking_meta(len(trained_models), img_size)
                meta.compile(
                    optimizer=tf.keras.optimizers.Adam(config.get("stacking_lr", 1e-4)),
                    loss=_dice_loss,
                    metrics=[_dice_coefficient],
                )
                # Train on train set
                stacked_train = np.concatenate(
                    [m.predict(X_train, verbose=0) for m in trained_models], axis=-1
                )
                meta.fit(
                    stacked_train, y_train,
                    validation_data=(stacked_inp, y_val),
                    epochs=config.get("stacking_epochs", 10),
                    batch_size=config.get("batch_size", 8),
                    verbose=1,
                )
                pred = meta.predict(stacked_inp, verbose=0)
                meta.save(str(ckpt_dir / "meta_learner.keras"))
            else:
                print(f"Unknown ensemble type: {ens_type}")
                continue

            dice = _compute_dice(y_val, pred)
            print(f"  Ensemble [{ens_type:10s}] val Dice: {dice:.4f}")
            ens_preds[ens_type] = pred
            ens_metrics.append({"ensemble": ens_type, "val_dice": dice})

        # Predictions plot (include best U-Net)
        preds_plot = {}
        if trained_models:
            best_idx = int(np.argmax(val_dices)) if val_dices else 0
            preds_plot[f"UNet-{loss_names[best_idx]}"] = trained_models[best_idx].predict(X_val, verbose=0)
        preds_plot.update(ens_preds)
        if len(X_val) > 0:
            _plot_predictions(X_val, y_val, preds_plot, fig_dir)

        pd.DataFrame(ens_metrics).to_csv(met_dir / "ensemble_metrics.csv", index=False)

        # Combined metrics
        all_rows = []
        for ln, d in zip(loss_names, val_dices):
            all_rows.append({"type": f"unet_{ln}", "val_dice": d})
        for row in ens_metrics:
            all_rows.append({"type": f"ens_{row['ensemble']}", "val_dice": row["val_dice"]})
        pd.DataFrame(all_rows).to_csv(met_dir / "metrics.csv", index=False)
        print(f"Metrics → {met_dir / 'metrics.csv'}")

    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
