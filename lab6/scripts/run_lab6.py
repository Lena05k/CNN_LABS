"""Lab 6 — PointNet for tree-species classification from 3-D point clouds.

This script uses TensorFlow / Keras (the original notebook stack).
Install dependencies:
    pip install tensorflow>=2.10.0 h5py>=3.7.0

Dataset:
    Place the h5 file from the course repository in lab6/data/:
        v1.h5  — odd  variant
        v2.h5  — even variant

Usage:
    python lab6/scripts/run_lab6.py
    python lab6/scripts/run_lab6.py --config lab6/configs/config.yaml
    python lab6/scripts/run_lab6.py --data lab6/data/v2.h5

Outputs:
    lab6/outputs/figures/training_history.png
    lab6/outputs/figures/confusion_matrix.png
    lab6/outputs/figures/point_cloud_samples.png
    lab6/outputs/checkpoints/pointnet_best.keras
    lab6/outputs/metrics/metrics.csv
    lab6/outputs/tables/training_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    for pkg in ("tensorflow", "h5py"):
        try:
            __import__(pkg)
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
    p = argparse.ArgumentParser(description="Lab 6 — PointNet tree classification")
    p.add_argument("--config", default=str(ROOT / "lab6" / "configs" / "config.yaml"))
    p.add_argument("--data",   default=None, help="Override data_file from config.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PointNet building blocks (TF/Keras)
# ---------------------------------------------------------------------------

def _build_pointnet(num_points: int, num_classes: int, dropout: float = 0.3):
    """Build the PointNet classification network using TF/Keras."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np

    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features: int, l2reg: float = 0.001) -> None:
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x   = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

        def get_config(self):
            return {"num_features": self.num_features, "l2reg": self.l2reg}

    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(inputs, num_features):
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg  = OrthogonalRegularizer(num_features)
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    inputs = keras.Input(shape=(num_points, 3))
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(dropout)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_history(history, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["sparse_categorical_accuracy"],     label="train")
    axes[0].plot(history.history["val_sparse_categorical_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history → {path}")


def _plot_confusion_matrix(cm, class_names, out_dir: Path) -> None:
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix → {path}")


def _plot_point_clouds(X, class_map, labels, out_dir: Path, n: int = 6) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4 * n, 4))
    for i in range(min(n, len(X))):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        pts = X[i]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(class_map.get(int(labels[i]), str(labels[i])), fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "point_cloud_samples.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Point cloud samples → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_deps()
    args = parse_args()

    import h5py
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    import yaml

    tf.random.set_seed(42)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = Path(args.data or ROOT / config["data_file"])
    if not data_path.exists():
        print(f"\nData file not found: {data_path}")
        print("Download the h5 file from the course repository:")
        print("  v1.h5 — odd variant  | v2.h5 — even variant")
        print(f"Place it in: {data_path.parent}/")
        sys.exit(0)

    # Create output dirs
    OUTPUTS = ROOT / "lab6" / "outputs"
    fig_dir  = OUTPUTS / "figures"
    ckpt_dir = OUTPUTS / "checkpoints"
    met_dir  = OUTPUTS / "metrics"
    tab_dir  = OUTPUTS / "tables"
    for d in (fig_dir, ckpt_dir, met_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path} …")
    with h5py.File(data_path, "r") as hf:
        X = hf["dataset_X"][:]
        Y_raw = hf["dataset_Y"].asstr()[:]

    classes_cfg: dict = config["classes"]          # {0: "Рябина", 1: "Ель", ...}
    name_to_idx = {v: k for k, v in classes_cfg.items()}
    # Filter to configured classes only
    valid_mask = np.array([y in name_to_idx for y in Y_raw])
    X = X[valid_mask]
    Y_raw = Y_raw[valid_mask]

    Y = np.array([name_to_idx[y] for y in Y_raw])
    # Re-index labels to 0..N-1
    unique_labels = sorted(set(Y.tolist()))
    label_remap   = {old: new for new, old in enumerate(unique_labels)}
    Y = np.array([label_remap[y] for y in Y])
    class_map = {label_remap[k]: v for k, v in classes_cfg.items() if k in label_remap}
    num_classes = len(class_map)
    class_names = [class_map[i] for i in range(num_classes)]
    num_points   = config["num_points"]

    print(f"Samples: {len(X)}  |  Classes: {num_classes}  |  Points/cloud: {num_points}")
    for i, name in class_map.items():
        print(f"  {name}: {(Y == i).sum()} samples")

    # Plot sample point clouds
    _plot_point_clouds(X, class_map, Y, fig_dir)

    # Train / test split via stratified k-fold (use last fold as test)
    skf = StratifiedKFold(n_splits=config.get("n_folds", 5))
    train_idx = test_idx = None
    for train_idx, test_idx in skf.split(X, Y):
        pass  # use last fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # Data augmentation: random jitter + random subsampling
    aug_n   = config.get("aug_n_repeats", 4)
    aug_sig = config.get("aug_sigma", 0.005)
    X_aug_list, y_aug_list = [X_train], [y_train]
    for _ in range(aug_n):
        pts_aug = []
        for cloud in X_train:
            idx = np.random.choice(num_points, size=num_points, replace=True)
            pts_aug.append(cloud[idx])
        pts_aug = np.array(pts_aug) + np.random.normal(0, aug_sig, (len(X_train), num_points, 3))
        X_aug_list.append(pts_aug)
        y_aug_list.append(y_train)
    X_aug = np.concatenate(X_aug_list).astype(np.float32)
    y_aug = np.concatenate(y_aug_list)

    # TF Datasets
    BATCH_SIZE = config.get("batch_size", 64)
    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_aug, y_aug))
                .shuffle(len(X_aug)).batch(BATCH_SIZE))
    test_ds  = (tf.data.Dataset
                .from_tensor_slices((X_test.astype(np.float32), y_test))
                .shuffle(len(X_test)).batch(BATCH_SIZE))

    # Build model
    model = _build_pointnet(num_points, num_classes)
    model.summary()

    lr  = config.get("learning_rate", 0.001)
    opt_name = config.get("optimizer", "sgd").lower()
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["sparse_categorical_accuracy"],
    )

    epochs = config.get("epochs", 30)
    ckpt_path = str(ckpt_dir / "pointnet_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_sparse_categorical_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nTraining for {epochs} epochs …")
    history = model.fit(
        train_ds, epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # Plots
    _plot_history(history, fig_dir)

    # Save training history CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(tab_dir / "training_history.csv")
    print(f"Training history CSV → {tab_dir / 'training_history.csv'}")

    # Load best model for evaluation
    best_model = tf.keras.models.load_model(ckpt_path)

    # Confusion matrix
    data_all = test_ds.take(len(test_ds))
    pts_list, lbl_list = [], []
    for pts, lbl in data_all:
        pts_list.append(pts.numpy())
        lbl_list.append(lbl.numpy())
    pts_all = np.concatenate(pts_list)
    lbl_all = np.concatenate(lbl_list)
    preds    = best_model.predict(pts_all, verbose=0)
    pred_cls = np.argmax(preds, axis=-1)

    cm = confusion_matrix(lbl_all, pred_cls)
    _plot_confusion_matrix(cm, class_names, fig_dir)

    # Final metrics
    accuracy = float((pred_cls == lbl_all).mean())
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    from sklearn.metrics import classification_report
    print(classification_report(lbl_all, pred_cls, target_names=class_names))

    met_df = pd.DataFrame([{
        "test_accuracy": accuracy,
        "optimizer": opt_name,
        "epochs": epochs,
        "num_classes": num_classes,
        "num_points": num_points,
    }])
    met_df.to_csv(met_dir / "metrics.csv", index=False)
    print(f"Metrics → {met_dir / 'metrics.csv'}")
    print(f"\nAll outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
