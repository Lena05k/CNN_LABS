"""Data loading and preprocessing for CIFAR-100."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T


def load_cifar100(data_dir: str | Path) -> dict:
    """Load raw CIFAR-100 train/test splits from pickle files."""
    raw: dict = {}
    for split in ("train", "test"):
        path = Path(data_dir) / "cifar-100-python" / split
        with open(path, "rb") as f:
            raw[split] = pickle.load(f, encoding="latin1")
    return raw


def make_subset(
    raw_split: dict, classes: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a CIFAR-100 split to the given fine-label class indices.

    Returns images in NHWC format and labels re-indexed to 0..N-1.
    """
    X = raw_split["data"].reshape(-1, 3, 32, 32)
    X = np.transpose(X, [0, 2, 3, 1])          # NCHW → NHWC
    y = np.array(raw_split["fine_labels"])

    mask = np.isin(y, classes)
    X = X[mask].copy()
    y = y[mask].copy()
    y = np.unique(y, return_inverse=True)[1]    # re-label to 0..len(classes)-1
    return X, y


# ---------------------------------------------------------------------------
# Lab 3: augmentation-aware dataset and dataloader builder
# ---------------------------------------------------------------------------

# Pre-defined augmentation pipelines (operate on CHW float tensors in [0, 1])
AUG_TRANSFORMS: dict[str, T.Compose | None] = {
    "none": None,
    "light": T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
    ]),
    "heavy": T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5),
    ]),
}


class CifarDatasetAug(Dataset):
    """CIFAR-100 subset dataset with optional online augmentation.

    Samples are stored as NHWC float tensors in [0, 255].
    When *transform* is given, it is applied in CHW [0, 1] space with
    probability *aug_prob* during training.

    Args:
        X: image tensor, shape (N, H, W, C), dtype float32, values [0, 255].
        y: one-hot label tensor, shape (N, num_classes), dtype float32.
        transform: torchvision transform (CHW float [0,1]) or None.
        aug_prob: probability of applying the transform per sample.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        transform: Callable | None = None,
        aug_prob: float = 0.5,
    ) -> None:
        assert X.shape[0] == y.shape[0], "X and y must have the same length"
        self.X = X
        self.y = y
        self.transform = transform
        self.aug_prob = aug_prob

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]                          # (H, W, C) float [0, 255]
        if self.transform is not None and torch.rand(1).item() < self.aug_prob:
            # CHW float [0, 1] → transform → back to HWC float [0, 255]
            x_chw = x.permute(2, 0, 1) / 255.0  # (C, H, W)
            x_chw = self.transform(x_chw)
            x = x_chw.permute(1, 2, 0) * 255.0  # (H, W, C)
        return x, self.y[idx]


def build_dataloaders_aug(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    batch_size: int = 128,
    aug_variant: str = "light",
    aug_prob: float = 0.5,
) -> dict[str, DataLoader]:
    """Build DataLoaders using CifarDatasetAug with the chosen augmentation.

    Args:
        aug_variant: one of 'none', 'light', 'heavy'.
        aug_prob: probability of applying augmentation per training sample.
    """
    if aug_variant not in AUG_TRANSFORMS:
        raise ValueError(f"aug_variant must be one of {list(AUG_TRANSFORMS)}")

    transform = AUG_TRANSFORMS[aug_variant]
    loaders: dict[str, DataLoader] = {}

    for (X, y), split in zip(
        [(train_X, train_y), (test_X, test_y)], ["train", "test"]
    ):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = F.one_hot(
            torch.tensor(y, dtype=torch.int64), num_classes=num_classes
        ).float()
        # augmentation only for training split
        ds_transform = transform if split == "train" else None
        dataset = CifarDatasetAug(tensor_x, tensor_y, ds_transform, aug_prob)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # safe default for all platforms
        )

    return loaders


def build_dataloaders(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    batch_size: int = 128,
) -> dict[str, DataLoader]:
    """Wrap numpy arrays into PyTorch DataLoaders with one-hot labels."""
    loaders: dict[str, DataLoader] = {}
    for (X, y), split in zip(
        [(train_X, train_y), (test_X, test_y)], ["train", "test"]
    ):
        tensor_x = torch.Tensor(X)
        tensor_y = F.one_hot(
            torch.tensor(y, dtype=torch.int64), num_classes=num_classes
        ).float()
        dataset = TensorDataset(tensor_x, tensor_y)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train")
        )
    return loaders
