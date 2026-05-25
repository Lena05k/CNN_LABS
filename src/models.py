"""Model definitions for Labs 1–8."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CIFAR100_MEAN = (0.5074, 0.4867, 0.4411)
_CIFAR100_STD = (0.2011, 0.1987, 0.2025)


class Normalize(nn.Module):
    """Per-channel normalization that flattens NHWC → (N, 3072).

    Used by MLP (Lab 1). Buffers move to GPU automatically with model.to(device).
    """

    def __init__(
        self,
        mean: tuple[float, ...] = _CIFAR100_MEAN,
        std: tuple[float, ...] = _CIFAR100_STD,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = (x - self.mean) / self.std
        return torch.flatten(x, start_dim=1)


class NormalizeCNN(nn.Module):
    """Per-channel normalization for CNN: NHWC uint8 → NCHW float.

    Used by CNN (Lab 2). Mean/std registered as (1,C,1,1) buffers for
    broadcast over spatial dims.
    """

    def __init__(
        self,
        mean: tuple[float, ...] = _CIFAR100_MEAN,
        std: tuple[float, ...] = _CIFAR100_STD,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std


class Cifar100CNN(nn.Module):
    """Two-layer CNN for CIFAR-100 subset classification (Lab 2).

    All three pooling variants produce identical output shape (N, hidden*8)
    before the final Linear, so weights are comparable across runs.

    pool_type:
        'avg'    — Conv(stride=4) → Conv → AvgPool2d(4)          [baseline]
        'max'    — Conv(stride=1) → MaxPool2d(4) → Conv → MaxPool2d(4)
        'stride' — Conv(stride=4) → Conv(stride=4)               [no pool layer]
    """

    POOL_TYPES = ("avg", "max", "stride")

    def __init__(
        self,
        hidden_size: int = 32,
        num_classes: int = 3,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f"pool_type must be one of {self.POOL_TYPES}")

        layers: list[nn.Module] = [NormalizeCNN()]

        if pool_type == "avg":
            layers += [
                nn.Conv2d(3, hidden_size, kernel_size=5, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(4),
            ]
        elif pool_type == "max":
            layers += [
                nn.Conv2d(3, hidden_size, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),
            ]
        else:  # stride
            layers += [
                nn.Conv2d(3, hidden_size, kernel_size=5, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=4, padding=1),
                nn.ReLU(),
            ]

        # All variants produce (N, hidden*2, 2, 2) → flatten → hidden*8
        layers += [
            nn.Flatten(),
            nn.Linear(hidden_size * 8, num_classes),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Cifar100CNNReg(nn.Module):
    """CNN with Dropout2d regularization for Lab 3.

    Architecture (hidden_size=64, as in Lab 3 notebook):
        NormalizeCNN
        → Conv2d(3, H, 3, stride=4)  → ReLU → Dropout2d(dropout_p)
        → Conv2d(H, H*2, 3, stride=1, padding=1) → ReLU
        → AvgPool2d(4) → Dropout2d(min(dropout_p + 0.1, 0.9))
        → Flatten → Linear(H*8, num_classes)

    Spatial trace (input 32×32):
        Conv1 stride=4: floor((32-3)/4)+1 = 8  → (H, 8, 8)
        Conv2 pad=1   : (8-3+2)/1+1    = 8  → (H*2, 8, 8)
        AvgPool4      : 8//4            = 2  → (H*2, 2, 2)
        Flatten       : H*2*4 = H*8
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_classes: int = 3,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        p2 = min(dropout_p + 0.1, 0.9)
        self.seq = nn.Sequential(
            NormalizeCNN(),
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Dropout2d(p=p2),
            nn.Flatten(),
            nn.Linear(hidden_size * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Cifar100MLP(nn.Module):
    """Fully-connected MLP for CIFAR-100 subset classification (Lab 1 Part 3).

    Architecture:
        Normalize → Linear(3072, hidden) → ReLU → Linear(hidden, num_classes)
    """

    def __init__(self, hidden_size: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.norm = Normalize()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(x))


# ---------------------------------------------------------------------------
# Lab 4: Transfer Learning
# ---------------------------------------------------------------------------

#: Supported pretrained model names → (hub_model_id, head_param_pattern)
_PRETRAINED_REGISTRY: dict[str, tuple[str, str]] = {
    "resnet20":    ("cifar100_resnet20",          "1.fc"),
    "mobilenetv2": ("cifar100_mobilenetv2_x0_5",  "1.classifier.1"),
}


def build_pretrained_model(
    model_name: str,
    num_classes: int,
) -> tuple[nn.Sequential, str]:
    """Load a CIFAR-100 pretrained model from torch.hub and replace its head.

    The model is wrapped as ``nn.Sequential(NormalizeCNN(), base_model)``
    so it accepts raw NHWC uint8 inputs (same as Labs 1–3).

    Args:
        model_name: ``'resnet20'`` (even variant) or ``'mobilenetv2'`` (odd variant).
        num_classes: number of output classes.

    Returns:
        model       — wrapped Sequential ready for training.
        head_pattern — substring of parameter names that belong to the new head;
                       use this to freeze/unfreeze selectively.

    Raises:
        RuntimeError if the hub download fails (no internet / rate limit).
    """
    if model_name not in _PRETRAINED_REGISTRY:
        raise ValueError(
            f"model_name must be one of {list(_PRETRAINED_REGISTRY)}, got '{model_name}'"
        )

    hub_id, head_pattern = _PRETRAINED_REGISTRY[model_name]
    base = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_id,
        pretrained=True,
        verbose=False,
    )

    if model_name == "resnet20":
        in_features = base.fc.in_features           # 64
        base.fc = nn.Linear(in_features, num_classes)
    else:  # mobilenetv2
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)

    model = nn.Sequential(NormalizeCNN(), base)
    return model, head_pattern


def freeze_backbone(model: nn.Sequential, head_pattern: str) -> list[nn.Parameter]:
    """Freeze all parameters except those whose name contains *head_pattern*.

    Returns:
        List of trainable (unfrozen) parameters for the optimizer.
    """
    for param in model.parameters():
        param.requires_grad = False
    trainable = []
    for name, param in model.named_parameters():
        if head_pattern in name:
            param.requires_grad = True
            trainable.append(param)
    return trainable


def unfreeze_all(model: nn.Module) -> list[nn.Parameter]:
    """Unfreeze all parameters and return them."""
    for param in model.parameters():
        param.requires_grad = True
    return list(model.parameters())


# ---------------------------------------------------------------------------
# Lab 5 Part 1: Image Autoencoder
# ---------------------------------------------------------------------------

class Cifar100AE(nn.Module):
    """Fully-connected Autoencoder for CIFAR-100 image embedding (Lab 5, Part 1).

    Architecture (hidden_size H=512):
        Normalize(NHWC→flat 3072)
        Encoder: Linear(3072,H) → ELU → Linear(H,H//2) → ELU → Linear(H//2,H//8) → Tanh
        Decoder: Linear(H//8,H//2) → ELU → Linear(H//2,H) → ELU → Linear(H,3072)

    forward() returns (reconstruction, embedding, normalized_input) — the
    training loop uses MSE(reconstruction, normalized_input) as the loss.
    """

    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        H = hidden_size
        self.norm = Normalize()                   # NHWC uint8 → flat float norm.
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, H), nn.ELU(),
            nn.Linear(H, H // 2),      nn.ELU(),
            nn.Linear(H // 2, H // 8), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(H // 8, H // 2), nn.ELU(),
            nn.Linear(H // 2, H),      nn.ELU(),
            nn.Linear(H, 32 * 32 * 3),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed   = self.norm(x)                   # (N, 3072)
        encoded  = self.encoder(normed)            # (N, H//8)
        decoded  = self.decoder(encoded)           # (N, 3072)
        return decoded, encoded, normed


# ---------------------------------------------------------------------------
# Lab 5 Part 2: Audio Denoising Autoencoder
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class DenoisingAE(nn.Module):
    """1-D Conv Autoencoder for audio spectrogram denoising (Lab 5, Part 2).

    Input/output shape: (batch, 2, F) where F = n_fft//2+1.
    The two channels are the real and imaginary parts of the STFT.

    With n_fft=512 → F=257:
        Encoder:
            Conv1d(2,   256, 3, stride=2, pad=1) → (B, 256, 129)
            Conv1d(256, 512, 3, stride=2, pad=1) → (B, 512, 65)
            Conv1d(512,1024, 3, stride=2, pad=1) → (B,1024, 33)
        Decoder:
            ConvTranspose1d(1024, 512, 3, stride=2, pad=1) → (B, 512, 65)
            ConvTranspose1d( 512, 256, 3, stride=2, pad=1) → (B, 256,129)
            ConvTranspose1d( 256,   2, 3, stride=2, pad=1) → (B,   2,257)
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2,    256,  kernel_size=3, stride=2, padding=1), Mish(),
            nn.Conv1d(256,  512,  kernel_size=3, stride=2, padding=1), Mish(),
            nn.Conv1d(512,  1024, kernel_size=3, stride=2, padding=1), Mish(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1), Mish(),
            nn.ConvTranspose1d(512,  256, kernel_size=3, stride=2, padding=1), Mish(),
            nn.ConvTranspose1d(256,  2,   kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Lab 8: LSTM for time-series forecasting
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Bidirectional LSTM for univariate time-series forecasting (Lab 8).

    Architecture:
        LSTM(input_size=1, hidden_size, num_layers, bidirectional=True)
        → Dropout(p)
        → Linear(2*hidden_size → num_features)

    Input  : (batch, seq_len)          — sliding-window targets
    Output : (batch, num_features)     — next *num_features* values
    Hidden state is managed externally (pass ``hidden`` to forward or None
    for zero initialisation).
    """

    def __init__(
        self,
        num_features: int = 1,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, num_features)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch, seq_len) → (batch, seq_len, 1)
        out, hidden = self.lstm(x.unsqueeze(-1), hidden)
        # Use only the last time-step output
        pred = self.fc(self.dropout(out[:, -1, :]))   # (batch, num_features)
        return pred, hidden
