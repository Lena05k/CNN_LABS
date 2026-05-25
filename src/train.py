"""Training loop with per-epoch metrics tracking (Labs 1–8)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        targets = labels.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(targets)

    return total_loss / len(loader), correct / total


def train(
    model: nn.Module,
    dataloaders: dict,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> dict[str, list[float]]:
    """Run the full training loop and return per-epoch metrics history."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config.get("momentum", 0.0),
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        tr_loss, tr_acc = _run_epoch(
            model, dataloaders["train"], criterion, device, optimizer
        )
        va_loss, va_acc = _run_epoch(
            model, dataloaders["test"], criterion, device
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>4}/{config['epochs']} | "
                f"train  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
                f"val    loss={va_loss:.4f}  acc={va_acc:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Lab 3: extended training loop (label smoothing, weight decay, LR scheduler,
#         optional TensorBoard logging)
# ---------------------------------------------------------------------------

def train_reg(
    model: nn.Module,
    dataloaders: dict,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
    tb_log_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Training loop for Lab 3 with regularisation support.

    Extra config keys recognised (beyond Lab 2):
        label_smoothing  (float, default 0.0)
        weight_decay     (float, default 0.0)
        scheduler_step   (int,   default 240)
        scheduler_gamma  (float, default 0.5)
        tb_log_dir       (str,   optional — override via argument)

    TensorBoard is written if ``tb_log_dir`` is given and ``tensorboard``
    package is installed (imported lazily so the rest works without it).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.0)
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("scheduler_step", 240),
        gamma=config.get("scheduler_gamma", 0.5),
    )

    # TensorBoard (optional)
    writer: Any = None
    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"TensorBoard logs → {tb_log_dir}")
        except ImportError:
            print("tensorboard package not found — skipping TB logging")

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(config["epochs"]):
        # --- train ---
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            targets = labels.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += len(targets)

            if writer is not None:
                step_acc = (preds == targets).float().mean().item()
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                writer.add_scalar("train/step_acc", step_acc, global_step)
            global_step += 1

        tr_loss = epoch_loss / len(dataloaders["train"])
        tr_acc = epoch_correct / epoch_total

        # --- val ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=-1)
                targets = labels.argmax(dim=-1)
                v_correct += (preds == targets).sum().item()
                v_total += len(targets)
        va_loss = v_loss / len(dataloaders["test"])
        va_acc = v_correct / v_total

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

        if writer is not None:
            writer.add_scalar("epoch/train_loss", tr_loss, epoch)
            writer.add_scalar("epoch/val_loss", va_loss, epoch)
            writer.add_scalar("epoch/train_acc", tr_acc, epoch)
            writer.add_scalar("epoch/val_acc", va_acc, epoch)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>4}/{config['epochs']} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val   loss={va_loss:.4f} acc={va_acc:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    if writer is not None:
        writer.close()
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Lab 4: Fine-tuning pretrained models
# ---------------------------------------------------------------------------

def _make_optimizer(
    params: list[nn.Parameter],
    optimizer_name: str,
    config: dict,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.get("lr", 3e-4),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-5),
        )
    elif name == "adam":
        return torch.optim.Adam(
            params,
            lr=config.get("lr_adam", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.get("lr_adam", 3e-4),
            weight_decay=config.get("weight_decay_adamw", 1e-2),
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose sgd | adam | adamw")


def train_finetune(
    model: nn.Module,
    dataloaders: dict,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
    optimizer_name: str = "sgd",
    tb_log_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Fine-tuning loop for Lab 4 (transfer learning).

    Works with whatever parameters currently have ``requires_grad=True``.
    Call ``freeze_backbone`` / ``unfreeze_all`` *before* this function to
    control which weights are updated.

    Extra config keys:
        lr           (float, default 3e-4)
        lr_adam      (float, default 3e-4, used for adam/adamw)
        momentum     (float, default 0.9)
        weight_decay (float, default 1e-5)
        weight_decay_adamw (float, default 1e-2)
        label_smoothing (float, default 0.1)
        scheduler_step  (int,   default 20)
        scheduler_gamma (float, default 0.5)
        epochs       (int, default 60)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train   = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_train:,}  |  Frozen: {n_frozen:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    optimizer = _make_optimizer(trainable, optimizer_name, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("scheduler_step", 20),
        gamma=config.get("scheduler_gamma", 0.5),
    )

    writer: Any = None
    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(log_dir=str(tb_log_dir))
        except ImportError:
            pass

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_acc = 0.0

    for epoch in range(config.get("epochs", 60)):
        # --- train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            preds      = out.argmax(-1)
            targets    = labels.argmax(-1)
            t_correct += (preds == targets).sum().item()
            t_total   += len(targets)
        tr_loss = t_loss  / len(dataloaders["train"])
        tr_acc  = t_correct / t_total

        # --- val ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                out   = model(inputs)
                v_loss   += criterion(out, labels).item()
                preds     = out.argmax(-1)
                targets   = labels.argmax(-1)
                v_correct += (preds == targets).sum().item()
                v_total   += len(targets)
        va_loss = v_loss  / len(dataloaders["test"])
        va_acc  = v_correct / v_total

        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

        if writer:
            writer.add_scalar("epoch/train_loss", tr_loss, epoch)
            writer.add_scalar("epoch/val_loss",   va_loss, epoch)
            writer.add_scalar("epoch/train_acc",  tr_acc,  epoch)
            writer.add_scalar("epoch/val_acc",    va_acc,  epoch)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>3}/{config.get('epochs', 60)} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val   loss={va_loss:.4f} acc={va_acc:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    if writer:
        writer.close()
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Lab 5: Autoencoder training (image AE)
# ---------------------------------------------------------------------------

def _r2_batch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R² for a single batch (tensors on any device, returns float)."""
    yt = y_true.detach().float()
    yp = y_pred.detach().float()
    ss_res = ((yt - yp) ** 2).sum().item()
    ss_tot = ((yt - yt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-9)


def train_ae(
    model: nn.Module,
    dataloaders: dict,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
    tb_log_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Training loop for the image autoencoder (Lab 5, Part 1).

    The model's ``forward()`` must return ``(reconstruction, embedding, normed_input)``.
    Loss  = MSE(reconstruction, normed_input).
    Metric = R² (coefficient of determination, higher is better, 1 = perfect).

    Config keys used:
        lr      (float, default 1e-3)
        epochs  (int,   default 200)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    writer: Any = None
    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(log_dir=str(tb_log_dir))
        except ImportError:
            pass

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []
    }
    best_val_r2 = -float("inf")

    for epoch in range(config.get("epochs", 200)):
        # --- train ---
        model.train()
        t_loss_sum, t_r2_sum, t_batches = 0.0, 0.0, 0
        for inputs, _ in dataloaders["train"]:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            out, _, normed = model(inputs)
            loss = criterion(out, normed)
            loss.backward()
            optimizer.step()
            t_loss_sum += loss.item()
            t_r2_sum   += _r2_batch(normed.cpu(), out.detach().cpu())
            t_batches  += 1
        tr_loss = t_loss_sum / t_batches
        tr_r2   = t_r2_sum  / t_batches

        # --- val ---
        model.eval()
        v_loss_sum, v_r2_sum, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, _ in dataloaders["test"]:
                inputs = inputs.to(device)
                out, _, normed = model(inputs)
                v_loss_sum += criterion(out, normed).item()
                v_r2_sum   += _r2_batch(normed.cpu(), out.cpu())
                v_batches  += 1
        va_loss = v_loss_sum / v_batches
        va_r2   = v_r2_sum  / v_batches

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_r2"].append(tr_r2)
        history["val_r2"].append(va_r2)

        if va_r2 > best_val_r2:
            best_val_r2 = va_r2
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

        if writer:
            writer.add_scalar("epoch/train_loss", tr_loss, epoch)
            writer.add_scalar("epoch/val_loss",   va_loss, epoch)
            writer.add_scalar("epoch/train_r2",   tr_r2,   epoch)
            writer.add_scalar("epoch/val_r2",     va_r2,   epoch)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:>4}/{config.get('epochs', 200)} | "
                f"train loss={tr_loss:.5f} R²={tr_r2:.4f} | "
                f"val   loss={va_loss:.5f} R²={va_r2:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    if writer:
        writer.close()
    print(f"\nBest val R²: {best_val_r2:.4f}")
    return history


# ---------------------------------------------------------------------------
# Lab 8: LSTM forecasting training loop
# ---------------------------------------------------------------------------

def train_lstm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
    tb_log_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Training loop for the LSTM time-series forecaster (Lab 8).

    Config keys used:
        lr            (float, default 0.01)
        epochs        (int,   default 100)
        clip_grad_norm (float, default 1.0)
        scheduler_step (int,   default 15)
        scheduler_gamma (float, default 0.1)

    Returns history dict with keys:
        train_loss, val_loss, train_r2, val_r2
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("scheduler_step", 15),
        gamma=config.get("scheduler_gamma", 0.1),
    )
    clip_norm = config.get("clip_grad_norm", 1.0)

    writer: Any = None
    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(log_dir=str(tb_log_dir))
        except ImportError:
            pass

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []
    }
    best_val_r2 = -float("inf")
    epochs = config.get("epochs", 100)

    for epoch in range(epochs):
        # --- train ---
        model.train()
        t_loss_sum, t_r2_sum, t_batches = 0.0, 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            t_loss_sum += loss.item()
            t_r2_sum   += _r2_batch(y_batch.cpu(), pred.detach().cpu())
            t_batches  += 1
        tr_loss = t_loss_sum / t_batches
        tr_r2   = t_r2_sum  / t_batches

        # --- val ---
        model.eval()
        v_loss_sum, v_r2_sum, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred, _ = model(x_batch)
                v_loss_sum += criterion(pred, y_batch).item()
                v_r2_sum   += _r2_batch(y_batch.cpu(), pred.cpu())
                v_batches  += 1
        va_loss = v_loss_sum / v_batches
        va_r2   = v_r2_sum  / v_batches

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_r2"].append(tr_r2)
        history["val_r2"].append(va_r2)

        if va_r2 > best_val_r2:
            best_val_r2 = va_r2
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

        if writer:
            writer.add_scalar("epoch/train_loss", tr_loss, epoch)
            writer.add_scalar("epoch/val_loss",   va_loss, epoch)
            writer.add_scalar("epoch/train_r2",   tr_r2,   epoch)
            writer.add_scalar("epoch/val_r2",     va_r2,   epoch)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>4}/{epochs} | "
                f"train loss={tr_loss:.5f} R²={tr_r2:.4f} | "
                f"val   loss={va_loss:.5f} R²={va_r2:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    if writer:
        writer.close()
    print(f"\nBest val R²: {best_val_r2:.4f}")
    return history
