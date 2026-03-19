"""
spectangle.utils.training
==========================
Generic PyTorch training loop for spectangle models.

Features
--------
* Works with all three architectures (UNet, ViT, PINN) via a unified interface.
* Supports PINN physics-loss mode via ``physics_mode=True``.
* Logs per-epoch metrics to a CSV file for easy analysis.
* Saves best-validation-loss checkpoints automatically.
* Provides a ``predict`` method for inference on a DataLoader.
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectangle.utils.metrics import cube_metrics


def _probe_conv3d_mps() -> bool:
    """Return True if Conv3d can run on MPS on the current PyTorch build."""
    try:
        x = torch.zeros(1, 1, 2, 2, 2, device="mps")
        conv = nn.Conv3d(1, 1, kernel_size=1, bias=False).to("mps")
        _ = conv(x)
        return True
    except (RuntimeError, NotImplementedError):
        return False


def get_device() -> torch.device:
    """Auto-detect the best available accelerator.

    Priority order: CUDA (NVIDIA / ROCm) → MPS (Apple Silicon) → CPU.

    On Apple Silicon, Conv3d on MPS requires **PyTorch ≥ 2.3**.  If an older
    build is detected, a warning is printed and the function falls back to CPU.
    Upgrade with::

        micromamba install pytorch>=2.3 -c pytorch

    Returns
    -------
    torch.device
        The best device available on the current machine.

    Examples
    --------
    >>> device = get_device()
    >>> print(device)
    device(type='mps')          # on an Apple Silicon Mac with PyTorch ≥ 2.3
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        if _probe_conv3d_mps():
            return torch.device("mps")
        warnings.warn(
            f"[spectangle] MPS is available but Conv3d is not supported on your "
            f"PyTorch build ({torch.__version__}).  PyTorch ≥ 2.3 is required.\n"
            f"Upgrade with:  micromamba install 'pytorch>=2.3' -c pytorch\n"
            f"Falling back to CPU for now.",
            stacklevel=2,
        )
    return torch.device("cpu")


class Trainer:
    """Generic trainer for spectangle 3-D reconstruction models.

    Parameters
    ----------
    model : nn.Module
        The network to train (UNet2Dto3D, SpectralViT, or PINN).
    loss_fn : callable
        Loss function ``(pred, target) → scalar``.
        For PINN: ``(pred, target, observed) → (total, breakdown_dict)``.
    optimizer : Optimizer
    scheduler : _LRScheduler, optional
    device : str or torch.device, optional
        Target device.  When ``None`` (default), ``get_device()`` is called
        automatically: CUDA → MPS (Apple Silicon) → CPU.
    physics_mode : bool
        If True, the trainer expects ``model.forward_with_physics_loss`` and
        a ``CombinedLoss`` that accepts an ``observed`` tensor.
    checkpoint_dir : str or Path
        Directory to save ``.pt`` checkpoint files.
    log_csv : str or Path, optional
        If set, per-epoch metrics are appended to this CSV file.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str | torch.device | None = None,
        physics_mode: bool = False,
        checkpoint_dir: str | Path = "checkpoints",
        log_csv: Optional[str | Path] = None,
    ) -> None:
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        print(f"[spectangle] Using device: {self.device}")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.physics_mode = physics_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = Path(log_csv) if log_csv else None

        self._best_val_loss = float("inf")
        self._history: list[Dict] = []

        # Initialise CSV log
        if self.log_csv:
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    # ------------------------------------------------------------------

    def _step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Single forward + loss for one batch.  Returns (loss, breakdown)."""
        x, y = x.to(self.device), y.to(self.device)

        if self.physics_mode:
            pred, phys_loss, rec_loss = self.model.forward_with_physics_loss(x, y)
            # Use CombinedLoss if passed, otherwise fall back to (rec + phys)
            try:
                total, breakdown = self.loss_fn(pred, y, x[:, :4])
            except TypeError:
                # Simple fallback
                total = rec_loss + self.model.lambda_physics * phys_loss
                breakdown = {"rec": rec_loss.item(), "physics": phys_loss.item()}
        else:
            pred = self.model(x)
            try:
                total, breakdown = self.loss_fn(pred, y)
            except (TypeError, ValueError):
                # Plain scalar loss function
                total = self.loss_fn(pred, y)
                breakdown = {}

        return total, breakdown

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for x, y in loader:
            self.optimizer.zero_grad()
            loss, _ = self._step(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item() * x.shape[0]
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        for x, y in loader:
            loss, _ = self._step(x, y)
            running_loss += loss.item() * x.shape[0]
        return running_loss / len(loader.dataset)

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        verbose: bool = True,
    ) -> list[Dict]:
        """Full training loop.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
        n_epochs : int
        verbose : bool

        Returns
        -------
        history : list of dict
            One dict per epoch with keys ``epoch``, ``train_loss``,
            ``val_loss``, ``lr``.
        """
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                # ReduceLROnPlateau requires the monitored metric; other
                # schedulers (CosineAnnealingLR, StepLR, …) take no argument.
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                ckpt = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    ckpt,
                )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "elapsed_s": elapsed,
            }
            self._history.append(row)

            if self.log_csv:
                with open(self.log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, val_loss, current_lr, elapsed])

            if verbose:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"train={train_loss:.5f} | val={val_loss:.5f} | "
                    f"lr={current_lr:.2e} | {elapsed:.1f}s"
                )

        return self._history

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a DataLoader.

        Returns
        -------
        preds : Tensor, shape (N, n_lambda, H, W)
        targets : Tensor, shape (N, n_lambda, H, W)
        """
        self.model.eval()
        all_preds, all_targets = [], []
        for x, y in loader:
            x = x.to(self.device)
            pred = self.model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y)
        return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)

    def load_best(self) -> None:
        """Load the best checkpoint saved during training."""
        ckpt = self.checkpoint_dir / "best_model.pt"
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        print(f"Loaded best model from epoch {state['epoch']} (val={state['val_loss']:.5f})")
