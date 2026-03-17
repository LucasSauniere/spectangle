#!/usr/bin/env python
"""
scripts/train.py
-----------------
Command-line training script for spectangle models.

Usage
-----
    python scripts/train.py --config configs/models/unet.yaml
    python scripts/train.py --config configs/models/pinn.yaml --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectangle.data_loaders import SpectangleDataModule
from spectangle.models import PINN, CombinedLoss, SpectralViT, UNet2Dto3D
from spectangle.utils.training import Trainer


def build_model(cfg: dict, scene_shape=None, spectrogram_shape=None) -> torch.nn.Module:
    m = cfg["model"]
    mtype = m["type"]
    n_lam = m.get("n_lambda", 128)
    in_ch = m.get("in_channels", 4)

    # scene_shape (ny, nx) — the unpadded cube size; used for center-cropping.
    # Prefer the value derived from the DataModule geometry; fall back to config.
    if scene_shape is None:
        sim_cfg = cfg.get("simulation", {})
        ny = sim_cfg.get("image_shape_ny", m.get("image_size", 128))
        nx = sim_cfg.get("image_shape_nx", m.get("image_size", 128))
        scene_shape = (ny, nx)

    if mtype == "unet":
        return UNet2Dto3D(
            in_channels=in_ch,
            n_lambda=n_lam,
            base_features=m.get("base_features", 32),
            depth=m.get("depth", 4),
            scene_shape=scene_shape,
        )
    elif mtype == "vit":
        patch_size = m.get("patch_size", 8)
        # ViT needs the padded input dimensions (H_spec, W_spec).
        # If spectrogram_shape is known (from the data module), round up to the
        # nearest multiple of patch_size so the patch grid divides evenly.
        if spectrogram_shape is not None:
            h_spec = ((spectrogram_shape[0] + patch_size - 1) // patch_size) * patch_size
            w_spec = ((spectrogram_shape[1] + patch_size - 1) // patch_size) * patch_size
        else:
            # Fall back: use config image_size (unpadded) — user must set
            # image_size_h/w in the YAML for padded inputs.
            h_spec = m.get("image_size_h", m.get("image_size", 128))
            w_spec = m.get("image_size_w", m.get("image_size", 128))
        return SpectralViT(
            in_channels=in_ch,
            image_size_h=h_spec,
            image_size_w=w_spec,
            patch_size=patch_size,
            embed_dim=m.get("embed_dim", 256),
            depth=m.get("depth", 6),
            n_heads=m.get("n_heads", 8),
            n_lambda=n_lam,
            mlp_ratio=m.get("mlp_ratio", 4.0),
            dropout=m.get("dropout", 0.1),
            scene_shape=scene_shape,
        )
    elif mtype == "pinn":
        return PINN(
            backbone=UNet2Dto3D(
                in_channels=in_ch,
                n_lambda=n_lam,
                base_features=m.get("base_features", 32),
                depth=m.get("depth", 4),
                scene_shape=scene_shape,
            ),
            in_channels=in_ch,
            n_lambda=n_lam,
            lambda_physics=m.get("lambda_physics", 0.1),
            psf_fwhm_pixels=m.get("psf_fwhm_pixels", 1.6),
            image_shape=scene_shape,
        )
    else:
        raise ValueError(f"Unknown model type: {mtype!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML model config path.")
    p.add_argument("--device", default=None, help="e.g. 'cuda:0' or 'cpu'.")
    p.add_argument("--n_epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    l_cfg = cfg.get("loss", {})
    o_cfg = cfg["output"]

    # Data
    dm = SpectangleDataModule(
        h5_path=d_cfg["h5_path"],
        split_ratios=tuple(d_cfg["split_ratios"]),
        batch_size=t_cfg["batch_size"],
        n_channels=cfg["model"].get("in_channels", 4),
        normalise=d_cfg.get("normalise", "per_sample"),
        num_workers=d_cfg.get("num_workers", 4),
        seed=t_cfg.get("seed", 42),
    )
    print(dm)

    # Derive scene geometry from the data module (reads HDF5 metadata).
    # Falls back to config values if the HDF5 was written without metadata.
    scene_shape       = dm.scene_shape        # (ny, nx) or None
    spectrogram_shape = dm.spectrogram_shape  # (ny+2·pad_y, nx+2·pad_x) or None
    if scene_shape is None:
        print("[spectangle] WARNING: scene_shape not found in HDF5 metadata; "
              "falling back to config values.")

    # Override n_lambda from actual data if available
    if dm.n_lambda is not None:
        cfg["model"]["n_lambda"] = dm.n_lambda

    # Model — geometry is read from HDF5 metadata for correctness
    model = build_model(cfg, scene_shape=scene_shape, spectrogram_shape=spectrogram_shape)
    n_params = model.parameter_count()
    print(f"[spectangle] Model: {cfg['model']['type']} | params={n_params:,}")

    # Loss
    physics_mode = cfg["model"]["type"] == "pinn"
    fwd_model = model.physics_model if physics_mode else None
    loss_fn = CombinedLoss(weights=l_cfg, forward_model=fwd_model)

    # Optimiser
    lr = t_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler
    sched_type   = t_cfg.get("lr_scheduler", "reduce_on_plateau")
    n_ep         = args.n_epochs or t_cfg["n_epochs"]
    warmup_steps = t_cfg.get("warmup_epochs", 0)

    if sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=t_cfg.get("lr_patience", 10),
            factor=t_cfg.get("lr_factor", 0.5),
        )
    elif sched_type == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_ep - warmup_steps, 1)
        )
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine
    else:
        scheduler = None

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        physics_mode=physics_mode,
        checkpoint_dir=o_cfg["checkpoint_dir"],
        log_csv=o_cfg.get("log_csv"),
    )

    n_epochs = args.n_epochs or t_cfg["n_epochs"]
    trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        n_epochs=n_epochs,
    )
    print("[spectangle] Training complete.")


if __name__ == "__main__":
    main()
