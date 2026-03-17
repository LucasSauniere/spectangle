spectangle — Spectroscopic Disentangling
========================================

Overview
--------
**spectangle** solves the slitless spectroscopy inverse problem for the Euclid
NISP instrument: reconstruct a deblended 3-D spectral cube (x, y, λ) from the
four K-sequence dispersed spectrograms produced by the RGS000 and RGS180 grisms
with GWA tilts of ±4°.

```
K-sequence: RGS000+0 → RGS180+4 → RGS000−4 → RGS180+0
```

Three deep-learning architectures are implemented and compared:

| Model              | Input  | Notes                                      |
|--------------------|--------|--------------------------------------------|
| `UNet2Dto3D`       | 4 or 5 | 2-D encoder → 3-D decoder via skip conn.   |
| `SpectralViT`      | 4 or 5 | Patch-token attention + spectral decoder   |
| `PINN`             | 4      | U-Net backbone + differentiable fwd model  |

Directory structure
-------------------
```
spectangle/          Python package
  simulations/       Numpy-based simulators (simple, complex, forward)
  models/            PyTorch models (unet, vit, pinn) + losses
  physics/           Dispersion, grism geometry, PSF utilities
  data_loaders/      HDF5 Dataset / DataModule for PyTorch
  utils/             Metrics, visualisation, Trainer

configs/
  models/            YAML model configs (unet.yaml, vit.yaml, pinn.yaml)
  simulations/       YAML simulation configs (simple_mini, complex_euclid)

notebooks/
  miniature_version/
    simulations/
      01_generate_mini_dataset.ipynb
      02_psf_and_sources.ipynb
      03_dispersion_traces.ipynb
    training_testing/
      01_train_unet_on_mini.ipynb
      02_train_vit_on_mini.ipynb
      03_train_pinn_on_mini.ipynb
  euclid_like_version/
    simulations/
      01_generate_euclid_like_dataset.ipynb
      02_psf_and_sources.ipynb
      03_dispersion_traces.ipynb
    training_testing/
      01_train_pinn_on_euclid_like.ipynb
      02_train_unet_on_euclid_like.ipynb
      03_train_vit_on_euclid_like.ipynb

scripts/
  run_simulation.py  CLI dataset generator
  train.py           CLI model trainer

data/
  raw/               HDF5 datasets (gitignored)
  interim/
  processed/
```

Environment setup (micromamba, zsh)
-------------------------------------
```zsh
# Create environment from environment.yml
micromamba create -n spectangle -f environment.yml

# Activate
micromamba activate spectangle

# Install the package in editable mode
pip install -e .
```

Quickstart
----------

### 1 — Generate a miniature dataset (fast, noiseless)
```zsh
python scripts/run_simulation.py \
    --config configs/simulations/simple_mini.yaml \
    --n_samples 100
```
Output: `data/raw/simple_mini_1000s.h5`

### 2 — Generate an Euclid-like dataset (noisy, multi-order)
```zsh
python scripts/run_simulation.py \
    --config configs/simulations/complex_euclid.yaml
```
Output: `data/raw/complex_euclid_200s.h5`

### 3 — Train a U-Net on the miniature dataset
```zsh
python scripts/train.py --config configs/models/unet.yaml
```

### 4 — Train a PINN on the Euclid-like dataset
```zsh
python scripts/train.py --config configs/models/pinn.yaml
```

Simulation module
-----------------
**Simple** (`SimpleSimulator`)
- Noiseless, 1st diffraction order only.
- Stars modelled as blackbody SEDs.
- Spectrograms are 128 pixels long (configurable).
- Ideal for rapid proof-of-concept ML training.

**Complex** (`ComplexSimulator`)
- Realistic NISP noise: Poisson sky background, read noise, dark current.
- Multiple diffraction orders: 0th-order (peanut PSF), 1st, 2nd.
- Configurable SEDs (blackbody or realistic astronomical templates).
- Exposure time parameterised (default: 565 s, matching Euclid NISP standard).

Both simulators save `.h5` (HDF5) files with the following structure:
```
/metadata          — simulation config as HDF5 attributes
/wavelengths       — (n_lambda,) float32
/samples/
    /0000/
        cube               — (n_lambda, ny, nx) float32  ← ground truth
        spectrograms       — (4, H_spec, W_spec) float32 ← network input
        spectrograms_clean — (4, ...) float32            ← complex only
        noise_maps         — (4, ...) float32            ← complex only
        direct_image       — (H_spec, W_spec) float32    ← optional
        source_xs, source_ys, source_temps               ← source catalogue
    /0001/ ...
```

Model input/output
------------------
| Tensor    | Shape                        | Description                  |
|-----------|------------------------------|------------------------------|
| `x`       | `(B, C_in, H_spec, W_spec)`  | Padded K-sequence images     |
| `y`       | `(B, n_lambda, ny, nx)`      | Ground-truth spectral cube   |
| `C_in`    | 4 or 5                       | +1 if direct image is used   |
| `H_spec`  | `ny + 2·pad_y`               | Padded spectrogram height    |
| `W_spec`  | `nx + 2·pad_x`               | Padded spectrogram width     |

Loss functions
--------------
| Loss                   | Key          | Notes                             |
|------------------------|--------------|-----------------------------------|
| MSE reconstruction     | `mse`        | Pixel-level fidelity              |
| SSIM reconstruction    | `ssim`       | Perceptual / structural fidelity  |
| Spectral smoothness    | `spectral`   | 2nd-order finite diff. along λ    |
| Physics residual (PINN)| `physics`    | `||A(ŷ) − x_obs||²`               |

Configure loss weights in the YAML model configs under the `loss:` key.

Physics-informed constraint (PINN)
-----------------------------------
The `DifferentiableForwardModel` inside `PINN` implements the full K-sequence
dispersion + PSF convolution in pure PyTorch so that gradients flow back to
the network weights.  The physics loss

    L_phys = λ_phys · ||A(ŷ) − x_obs||²

forces the predicted cube to be *consistent with the observed spectrograms*
even without ground-truth labels — enabling self-supervised fine-tuning at
inference time.

Notebooks
---------
Each notebook is self-contained and walks through one specific step:

**Miniature (fast, noiseless)**
1. `01_generate_mini_dataset` — Run SimpleSimulator, inspect HDF5 output.
2. `02_psf_and_sources`       — Visualise PSF, SED sampling, scene construction.
3. `03_dispersion_traces`     — Plot K-sequence traces and spectrogram geometry.
4. `01_train_unet_on_mini`    — Train + evaluate U-Net.
5. `02_train_vit_on_mini`     — Train + evaluate ViT + attention map visualisation.
6. `03_train_pinn_on_mini`    — Train PINN, inspect physics residuals, λ-ablation.

**Euclid-like (noisy, multi-order)**
1. `01_generate_euclid_like_dataset` — Run ComplexSimulator with NISP noise.
2. `02_psf_and_sources`              — PSF, multi-order anatomy, SED diversity.
3. `03_dispersion_traces`            — Multi-order traces, pixel-budget analysis.
4. `01_train_pinn_on_euclid_like`    — PINN training on noisy data.
5. `02_train_unet_on_euclid_like`    — U-Net with 5-channel input.
6. `03_train_vit_on_euclid_like`     — ViT + noise-robustness analysis + comparison table.

Contact
-------
Internal research codebase — Euclid NISP spectroscopy group.
For questions open an issue or contact the maintainer.
