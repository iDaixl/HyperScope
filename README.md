# HyperScope

Deep-learning-based hyperspectral reconstruction and spectral unmixing for multi-fluorophore microscopy.

HyperScope reconstructs a 46-band hyperspectral image from a coded 2D measurement with S2RNet, renders selected spectral bands as an RGB image, and optionally performs endmember-based fluorescence unmixing. This repository contains the model architecture and the complete training and inference pipeline.

> This repository is prepared as a research-code release. The pretrained checkpoint and calibrated sensing mask are experiment-specific and are not included in Git.

## Highlights

- 46-band hyperspectral reconstruction with the S2RNet architecture.
- GPU-accelerated inference and spectral unmixing.
- Multi-GPU distributed training with PyTorch DDP.
- TIFF, RGB, per-band, and unmixed-component export.
- Support for 2048 × 2048 coded measurements in the example pipeline.

## Repository structure

```text
HyperScope/
├── architecture/                    # S2RNet architecture and network modules
├── input_img/                       # Place coded measurements here
├── mask_dir/                        # Place the calibrated sensing mask here
├── model_zoo/                       # Place the pretrained checkpoint here
├── spectral_unmixing/               # Endmember spectra for fluorescence unmixing
├── DataProcess.py                   # Forward model and data-processing utilities
├── getdataset.py                    # HDF5 training/validation dataset loaders
├── inference.py                     # Reconstruction and export pipeline
├── test_inference_with_unmixing.py  # End-to-end inference example
├── train.py                         # Multi-GPU training entry point
└── train.sh                         # Example training command
```

## Requirements

The code is intended for a CUDA-capable Linux workstation and has also been syntax-checked on Windows. A typical environment is:

- Python 3.9+
- PyTorch 2.x
- CUDA 12.x
- NVIDIA GPU with sufficient memory for 2048 × 2048 inference

Create an environment and install the Python dependencies:

```bash
conda create -n hyperscope python=3.9 -y
conda activate hyperscope

# Install the PyTorch build matching your CUDA driver first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

If your system uses a CUDA version other than 12.x, replace `cupy-cuda12x` with the matching CuPy package.

## Required files

Before inference, add the following experiment-specific files:

```text
model_zoo/net.pth
mask_dir/6504pro_10x_flipud.mat
```

The mask MAT file must contain a `mask` array with 46 spectral channels. The checkpoint must be compatible with the S2RNet configuration in `architecture/__init__.py`.

## Quick start

1. Clone the repository and install the dependencies.
2. Put the pretrained checkpoint and calibrated mask at the paths shown above.
3. Put one or more coded TIFF measurements in `input_img/`.
4. Run the end-to-end example:

```bash
python test_inference_with_unmixing.py
```

The script reads TIFF images from `input_img/` and writes results to `input_img/result_wt_unmixing/`:

- `rgb/`: RGB composite and selected spectral bands;
- `unmix/`: fluorescence abundance maps and, when enabled, the reconstructed hyperspectral TIFF.

The bundled unmixing example uses `spectral_unmixing/Cell_unmixing.csv` and the component order `Nuclei`, `Lyso`, `Mito`, and `Tublin` (the spelling is retained for compatibility with the current data file).

## Training

Training samples are HDF5 files containing an `hsi` dataset in channel-first form (`C × H × W`). Place the training and validation files in separate directories, then update the paths in `train.sh` or pass them directly to `train.py`.

Example:

```bash
bash train.sh
```

For multi-GPU training, `train.py` launches one process per ID supplied with `--gpu_id`. Adjust batch size, patch size, and GPU IDs to fit the available hardware.

## Notes

- The repository does not include large training datasets, pretrained weights, or calibrated masks.
- Raw measurements and generated result images are intentionally excluded to avoid publishing experiment data.
- Inference currently targets CUDA devices; CPU-only execution is not part of the tested workflow.
- Generated results and checkpoints are excluded by `.gitignore` to keep the repository lightweight.

## Acknowledgement

The organization of this research-code release was inspired by the public [Hypervision](https://github.com/bianlab/Hypervision) repository.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by the repository owner.
