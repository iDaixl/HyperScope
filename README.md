# Single-shot high-throughput hyperspectral microscopy for highly-multiplexed fluorescence imaging (HyperScope)

Liheng Bian * , Xilong Dai * ,  Tong Liao, Yinghui Lv, Ruoyao Zhang, Yibo Feng, Lianjie Li, Meng Li, Wenhui Liu, Jun Zhang. (* Equal contributions)

This is the official implementation of "Single-shot high-throughput hyperspectral microscopy for highly-multiplexed fluorescence imaging".  HyperScope reconstructs a 46-band hyperspectral datacube from a single coded 2D measurement using S^2^RNet, with optional RGB visualization from selected spectral bands, followed by spectral unmixing for multiplexed fluorescence imaging. This repository provides the S^2^RNet architecture together with the complete training and inference pipeline.

## 📁Repository structure

```text
HyperScope/
├── architecture/                    # S2RNet architecture and network modules
├── input_img/                       # Place coded measurements here
├── mask_dir/                        # Place the calibrated sensing mask here
├── model_zoo/                       # Place the pretrained checkpoint here
├── spectral_unmixing/               # Endmember spectra for fluorescence unmixing
├── DataProcess.py                   # Model and data-processing utilities
├── getdataset.py                    # Dataset loaders
├── inference.py                     # Reconstruction and export pipeline
├── test_inference_with_unmixing.py  # End-to-end inference example
├── train.py                         # Multi-GPU training entry point
└── train.sh                         # Example training command
```



## 🚀Quick start

### Versions the code has been tested on

- The S^2^RNet has been tested on Windows 10 or Ubuntu 20.04.1. The network has been tested on CUDA 12.4, pytorch 2.4.1, torchvision 0.19.1, python 3.8.20, opencv-python 4.11.0.86, Cupy 12.x.

### Required files

Before inference, add the following files:

```text
Model: ./model_zoo/net.pth
Spectral mask: ./mask_dir/6504pro_10x_flipud.mat
```

The mask MAT file must contain a `mask` array with 46 spectral channels. The checkpoint must be compatible with the S^2^RNet configuration in `architecture/__init__.py`.

### Test HyperScope with real-world data

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

The bundled unmixing example uses `spectral_unmixing/Cell_unmixing.csv` and the component order `Nuclei`, `Lyso`, `Mito`, and `Tubulin`.

### Training

Training samples are HDF5 files containing an `hsi` dataset in channel-first form (`C × H × W`). Place the training and validation files in separate directories, then update the paths in `train.sh` or pass them directly to `train.py`.

Example:

```bash
bash train.sh
```

For multi-GPU training, `train.py` launches one process per ID supplied with `--gpu_id`. Adjust batch size, patch size, and GPU IDs to fit the available hardware.

## 📬Contact

- For questions, please contact: [idaixl@126.com](mailto:idaixl@126.com) or open an issue on this GitHub repository.