# CVST

CVST is a PyTorch implementation of the accepted IEEE TWC paper "Context video semantic transmission with variable length and rate coding over MIMO channels". It combines an image model for the initial frame with motion and context coding for subsequent video frames.

## Project structure

- `train.py` — distributed training and evaluation entry point
- `net/` — CVST, image, channel, encoder, and decoder models
- `data/` — Vimeo and UVG/HEVC dataset loaders and augmentation
- `loss/` — MSE and MS-SSIM distortion losses
- `modules/`, `layers/`, `utils_entropy/` — transform and entropy-coding components
- `DVC/` — optical-flow and motion-estimation utilities
- `*.pth` — pretrained checkpoints stored with Git LFS

## Setup

Install a CUDA-compatible PyTorch environment and the main dependencies: `torchvision`, `mmcv` with compiled ops, `compressai`, `timm`, `einops`, `scipy`, `numpy`, `opencv-python`, `imageio`, `lpips`, and `pytorch-msssim`.

Clone the repository and download the checkpoints:

```bash
git clone https://github.com/xie233333/CVST.git
cd CVST
git lfs pull
```

Update the dataset paths in `train.py`, then provide `H_MIMO8.mat` and `mimo_seq.npy` in the project root. The current configuration expects eight CUDA devices and runs evaluation (`args.training = False`):

```bash
torchrun --nproc_per_node=8 train.py --channel-type MIMO
```

To train the model, set `args.training = True` in the configuration and adjust the dataset, GPU, and optimization settings for your environment.
