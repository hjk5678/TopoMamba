# TopoMamba

> Environment prerequisite: this project relies on the VMamba runtime and its selective scan CUDA kernel. Please set up the VMamba environment first: [VMamba](https://github.com/MzeroMiko/VMamba).

TopoMamba is a remote-sensing semantic segmentation project for ISPRS-style datasets such as Potsdam and Vaihingen. The model combines segmentation, boundary guidance, and topology-aware connection prediction.

## Environment

This repository was developed on a CUDA/PyTorch environment with VMamba components available. A recommended setup flow is:

```bash
conda create -n topomamba python=3.9 -y
conda activate topomamba

# Install PyTorch according to your CUDA version.
# Example only:
pip install torch torchvision torchaudio

# Install VMamba and its selective_scan kernel first.
# See: https://github.com/MzeroMiko/VMamba

pip install -r requirements.txt
```

Note: `requirements.txt` is an exported environment snapshot. If local path dependencies such as `selective-scan @ file:///.../VMamba/...` fail on your machine, install the corresponding VMamba kernel from your own VMamba checkout and then rerun the remaining dependency installation.

## Dataset Layout

For Potsdam and Vaihingen, pass `--data_root` to a dataset directory with this structure:

```text
Potsdam/
  Images/
    xxx_RGB.tif
  Labels/
    xxx_label.tif
```

The same layout applies to Vaihingen:

```text
Vaihingen/
  Images/
  Labels/
```

The training script also supports pre-cropped patches with `--pre_cropped` and `--processed_dir`. The expected processed directory is:

```text
processed/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

## Training

Single-GPU training:

```bash
python train.py \
  --dataset potsdam \
  --data_root /path/to/Potsdam \
  --epochs 100 \
  --batch_size 8 \
  --crop_size 512 \
  --save_dir checkpoints
```

Multi-GPU training with PyTorch DDP:

```bash
torchrun --nproc_per_node=4 train.py \
  --dataset potsdam \
  --data_root /path/to/Potsdam \
  --epochs 100 \
  --batch_size 8 \
  --crop_size 512 \
  --save_dir checkpoints
```

Useful options:

- `--dataset`: `potsdam`, `vaihingen`, or `loveda`
- `--pre_cropped`: use offline cropped patches
- `--processed_dir`: path to processed patches
- `--strong_aug`: enable stronger data augmentation
- `--resume`: resume from a checkpoint
- `--resume_model_only`: load model weights only
- `--use_rmp_vss`: enable residual multi-path VSS branch
- `--use_gia`: enable graph interaction attention
- `--skip_mode`: `ssam`, `basic`, or `encoder`

## Evaluation

Evaluate a checkpoint:

```bash
python tools/eval_checkpoint.py \
  --dataset potsdam \
  --checkpoint checkpoints/topomamba_potsdam_best.pth \
  --data_root /path/to/Potsdam
```

Generate confusion-matrix results:

```bash
python tools/eval_confusion_matrix.py \
  --dataset potsdam \
  --checkpoint checkpoints/topomamba_potsdam_best.pth \
  --data_root /path/to/Potsdam \
  --save_dir eval_results
```

## Inference

Run sliding-window inference on one image:

```bash
python inference.py \
  --dataset vaihingen \
  --image /path/to/image.tif \
  --checkpoint checkpoints/topomamba_vaihingen_best.pth \
  --output_dir output \
  --crop_size 512 \
  --stride 256
```

If `--image` is not provided, `inference.py` tries to select an image from `--data_root`.

## Project Structure

```text
TopoMamba/
  data/                 Dataset loading and transforms
  models/               TopoMamba model, backbone, encoder, decoder modules
  tools/                Evaluation and preprocessing scripts
  utils/                Losses and topology configuration
  train.py              Training entry point
  inference.py          Sliding-window inference entry point
  requirements.txt      Python dependency snapshot
```

## Notes

- Checkpoints, training outputs, caches, and local IDE files are ignored by `.gitignore`.
- The project contains some default paths from the original development machine. Prefer passing explicit paths such as `--data_root`, `--processed_dir`, and `--checkpoint` when running on a new machine.
- VMamba and `selective_scan` must match your CUDA/PyTorch setup, so follow the VMamba installation instructions carefully.
