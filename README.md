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
- `--rmp_window_size`: local-window Cross VSS window size (default: `8`)
- `--rmp_atrous_rate`: Atrous Cross VSS sampling rate (default: `2`)
- `--use_cluster_gcn`: enable attention-free multi-scale cluster GCN (MS-CGC)
- `--cluster_counts S1 S2 S3 S4`: region-node counts (default: `256 128 64 32`)
- `--cluster_graph_dim`: graph-node feature width (default: `64`)
- `--cluster_iters`: hard-clustering iterations (default: `2`)
- `--cluster_spatial_weight`: coordinate weight used by hard clustering (default: `0.5`)

The current residual multi-path VSS layout uses four complementary paths:
global Cross, shifted local-window Cross, bidirectional diagonal Cross, and
Atrous Cross. Checkpoints produced by the earlier
cross/unidirectional/bidirectional/rotated-cross layout are not structurally
compatible with this version and should not be used to resume an RMP run.
The current `bau_classic_unet4_local_boundary_v1` decoder uses four RMTPB skip features
S1-S4, a separate H/64 bottleneck below S4, and four corresponding BAU stages:
bottleneck+S4, then S3, S2, and S1. There is no parallel Detail Stem.
Checkpoints from earlier decoder layouts are intentionally incompatible.

There is no image-level boundary-prior branch. Each BAU derives its local
boundary gate only from the corresponding encoder skip using Sobel and
multi-scale LoG responses.
The legacy ResNet0/SSA-M skip path has been removed; its checkpoints are no
longer supported by this source tree.

MS-CGC applies independent hard feature/coordinate clustering to encoder
features S1-S4, builds graphs from spatially touching regions, runs two
ordinary normalized-adjacency GCN layers, and broadcasts the region features
back through a bounded residual. It contains no Q/K/V or multi-head attention.
The former GIA multi-head-attention path has been removed. Checkpoints that
contain GIA parameters are intentionally unsupported by this source tree.

Before training the new layout, run its forward/backward smoke test:

```bash
python tools/test_rmp_scans.py
python tools/test_decoder.py
python tools/test_cluster_graph.py
```

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
