import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import ISPRS_PALETTE, Normalize, ToTensor
from models.build_model import (
    CLUSTER_GRAPH_LAYOUT,
    DECODER_LAYOUT,
    RMP_SCAN_LAYOUT,
    TopoMamba,
)
from utils.topology_configs import get_topology_pairs


DEFAULT_IMAGE_DIRS = {
    "potsdam": "/data/BUAS/HJK/TopoMamba/data/Potsdam/Images",
    "vaihingen": "/data/BUAS/HJK/TopoMamba/data/Vaihingen/Images",
}


def label_to_color(label):
    color_map = np.zeros(
        (label.shape[0], label.shape[1], 3),
        dtype=np.uint8,
    )

    for rgb, class_id in ISPRS_PALETTE.items():
        color_map[label == class_id] = list(rgb)

    return color_map


def unpack_seg_logits(outputs):
    if isinstance(outputs, dict):
        if "seg_logits" not in outputs:
            raise KeyError("Model output dict does not contain 'seg_logits'.")
        return outputs["seg_logits"]

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise ValueError("Model output tuple/list is empty.")
        return outputs[0]

    if torch.is_tensor(outputs):
        return outputs

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def get_default_checkpoint(dataset):
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    candidates = [
        os.path.join(checkpoint_dir, f"topomamba_{dataset}_latest.pth"),
        os.path.join(checkpoint_dir, f"topomamba_{dataset}_best.pth"),
        os.path.join(checkpoint_dir, f"topomamba_{dataset}_epoch100.pth"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        topology_pairs = checkpoint.get("topology_pairs")
        checkpoint_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        topology_pairs = None
        checkpoint_args = {}

    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    return state_dict, topology_pairs, checkpoint_args


def get_image_path(args):
    if args.image is not None:
        return args.image

    data_root = args.data_root or DEFAULT_IMAGE_DIRS[args.dataset]
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Image directory not found: {data_root}")

    valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    all_imgs = sorted(
        f for f in os.listdir(data_root)
        if f.lower().endswith(valid_exts)
    )

    if not all_imgs:
        raise FileNotFoundError(f"No image files found in: {data_root}")

    img_path = os.path.join(data_root, all_imgs[0])
    print(f"Auto-selected image: {img_path}")

    return img_path


def sliding_window_inference(
    model,
    img,
    crop_size=1024,
    stride=512,
    num_classes=6,
    device="cuda",
):
    model.eval()

    orig_h, orig_w = img.shape[:2]
    h, w = orig_h, orig_w

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT,
        )
        h, w = img.shape[:2]

    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    y_starts = [y for y in range(0, h, stride) if y + crop_size <= h]
    x_starts = [x for x in range(0, w, stride) if x + crop_size <= w]

    if not y_starts or y_starts[-1] < h - crop_size:
        y_starts.append(h - crop_size)
    if not x_starts or x_starts[-1] < w - crop_size:
        x_starts.append(w - crop_size)

    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    to_tensor = ToTensor()
    dummy_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)

    total_windows = len(y_starts) * len(x_starts)
    pbar = tqdm(
        total=total_windows,
        desc="Sliding-window inference",
        leave=True,
    )

    with torch.inference_mode():
        for y in y_starts:
            for x in x_starts:
                crop = img[y:y + crop_size, x:x + crop_size, :]

                crop_norm, _ = normalize(crop, dummy_mask)
                crop_tensor, _ = to_tensor(crop_norm, dummy_mask)
                crop_tensor = crop_tensor.unsqueeze(0).to(device)

                outputs = model(crop_tensor)
                seg_logits = unpack_seg_logits(outputs)

                if seg_logits.shape[-2:] != (crop_size, crop_size):
                    seg_logits = F.interpolate(
                        seg_logits,
                        size=(crop_size, crop_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                probs = (
                    torch.softmax(seg_logits, dim=1)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )

                prob_map[y:y + crop_size, x:x + crop_size] += probs
                count_map[y:y + crop_size, x:x + crop_size] += 1.0
                pbar.update(1)

    pbar.close()

    count_map = np.maximum(count_map, 1.0)
    prob_map /= count_map[..., np.newaxis]

    pred = np.argmax(prob_map, axis=-1).astype(np.uint8)
    pred = pred[:orig_h, :orig_w]

    return pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="vaihingen",
        choices=["potsdam", "vaihingen"],
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Image directory used when --image is not provided.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 6

    checkpoint_path = args.checkpoint or get_default_checkpoint(args.dataset)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    try:
        img_path = get_image_path(args)
    except FileNotFoundError as exc:
        print(exc)
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Failed to read image: {img_path}")
        return

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Image: {img_path}")
    print(f"Image shape: {img.shape}")
    print(f"Checkpoint: {checkpoint_path}")

    state_dict, checkpoint_topology_pairs, checkpoint_args = load_checkpoint(
        checkpoint_path
    )

    if checkpoint_args.get("use_gia", False) or any(
        key.replace("module.", "").startswith("graph_interaction.")
        for key in state_dict
    ):
        raise RuntimeError(
            "This checkpoint contains the removed GIA attention path and is "
            "not compatible with the attention-free MS-CGC model."
        )

    if checkpoint_args.get("use_rmp_vss", False):
        checkpoint_layout = checkpoint_args.get("rmp_scan_layout")
        if checkpoint_layout != RMP_SCAN_LAYOUT:
            raise RuntimeError(
                "Checkpoint multi-path layout is incompatible with the current "
                f"model: checkpoint={checkpoint_layout!r}, "
                f"current={RMP_SCAN_LAYOUT!r}."
            )

    checkpoint_decoder_layout = checkpoint_args.get("decoder_layout")
    if checkpoint_decoder_layout != DECODER_LAYOUT:
        raise RuntimeError(
            "Checkpoint decoder layout is incompatible with the current "
            f"model: checkpoint={checkpoint_decoder_layout!r}, "
            f"current={DECODER_LAYOUT!r}."
        )

    if checkpoint_args.get("use_cluster_gcn", False):
        checkpoint_cluster_layout = checkpoint_args.get("cluster_graph_layout")
        if checkpoint_cluster_layout != CLUSTER_GRAPH_LAYOUT:
            raise RuntimeError(
                "Checkpoint cluster graph layout is incompatible with the "
                f"current model: checkpoint={checkpoint_cluster_layout!r}, "
                f"current={CLUSTER_GRAPH_LAYOUT!r}."
            )

    topology_pairs = checkpoint_topology_pairs or get_topology_pairs(args.dataset)

    model = TopoMamba(
        num_classes=num_classes,
        topology_pairs=topology_pairs,
        cnn_pretrained=False,
        use_rmp_vss=checkpoint_args.get("use_rmp_vss", False),
        rmp_num_paths=checkpoint_args.get("rmp_num_paths", 4),
        rmp_window_size=checkpoint_args.get("rmp_window_size", 8),
        rmp_atrous_rate=checkpoint_args.get("rmp_atrous_rate", 2),
        use_cluster_gcn=checkpoint_args.get("use_cluster_gcn", False),
        cluster_counts=checkpoint_args.get(
            "cluster_counts",
            [256, 128, 64, 32],
        ),
        cluster_graph_dim=checkpoint_args.get("cluster_graph_dim", 64),
        cluster_iters=checkpoint_args.get("cluster_iters", 2),
        cluster_spatial_weight=checkpoint_args.get(
            "cluster_spatial_weight",
            0.5,
        ),
    )

    load_info = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_info, "missing_keys", [])
    unexpected = getattr(load_info, "unexpected_keys", [])

    if missing:
        print(f"Missing keys: {len(missing)}")
        for key in missing[:20]:
            print(f"  {key}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in unexpected[:20]:
            print(f"  {key}")

    model.to(device)
    model.eval()
    print(f"Model loaded. topology channels: {len(topology_pairs)}")
    print(
        "Architecture flags: "
        f"use_rmp_vss={checkpoint_args.get('use_rmp_vss', False)}, "
        f"use_cluster_gcn={checkpoint_args.get('use_cluster_gcn', False)}"
    )
    print(f"Decoder layout: {checkpoint_args.get('decoder_layout')}")
    if checkpoint_args.get("use_cluster_gcn", False):
        print(
            "Cluster graph: "
            f"{checkpoint_args.get('cluster_graph_layout')} | "
            f"counts={checkpoint_args.get('cluster_counts')} | "
            f"dim={checkpoint_args.get('cluster_graph_dim', 64)}"
        )
    if checkpoint_args.get("use_rmp_vss", False):
        print(
            "RMP scan layout: "
            f"{checkpoint_args.get('rmp_scan_layout')} | "
            f"window={checkpoint_args.get('rmp_window_size', 8)} | "
            f"atrous_rate={checkpoint_args.get('rmp_atrous_rate', 2)}"
        )

    pred = sliding_window_inference(
        model,
        img,
        crop_size=args.crop_size,
        stride=args.stride,
        num_classes=num_classes,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    label_path = os.path.join(args.output_dir, f"{base_name}_label.png")
    cv2.imwrite(label_path, pred)
    print(f"Saved label map: {label_path}")

    color_pred = label_to_color(pred)
    color_path = os.path.join(args.output_dir, f"{base_name}_color.png")
    cv2.imwrite(color_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
    print(f"Saved color map: {color_path}")

    print("Inference finished.")


if __name__ == "__main__":
    main()
