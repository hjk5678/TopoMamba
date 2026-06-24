import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/data/BUAS/HJK/TopoMamba")

from data.dataset import ISPRSDataset, build_dataset, get_transform
from models.build_model import TopoMamba
from utils.losses import TopoMambaLoss
from utils.topology_configs import get_topology_pairs


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["seg_logits"], outputs["conn_logits"], outputs["edge_preds"]

    if isinstance(outputs, (tuple, list)):
        return outputs[0], outputs[1], outputs[2]

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def sanitize_mask(mask, num_classes, ignore_index=255):
    invalid = (mask != ignore_index) & ((mask < 0) | (mask >= num_classes))
    mask[invalid] = ignore_index
    return mask


def compute_class_stats(pred, target, num_classes, ignore_index=255):
    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    tp = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)

    for c in range(num_classes):
        tp[c] = ((pred == c) & (target == c)).sum()
        fp[c] = ((pred == c) & (target != c)).sum()
        fn[c] = ((pred != c) & (target == c)).sum()

    return tp, fp, fn


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index=255):
    model.eval()

    total_loss = 0.0
    total_seen = 0

    tp_all = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp_all = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn_all = torch.zeros(num_classes, dtype=torch.float64, device=device)

    correct_all = torch.tensor(0.0, dtype=torch.float64, device=device)
    valid_all = torch.tensor(0.0, dtype=torch.float64, device=device)

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.long().to(device, non_blocking=True)
        mask = sanitize_mask(mask, num_classes, ignore_index=ignore_index)

        with torch.inference_mode():
            outputs = model(img)
            seg_logits, conn_logits, edge_preds = unpack_outputs(outputs)

            loss_dict = criterion(
                seg_logits,
                mask,
                conn_logits,
                edge_preds
            )

            loss = loss_dict["total_loss"]

        if seg_logits.shape[-2:] != mask.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        pred = seg_logits.argmax(dim=1)

        valid = mask != ignore_index

        correct_all += ((pred == mask) & valid).sum()
        valid_all += valid.sum()

        tp, fp, fn = compute_class_stats(
            pred,
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        tp_all += tp
        fp_all += fp
        fn_all += fn

        bs = img.size(0)
        total_loss += loss.detach().item() * bs
        total_seen += bs

        pbar.set_postfix({
            "loss": f"{loss.detach().item():.4f}"
        })

    avg_loss = total_loss / max(total_seen, 1)

    oa = (correct_all / valid_all).item() if valid_all.item() > 0 else 0.0

    ious = []
    accs = []
    f1s = []

    for c in range(num_classes):
        tp = tp_all[c]
        fp = fp_all[c]
        fn = fn_all[c]

        union = tp + fp + fn
        if union > 0:
            iou = tp / union
            ious.append(iou.item())
        else:
            ious.append(float("nan"))

        denom_acc = tp + fn
        if denom_acc > 0:
            acc = tp / denom_acc
            accs.append(acc.item())
        else:
            accs.append(float("nan"))

        denom_p = tp + fp
        denom_r = tp + fn

        prec = tp / denom_p if denom_p > 0 else torch.tensor(0.0, device=device)
        rec = tp / denom_r if denom_r > 0 else torch.tensor(0.0, device=device)

        if (prec + rec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
            f1s.append(f1.item())
        else:
            f1s.append(float("nan"))

    miou = float(np.nanmean(ious))
    macc = float(np.nanmean(accs))
    mf1 = float(np.nanmean(f1s))

    return {
        "loss": avg_loss,
        "oa": oa,
        "miou": miou,
        "macc": macc,
        "mf1": mf1,
        "ious": ious,
        "accs": accs,
        "f1s": f1s,
    }


def build_val_dataset(args, num_classes):
    base = "/data/BUAS/HJK/TopoMamba/data"

    if args.dataset == "potsdam":
        root = args.data_root if args.data_root else os.path.join(base, "Potsdam")
    elif args.dataset == "vaihingen":
        root = args.data_root if args.data_root else os.path.join(base, "Vaihingen")
    elif args.dataset == "loveda":
        root = args.data_root if args.data_root else os.path.join(base, "Love DA")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.dataset in ["potsdam", "vaihingen"]:
        img_dir = os.path.join(root, "Images")
        label_dir = os.path.join(root, "Labels")

        if args.pre_cropped:
            if args.processed_dir is None:
                processed_dir = os.path.join(root, "processed", "val")
            else:
                processed_dir = os.path.join(args.processed_dir, "val")

            if not os.path.exists(processed_dir):
                raise FileNotFoundError(f"验证预切块目录不存在: {processed_dir}")

            val_set = ISPRSDataset(
                img_dir=img_dir,
                label_dir=label_dir,
                split="val",
                crop_size=args.crop_size,
                transform=get_transform("val"),
                pre_cropped=True,
                processed_dir=processed_dir
            )
        else:
            val_set = ISPRSDataset(
                img_dir=img_dir,
                label_dir=label_dir,
                split="val",
                crop_size=args.crop_size,
                transform=get_transform("val"),
                pre_cropped=False
            )
    else:
        val_set = build_dataset(
            "loveda",
            root,
            split="val",
            pre_cropped=args.pre_cropped,
            processed_dir=args.processed_dir
        )

    return val_set, root


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="potsdam",
                        choices=["potsdam", "vaihingen", "loveda"])
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)

    parser.add_argument("--pre_cropped", action="store_true", default=False)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.05)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset in ["potsdam", "vaihingen"]:
        num_classes = 6
        class_names = [
            "impervious_surface",
            "building",
            "low_vegetation",
            "tree",
            "car",
            "clutter"
        ]
    elif args.dataset == "loveda":
        num_classes = 7
        class_names = [
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agriculture"
        ]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("=" * 80)
    print(f"Dataset    : {args.dataset}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print("=" * 80)

    val_set, root = build_val_dataset(args, num_classes)

    print(f"Data root  : {root}")
    print(f"Val samples: {len(val_set)}")

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        checkpoint_args = checkpoint.get("args", {})
        topology_pairs = checkpoint.get("topology_pairs")
    else:
        state_dict = checkpoint
        checkpoint_args = {}
        topology_pairs = None

    topology_pairs = topology_pairs or get_topology_pairs(args.dataset)

    model = TopoMamba(
        num_classes=num_classes,
        use_msse=True,
        topology_pairs=topology_pairs,
        cnn_pretrained=False,
        use_rmp_vss=checkpoint_args.get("use_rmp_vss", False),
        rmp_num_paths=checkpoint_args.get("rmp_num_paths", 4),
        use_gia=checkpoint_args.get("use_gia", False),
        gia_dim=checkpoint_args.get("gia_dim", 64),
        gia_heads=checkpoint_args.get("gia_heads", 4),
        skip_mode=checkpoint_args.get("skip_mode", "ssam"),
    )

    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("前 20 个 Missing keys:")
        for k in missing[:20]:
            print("  ", k)

    if len(unexpected) > 0:
        print("前 20 个 Unexpected keys:")
        for k in unexpected[:20]:
            print("  ", k)

    model = model.to(device)

    criterion = TopoMambaLoss(
        num_classes=num_classes,
        lambda_edge=args.lambda_edge,
        lambda_conn=args.lambda_conn,
        ignore_index=255,
        topology_pairs=topology_pairs
    ).to(device)

    metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        ignore_index=255
    )

    print("=" * 80)
    print(
        f"Val | "
        f"Loss: {metrics['loss']:.4f} | "
        f"OA: {metrics['oa']:.4f} | "
        f"mIoU: {metrics['miou']:.4f} | "
        f"mAcc: {metrics['macc']:.4f} | "
        f"mF1: {metrics['mf1']:.4f}"
    )

    print("-" * 80)
    print("Per-class IoU:")
    for i, iou in enumerate(metrics["ious"]):
        print(f"  Class {i} ({class_names[i]}): {iou:.4f}")

    print("-" * 80)
    print("Per-class Acc:")
    for i, acc in enumerate(metrics["accs"]):
        print(f"  Class {i} ({class_names[i]}): {acc:.4f}")

    print("-" * 80)
    print("Per-class F1:")
    for i, f1 in enumerate(metrics["f1s"]):
        print(f"  Class {i} ({class_names[i]}): {f1:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
