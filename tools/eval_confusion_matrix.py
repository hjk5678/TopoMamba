import os
import sys
import argparse
import csv
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/data/BUAS/HJK/TopoMamba")

from data.dataset import ISPRSDataset, build_dataset, get_transform
from models.build_model import (
    CLUSTER_GRAPH_LAYOUT,
    DECODER_LAYOUT,
    RMP_SCAN_LAYOUT,
    TopoMamba,
)
from utils.losses import TopoMambaLoss
from utils.topology_configs import get_topology_pairs


# ============================================================
# 1. 工具函数
# ============================================================
def unpack_outputs(outputs):
    """
    兼容 tuple / dict 两种模型输出。
    """
    if isinstance(outputs, dict):
        return outputs["seg_logits"], outputs["conn_logits"], outputs["edge_preds"]

    if isinstance(outputs, (tuple, list)):
        return outputs[0], outputs[1], outputs[2]

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def sanitize_mask(mask, num_classes, ignore_index=255):
    """
    保留 ignore_index=255。
    其余非法标签全部置为 255。
    """
    invalid = (mask != ignore_index) & ((mask < 0) | (mask >= num_classes))
    mask[invalid] = ignore_index
    return mask


def get_class_names(dataset):
    if dataset in ["potsdam", "vaihingen"]:
        return [
            "impervious_surface",
            "building",
            "low_vegetation",
            "tree",
            "car",
            "clutter"
        ]

    if dataset == "loveda":
        return [
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agriculture"
        ]

    raise ValueError(f"Unknown dataset: {dataset}")


def get_num_classes(dataset):
    if dataset in ["potsdam", "vaihingen"]:
        return 6
    if dataset == "loveda":
        return 7
    raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================
# 2. 构建验证集
# ============================================================
def build_val_dataset(args):
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
                processed_name = (
                    "processed"
                    if args.crop_size == 512
                    else f"processed_{args.crop_size}"
                )
                processed_dir = os.path.join(root, processed_name, "val")
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


# ============================================================
# 3. 混淆矩阵与指标
# ============================================================
def update_confusion_matrix(confusion, pred, target, num_classes, ignore_index=255):
    """
    confusion[true_class, pred_class]

    Args:
        confusion: [C, C] torch.float64
        pred:      [B, H, W]
        target:    [B, H, W]
    """
    valid = target != ignore_index

    pred = pred[valid].view(-1)
    target = target[valid].view(-1)

    keep = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)

    pred = pred[keep]
    target = target[keep]

    # 使用 bincount 快速统计 C*C 个格子
    indices = target * num_classes + pred
    bins = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    ).double()

    confusion += bins.reshape(num_classes, num_classes)


def compute_metrics_from_confusion(confusion):
    """
    confusion[true, pred]

    返回：
        OA, per-class IoU, Acc, F1, mIoU, mAcc, mF1
    """
    tp = torch.diag(confusion)

    # 每一列代表被预测为某类的总数
    pred_sum = confusion.sum(dim=0)

    # 每一行代表真实为某类的总数
    target_sum = confusion.sum(dim=1)

    fp = pred_sum - tp
    fn = target_sum - tp

    union = tp + fp + fn

    ious = []
    accs = []
    f1s = []
    precisions = []
    recalls = []

    for c in range(confusion.shape[0]):
        if union[c] > 0:
            iou = tp[c] / union[c]
        else:
            iou = torch.tensor(float("nan"), device=confusion.device, dtype=torch.float64)

        if target_sum[c] > 0:
            acc = tp[c] / target_sum[c]
            rec = tp[c] / target_sum[c]
        else:
            acc = torch.tensor(float("nan"), device=confusion.device, dtype=torch.float64)
            rec = torch.tensor(float("nan"), device=confusion.device, dtype=torch.float64)

        if pred_sum[c] > 0:
            prec = tp[c] / pred_sum[c]
        else:
            prec = torch.tensor(float("nan"), device=confusion.device, dtype=torch.float64)

        if torch.isfinite(prec) and torch.isfinite(rec) and (prec + rec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = torch.tensor(float("nan"), device=confusion.device, dtype=torch.float64)

        ious.append(iou.item())
        accs.append(acc.item())
        f1s.append(f1.item())
        precisions.append(prec.item())
        recalls.append(rec.item())

    total_correct = tp.sum()
    total_pixels = confusion.sum()

    oa = (total_correct / total_pixels).item() if total_pixels > 0 else 0.0

    miou = float(np.nanmean(ious))
    macc = float(np.nanmean(accs))
    mf1 = float(np.nanmean(f1s))

    return {
        "oa": oa,
        "miou": miou,
        "macc": macc,
        "mf1": mf1,
        "ious": ious,
        "accs": accs,
        "f1s": f1s,
        "precisions": precisions,
        "recalls": recalls,
    }


def normalize_confusion_by_row(confusion):
    """
    行归一化：
        每一行是真实类别。
        normalized[true, pred] 表示真实 true 被预测成 pred 的比例。
    """
    row_sum = confusion.sum(dim=1, keepdim=True)
    normalized = confusion / torch.clamp(row_sum, min=1.0)
    return normalized


# ============================================================
# 4. 保存 CSV
# ============================================================
def save_matrix_csv(matrix, class_names, save_path):
    """
    保存混淆矩阵 CSV。
    """
    matrix_np = matrix.cpu().numpy()

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["true\\pred"] + class_names
        writer.writerow(header)

        for i, name in enumerate(class_names):
            row = [name] + matrix_np[i].tolist()
            writer.writerow(row)


def save_metrics_csv(metrics, class_names, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "IoU",
            "Acc",
            "Precision",
            "Recall",
            "F1"
        ])

        for i, name in enumerate(class_names):
            writer.writerow([
                i,
                name,
                metrics["ious"][i],
                metrics["accs"][i],
                metrics["precisions"][i],
                metrics["recalls"][i],
                metrics["f1s"][i],
            ])

        writer.writerow([])
        writer.writerow(["OA", metrics["oa"]])
        writer.writerow(["mIoU", metrics["miou"]])
        writer.writerow(["mAcc", metrics["macc"]])
        writer.writerow(["mF1", metrics["mf1"]])


# ============================================================
# 5. 验证主函数
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index=255):
    model.eval()

    total_loss = 0.0
    total_seen = 0

    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.float64,
        device=device
    )

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.long().to(device, non_blocking=True)
        mask = sanitize_mask(mask, num_classes, ignore_index=ignore_index)

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

        update_confusion_matrix(
            confusion=confusion,
            pred=pred,
            target=mask,
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        bs = img.size(0)
        total_loss += loss.detach().item() * bs
        total_seen += bs

        pbar.set_postfix({
            "loss": f"{loss.detach().item():.4f}"
        })

    avg_loss = total_loss / max(total_seen, 1)
    metrics = compute_metrics_from_confusion(confusion)
    metrics["loss"] = avg_loss

    return metrics, confusion


# ============================================================
# 6. main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="potsdam",
        choices=["potsdam", "vaihingen", "loveda"]
    )

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)

    parser.add_argument("--pre_cropped", action="store_true", default=False)

    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.05)

    parser.add_argument("--save_dir", type=str, default="eval_results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = get_num_classes(args.dataset)
    class_names = get_class_names(args.dataset)

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 100)
    print(f"Dataset    : {args.dataset}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Save dir   : {args.save_dir}")
    print("=" * 100)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    val_set, root = build_val_dataset(args)

    print(f"Data root  : {root}")
    print(f"Val samples: {len(val_set)}")
    print("=" * 100)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
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

    model = TopoMamba(
        num_classes=num_classes,
        use_msse=True,
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

    print("=" * 100)

    model = model.to(device)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    criterion = TopoMambaLoss(
        num_classes=num_classes,
        lambda_edge=args.lambda_edge,
        lambda_conn=args.lambda_conn,
        ignore_index=255,
        topology_pairs=topology_pairs
    ).to(device)

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    metrics, confusion = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        ignore_index=255
    )

    confusion_norm = normalize_confusion_by_row(confusion)

    # -------------------------------------------------
    # Print metrics
    # -------------------------------------------------
    print("=" * 100)
    print(
        f"Val | "
        f"Loss: {metrics['loss']:.4f} | "
        f"OA: {metrics['oa']:.4f} | "
        f"mIoU: {metrics['miou']:.4f} | "
        f"mAcc: {metrics['macc']:.4f} | "
        f"mF1: {metrics['mf1']:.4f}"
    )

    print("-" * 100)
    print("Per-class metrics:")
    for i, name in enumerate(class_names):
        print(
            f"Class {i:02d} ({name:18s}) | "
            f"IoU: {metrics['ious'][i]:.4f} | "
            f"Acc: {metrics['accs'][i]:.4f} | "
            f"Prec: {metrics['precisions'][i]:.4f} | "
            f"Rec: {metrics['recalls'][i]:.4f} | "
            f"F1: {metrics['f1s'][i]:.4f}"
        )

    print("-" * 100)
    print("Confusion Matrix: rows=true labels, cols=pred labels")
    print(confusion.cpu().numpy().astype(np.int64))

    print("-" * 100)
    print("Row-normalized Confusion Matrix:")
    print("每一行表示：真实类别被预测成各类别的比例。")
    print(np.round(confusion_norm.cpu().numpy(), 4))

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    raw_csv = os.path.join(args.save_dir, f"{args.dataset}_confusion_matrix.csv")
    norm_csv = os.path.join(args.save_dir, f"{args.dataset}_confusion_matrix_normalized.csv")
    metrics_csv = os.path.join(args.save_dir, f"{args.dataset}_per_class_metrics.csv")

    save_matrix_csv(
        matrix=confusion,
        class_names=class_names,
        save_path=raw_csv
    )

    save_matrix_csv(
        matrix=confusion_norm,
        class_names=class_names,
        save_path=norm_csv
    )

    save_metrics_csv(
        metrics=metrics,
        class_names=class_names,
        save_path=metrics_csv
    )

    print("-" * 100)
    print(f"Saved raw confusion matrix       : {raw_csv}")
    print(f"Saved normalized confusion matrix: {norm_csv}")
    print(f"Saved per-class metrics          : {metrics_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
