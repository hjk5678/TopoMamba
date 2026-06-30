import os
import sys
import argparse
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# 项目根目录
sys.path.append("/data/BUAS/HJK/TopoMamba")

from data.dataset import (
    build_dataset,
    ISPRSDataset,
    get_transform,
)

from models.build_model import (
    CLUSTER_GRAPH_LAYOUT,
    DECODER_LAYOUT,
    RMP_SCAN_LAYOUT,
    TopoMamba,
)
from utils.losses import TopoMambaLoss
from utils.topology_configs import (
    get_topology_pairs,
    describe_topology_pairs,
)


# ============================================================
# 0. DDP 工具函数
# ============================================================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def setup_distributed(args):
    """
    支持两种模式：

    1. DDP:
       torchrun --nproc_per_node=4 train.py ...

    2. 单卡:
       python train.py ...
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)

            dist.init_process_group(
                backend="nccl",
                init_method="env://"
            )
            dist.barrier()
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_sum_tensor(tensor):
    """
    DDP 下求和；单卡直接返回。
    """
    if is_dist_avail_and_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def get_raw_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def set_seed(seed, rank=0):
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


# ============================================================
# 1. 验证集 sampler：不重复样本
# ============================================================
class DistributedEvalSampler(Sampler):
    """
    验证集 DDP sampler。

    默认 DistributedSampler 会补齐样本，导致验证集重复计算。
    这个 sampler 不补齐样本。
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset

        if num_replicas is None:
            num_replicas = get_world_size()

        if rank is None:
            rank = get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = max(
            0,
            (len(self.dataset) - self.rank + self.num_replicas - 1)
            // self.num_replicas
        )

    def __iter__(self):
        indices = list(
            range(
                self.rank,
                len(self.dataset),
                self.num_replicas
            )
        )
        return iter(indices)

    def __len__(self):
        return self.num_samples


# ============================================================
# 2. 输出解析 / 指标函数
# ============================================================
def unpack_outputs(outputs):
    """
    当前 v5 模型输出：

    dict:
        {
            "seg_logits":  [B, num_classes, H, W],
            "edge_preds":  list of edge maps,
            "conn_logits": list of MSTC/topology logits
        }

    返回：
        seg_logits, conn_logits, edge_preds
    """
    if isinstance(outputs, dict):
        seg_logits = outputs["seg_logits"]
        conn_logits = outputs.get("conn_logits", None)
        edge_preds = outputs.get("edge_preds", None)

        return seg_logits, conn_logits, edge_preds

    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 3:
            return outputs[0], outputs[1], outputs[2]

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def sanitize_mask(mask, num_classes, ignore_index=255):
    """
    防止脏标签导致 CrossEntropy 越界。
    """
    invalid = (
        (mask != ignore_index)
        & (
            (mask < 0)
            | (mask >= num_classes)
        )
    )

    mask[invalid] = ignore_index

    return mask


def compute_class_stats(pred, target, num_classes, ignore_index=255):
    """
    返回每类 tp/fp/fn。
    """
    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    tp = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=pred.device
    )
    fp = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=pred.device
    )
    fn = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=pred.device
    )

    for c in range(num_classes):
        tp[c] = ((pred == c) & (target == c)).sum()
        fp[c] = ((pred == c) & (target != c)).sum()
        fn[c] = ((pred != c) & (target == c)).sum()

    return tp, fp, fn


def get_num_classes(dataset):
    dataset = dataset.lower()

    if dataset in ["potsdam", "vaihingen"]:
        return 6

    if dataset == "loveda":
        return 7

    raise ValueError(f"Unknown dataset: {dataset}")


def get_class_names(dataset):
    dataset = dataset.lower()

    if dataset in ["potsdam", "vaihingen"]:
        return [
            "impervious_surface",
            "building",
            "low_vegetation",
            "tree",
            "car",
            "clutter",
        ]

    if dataset == "loveda":
        return [
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agriculture",
        ]

    return [f"class_{i}" for i in range(100)]


# ============================================================
# 3. 数值稳定性检查
# ============================================================
def model_is_finite(model):
    """
    检查模型 state_dict 里所有浮点 tensor 是否 finite。

    重要：
        BN running_mean / running_var 也在 state_dict 里。
        如果 forward 中出现 NaN，BN running stats 可能已经被污染。
    """
    raw_model = get_raw_model(model)

    for name, tensor in raw_model.state_dict().items():
        if not torch.is_tensor(tensor):
            continue

        # 只检查浮点 tensor。
        # num_batches_tracked 这类 int tensor 不需要检查。
        if not tensor.is_floating_point():
            continue

        if not torch.isfinite(tensor).all():
            if is_main_process():
                print(f"[FiniteCheck] non-finite tensor detected: {name}")
            return False

    return True


def assert_model_finite(model, message="model has non-finite tensors"):
    if not model_is_finite(model):
        raise RuntimeError(message)


# ============================================================
# 4. 参数分组优化器
# ============================================================
def build_param_group_optimizer(
    model,
    base_lr=1e-4,
    weight_decay=0.01,
    backbone_lr_mult=0.25,
    new_module_lr_mult=1.0,
    vss_lr_mult=0.5,
    is_main=True,
    **kwargs,
):
    """
    RMTPB 版本参数分组优化器。

    兼容两种调用方式：

    方式 1：
        build_param_group_optimizer(model, args, is_main=is_main)

    方式 2：
        build_param_group_optimizer(
            model,
            base_lr=1e-4,
            weight_decay=0.01,
            backbone_lr_mult=0.25,
            new_module_lr_mult=1.0,
            is_main=is_main,
        )

    当前参数分组：
        1. stages.*.cnn_branch
            ResNet18 ImageNet pretrained CNN branch
            小学习率：base_lr * backbone_lr_mult

        2. stages.*.vmamba_branch
            RMTPB VSS / VMamba branch
            中等学习率：base_lr * vss_lr_mult

        3. 其他模块
            stem / mergings / fusion / alpha / bottleneck /
            BGSM / MS-CGC / MSTC / decoder
            大学习率：base_lr * new_module_lr_mult
    """

    # =====================================================
    # 兼容旧调用：
    #   build_param_group_optimizer(model, args, is_main=is_main)
    #
    # 此时：
    #   model 是模型
    #   base_lr 是 argparse.Namespace
    # =====================================================
    if hasattr(base_lr, "lr"):
        args = base_lr

        base_lr = args.lr
        weight_decay = args.weight_decay
        backbone_lr_mult = args.backbone_lr_mult
        new_module_lr_mult = args.new_module_lr_mult

        # 你的 argparse 里目前没有 vss_lr_mult，
        # 所以这里用默认 0.5。
        vss_lr_mult = getattr(args, "vss_lr_mult", vss_lr_mult)

    # =====================================================
    # 兼容另一种可能的旧调用：
    #   build_param_group_optimizer(args, model, is_main=is_main)
    # =====================================================
    if not hasattr(model, "named_parameters") and hasattr(model, "lr"):
        args = model
        model = base_lr

        base_lr = args.lr
        weight_decay = args.weight_decay
        backbone_lr_mult = args.backbone_lr_mult
        new_module_lr_mult = args.new_module_lr_mult
        vss_lr_mult = getattr(args, "vss_lr_mult", vss_lr_mult)

    # 转成 float，避免 Namespace / str 类型问题
    base_lr = float(base_lr)
    weight_decay = float(weight_decay)
    backbone_lr_mult = float(backbone_lr_mult)
    new_module_lr_mult = float(new_module_lr_mult)
    vss_lr_mult = float(vss_lr_mult)

    pretrained_cnn_params = []
    vss_params = []
    new_module_params = []
    other_params = []

    pretrained_cnn_names = []
    vss_names = []
    new_module_names = []
    other_names = []

    new_module_keywords = [
        "stem",
        "mergings",

        "fusion",
        "alpha",


        "cluster_graph",
        "mstc_heads",
        "conn_heads",
        "dcpm_heads",

        "decoder",
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # -------------------------------------------------
        # 1. RMTPB CNN branch
        # 已加载 ResNet18 ImageNet 预训练权重
        #
        # example:
        #   stages.0.0.cnn_branch.0.conv1.weight
        #   stages.2.3.cnn_branch.1.bn2.weight
        # -------------------------------------------------
        if "stages" in name and "cnn_branch" in name:
            pretrained_cnn_params.append(param)
            pretrained_cnn_names.append(name)

        # -------------------------------------------------
        # 2. RMTPB VSS / VMamba branch
        #
        # example:
        #   stages.0.0.vmamba_branch.0.norm.weight
        #   stages.1.1.vmamba_branch.0.op.x_proj.weight
        # -------------------------------------------------
        elif "stages" in name and "vmamba_branch" in name:
            vss_params.append(param)
            vss_names.append(name)

        # -------------------------------------------------
        # 3. 其他新模块
        # -------------------------------------------------
        elif any(key in name for key in new_module_keywords):
            new_module_params.append(param)
            new_module_names.append(name)

        # -------------------------------------------------
        # 4. 兜底
        # 如果这里很多，说明还有模块没正确分组。
        # -------------------------------------------------
        else:
            other_params.append(param)
            other_names.append(name)

    param_groups = []

    if len(pretrained_cnn_params) > 0:
        param_groups.append({
            "params": pretrained_cnn_params,
            "lr": base_lr * backbone_lr_mult,
            "weight_decay": weight_decay,
            "name": "resnet18_pretrained_cnn_branch",
        })

    if len(vss_params) > 0:
        param_groups.append({
            "params": vss_params,
            "lr": base_lr * vss_lr_mult,
            "weight_decay": weight_decay,
            "name": "rmtpb_vss_branch",
        })

    if len(new_module_params) > 0:
        param_groups.append({
            "params": new_module_params,
            "lr": base_lr * new_module_lr_mult,
            "weight_decay": weight_decay,
            "name": "new_modules",
        })

    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": base_lr * new_module_lr_mult,
            "weight_decay": weight_decay,
            "name": "others",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if is_main:
        print("=" * 80)
        print("[Optimizer] RMTPB parameter groups")
        print(f"  base_lr:                 {base_lr}")
        print(f"  pretrained CNN lr:       {base_lr * backbone_lr_mult}")
        print(f"  VSS branch lr:           {base_lr * vss_lr_mult}")
        print(f"  new module lr:           {base_lr * new_module_lr_mult}")
        print(f"  weight_decay:            {weight_decay}")
        print(f"  pretrained CNN tensors:  {len(pretrained_cnn_names)}")
        print(f"  VSS branch tensors:      {len(vss_names)}")
        print(f"  new module tensors:      {len(new_module_names)}")
        print(f"  other tensors:           {len(other_names)}")

        if len(other_names) > 0:
            print("[Optimizer] First other params:")
            for n in other_names[:80]:
                print("   ", n)

        print("=" * 80)

    return optimizer


def get_current_lrs(optimizer):
    lr_info = {}

    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{i}")
        lr_info[name] = group["lr"]

    return lr_info


# ============================================================
# 5. 训练一个 epoch
# ============================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    num_classes,
    epoch,
    ignore_index=255,
    grad_clip=0.3,
    debug_shapes=False,
    amp_enabled=True,
    max_nonfinite_count=1,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_edge = 0.0
    total_conn = 0.0
    total_seen = 0.0

    debug_printed = False
    nonfinite_count = 0

    if is_main_process():
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True
        )
    else:
        pbar = loader

    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.long().to(device, non_blocking=True)

        mask = sanitize_mask(
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(img)

            if (
                is_main_process()
                and debug_shapes
                and epoch == 1
                and not debug_printed
            ):
                print("=" * 80)
                print("[Debug] model output type:", type(outputs))

                if isinstance(outputs, dict):
                    print("[Debug] output keys:", outputs.keys())

                    for k, v in outputs.items():
                        if torch.is_tensor(v):
                            print(f"[Debug] {k}: {tuple(v.shape)}")
                        elif isinstance(v, (list, tuple)):
                            print(f"[Debug] {k}: list/tuple length={len(v)}")

                            for i, item in enumerate(v):
                                if torch.is_tensor(item):
                                    print(f"    {k}[{i}]: {tuple(item.shape)}")
                        else:
                            print(f"[Debug] {k}: {type(v)}")

                print("=" * 80)
                debug_printed = True

            seg_logits, conn_logits, edge_preds = unpack_outputs(outputs)

            loss_dict = criterion(
                seg_logits,
                mask,
                conn_logits=conn_logits,
                edge_preds=edge_preds
            )

            loss = loss_dict["total_loss"]

        # -------------------------------------------------
        # non-finite loss DDP 同步检测
        # -------------------------------------------------
        finite_flag = torch.tensor(
            float(torch.isfinite(loss)),
            device=device
        )

        if is_dist_avail_and_initialized():
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        if finite_flag.item() == 0:
            nonfinite_count += 1

            if is_main_process():
                print(
                    f"[Warning] non-finite loss detected at epoch {epoch}, "
                    f"count={nonfinite_count}/{max_nonfinite_count}"
                )

            optimizer.zero_grad(set_to_none=True)

            # 关键：检查模型是否已经被污染。
            # 如果 BN running stats 已经 NaN，必须立刻停止。
            if not model_is_finite(model):
                raise RuntimeError(
                    f"Model already contains non-finite tensors at epoch {epoch}. "
                    f"Stop training to prevent checkpoint pollution."
                )

            if nonfinite_count >= max_nonfinite_count:
                raise RuntimeError(
                    f"Too many non-finite losses in epoch {epoch}. "
                    f"Stop training to prevent checkpoint pollution."
                )

            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        # optimizer.step 后再次检查，防止参数被更新成 NaN。
        if not model_is_finite(model):
            raise RuntimeError(
                f"Model became non-finite after optimizer step at epoch {epoch}."
            )

        bs = img.size(0)
        total_seen += bs

        edge_value = loss_dict.get(
            "edge_loss",
            loss_dict.get("lbs_loss", torch.tensor(0.0, device=device))
        )

        total_loss += to_float(loss_dict["total_loss"]) * bs
        total_ce += to_float(loss_dict["ce_loss"]) * bs
        total_dice += to_float(loss_dict["dice_loss"]) * bs
        total_edge += to_float(edge_value) * bs
        total_conn += to_float(loss_dict["conn_loss"]) * bs

        if is_main_process():
            pbar.set_postfix({
                "loss": f"{to_float(loss_dict['total_loss']):.4f}",
                "ce": f"{to_float(loss_dict['ce_loss']):.4f}",
                "dice": f"{to_float(loss_dict['dice_loss']):.4f}",
                "edge": f"{to_float(edge_value):.4f}",
                "conn": f"{to_float(loss_dict['conn_loss']):.4f}",
            })

    stat = torch.tensor(
        [
            total_loss,
            total_ce,
            total_dice,
            total_edge,
            total_conn,
            total_seen,
        ],
        dtype=torch.float64,
        device=device
    )

    stat = reduce_sum_tensor(stat)

    seen = max(stat[5].item(), 1.0)

    return {
        "loss": stat[0].item() / seen,
        "ce": stat[1].item() / seen,
        "dice": stat[2].item() / seen,
        "edge": stat[3].item() / seen,
        "conn": stat[4].item() / seen,
    }


# ============================================================
# 6. 验证
# ============================================================
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    num_classes,
    ignore_index=255,
    amp_enabled=True,
):
    model.eval()

    total_loss = 0.0
    total_seen = 0.0

    tp_all = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device
    )
    fp_all = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device
    )
    fn_all = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device
    )

    correct_all = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device
    )
    valid_all = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device
    )

    if is_main_process():
        pbar = tqdm(
            loader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True
        )
    else:
        pbar = loader

    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.long().to(device, non_blocking=True)

        mask = sanitize_mask(
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        with autocast(enabled=amp_enabled):
            outputs = model(img)
            seg_logits, conn_logits, edge_preds = unpack_outputs(outputs)

            loss_dict = criterion(
                seg_logits,
                mask,
                conn_logits=conn_logits,
                edge_preds=edge_preds
            )

            loss = loss_dict["total_loss"]

        if not torch.isfinite(loss):
            if is_main_process():
                print("[Validation] non-finite loss detected, skip this batch.")
            continue

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
        total_seen += bs
        total_loss += to_float(loss) * bs

    stat = torch.tensor(
        [
            total_loss,
            total_seen,
            correct_all.item(),
            valid_all.item(),
        ],
        dtype=torch.float64,
        device=device
    )

    stat = reduce_sum_tensor(stat)

    tp_all = reduce_sum_tensor(tp_all)
    fp_all = reduce_sum_tensor(fp_all)
    fn_all = reduce_sum_tensor(fn_all)

    seen = max(stat[1].item(), 1.0)
    avg_loss = stat[0].item() / seen

    correct = stat[2].item()
    valid_pixels = max(stat[3].item(), 1.0)
    oa = correct / valid_pixels

    ious = []
    accs = []
    f1s = []

    for c in range(num_classes):
        tp = tp_all[c].item()
        fp = fp_all[c].item()
        fn = fn_all[c].item()

        iou = tp / max(tp + fp + fn, 1.0)
        acc = tp / max(tp + fn, 1.0)
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)

        ious.append(iou)
        accs.append(acc)
        f1s.append(f1)

    miou = float(np.nanmean(ious)) if len(ious) > 0 else 0.0
    macc = float(np.nanmean(accs)) if len(accs) > 0 else 0.0
    mf1 = float(np.nanmean(f1s)) if len(f1s) > 0 else 0.0

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


# ============================================================
# 7. 可选 ResNet18 预训练检查
# ============================================================
def try_load_resnet18_weights(model):
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        pretrained_resnet = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
    except Exception:
        import torchvision.models as models
        pretrained_resnet = models.resnet18(pretrained=True)

    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()

    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    if is_main_process():
        print(
            f"[Pretrain] 实际加载 ResNet-18 参数数量: "
            f"{len(filtered_dict)}"
        )

        if len(filtered_dict) > 0:
            print("[Pretrain] 前几个加载的 key:")

            for k in list(filtered_dict.keys())[:10]:
                print("  ", k)
        else:
            print("[Pretrain] 没有匹配到可加载的 ResNet-18 参数。")


# ============================================================
# 8. 数据集构建
# ============================================================
def build_train_val_datasets(args, num_classes):
    base = "/data/BUAS/HJK/TopoMamba/data"

    if args.dataset == "potsdam":
        default_root = os.path.join(base, "Potsdam")
    elif args.dataset == "vaihingen":
        default_root = os.path.join(base, "Vaihingen")
    elif args.dataset == "loveda":
        default_root = os.path.join(base, "Love DA")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    root = args.data_root if args.data_root is not None else default_root

    if is_main_process():
        print(f"数据集: {args.dataset}")
        print(f"路径: {root}")
        print(f"类别数: {num_classes}")

    # -------------------------------------------------
    # LoveDA
    # -------------------------------------------------
    if args.dataset.lower() == "loveda":
        train_set = build_dataset(
            "loveda",
            root,
            split="train",
            pre_cropped=args.pre_cropped,
            processed_dir=args.processed_dir,
            crop_size=args.crop_size,
            strong_aug=args.strong_aug
        )

        val_set = build_dataset(
            "loveda",
            root,
            split="val",
            pre_cropped=args.pre_cropped,
            processed_dir=args.processed_dir,
            crop_size=args.crop_size,
            strong_aug=False
        )

        return train_set, val_set, root

    # -------------------------------------------------
    # ISPRS: Potsdam / Vaihingen
    # -------------------------------------------------
    img_dir = os.path.join(root, "Images")
    label_dir = os.path.join(root, "Labels")

    if args.pre_cropped:
        if args.processed_dir is None:
            processed_name = (
                "processed"
                if args.crop_size == 512
                else f"processed_{args.crop_size}"
            )
            train_processed_dir = os.path.join(
                root,
                processed_name,
                "train"
            )
            val_processed_dir = os.path.join(
                root,
                processed_name,
                "val"
            )
        else:
            train_processed_dir = os.path.join(
                args.processed_dir,
                "train"
            )
            val_processed_dir = os.path.join(
                args.processed_dir,
                "val"
            )

        if not os.path.exists(train_processed_dir):
            raise FileNotFoundError(
                f"预切块训练目录不存在: {train_processed_dir}"
            )

        train_set = ISPRSDataset(
            img_dir,
            label_dir,
            split="train",
            crop_size=args.crop_size,
            transform=get_transform(
                "train",
                crop_size=args.crop_size,
                strong_aug=args.strong_aug
            ),
            pre_cropped=True,
            processed_dir=train_processed_dir
        )

        if os.path.exists(val_processed_dir):
            val_set = ISPRSDataset(
                img_dir,
                label_dir,
                split="val",
                crop_size=args.crop_size,
                transform=get_transform(
                    "val",
                    crop_size=args.crop_size,
                    strong_aug=False
                ),
                pre_cropped=True,
                processed_dir=val_processed_dir
            )
        else:
            if is_main_process():
                print(
                    f"[Warning] 验证预切块目录不存在: "
                    f"{val_processed_dir}"
                )
                print("[Warning] 回退到在线中心裁剪验证模式。")

            val_set = ISPRSDataset(
                img_dir,
                label_dir,
                split="val",
                crop_size=args.crop_size,
                transform=get_transform(
                    "val",
                    crop_size=args.crop_size,
                    strong_aug=False
                ),
                pre_cropped=False
            )

    else:
        train_set = build_dataset(
            args.dataset,
            root,
            split="train",
            pre_cropped=False,
            crop_size=args.crop_size,
            strong_aug=args.strong_aug
        )

        val_set = build_dataset(
            args.dataset,
            root,
            split="val",
            pre_cropped=False,
            crop_size=args.crop_size,
            strong_aug=False
        )

    return train_set, val_set, root


# ============================================================
# 9. Checkpoint 保存
# ============================================================
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_miou,
    args,
    topology_pairs
):
    raw_model = get_raw_model(model)

    torch.save(
        {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
            "topology_pairs": topology_pairs,
        },
        path
    )


# ============================================================
# 10. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="potsdam",
        choices=["potsdam", "vaihingen", "loveda"]
    )

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument(
        "--pre_cropped",
        action="store_true",
        default=False,
        help="使用离线预切块数据"
    )

    parser.add_argument(
        "--strong_aug",
        action="store_true",
        default=False,
        help="使用强数据增强"
    )

    # -------------------------------------------------
    # Optimizer / LR
    # -------------------------------------------------
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=0.5,
        help="backbone 学习率倍率"
    )

    parser.add_argument(
        "--new_module_lr_mult",
        type=float,
        default=1.0,
        help="BGSM/MSTC/decoder 等新模块学习率倍率"
    )

    # -------------------------------------------------
    # Loss weights
    # -------------------------------------------------
    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.05)

    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=False,
        help="是否使用类别权重"
    )

    # -------------------------------------------------
    # Stable training options
    # -------------------------------------------------
    parser.add_argument(
        "--no_amp",
        action="store_true",
        default=False,
        help="关闭 AMP，使用 FP32 训练"
    )

    parser.add_argument(
        "--max_nonfinite_count",
        type=int,
        default=1,
        help="一个 epoch 内允许的 non-finite loss 次数，超过则停止"
    )

    # -------------------------------------------------
    # Train options
    # -------------------------------------------------
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        default=False,
        help="只加载模型权重，不恢复 optimizer/scheduler/scaler"
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=0.3)

    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        default=False,
        help="DDP 是否启用 find_unused_parameters"
    )

    parser.add_argument(
        "--pretrain_resnet18",
        action="store_true",
        default=False,
        help="尝试加载 ResNet-18 权重"
    )

    parser.add_argument(
        "--debug_shapes",
        action="store_true",
        default=False,
        help="第一个 epoch 打印模型输出 shape"
    )

    parser.add_argument(
        "--use_rmp_vss",
        action="store_true",
        default=False,
        help="启用 residual multi-path VSS encoder branch"
    )

    parser.add_argument(
        "--rmp_num_paths",
        type=int,
        default=4,
        choices=[4],
        help="residual multi-path VSS 固定使用四条互补路径"
    )

    parser.add_argument(
        "--rmp_window_size",
        type=int,
        default=8,
        help="局部窗口 Cross VSS 的特征图窗口大小"
    )

    parser.add_argument(
        "--rmp_atrous_rate",
        type=int,
        default=2,
        help="Atrous Cross VSS 的空间采样率"
    )

    parser.add_argument(
        "--use_cluster_gcn",
        action="store_true",
        default=False,
        help="启用四尺度硬聚类区域图卷积（MS-CGC，无注意力）"
    )

    parser.add_argument(
        "--cluster_counts",
        type=int,
        nargs=4,
        default=[256, 128, 64, 32],
        metavar=("S1", "S2", "S3", "S4"),
        help="S1-S4 的区域聚类节点数"
    )

    parser.add_argument(
        "--cluster_graph_dim",
        type=int,
        default=64,
        help="MS-CGC 区域节点特征维度"
    )

    parser.add_argument(
        "--cluster_iters",
        type=int,
        default=2,
        help="每个尺度的硬聚类迭代次数"
    )

    parser.add_argument(
        "--cluster_spatial_weight",
        type=float,
        default=0.5,
        help="硬聚类描述子中的归一化坐标权重"
    )

    args = parser.parse_args()
    args.rmp_scan_layout = RMP_SCAN_LAYOUT
    args.decoder_layout = DECODER_LAYOUT
    args.cluster_graph_layout = CLUSTER_GRAPH_LAYOUT

    device = setup_distributed(args)
    set_seed(args.seed, get_rank())

    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    if is_main_process():
        print("=" * 80)
        print("[Config]")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"  amp_enabled: {amp_enabled}")
        print("=" * 80)

    # -------------------------------------------------
    # Dataset / topology
    # -------------------------------------------------
    num_classes = get_num_classes(args.dataset)
    class_names = get_class_names(args.dataset)

    topology_pairs = get_topology_pairs(args.dataset)

    if is_main_process():
        print(describe_topology_pairs(topology_pairs))

    train_set, val_set, root = build_train_val_datasets(
        args,
        num_classes
    )

    if is_main_process():
        print(f"训练集大小: {len(train_set)}")
        print(f"验证集大小: {len(val_set)}")

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------
    if args.distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )

        val_sampler = DistributedEvalSampler(
            val_set,
            num_replicas=get_world_size(),
            rank=get_rank()
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = TopoMamba(
        num_classes=num_classes,
        topology_pairs=topology_pairs,
        use_rmp_vss=args.use_rmp_vss,
        rmp_num_paths=args.rmp_num_paths,
        rmp_window_size=args.rmp_window_size,
        rmp_atrous_rate=args.rmp_atrous_rate,
        use_cluster_gcn=args.use_cluster_gcn,
        cluster_counts=args.cluster_counts,
        cluster_graph_dim=args.cluster_graph_dim,
        cluster_iters=args.cluster_iters,
        cluster_spatial_weight=args.cluster_spatial_weight,
    )

    if args.pretrain_resnet18:
        try_load_resnet18_weights(model)

    model = model.to(device)

    assert_model_finite(
        model,
        message="Initial model contains non-finite tensors."
    )

    # -------------------------------------------------
    # Optimizer before DDP
    # -------------------------------------------------
    optimizer = build_param_group_optimizer(
        model,
        args,
        is_main=is_main_process()
    )

    # -------------------------------------------------
    # DDP
    # -------------------------------------------------
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.find_unused_parameters
        )

    # -------------------------------------------------
    # Class weights
    # -------------------------------------------------
    class_weights = None

    if args.use_class_weights:
        if args.dataset.lower() in ["potsdam", "vaihingen"]:
            class_weights = [1.0, 1.0, 1.15, 1.15, 1.3, 1.6]
        elif args.dataset.lower() == "loveda":
            class_weights = [1.0, 1.2, 1.2, 1.4, 1.3, 1.3, 1.3]
        else:
            class_weights = None

        if is_main_process():
            print(f"[Loss] class_weights = {class_weights}")

    # -------------------------------------------------
    # Criterion
    # -------------------------------------------------
    criterion = TopoMambaLoss(
        num_classes=num_classes,
        lambda_edge=args.lambda_edge,
        lambda_conn=args.lambda_conn,
        ignore_index=255,
        class_weights=class_weights,
        topology_pairs=topology_pairs
    ).to(device)

    # -------------------------------------------------
    # Scheduler / AMP scaler
    # -------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_miou = 0.0

    # -------------------------------------------------
    # Resume
    # -------------------------------------------------
    if args.resume is not None and os.path.isfile(args.resume):
        map_location = {
            "cuda:%d" % 0: "cuda:%d" % args.local_rank
        } if args.distributed else device

        checkpoint = torch.load(
            args.resume,
            map_location=map_location
        )

        checkpoint_state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        checkpoint_args = (
            checkpoint.get("args", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        if checkpoint_args.get("use_gia", False) or any(
            key.replace("module.", "").startswith("graph_interaction.")
            for key in checkpoint_state
        ):
            raise RuntimeError(
                "The resume checkpoint contains the removed GIA attention "
                "path and cannot be loaded by the MS-CGC model."
            )

        if args.use_rmp_vss and isinstance(checkpoint, dict):
            saved_args = checkpoint.get("args", {})
            saved_uses_rmp = saved_args.get("use_rmp_vss", False)
            saved_layout = saved_args.get("rmp_scan_layout")

            if saved_uses_rmp and saved_layout != RMP_SCAN_LAYOUT:
                raise RuntimeError(
                    "The checkpoint uses the legacy multi-path VSS layout and "
                    "cannot be resumed as global/local/diagonal/atrous VSS. "
                    "Start a new training run."
                )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            saved_args = checkpoint.get("args", {})
            saved_decoder_layout = saved_args.get("decoder_layout")
            if saved_decoder_layout != DECODER_LAYOUT:
                raise RuntimeError(
                    "The checkpoint uses an incompatible decoder layout: "
                    f"checkpoint={saved_decoder_layout!r}, "
                    f"current={DECODER_LAYOUT!r}. Start a new training run."
                )

            saved_uses_cluster_gcn = saved_args.get(
                "use_cluster_gcn",
                False,
            )
            saved_cluster_layout = saved_args.get("cluster_graph_layout")
            if (
                saved_uses_cluster_gcn
                and saved_cluster_layout != CLUSTER_GRAPH_LAYOUT
            ):
                raise RuntimeError(
                    "The checkpoint uses an incompatible cluster graph layout: "
                    f"checkpoint={saved_cluster_layout!r}, "
                    f"current={CLUSTER_GRAPH_LAYOUT!r}."
                )

        raw_model = get_raw_model(model)

        if "model" in checkpoint:
            load_info = raw_model.load_state_dict(
                checkpoint["model"],
                strict=False
            )
        else:
            load_info = raw_model.load_state_dict(
                checkpoint,
                strict=False
            )

        missing = getattr(load_info, "missing_keys", [])
        unexpected = getattr(load_info, "unexpected_keys", [])

        if is_main_process():
            print(f"[Resume] Missing keys: {len(missing)}")
            print(f"[Resume] Unexpected keys: {len(unexpected)}")

            if len(missing) > 0:
                print("[Resume] 前 20 个 missing keys:")
                for k in missing[:20]:
                    print("  ", k)

            if len(unexpected) > 0:
                print("[Resume] 前 20 个 unexpected keys:")
                for k in unexpected[:20]:
                    print("  ", k)

        assert_model_finite(
            model,
            message="Resumed model contains non-finite tensors."
        )

        if args.resume_model_only:
            if is_main_process():
                print(
                    "[Resume] resume_model_only=True，只加载模型权重，"
                    "不加载 optimizer/scheduler/scaler。"
                )
        else:
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    if is_main_process():
                        print(f"[Resume] optimizer 加载失败，跳过: {e}")

            if "scheduler" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except Exception as e:
                    if is_main_process():
                        print(f"[Resume] scheduler 加载失败，跳过: {e}")

            if "scaler" in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    if is_main_process():
                        print(f"[Resume] scaler 加载失败，跳过: {e}")

            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1

            if "best_miou" in checkpoint:
                best_miou = checkpoint["best_miou"]

        if args.resume_model_only:
            # 模型权重恢复，但优化器重新开始。
            # 为了避免继续沿用旧 scheduler 阶段，这里从 epoch 0 重新算。
            start_epoch = 0

            if "best_miou" in checkpoint:
                best_miou = checkpoint["best_miou"]

        if is_main_process():
            print(f"[Resume] start_epoch = {start_epoch}")
            print(f"[Resume] best_mIoU = {best_miou:.4f}")

    # -------------------------------------------------
    # Save dir
    # -------------------------------------------------
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    if is_dist_avail_and_initialized():
        dist.barrier()

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    try:
        for epoch in range(start_epoch, args.epochs):
            if args.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t0 = time.time()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                num_classes=num_classes,
                epoch=epoch + 1,
                ignore_index=255,
                grad_clip=args.grad_clip,
                debug_shapes=args.debug_shapes,
                amp_enabled=amp_enabled,
                max_nonfinite_count=args.max_nonfinite_count,
            )

            scheduler.step()

            elapsed = time.time() - t0
            lr_info = get_current_lrs(optimizer)
            lr_backbone = lr_info.get(
                "resnet18_pretrained_cnn_branch",
                lr_info.get("backbone", 0.0)
            )
            lr_vss = lr_info.get("rmtpb_vss_branch", 0.0)
            lr_new = lr_info.get("new_modules", 0.0)

            if is_main_process():
                print(
                    f"Epoch {epoch + 1:3d}/{args.epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"CE: {train_metrics['ce']:.4f} | "
                    f"Dice: {train_metrics['dice']:.4f} | "
                    f"Edge: {train_metrics['edge']:.4f} | "
                    f"Conn: {train_metrics['conn']:.4f} | "
                    f"LR_backbone: {lr_backbone:.6e} | "
                    f"LR_vss: {lr_vss:.6e} | "
                    f"LR_new: {lr_new:.6e} | "
                    f"Time: {elapsed:.1f}s"
                )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------
            do_val = (
                ((epoch + 1) % args.val_every == 0)
                or (epoch == args.epochs - 1)
            )

            if do_val:
                val_metrics = validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=num_classes,
                    ignore_index=255,
                    amp_enabled=amp_enabled,
                )

                if is_main_process():
                    print(
                        f"验证 | "
                        f"Loss: {val_metrics['loss']:.4f} | "
                        f"OA: {val_metrics['oa']:.4f} | "
                        f"mIoU: {val_metrics['miou']:.4f} | "
                        f"mAcc: {val_metrics['macc']:.4f} | "
                        f"mF1: {val_metrics['mf1']:.4f}"
                    )

                    print("Per-class IoU:")
                    for i in range(num_classes):
                        name = class_names[i] if i < len(class_names) else f"class_{i}"
                        print(
                            f"  Class {i} ({name}): "
                            f"{val_metrics['ious'][i]:.4f}"
                        )

                    print("Per-class Acc:")
                    for i in range(num_classes):
                        name = class_names[i] if i < len(class_names) else f"class_{i}"
                        print(
                            f"  Class {i} ({name}): "
                            f"{val_metrics['accs'][i]:.4f}"
                        )

                    print("Per-class F1:")
                    for i in range(num_classes):
                        name = class_names[i] if i < len(class_names) else f"class_{i}"
                        print(
                            f"  Class {i} ({name}): "
                            f"{val_metrics['f1s'][i]:.4f}"
                        )

                    # -------------------------------------------------
                    # Check finite before saving
                    # -------------------------------------------------
                    finite_ok = model_is_finite(model)

                    if not finite_ok:
                        print(
                            "[Checkpoint] 当前模型存在 non-finite tensor，"
                            "跳过所有 checkpoint 保存。"
                        )
                    else:
                        # Save best
                        is_best = val_metrics["miou"] > best_miou

                        if is_best:
                            best_miou = val_metrics["miou"]

                            best_path = os.path.join(
                                args.save_dir,
                                f"topomamba_{args.dataset}_best.pth"
                            )

                            save_checkpoint(
                                path=best_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                epoch=epoch,
                                best_miou=best_miou,
                                args=args,
                                topology_pairs=topology_pairs
                            )

                            print(
                                f"[Best] 保存最佳模型到: {best_path} | "
                                f"best_mIoU = {best_miou:.4f}"
                            )

                        # Save latest
                        latest_path = os.path.join(
                            args.save_dir,
                            f"topomamba_{args.dataset}_latest.pth"
                        )

                        save_checkpoint(
                            path=latest_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            epoch=epoch,
                            best_miou=best_miou,
                            args=args,
                            topology_pairs=topology_pairs
                        )

            if is_dist_avail_and_initialized():
                dist.barrier()

        if is_main_process():
            print("=" * 80)
            print(f"训练完成，best_mIoU = {best_miou:.4f}")
            print("=" * 80)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
