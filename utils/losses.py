# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础工具
# ============================================================
def _resize_like(logits, target_hw):
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False
        )
    return logits


def _valid_mask_from_label(label, ignore_index=255):
    return (label != ignore_index).unsqueeze(1).float()


def downsample_label_nearest(label, size):
    """
    label: [B, H, W]
    size:  (h, w)

    return:
        [B, h, w]
    """
    label_small = F.interpolate(
        label.unsqueeze(1).float(),
        size=size,
        mode="nearest"
    ).squeeze(1).long()

    return label_small


# ============================================================
# 2. 多类 Dice Loss
# ============================================================
def multiclass_dice_loss(
    logits,
    target,
    num_classes,
    ignore_index=255,
    eps=1e-6
):
    logits = _resize_like(logits, target.shape[-2:])

    valid = target != ignore_index

    target_safe = target.clone()
    target_safe[~valid] = 0

    prob = torch.softmax(logits, dim=1)

    one_hot = F.one_hot(
        target_safe.long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    valid = valid.unsqueeze(1).float()

    prob = prob * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)

    intersection = torch.sum(prob * one_hot, dim=dims)
    cardinality = torch.sum(prob + one_hot, dim=dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return 1.0 - dice.mean()


# ============================================================
# 3. 二分类 masked BCE / Dice
# ============================================================
def masked_bce_with_logits(
    logits,
    target,
    valid_mask=None,
    pos_weight=None,
    eps=1e-6
):
    """
    logits: [B, K, H, W]
    target: [B, K, H, W]
    valid_mask: [B, 1, H, W] or [B, K, H, W]
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight
    )

    if valid_mask is not None:
        if valid_mask.shape[1] == 1 and logits.shape[1] != 1:
            valid_mask = valid_mask.expand_as(logits)

        loss = loss * valid_mask
        denom = valid_mask.sum().clamp_min(eps)
        return loss.sum() / denom

    return loss.mean()


def binary_dice_loss_with_logits(
    logits,
    target,
    valid_mask=None,
    eps=1e-6
):
    prob = torch.sigmoid(logits)

    if valid_mask is not None:
        if valid_mask.shape[1] == 1 and logits.shape[1] != 1:
            valid_mask = valid_mask.expand_as(logits)

        prob = prob * valid_mask
        target = target * valid_mask

    dims = (0, 2, 3)

    intersection = torch.sum(prob * target, dim=dims)
    union = torch.sum(prob + target, dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()


# ============================================================
# 4. 普通边界 GT
# ============================================================
def generate_edge_gt(label, ignore_index=255):
    """
    label: [B, H, W]
    return: [B, 1, H, W]
    """
    B, H, W = label.shape

    edge = torch.zeros(
        (B, H, W),
        dtype=torch.float32,
        device=label.device
    )

    valid = label != ignore_index

    # 横向边界
    diff_h = (
        (label[:, :, 1:] != label[:, :, :-1]) &
        valid[:, :, 1:] &
        valid[:, :, :-1]
    )

    edge[:, :, 1:] = torch.maximum(
        edge[:, :, 1:],
        diff_h.float()
    )
    edge[:, :, :-1] = torch.maximum(
        edge[:, :, :-1],
        diff_h.float()
    )

    # 纵向边界
    diff_v = (
        (label[:, 1:, :] != label[:, :-1, :]) &
        valid[:, 1:, :] &
        valid[:, :-1, :]
    )

    edge[:, 1:, :] = torch.maximum(
        edge[:, 1:, :],
        diff_v.float()
    )
    edge[:, :-1, :] = torch.maximum(
        edge[:, :-1, :],
        diff_v.float()
    )

    return edge.unsqueeze(1)


# ============================================================
# 5. 通用 Class-aware mstc GT
# ============================================================
def _any_semantic_boundary(
    label,
    offsets,
    ignore_index=255
):
    """
    任意语义类别之间的边界。

    label: [B, H, W]
    return: [B, H, W]
    """
    B, H, W = label.shape

    out = torch.zeros(
        (B, H, W),
        dtype=torch.float32,
        device=label.device
    )

    valid = label != ignore_index

    for dy, dx in offsets:
        y1_start = max(0, dy)
        y1_end = H + min(0, dy)

        x1_start = max(0, dx)
        x1_end = W + min(0, dx)

        y2_start = max(0, -dy)
        y2_end = H - max(0, dy)

        x2_start = max(0, -dx)
        x2_end = W - max(0, dx)

        l1 = label[:, y1_start:y1_end, x1_start:x1_end]
        l2 = label[:, y2_start:y2_end, x2_start:x2_end]

        v1 = valid[:, y1_start:y1_end, x1_start:x1_end]
        v2 = valid[:, y2_start:y2_end, x2_start:x2_end]

        diff = (l1 != l2) & v1 & v2

        out[:, y1_start:y1_end, x1_start:x1_end] = torch.maximum(
            out[:, y1_start:y1_end, x1_start:x1_end],
            diff.float()
        )

        out[:, y2_start:y2_end, x2_start:x2_end] = torch.maximum(
            out[:, y2_start:y2_end, x2_start:x2_end],
            diff.float()
        )

    return out


def _pair_boundary(
    label,
    class_a_set,
    class_b_set,
    offsets,
    ignore_index=255
):
    """
    指定两个类别集合之间的边界。

    label: [B, H, W]
    class_a_set: list[int]
    class_b_set: list[int]
    return: [B, H, W]
    """
    B, H, W = label.shape

    out = torch.zeros(
        (B, H, W),
        dtype=torch.float32,
        device=label.device
    )

    valid = label != ignore_index

    a_mask = torch.zeros_like(label, dtype=torch.bool)
    b_mask = torch.zeros_like(label, dtype=torch.bool)

    for c in class_a_set:
        a_mask |= (label == int(c))

    for c in class_b_set:
        b_mask |= (label == int(c))

    for dy, dx in offsets:
        y1_start = max(0, dy)
        y1_end = H + min(0, dy)

        x1_start = max(0, dx)
        x1_end = W + min(0, dx)

        y2_start = max(0, -dy)
        y2_end = H - max(0, dy)

        x2_start = max(0, -dx)
        x2_end = W - max(0, dx)

        l1_a = a_mask[:, y1_start:y1_end, x1_start:x1_end]
        l1_b = b_mask[:, y1_start:y1_end, x1_start:x1_end]

        l2_a = a_mask[:, y2_start:y2_end, x2_start:x2_end]
        l2_b = b_mask[:, y2_start:y2_end, x2_start:x2_end]

        v1 = valid[:, y1_start:y1_end, x1_start:x1_end]
        v2 = valid[:, y2_start:y2_end, x2_start:x2_end]

        pair_diff = (
            ((l1_a & l2_b) | (l1_b & l2_a)) &
            v1 &
            v2
        )

        out[:, y1_start:y1_end, x1_start:x1_end] = torch.maximum(
            out[:, y1_start:y1_end, x1_start:x1_end],
            pair_diff.float()
        )

        out[:, y2_start:y2_end, x2_start:x2_end] = torch.maximum(
            out[:, y2_start:y2_end, x2_start:x2_end],
            pair_diff.float()
        )

    return out


def generate_class_aware_conn_gt(
    label,
    topology_pairs,
    ignore_index=255
):
    """
    根据 topology_pairs 动态生成 mstc GT。

    label:
        [B, H, W] or [H, W]

    topology_pairs:
        [
            ("any_boundary", None, None),
            ("low_tree", [2], [3]),
            ...
        ]

    return:
        conn_gt: [B, K, H, W]
    """
    if label.dim() == 2:
        label = label.unsqueeze(0)

    label = label.long()

    offsets_8 = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    maps = []

    for name, class_a_set, class_b_set in topology_pairs:
        if class_a_set is None or class_b_set is None:
            boundary = _any_semantic_boundary(
                label,
                offsets=offsets_8,
                ignore_index=ignore_index
            )
        else:
            boundary = _pair_boundary(
                label,
                class_a_set=class_a_set,
                class_b_set=class_b_set,
                offsets=offsets_8,
                ignore_index=ignore_index
            )

        maps.append(boundary)

    if len(maps) == 0:
        raise RuntimeError("topology_pairs 为空，无法生成 mstc GT。")

    conn_gt = torch.stack(maps, dim=1)

    return conn_gt.float()


def generate_conn_gt(
    label,
    topology_pairs=None,
    num_classes=None,
    ignore_index=255
):
    """
    兼容旧接口。

    新版本返回：
        conn_gt: [B, K, H, W]
    """
    if topology_pairs is None:
        topology_pairs = [
            ("any_boundary", None, None),
        ]

    return generate_class_aware_conn_gt(
        label=label,
        topology_pairs=topology_pairs,
        ignore_index=ignore_index
    )




# ============================================================
# 7. TopoMambaLoss
# ============================================================
class TopoMambaLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_edge=0.2,
        lambda_conn=0.1,
        ignore_index=255,
        class_weights=None,
        topology_pairs=None,
        conn_scale_weights=None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_edge = lambda_edge
        self.lambda_conn = lambda_conn
        self.ignore_index = ignore_index

        if topology_pairs is None:
            topology_pairs = [
                ("any_boundary", None, None),
            ]

        self.topology_pairs = topology_pairs

        if conn_scale_weights is None:
            conn_scale_weights = [0.4, 0.3, 0.2, 0.1]

        self.conn_scale_weights = conn_scale_weights

        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"class_weights 长度必须等于 num_classes, "
                    f"当前 len(class_weights)={len(class_weights)}, "
                    f"num_classes={num_classes}"
                )

            class_weights = torch.tensor(
                class_weights,
                dtype=torch.float32
            )

            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # 主分割 CE Loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index
        )

    def _single_conn_loss(self, conn_logit, target, device):
        """
        单尺度 mstc loss。

        conn_logit:
            [B, K, h, w]
        target:
            [B, H, W]
        """
        if conn_logit is None:
            return torch.tensor(0.0, device=device)

        target_i = downsample_label_nearest(
            target,
            size=conn_logit.shape[-2:]
        )

        conn_gt_i = generate_class_aware_conn_gt(
            label=target_i,
            topology_pairs=self.topology_pairs,
            ignore_index=self.ignore_index
        )

        valid_mask_i = _valid_mask_from_label(
            target_i,
            ignore_index=self.ignore_index
        )

        if conn_logit.shape[1] != conn_gt_i.shape[1]:
            raise RuntimeError(
                f"conn_logits 通道数和 conn_gt 不一致: "
                f"conn_logits={tuple(conn_logit.shape)}, "
                f"conn_gt={tuple(conn_gt_i.shape)}. "
                f"mstc 输出通道数应等于 len(topology_pairs)={len(self.topology_pairs)}。"
            )

        bce_conn = masked_bce_with_logits(
            conn_logit,
            conn_gt_i,
            valid_mask=valid_mask_i
        )

        dice_conn = binary_dice_loss_with_logits(
            conn_logit,
            conn_gt_i,
            valid_mask=valid_mask_i
        )

        return bce_conn + dice_conn

    def _multi_scale_conn_loss(self, conn_logits, target, device):
        if conn_logits is None:
            return torch.tensor(0.0, device=device)

        if isinstance(conn_logits, (list, tuple)):
            num_scales = len(conn_logits)

            if num_scales == 0:
                return torch.tensor(0.0, device=device)

            if len(self.conn_scale_weights) == num_scales:
                scale_weights = self.conn_scale_weights
            else:
                scale_weights = [1.0 / num_scales for _ in range(num_scales)]

            losses = []

            for w, conn_logit in zip(scale_weights, conn_logits):
                losses.append(
                    float(w) * self._single_conn_loss(
                        conn_logit=conn_logit,
                        target=target,
                        device=device
                    )
                )

            return sum(losses)

        return self._single_conn_loss(
            conn_logit=conn_logits,
            target=target,
            device=device
        )

    def forward(
        self,
        seg_logits,
        target,
        conn_logits=None,
        edge_preds=None
    ):
        target = target.long()
        device = seg_logits.device

        # ----------------------------------------------------
        # 1. Segmentation Loss: CE + Dice
        # ----------------------------------------------------
        seg_logits = _resize_like(
            seg_logits,
            target.shape[-2:]
        )

        ce = self.ce_loss(
            seg_logits,
            target
        )

        dice = multiclass_dice_loss(
            logits=seg_logits,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index
        )

        seg_loss = ce + dice

        # ----------------------------------------------------
        # 2. LBS Edge Loss
        # ----------------------------------------------------
        edge_loss = torch.tensor(0.0, device=device)

        if edge_preds is not None and self.lambda_edge > 0:
            edge_gt = generate_edge_gt(
                target,
                ignore_index=self.ignore_index
            )

            valid_mask = _valid_mask_from_label(
                target,
                ignore_index=self.ignore_index
            )

            if torch.is_tensor(edge_preds):
                edge_preds_list = [edge_preds]
            else:
                edge_preds_list = list(edge_preds)

            edge_losses = []

            for edge_logit in edge_preds_list:
                if edge_logit is None:
                    continue

                if edge_logit.dim() == 3:
                    edge_logit = edge_logit.unsqueeze(1)

                edge_logit = _resize_like(
                    edge_logit,
                    target.shape[-2:]
                )

                bce_edge = masked_bce_with_logits(
                    edge_logit,
                    edge_gt,
                    valid_mask=valid_mask
                )

                dice_edge = binary_dice_loss_with_logits(
                    edge_logit,
                    edge_gt,
                    valid_mask=valid_mask
                )

                edge_losses.append(bce_edge + dice_edge)

            if len(edge_losses) > 0:
                edge_loss = sum(edge_losses) / len(edge_losses)

        # ----------------------------------------------------
        # 3. Multi-scale mstc
        # Loss
        # ----------------------------------------------------
        conn_loss = torch.tensor(0.0, device=device)

        if conn_logits is not None and self.lambda_conn > 0:
            conn_loss = self._multi_scale_conn_loss(
                conn_logits=conn_logits,
                target=target,
                device=device
            )

        # ----------------------------------------------------
        # 4. Total Loss
        # ----------------------------------------------------
        total_loss = (
            seg_loss
            + self.lambda_edge * edge_loss
            + self.lambda_conn * conn_loss
        )

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss.detach(),
            "ce_loss": ce.detach(),
            "dice_loss": dice.detach(),
            "edge_loss": edge_loss.detach(),
            "lbs_loss": edge_loss.detach(),
            "conn_loss": conn_loss.detach(),
        }
