import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_edge_gt(label: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    label: (B, H, W) long tensor, 0..num_classes-1 or 255
    returns: (B, 1, H, W) float tensor, boundary map
    """
    B, H, W = label.shape
    device = label.device
    # 忽略255：将其设为0（背景），但在后续会通过mask过滤
    valid_mask = (label != 255).float()  # (B, H, W)
    label_safe = torch.where(label == 255, torch.zeros_like(label), label)  # 255 -> 0

    # 使用 max pooling 实现膨胀，-max pooling 实现腐蚀（对类别标签需小心）
    # 更稳健：将标签转为 one-hot 形式，对每个类别分别做膨胀/腐蚀再合并
    # 但为简单，用形态学近似：对每个像素周围区域判断是否与中心不同。
    # 我们直接用标签值与邻居比较生成边界，这比形态学快且容易忽略255。

    # 用绝对差值生成边界：如果像素与4邻域有类别不同，则为边界
    # 4邻域差异
    diff_right = (label_safe[:, :, :-1] != label_safe[:, :, 1:]).float()
    diff_down = (label_safe[:, :-1, :] != label_safe[:, 1:, :]).float()
    edge = torch.zeros_like(label_safe, dtype=torch.float32)
    edge[:, :, :-1] = edge[:, :, :-1] + diff_right
    edge[:, :, 1:] = edge[:, :, 1:] + diff_right
    edge[:, :-1, :] = edge[:, :-1, :] + diff_down
    edge[:, 1:, :] = edge[:, 1:, :] + diff_down
    edge = (edge > 0).float()   # 边界为1，内部为0

    # 排除255像素
    edge = edge * valid_mask
    return edge.unsqueeze(1)   # (B, 1, H, W)


def generate_conn_gt(label: torch.Tensor) -> tuple:
    """
    显存优化版的 4-连接和 8-连接真值生成
    """
    if label.dim() == 2:
        label = label.unsqueeze(0)
    B, H, W = label.shape

    # 使用 bool 类型初始化，极大节省显存
    GT4 = torch.zeros_like(label, dtype=torch.bool)

    # 判断邻居是否相同 (仍为 bool)
    right = (label[:, :, :-1] == label[:, :, 1:])
    down = (label[:, :-1, :] == label[:, 1:, :])

    # 布尔或运算（只要某方向有相同类别的像素，即为 True）
    GT4[:, :, :-1] |= right
    GT4[:, :, 1:] |= right
    GT4[:, :-1, :] |= down
    GT4[:, 1:, :] |= down

    GT8 = GT4.clone()

    diag1 = (label[:, :-1, :-1] == label[:, 1:, 1:])  # 右下
    diag2 = (label[:, :-1, 1:] == label[:, 1:, :-1])  # 左下

    GT8[:, :-1, :-1] |= diag1
    GT8[:, 1:, 1:] |= diag1
    GT8[:, :-1, 1:] |= diag2
    GT8[:, 1:, :-1] |= diag2

    # 最后一步才转换为 Float，喂给 BCE Loss
    return GT4.float(), GT8.float()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    支持 ignore_index 的多类别 Dice 损失
    """
    # 1. 经过 Softmax 得到预测概率
    pred_soft = F.softmax(pred, dim=1)

    # 2. 创建一个掩码，标记出有效的像素 (非 255 的地方为 True)
    valid_mask = (target != ignore_index).unsqueeze(1).float()  # (B, 1, H, W)

    # 3. 将 target 中所有的 255 截断成 0 (或任何合法值)
    # 这样 F.one_hot 就绝对不会再报越界错误了
    target_safe = torch.clamp(target, min=0, max=num_classes - 1)

    # 生成 One-Hot 编码
    target_one_hot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # 4. 将 target 和 pred 中属于忽略区域的像素全部清零！
    # 这样它们既不会增加 intersection，也不会增加 union
    target_one_hot = target_one_hot * valid_mask
    pred_soft = pred_soft * valid_mask

    # 5. 计算交集和并集 (在空间维度上求和)
    intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))  # (B, C)
    union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    # 6. 计算 Dice 系数
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1. - dice.mean()


class TopoMambaLoss(nn.Module):
    """
    TopoMamba 总损失
    包含:
        - 交叉熵损失 (主分割)
        - Dice 损失
        - LBS 边界损失 (4 个辅助头)
        - DCPM 连接性损失
    """

    def __init__(self, num_classes: int, lambda_edge: float = 0.2, lambda_conn: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_edge = lambda_edge
        self.lambda_conn = lambda_conn
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, seg_logits: torch.Tensor, mask: torch.Tensor,
                conn_logits: torch.Tensor, edge_preds: list) -> dict:

        # 🌟 核心修复 1：将预测的 seg_logits 强行上采样到真实标签 mask 的尺寸
        if seg_logits.shape[2:] != mask.shape[1:]:
            seg_logits = F.interpolate(seg_logits, size=mask.shape[1:], mode='bilinear', align_corners=False)

        # 🌟 核心修复 2：将预测的 conn_logits 也对齐到 mask 的尺寸
        if conn_logits.shape[2:] != mask.shape[1:]:
            conn_logits = F.interpolate(conn_logits, size=mask.shape[1:], mode='bilinear', align_corners=False)

        # 现在的 H, W 已经和 mask 完美一致了（都是 512）
        B, C, H, W = seg_logits.shape

        # 1. 交叉熵损失 (现在绝不会报错了)
        ce_loss = self.ce_loss(seg_logits, mask)

        # 2. Dice 损失
        dice = dice_loss(seg_logits, mask, self.num_classes)

        # 3. LBS 边界损失
        edge_gt = generate_edge_gt(mask, kernel_size=3)  # (B, 1, 512, 512)
        lbs_loss = 0.
        for pred_edge in edge_preds:
            # 你这里之前写得很好，已经自带了 F.interpolate，所以辅助头没报错
            pred_up = F.interpolate(pred_edge, size=(H, W), mode='bilinear', align_corners=False)
            lbs_loss += F.binary_cross_entropy_with_logits(pred_up, edge_gt)
        lbs_loss = lbs_loss / len(edge_preds)

        # 4. 连接性损失
        gt4, gt8 = generate_conn_gt(mask)  # (B, 512, 512)
        conn_loss = F.binary_cross_entropy_with_logits(conn_logits[:, 0, :, :], gt4) + \
                    F.binary_cross_entropy_with_logits(conn_logits[:, 1, :, :], gt8)

        # 5. 总损失
        total_loss = ce_loss + dice + self.lambda_edge * lbs_loss + self.lambda_conn * conn_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.item(),
            'dice_loss': dice.item(),
            'lbs_loss': lbs_loss.item(),
            'conn_loss': conn_loss.item(),
        }