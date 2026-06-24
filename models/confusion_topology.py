import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础卷积块
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        act=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# 2. Confusion-Aware Refinement Module
# ============================================================
class ConfusionAwareRefinementModule(nn.Module):
    """
    CARM: Confusion-Aware Refinement Module

    目的：
        专门处理 Potsdam 中容易混淆的类别：
        0 impervious_surface
        2 low_vegetation
        3 tree
        5 clutter

    输入：
        feat:            [B, C, H, W] decoder 输出特征
        base_seg_logits: [B, num_classes, H, W] 原始分割 logits

    输出：
        refined_seg_logits: [B, num_classes, H, W]
        confusion_logits:   [B, K, H, W]
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        confusion_classes=(0, 2, 3, 5),
        hidden_channels=None,
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.confusion_classes = list(confusion_classes)
        self.num_confusion_classes = len(self.confusion_classes)

        if hidden_channels is None:
            hidden_channels = max(in_channels, 32)

        # ----------------------------------------------------
        # 1. 易混类别辅助分类头
        # ----------------------------------------------------
        self.confusion_head = nn.Sequential(
            ConvBNAct(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                hidden_channels,
                self.num_confusion_classes,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # 2. 使用易混类别概率作为先验，引导主分割 logits 修正
        # ----------------------------------------------------
        refine_in_channels = in_channels + self.num_confusion_classes

        self.refine_conv = nn.Sequential(
            ConvBNAct(
                refine_in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                hidden_channels,
                num_classes,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # 3. 残差强度，初始为 0，避免刚开始破坏原模型
        # ----------------------------------------------------
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat, base_seg_logits):
        confusion_logits = self.confusion_head(feat)

        confusion_prob = torch.softmax(
            confusion_logits,
            dim=1
        )

        refine_input = torch.cat(
            [feat, confusion_prob],
            dim=1
        )

        delta_logits = self.refine_conv(refine_input)

        refined_seg_logits = base_seg_logits + self.gamma * delta_logits

        return refined_seg_logits, confusion_logits


# ============================================================
# 3. Class-Aware mstc
# ============================================================
class ClassAwaremstc(nn.Module):
    """
    Class-Aware mstc

    代替原来的 2-channel mstc。

    输出 4 个通道：
        channel 0: any semantic discontinuity
        channel 1: low_vegetation-tree discontinuity
        channel 2: clutter-vegetation discontinuity
        channel 3: clutter-impervious discontinuity

    对应 Potsdam 类别：
        0 impervious_surface
        1 building
        2 low_vegetation
        3 tree
        4 car
        5 clutter
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=4,
        dropout=0.1
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels, 32)

        self.local_branch = nn.Sequential(
            ConvBNAct(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            )
        )

        self.context_branch_1 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.context_branch_2 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=4,
            dilation=4
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                hidden_channels * 3,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1
            )
        )

    def forward(self, feat):
        local_feat = self.local_branch(feat)
        context_feat_1 = self.context_branch_1(feat)
        context_feat_2 = self.context_branch_2(feat)

        fused = torch.cat(
            [local_feat, context_feat_1, context_feat_2],
            dim=1
        )

        logits = self.fuse(fused)

        return logits