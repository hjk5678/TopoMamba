import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(8, in_channels // reduction)
        self.conv_compress = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        z_h = torch.mean(x, dim=3, keepdim=True)          # (B, C, H, 1)
        z_w = torch.mean(x, dim=2, keepdim=True)          # (B, C, 1, W)
        z_h_perm = z_h.permute(0, 1, 3, 2)                # (B, C, 1, H)
        z_cat = torch.cat([z_h_perm, z_w], dim=3)         # (B, C, 1, H+W)

        f = self.conv_compress(z_cat)
        f_h, f_w = torch.split(f, [H, W], dim=3)
        f_h = f_h.permute(0, 1, 3, 2)                     # (B, C_reduced, H, 1)
        g_h = torch.sigmoid(self.conv_h(f_h))             # (B, C, H, 1)
        g_w = torch.sigmoid(self.conv_w(f_w))             # (B, C, 1, W)
        return x * g_h * g_w


class MSSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 1×1 卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 膨胀金字塔分支
        self.dil_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for d in [1, 2, 4, 6]
        ])
        self.dil_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.ca = CoordinateAttention(in_channels)

        # 全局池化分支 – 注意：此处不能有 BatchNorm，否则在 1×1 特征图上会报错
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            # BatchNorm 已移除，避免 1×1 特征图上的 batch size 问题
        )

        # 三路融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 分支1：简单 1×1 卷积
        f1 = self.conv1x1(x)                               # (B, C, H, W)

        # 分支2：多尺度膨胀卷积 + 坐标注意力
        dil_outs = [conv(x) for conv in self.dil_convs]    # 各 (B, C, H, W)
        f_cat = torch.cat(dil_outs, dim=1)                 # (B, 4C, H, W)
        f_cat = self.dil_fuse(f_cat)                       # (B, C, H, W)
        f_multi = self.ca(f_cat)                           # 坐标注意力增强

        # 分支3：全局平均池化
        f_gap = self.gap(x)                                # (B, C, 1, 1)
        f_gap = self.gap_conv(f_gap)                       # (B, C, 1, 1)
        f_global = F.interpolate(f_gap, size=(H, W), mode='bilinear', align_corners=True)

        # 三路合并 + 残差
        out = torch.cat([f1, f_multi, f_global], dim=1)    # (B, 3C, H, W)
        out = self.final_conv(out)                         # (B, C, H, W)
        return out + x