import torch
import torch.nn as nn
import torch.nn.functional as F


class SSA_M(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 🌟 修复 1：增加高层特征的通道降维层，将其与低层特征通道数对齐
        self.align_channels = nn.Conv2d(high_ch, low_ch, kernel_size=1, bias=False)

        # 因为对齐后高低特征都是 low_ch，所以合并后的通道数是 low_ch * 2
        self.match_conv = nn.Sequential(
            nn.Conv2d(low_ch * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.diff_conv = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 融合时输入是 L_edge (low_ch) + H_body (low_ch)
        self.fuse_conv = nn.Conv2d(low_ch * 2, out_ch, kernel_size=1, bias=False)

        # 🌟 修复 2：在 __init__ 中定义残差连接（Shortcut）所需的对齐层
        if low_ch != out_ch:
            self.shortcut = nn.Conv2d(low_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()  # 如果通道数一致，直接使用恒等映射

    def forward(self, L, H):
        # 1. 先将高层特征对齐到低层的通道数 (上一轮修好的部分)
        H_aligned = self.align_channels(H)

        # 🌟 核心修复：抛弃固定的 scale_factor=2，改为动态目标尺寸插值
        # L.shape[2:] 取的是 L 的 (Height, Width)
        # 这样无论 H 是比 L 小，还是和 L 一样大，H_up 最终一定会变成和 L 相同的宽和高！
        H_up = F.interpolate(H_aligned, size=L.shape[2:], mode='bilinear', align_corners=False)

        # 2. 现在的 L 和 H_up 在 [Batch, Channel, Height, Width] 上都完美匹配了
        # 可以安全拼接！✅
        M = self.match_conv(torch.cat([L, H_up], dim=1))

        L_edge = L - L * M
        H_body = H_up + H_up * M

        # 可以安全相减！✅
        D = self.diff_conv(torch.abs(L - H_up))

        fused = self.fuse_conv(torch.cat([L_edge, H_body], dim=1))

        G = D * fused

        # 正常的残差连接
        G = G + self.shortcut(L)

        return G