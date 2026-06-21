import torch
import torch.nn as nn
import torch.nn.functional as F


class SSA_M(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.align_channels = nn.Conv2d(high_ch, low_ch, kernel_size=1, bias=False)
        self.match_conv = nn.Sequential(
            nn.Conv2d(low_ch * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.diff_conv = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(low_ch * 2, out_ch, kernel_size=1, bias=False)
        if low_ch != out_ch:
            self.shortcut = nn.Conv2d(low_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, L, H):
        H_aligned = self.align_channels(H)
        H_up = F.interpolate(H_aligned, size=L.shape[2:], mode='bilinear', align_corners=True)

        M = self.match_conv(torch.cat([L, H_up], dim=1))

        L_edge = L - L * M
        H_body = H_up + H_up * M

        D = self.diff_conv(torch.abs(L - H_up))

        fused = self.fuse_conv(torch.cat([L_edge, H_body], dim=1))
        G = D * fused

        G = G + self.shortcut(L)

        return G
