import torch
import torch.nn as nn
from models.backbone.vmamba import VSSBlock

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class DGF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_interact = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        self.norm = nn.LayerNorm(channels)

        # ⭐ learnable fusion scale（关键稳定器）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_local, F_global):

        # ⭐ stabilize global feature
        F_global = torch.tanh(F_global)   # 比 sigmoid 更稳定

        G = torch.sigmoid(self.channel_interact(F_global))

        LG = G * F_local
        LG = self.channel_interact(LG)

        out = F_local + self.gamma * LG   # ⭐ controlled residual fusion

        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

class TPDBlock(nn.Module):
    def __init__(self, dim, num_cnn_blocks=3, num_vss_blocks=2):
        super().__init__()

        self.cnn_branch = nn.Sequential(
            *[ResBlock(dim) for _ in range(num_cnn_blocks)]
        )

        self.vmamba_branch = nn.Sequential(
            *[VSSBlock(dim) for _ in range(num_vss_blocks)]
        )

        self.fusion = DGF(dim)

        # ⭐ learnable residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        F_local = self.cnn_branch(x)

        # ⭐ stabilize VSS input
        F_global = self.vmamba_branch(x)
        F_global = torch.clamp(F_global, -10, 10)

        fused = self.fusion(F_local, F_global)

        # ⭐ controlled residual (VERY IMPORTANT)
        return x + self.alpha * fused

class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.reduction = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reduction(x)

def build_encoder_stages(dims, depths):
    """
    dims: [C1, C2, C3, C4]
    depths: [N1, N2, N3, N4] 每个stage的TPDB数量
    returns: nn.ModuleList stages, nn.ModuleList mergings
    """
    stages = nn.ModuleList()
    mergings = nn.ModuleList()
    for i in range(4):
        stage = nn.Sequential(*[TPDBlock(dims[i]) for _ in range(depths[i])])
        stages.append(stage)
        if i < 3:
            merge = PatchMerging(dims[i], dims[i+1])
            mergings.append(merge)
    return stages, mergings