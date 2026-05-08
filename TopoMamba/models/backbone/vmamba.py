import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SS2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        outs = []
        for flip_dims in [[], [2], [3], [2, 3]]:
            xi = x
            if flip_dims:
                xi = torch.flip(xi, dims=flip_dims)

            xi = xi.permute(0, 2, 3, 1).contiguous()
            xi = xi.view(B, H * W, C).float()   # 🔥关键
            xi = self.mamba(xi)

            xi = xi.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            outs.append(xi)

        out = torch.stack(outs, dim=0).mean(dim=0)

        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.proj(out)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

class VSSBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state, d_conv, expand)

        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        identity = x

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.ss2d(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc2(self.act(self.fc1(x)))
        x = x.permute(0, 3, 1, 2).contiguous()

        return x + identity