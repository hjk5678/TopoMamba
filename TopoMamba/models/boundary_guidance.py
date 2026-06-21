
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedLoGExtractor(nn.Module):
    """
    固定 LoG / Laplacian 边界提取器。

    输入:
        x: [B, 3, H, W]

    输出:
        edge: [B, 1, H, W]

    说明:
        这里用 Laplacian 近似 LoG 的高频边界响应。
        优点是简单稳定，不引入额外复杂依赖。
    """

    def __init__(self):
        super().__init__()

        kernel = torch.tensor(
            [
                [0.0,  1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0,  1.0, 0.0],
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # RGB -> grayscale
        if x.shape[1] == 3:
            gray = (
                0.299 * x[:, 0:1]
                + 0.587 * x[:, 1:2]
                + 0.114 * x[:, 2:3]
            )
        else:
            gray = x.mean(dim=1, keepdim=True)

        edge = F.conv2d(
            gray,
            self.kernel,
            padding=1
        )

        edge = edge.abs()

        # 每张图归一化，避免数值范围不稳定
        B = edge.shape[0]
        edge_flat = edge.view(B, -1)

        edge_min = edge_flat.min(dim=1)[0].view(B, 1, 1, 1)
        edge_max = edge_flat.max(dim=1)[0].view(B, 1, 1, 1)

        edge = (edge - edge_min) / (edge_max - edge_min + 1e-6)

        return edge


class BoundaryPriorExtractor(nn.Module):
    """
    边界先验提取器。

    由两部分组成：
        1. 固定 LoG/Laplacian 边界响应
        2. 可学习卷积边界响应

    输出:
        boundary_prior: [B, 1, H, W]
    """

    def __init__(self, in_channels=3, hidden_channels=16):
        super().__init__()

        self.fixed_log = FixedLoGExtractor()

        self.learnable_edge = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1
            )
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        fixed_edge = self.fixed_log(x)
        learned_edge = self.learnable_edge(x)

        edge_logits = self.fuse(
            torch.cat(
                [fixed_edge, learned_edge],
                dim=1
            )
        )

        boundary_prior = torch.sigmoid(edge_logits)

        return boundary_prior


class BoundaryStateModulator(nn.Module):
    """
    Boundary-Guided State Modulation, stable version.

    核心改动：
        1. gamma 使用 tanh 限幅，避免 BGSM 后期放大特征导致 NaN；
        2. 使用 GroupNorm，避免小 batch 下 BatchNorm running stats 不稳定；
        3. forward 里对异常数值做 nan_to_num 保护。

    公式：
        feat_out = feat + gamma * gate * value

    其中：
        gamma = tanh(gamma_raw) * max_gamma

    这样 gamma 永远限制在 [-max_gamma, max_gamma]。
    """

    def __init__(
        self,
        channels,
        hidden_channels=None,
        max_gamma=0.1,
        init_gamma=1e-3
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(16, channels // 4)

        self.max_gamma = float(max_gamma)

        def make_gn(c):
            groups = min(32, c)
            while c % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, c)

        self.boundary_proj = nn.Sequential(
            nn.Conv2d(
                1,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            make_gn(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            make_gn(channels),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False
            ),
            make_gn(channels),
            nn.Sigmoid()
        )

        self.boundary_value = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            make_gn(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            make_gn(channels),
        )

        # 让 tanh(gamma_raw) * max_gamma ≈ init_gamma
        init_ratio = init_gamma / self.max_gamma
        init_ratio = max(min(init_ratio, 0.999), -0.999)
        init_raw = torch.atanh(torch.tensor(init_ratio, dtype=torch.float32))

        self.gamma_raw = nn.Parameter(init_raw.view(1))

    def forward(self, feat, boundary):
        feat = torch.nan_to_num(
            feat,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )

        boundary = torch.nan_to_num(
            boundary,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        if boundary.shape[-2:] != feat.shape[-2:]:
            boundary = F.interpolate(
                boundary,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        boundary_feat = self.boundary_proj(boundary)

        fusion = torch.cat(
            [feat, boundary_feat],
            dim=1
        )

        gate = self.gate(fusion)
        value = self.boundary_value(fusion)

        gamma = torch.tanh(self.gamma_raw) * self.max_gamma

        feat_out = feat + gamma * gate * value

        feat_out = torch.nan_to_num(
            feat_out,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )

        return feat_out