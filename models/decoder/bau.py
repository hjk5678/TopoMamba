import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels, max_groups=32):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _make_log_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generate a zero-mean 2D Laplacian-of-Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    laplacian = (xx.square() + yy.square() - 2.0 * sigma ** 2) / (sigma ** 4)
    kernel = laplacian * gaussian
    return kernel - kernel.mean()


class BAUBlock(nn.Module):
    """
    Boundary-aware residual upsampling block.

    The skip feature is first projected into a learnable one-channel edge
    source. Fixed Sobel and multi-scale LoG responses are normalized and fused
    into a local boundary gate. The gate modulates the upsampled decoder feature
    through a bounded residual scale initialized near zero, so flat regions are
    not amplified by default.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        log_sigmas=(0.8, 1.2, 1.6),
        kernel_size=7,
        use_sobel=True,
        max_boundary_gamma=0.5,
        init_boundary_gamma=1e-3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"LoG kernel_size must be odd, got {kernel_size}")
        if len(log_sigmas) == 0:
            raise ValueError("log_sigmas must contain at least one scale")
        if max_boundary_gamma <= 0:
            raise ValueError(
                f"max_boundary_gamma must be positive, got {max_boundary_gamma}"
            )

        self.use_sobel = bool(use_sobel)
        self.kernel_size = int(kernel_size)
        self.max_boundary_gamma = float(max_boundary_gamma)

        self.x_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            make_group_norm(out_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            make_group_norm(out_channels),
        )

        edge_hidden = max(8, min(32, skip_channels // 4))
        self.edge_source = nn.Sequential(
            nn.Conv2d(skip_channels, edge_hidden, kernel_size=1, bias=False),
            make_group_norm(edge_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(edge_hidden, 1, kernel_size=1, bias=True),
        )

        sobel_kernels = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
                [[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("sobel_kernels", sobel_kernels)

        log_kernels = torch.stack(
            [
                _make_log_kernel(kernel_size, float(sigma))
                for sigma in log_sigmas
            ],
            dim=0,
        ).unsqueeze(1)
        self.register_buffer("log_kernels", log_kernels)

        self.log_scale_logits = nn.Parameter(torch.zeros(len(log_sigmas)))
        self.edge_mix_logits = nn.Parameter(
            torch.tensor([math.log(0.8), math.log(0.2)], dtype=torch.float32)
        )

        gate_hidden = max(8, out_channels // 8)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, gate_hidden, kernel_size=3, padding=1, bias=False),
            make_group_norm(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(out_channels),
        )
        self.out_act = nn.GELU()

        init_ratio = init_boundary_gamma / self.max_boundary_gamma
        init_ratio = max(min(init_ratio, 0.999), -0.999)
        init_raw = torch.atanh(torch.tensor(init_ratio, dtype=torch.float32))
        self.boundary_gamma_raw = nn.Parameter(init_raw.view(1))

    @staticmethod
    def _normalize_response(edge):
        edge_min = edge.amin(dim=(2, 3), keepdim=True)
        edge_max = edge.amax(dim=(2, 3), keepdim=True)
        return (edge - edge_min) / (edge_max - edge_min + 1e-6)

    def _extract_sobel_edge(self, source):
        if not self.use_sobel:
            return torch.zeros_like(source)

        source = F.pad(source, (1, 1, 1, 1), mode="replicate")
        responses = F.conv2d(source, self.sobel_kernels)
        edge = torch.sqrt(responses.square().mean(dim=1, keepdim=True) + 1e-12)
        return self._normalize_response(edge)

    def _extract_log_edge(self, source):
        padding = self.kernel_size // 2
        source = F.pad(
            source,
            (padding, padding, padding, padding),
            mode="replicate",
        )
        responses = F.conv2d(source, self.log_kernels).abs()
        weights = torch.softmax(self.log_scale_logits, dim=0)
        edge = (responses * weights.view(1, -1, 1, 1)).sum(
            dim=1,
            keepdim=True,
        )
        return self._normalize_response(edge)

    def forward(self, x, skip):
        x_up = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x_feat = self.x_proj(x_up)
        skip_feat = self.skip_proj(skip)

        edge_source = torch.nan_to_num(
            self.edge_source(skip),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        sobel_edge = self._extract_sobel_edge(edge_source)
        log_edge = self._extract_log_edge(edge_source)
        mix = torch.softmax(self.edge_mix_logits, dim=0)
        fixed_edge = mix[0] * sobel_edge + mix[1] * log_edge

        gate = self.edge_gate(fixed_edge)

        boundary_gamma = (
            torch.tanh(self.boundary_gamma_raw) * self.max_boundary_gamma
        )
        x_guided = x_feat + boundary_gamma * gate * x_feat

        base = x_feat + skip_feat
        delta = self.fuse(torch.cat([x_guided, skip_feat], dim=1))
        return self.out_act(base + delta)
