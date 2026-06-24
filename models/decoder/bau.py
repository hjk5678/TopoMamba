import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成一个二维高斯核（用于构建 LoG）。"""
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _make_log_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成一个二维高斯拉普拉斯核。"""
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    # 拉普拉斯：核中心为负，周围为正
    laplace_kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (sigma ** 4)
    log_kernel = laplace_kernel * g
    # 归一化使核的总和为零（保持 DC 分量不变）
    log_kernel = log_kernel - log_kernel.mean()
    return log_kernel


class BAUBlock(nn.Module):
    """
    Boundary-Aware Upsample Block with LoG + Sobel edge priors.

    Args:
        in_channels: 输入特征通道数（上采样源）。
        skip_channels: 跳跃特征通道数（来自 SSA‑M 的 G）。
        out_channels: 输出特征通道数。
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 log_sigmas=(1.0, 2.0, 4.0), kernel_size=7,
                 use_sobel=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_sobel = use_sobel

        # ---------- Sobel 核（固定） ----------
        if use_sobel:
            sobel_x = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1., -2., -1.],
                                    [0., 0., 0.],
                                    [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # ---------- 多尺度 LoG 核 ----------
        self.log_sigmas = log_sigmas
        self.n_log = len(log_sigmas)
        log_kernels = []
        for sigma in log_sigmas:
            k = _make_log_kernel(kernel_size, sigma).float()
            log_kernels.append(k.view(1, 1, kernel_size, kernel_size))
        # 堆叠成 (n_log, 1, 1, ksize, ksize)，但 conv2d 只支持单输出通道，我们分别处理
        self.log_kernels = nn.ParameterList([
            nn.Parameter(k, requires_grad=False) for k in log_kernels
        ])
        # 可学习的尺度融合权重（初始化为相等）
        self.log_weights = nn.Parameter(torch.ones(self.n_log) / self.n_log)
        # 可学习的 Sobel 与 LoG 混合权重（初始偏向 Sobel 以保持稳定）
        self.fuse_alpha = nn.Parameter(torch.tensor(0.8))  # Sobel 权重
        self.fuse_beta = nn.Parameter(torch.tensor(0.2))  # LoG 权重

    def _extract_sobel_edge(self, x):
        """x: (B, C, H, W) -> (B, 1, H, W) 多方向 Sobel 梯度幅值"""
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        # 定义四个方向的 Sobel 核（3x3）
        sobel_0 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                                3)
        sobel_90 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1,
                                                                                                                 3, 3)
        sobel_45 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32, device=x.device).view(1, 1,
                                                                                                                 3, 3)
        sobel_135 = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32, device=x.device).view(1, 1,
                                                                                                                  3, 3)

        # 对每个方向做卷积，得到梯度分量
        g0 = F.conv2d(x, sobel_0, padding=1)
        g90 = F.conv2d(x, sobel_90, padding=1)
        g45 = F.conv2d(x, sobel_45, padding=1)
        g135 = F.conv2d(x, sobel_135, padding=1)

        # 合成边缘强度（平方和的平方根，类似 Canny）
        E = torch.sqrt(g0 ** 2 + g90 ** 2 + g45 ** 2 + g135 ** 2 + 1e-8)
        return E

    def _extract_log_edge(self, x):
        """x: (B, C, H, W) -> (B, 1, H, W) 多尺度 LoG 响应融合。"""
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        responses = []
        for idx, kernel in enumerate(self.log_kernels):
            # 对每个通道分别卷积后求和（因为输入为 1 通道）
            resp = F.conv2d(x, kernel, padding=kernel.shape[-1] // 2)
            responses.append(resp.abs())  # 取绝对值作为边缘强度
        # 加权融合（log_weights 经过 softmax 保证和为 1）
        w = torch.softmax(self.log_weights, dim=0)
        E_log = sum(w[i] * responses[i] for i in range(self.n_log))
        return E_log

    def forward(self, x, G):
        # 1. 上采样
        x_up = F.interpolate(
            x,
            size=G.shape[-2:],
            mode='bilinear',
            align_corners=True
        )  # (B, C, H, W)

        # 2. 提取边缘先验（Sobel + LoG）
        E_sobel = self._extract_sobel_edge(G)  # (B, 1, H, W)
        E_log = self._extract_log_edge(G)  # (B, 1, H, W)

        # 融合两种先验（通过可学习权重）
        alpha = torch.sigmoid(self.fuse_alpha)
        beta = torch.sigmoid(self.fuse_beta)
        E_fused = alpha * E_sobel + beta * E_log

        # 归一化到 [0,1] 并作为增强系数
        E_fused = torch.sigmoid(E_fused)  # (B, 1, H, W)

        # 3. 边界感知增强
        x_edge = x_up * (1.0 + E_fused)

        # 4. 与跳跃特征融合
        out = torch.cat([x_edge, G], dim=1)
        return self.conv(out)
