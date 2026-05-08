import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeExtractor(nn.Module):
    """边缘提取器：安全、高效且支持混合精度"""

    def __init__(self):
        super().__init__()
        # 定义 Sobel 算子
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        # 注册为 Buffer，这样它们会随着模型一起移动到 GPU，且不会被优化器更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_x.transpose(2, 3))

    def forward(self, feat):
        # 压缩通道至 1
        if feat.shape[1] > 1:
            feat = feat.mean(dim=1, keepdim=True)

        # 动态对齐数据类型 (完美适配 FP16/AMP 训练)
        weight_x = self.sobel_x.to(dtype=feat.dtype)
        weight_y = self.sobel_y.to(dtype=feat.dtype)

        gx = F.conv2d(feat, weight_x, padding=1)
        gy = F.conv2d(feat, weight_y, padding=1)

        # 🌟 核心修复：加上 1e-6 防止除 0 导致梯度 NaN
        E = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return E


class BAUBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 移除了硬编码的 Upsample
        self.edge_extractor = EdgeExtractor()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, G):
        # 🌟 核心修复：动态上采样 x，强行对齐到 G 的尺寸，杜绝一切尺寸不匹配
        x_up = F.interpolate(x, size=G.shape[2:], mode='bilinear', align_corners=False)

        # 提取边界注意力
        E = self.edge_extractor(G)
        E = torch.sigmoid(E)

        # 此时 x_up 的尺寸和 E (继承自 G) 完美一致，安全相乘
        x_edge = x_up * (1.0 + E)

        # 此时 x_edge 的尺寸和 G 完美一致，安全拼接
        out = self.conv(torch.cat([x_edge, G], dim=1))

        return out