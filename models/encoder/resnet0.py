import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels, max_groups=8):
    """
    小 batch 训练时，GroupNorm 通常比 BatchNorm 更稳定。
    DDP 下每张卡 batch_size=2，建议这里用 GroupNorm。
    """
    groups = min(max_groups, num_channels)

    while num_channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, num_channels)


class ConvGNAct(nn.Module):
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

        self.norm = make_group_norm(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SEModule(nn.Module):
    """
    轻量通道注意力。
    对遥感中的车、建筑边缘、植被纹理有一定帮助。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class DepthwiseResidualBlock(nn.Module):
    """
    高分辨率残差块。

    使用 depthwise separable conv：
        1. 保持 H,W 不变；
        2. 增强局部纹理建模；
        3. 比普通 3x3 卷积更省显存和计算量。
    """
    def __init__(self, channels, dilation=1, use_se=True):
        super().__init__()

        padding = dilation

        self.dwconv = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=channels,
            act=True
        )

        self.pwconv = ConvGNAct(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=False
        )

        self.se = SEModule(channels) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)

        # LayerScale，防止新分支初期扰动太大
        self.gamma = nn.Parameter(1e-3 * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        identity = x

        out = self.dwconv(x)
        out = self.pwconv(out)
        out = self.se(out)

        out = identity + self.gamma * out
        out = self.act(out)

        return out


class MultiScaleDetailBlock(nn.Module):
    """
    多尺度细节增强模块。

    三个分支：
        dilation=1：局部纹理
        dilation=2：中尺度边界
        dilation=3：更大感受野

    输出仍然是 [B, C, H, W]。
    """
    def __init__(self, channels):
        super().__init__()

        self.branch1 = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=channels
        )

        self.branch2 = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels
        )

        self.branch3 = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=channels
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            make_group_norm(channels),
            nn.SiLU(inplace=True)
        )

        self.se = SEModule(channels)

        self.gamma = nn.Parameter(1e-3 * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        identity = x

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(out)
        out = self.se(out)

        out = identity + self.gamma * out

        return out


class ResNet0(nn.Module):
    """
    Stronger high-resolution shallow CNN branch.

    替换原始 ResNet0，但保持接口完全一致：

        输入:
            x: [B, 3, H, W]

        输出:
            out: [B, 32, H, W]

    设计目标：
        1. 保留全分辨率细节；
        2. 增强建筑边缘、车、小目标、植被纹理；
        3. 不改变后续 f0_down / SSA-M / decoder；
        4. 比完整 ResNet 更轻，不明显增加显存压力。
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=32,
        num_blocks=3,
        use_multiscale=True
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvGNAct(
                out_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        blocks = []

        for i in range(num_blocks):
            # 前两个块 dilation=1，第三个块 dilation=2
            dilation = 1 if i < 2 else 2
            blocks.append(
                DepthwiseResidualBlock(
                    channels=out_ch,
                    dilation=dilation,
                    use_se=True
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.multiscale = (
            MultiScaleDetailBlock(out_ch)
            if use_multiscale
            else nn.Identity()
        )

        self.out_proj = nn.Sequential(
            ConvGNAct(
                out_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=1,
                bias=True
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.multiscale(x)
        x = self.out_proj(x)

        # 防止极端情况下出现 nan/inf
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )

        return x
