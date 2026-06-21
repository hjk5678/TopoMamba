import torch
import torch.nn as nn


def make_group_norm(num_channels, max_groups=32):
    groups = min(max_groups, num_channels)

    while num_channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, num_channels)


class MSTC(nn.Module):
    """
    MSTC: Multi-Scale Topological Constraint Module.

    稳定版改动：
        1. BatchNorm2d -> GroupNorm；
        2. 小 batch DDP 下不再依赖 BN running_mean / running_var；
        3. forward 中加入 nan_to_num，避免辅助分支传播非有限值；
        4. 输出 logits 做轻微 clamp，避免极端 logits。
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=4,
        dropout=0.1
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(32, in_channels // 2)

        self.local_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.context_branch_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.context_branch_2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=4,
                dilation=4,
                bias=False
            ),
            make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 3,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout),

            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )

        local_feat = self.local_branch(x)
        context_feat_1 = self.context_branch_1(x)
        context_feat_2 = self.context_branch_2(x)

        feat = torch.cat(
            [local_feat, context_feat_1, context_feat_2],
            dim=1
        )

        logits = self.fuse(feat)

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0
        )

        logits = torch.clamp(
            logits,
            min=-30.0,
            max=30.0
        )

        return logits