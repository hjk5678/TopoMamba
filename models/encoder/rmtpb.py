import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from models.backbone.vmamba import VSSBlock


# ============================================================
# ResNet18-style BasicBlock
# ============================================================
class ResBlock(nn.Module):
    """
    ResNet18-style BasicBlock.

    保持输入输出通道一致，不做下采样：
        input:  [B, C, H, W]
        output: [B, C, H, W]

    注意：
        RMTPB 的下采样由外部 PatchMerging 完成，
        所以这里 stride 固定为 1。
    """

    expansion = 1

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


# ============================================================
# Dynamic Gated Fusion
# ============================================================
class _OldDGF(nn.Module):
    """
    Dynamic Gated Fusion.

    F_local:
        CNN local branch feature

    F_global:
        VMamba / VSS global branch feature

    输出：
        融合后的 local-global feature
    """

    def __init__(self, channels):
        super().__init__()

        self.channel_interact = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

        self.norm = nn.LayerNorm(channels)

        # learnable fusion scale
        # 初始为 0，训练初期不破坏 CNN local feature
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_local, F_global):
        # stabilize global feature
        F_global = torch.tanh(F_global)

        gate = torch.sigmoid(self.channel_interact(F_global))

        local_global = gate * F_local
        local_global = self.channel_interact(local_global)

        out = F_local + self.gamma * local_global

        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


class DGF(nn.Module):
    """
    Learnable multi-path fusion.

    Inputs can be:
        CNN local + VSS global
    or:
        CNN local + VSS global + each VSS path feature.

    The module predicts sample-wise, channel-wise softmax weights over paths.
    """

    def __init__(
        self,
        channels,
        input_channels=None,
        reduction=4,
        init_gamma=1e-3,
    ):
        super().__init__()

        if input_channels is None:
            input_channels = [channels, channels]

        self.channels = int(channels)
        self.input_channels = [int(ch) for ch in input_channels]
        self.num_inputs = len(self.input_channels)

        if self.num_inputs < 2:
            raise ValueError("DGF needs at least two input features.")

        self.input_proj = nn.ModuleList()
        for in_ch in self.input_channels:
            if in_ch == self.channels:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_ch,
                            self.channels,
                            kernel_size=1,
                            bias=False
                        ),
                        nn.BatchNorm2d(self.channels),
                        nn.ReLU(inplace=True),
                    )
                )

        hidden = max(self.channels // reduction, 16)

        self.score_mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.channels),
        )

        self.path_prior = nn.Parameter(
            torch.zeros(self.num_inputs, self.channels)
        )

        with torch.no_grad():
            self.path_prior[0].fill_(0.5)

        self.refine = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
                bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.norm = nn.LayerNorm(self.channels)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, F_local, F_global, path_features=None):
        if path_features is None:
            path_features = []

        features = [F_local, F_global] + list(path_features)

        if len(features) != self.num_inputs:
            raise ValueError(
                f"DGF expected {self.num_inputs} features, "
                f"got {len(features)}."
            )

        projected = []
        descriptors = []

        for idx, (feat, proj) in enumerate(zip(features, self.input_proj)):
            feat = torch.nan_to_num(
                feat,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4
            )

            if idx > 0:
                feat = torch.tanh(feat)

            feat = proj(feat.contiguous())
            projected.append(feat)
            descriptors.append(feat.mean(dim=(2, 3)))

        desc = torch.stack(descriptors, dim=1)
        scores = self.score_mlp(desc)
        scores = scores + self.path_prior.unsqueeze(0)
        weights = torch.softmax(scores, dim=1)

        fused = 0.0
        for idx, feat in enumerate(projected):
            weight = weights[:, idx].unsqueeze(-1).unsqueeze(-1)
            fused = fused + weight * feat

        fused = self.refine(fused)
        out = F_local + self.gamma * fused

        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


def _build_vss_block(
    dim,
    drop_path=0.0,
    use_checkpoint=False,
    forward_type="v05_noz",
    ssm_conv=3,
):
    """Build one channel-first VSS block with the project defaults."""
    return VSSBlock(
        hidden_dim=dim,
        drop_path=drop_path,
        channel_first=True,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=ssm_conv,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type=forward_type,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        use_checkpoint=use_checkpoint,
        post_norm=False,
    )


class LocalWindowCrossVSS(nn.Module):
    """Run cross scan independently inside non-overlapping local windows."""

    def __init__(
        self,
        dim,
        num_blocks=1,
        window_size=8,
        shifted=False,
        drop_path=0.0,
        use_checkpoint=False,
    ):
        super().__init__()

        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.window_size = int(window_size)
        self.shifted = bool(shifted)
        self.blocks = nn.Sequential(*[
            _build_vss_block(
                dim=dim,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
                forward_type="v05_noz",
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.window_size, max(H, W))
        shift = ws // 2 if self.shifted and ws > 1 else 0

        pad_left = shift
        pad_top = shift
        pad_right = (ws - (W + pad_left) % ws) % ws
        pad_bottom = (ws - (H + pad_top) % ws) % ws

        if pad_left or pad_right or pad_top or pad_bottom:
            x = F.pad(
                x,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )

        Hp, Wp = x.shape[-2:]
        nh, nw = Hp // ws, Wp // ws

        windows = (
            x.view(B, C, nh, ws, nw, ws)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(B * nh * nw, C, ws, ws)
        )

        windows = self.blocks(windows)

        x = (
            windows.view(B, nh, nw, C, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(B, C, Hp, Wp)
        )

        return x[:, :, pad_top:pad_top + H, pad_left:pad_left + W]


class DiagonalCrossVSS(nn.Module):
    """
    Bidirectionally scan the main- and anti-diagonal zigzag routes.

    Spatial tokens are reordered into a diagonal sequence before a
    bidirectional VSS block and restored to their original coordinates after
    scanning. The VSS depthwise convolution is disabled on this path because
    convolution on the reordered grid would not represent local 2D neighbors.
    """

    def __init__(
        self,
        dim,
        num_blocks=1,
        drop_path=0.0,
        use_checkpoint=False,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"DiagonalCrossVSS needs an even dim, got {dim}")

        branch_dim = dim // 2
        self.main_blocks = nn.Sequential(*[
            _build_vss_block(
                dim=branch_dim,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
                forward_type="v052d_noz",
                ssm_conv=1,
            )
            for _ in range(num_blocks)
        ])
        self.anti_blocks = nn.Sequential(*[
            _build_vss_block(
                dim=branch_dim,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
                forward_type="v052d_noz",
                ssm_conv=1,
            )
            for _ in range(num_blocks)
        ])
        self._index_cache = {}

    @staticmethod
    def _build_diagonal_indices(H, W, device):
        main_order = []

        for diagonal in range(H + W - 1):
            row_start = max(0, diagonal - W + 1)
            row_end = min(H - 1, diagonal)
            coords = [
                (row, diagonal - row)
                for row in range(row_start, row_end + 1)
            ]

            # Zigzag between neighboring diagonals to avoid a large jump at
            # every diagonal boundary.
            if diagonal % 2 == 0:
                coords.reverse()

            main_order.extend(row * W + col for row, col in coords)

        main = torch.tensor(main_order, dtype=torch.long, device=device)
        rows = torch.div(main, W, rounding_mode="floor")
        cols = main.remainder(W)
        anti = rows * W + (W - 1 - cols)

        arange = torch.arange(H * W, dtype=torch.long, device=device)
        main_inverse = torch.empty_like(main)
        anti_inverse = torch.empty_like(anti)
        main_inverse[main] = arange
        anti_inverse[anti] = arange

        return main, main_inverse, anti, anti_inverse

    def _get_indices(self, H, W, device):
        key = (H, W, device.type, device.index)
        indices = self._index_cache.get(key)

        if indices is None:
            indices = self._build_diagonal_indices(H, W, device)
            self._index_cache[key] = indices

        return indices

    @staticmethod
    def _scan_branch(x, blocks, order, inverse):
        B, C, H, W = x.shape
        ordered = x.flatten(2).index_select(-1, order).view(B, C, H, W)
        ordered = blocks(ordered)
        restored = ordered.flatten(2).index_select(-1, inverse)
        return restored.view(B, C, H, W)

    def forward(self, x):
        _, _, H, W = x.shape
        main_idx, main_inv, anti_idx, anti_inv = self._get_indices(
            H,
            W,
            x.device,
        )
        main_x, anti_x = torch.chunk(x, 2, dim=1)
        main_out = self._scan_branch(
            main_x,
            self.main_blocks,
            main_idx,
            main_inv,
        )
        anti_out = self._scan_branch(
            anti_x,
            self.anti_blocks,
            anti_idx,
            anti_inv,
        )
        return torch.cat([main_out, anti_out], dim=1)


class AtrousCrossVSS(nn.Module):
    """Cross scan on interleaved spatial lattices with a configurable rate."""

    def __init__(
        self,
        dim,
        num_blocks=1,
        rate=2,
        drop_path=0.0,
        use_checkpoint=False,
    ):
        super().__init__()

        if rate < 1:
            raise ValueError(f"atrous rate must be positive, got {rate}")

        self.rate = int(rate)
        self.blocks = nn.Sequential(*[
            _build_vss_block(
                dim=dim,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
                forward_type="v05_noz",
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        if self.rate == 1:
            return self.blocks(x)

        B, C, H, W = x.shape
        rate = self.rate
        pad_bottom = (rate - H % rate) % rate
        pad_right = (rate - W % rate) % rate

        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")

        Hp, Wp = x.shape[-2:]
        gh, gw = Hp // rate, Wp // rate

        lattices = (
            x.view(B, C, gh, rate, gw, rate)
            .permute(0, 3, 5, 1, 2, 4)
            .contiguous()
            .view(B * rate * rate, C, gh, gw)
        )

        lattices = self.blocks(lattices)

        x = (
            lattices.view(B, rate, rate, C, gh, gw)
            .permute(0, 3, 4, 1, 5, 2)
            .contiguous()
            .view(B, C, Hp, Wp)
        )

        return x[:, :, :H, :W]


class ResidualMultiPathVSS(nn.Module):
    """
    Residual multi-path VSS wrapper inspired by multi-path vision Mamba.

    The input channels are split into four complementary paths:
        path 1: global cross scan
        path 2: local-window cross scan
        path 3: main/anti-diagonal bidirectional scan
        path 4: atrous cross scan

    The paths are fused back with a controlled residual scale initialized
    near zero.
    """

    def __init__(
        self,
        dim,
        num_blocks=1,
        num_paths=4,
        drop_path=0.0,
        use_checkpoint=False,
        local_window_size=8,
        local_window_shift=False,
        atrous_rate=2,
    ):
        super().__init__()

        if num_paths != 4:
            raise ValueError(
                "The redesigned multi-path VSS uses exactly four paths: "
                f"global/local/diagonal/atrous, got num_paths={num_paths}."
            )

        if dim % num_paths != 0:
            raise ValueError(
                f"dim must be divisible by num_paths, got "
                f"dim={dim}, num_paths={num_paths}"
            )

        self.num_paths = int(num_paths)
        self.path_dim = dim // self.num_paths
        self.path_names = (
            "global_cross",
            "local_window_cross",
            "diagonal_cross",
            "atrous_cross",
        )

        self.paths = nn.ModuleList([
            nn.Sequential(*[
                _build_vss_block(
                    dim=self.path_dim,
                    drop_path=drop_path,
                    use_checkpoint=use_checkpoint,
                    forward_type="v05_noz",
                )
                for _ in range(num_blocks)
            ]),
            LocalWindowCrossVSS(
                dim=self.path_dim,
                num_blocks=num_blocks,
                window_size=local_window_size,
                shifted=local_window_shift,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
            ),
            DiagonalCrossVSS(
                dim=self.path_dim,
                num_blocks=num_blocks,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
            ),
            AtrousCrossVSS(
                dim=self.path_dim,
                num_blocks=num_blocks,
                rate=atrous_rate,
                drop_path=drop_path,
                use_checkpoint=use_checkpoint,
            ),
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward_paths(self, x):
        chunks = torch.chunk(x, self.num_paths, dim=1)

        outs = []
        for path, chunk in zip(self.paths, chunks):
            out = path(chunk.contiguous())
            outs.append(out)

        return outs

    def forward_with_paths(self, x):
        outs = self.forward_paths(x)

        out = torch.cat(outs, dim=1)
        out = self.fuse(out)

        return x + self.gamma * out, outs

    def forward(self, x):
        out, _ = self.forward_with_paths(x)
        return out


# ============================================================
# RMTPBlock
# ============================================================
class RMTPBlock(nn.Module):
    """
    Residual Multi-path Topological Perception Block.

    保留你的原始设计：
        1. CNN branch: ResNet18-style local branch
        2. VMamba branch: official VSSBlock global branch
        3. DGF learnable multi-path fusion
        4. controlled residual

    输入输出：
        x:   [B, C, H, W]
        out: [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        num_cnn_blocks=2,
        num_vss_blocks=2,
        vss_use_checkpoint=False,
        drop_path=0.0,
        use_rmp_vss=False,
        rmp_num_paths=4,
        rmp_window_size=8,
        rmp_atrous_rate=2,
        rmp_window_shift=False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.use_rmp_vss = bool(use_rmp_vss)
        self.rmp_num_paths = int(rmp_num_paths)

        # ----------------------------------------------------
        # CNN local branch
        # ----------------------------------------------------
        self.cnn_branch = nn.Sequential(
            *[
                ResBlock(dim)
                for _ in range(num_cnn_blocks)
            ]
        )

        # ----------------------------------------------------
        # VMamba global branch
        #
        # 这里必须 channel_first=True，
        # 因为 RMTPB 输入是 [B, C, H, W]。
        #
        # 参数与 vssm1_tiny_0230s_ckpt_epoch_264.pth 对齐：
        #   ssm_d_state=1
        #   ssm_ratio=1.0
        #   forward_type="v05_noz"
        # ----------------------------------------------------
        if self.use_rmp_vss:
            self.vmamba_branch = ResidualMultiPathVSS(
                dim=dim,
                num_blocks=num_vss_blocks,
                num_paths=self.rmp_num_paths,
                drop_path=drop_path,
                use_checkpoint=vss_use_checkpoint,
                local_window_size=rmp_window_size,
                local_window_shift=rmp_window_shift,
                atrous_rate=rmp_atrous_rate,
            )
        else:
            self.vmamba_branch = nn.Sequential(
                *[
                    VSSBlock(
                        hidden_dim=dim,
                        drop_path=drop_path,
                        channel_first=True,

                        ssm_d_state=1,
                        ssm_ratio=1.0,
                        ssm_dt_rank="auto",
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=3,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init="v0",
                        forward_type="v05_noz",

                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,

                        use_checkpoint=vss_use_checkpoint,
                        post_norm=False,
                    )
                    for _ in range(num_vss_blocks)
                ]
            )

        if self.use_rmp_vss:
            fusion_input_channels = [
                dim,
                dim,
            ] + [
                dim // self.rmp_num_paths
                for _ in range(self.rmp_num_paths)
            ]
        else:
            fusion_input_channels = [dim, dim]

        self.fusion = DGF(
            dim,
            input_channels=fusion_input_channels
        )

        # learnable residual scale
        # 初始为 0，避免一开始扰动主干特征
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        F_local = self.cnn_branch(x)

        path_features = None

        if self.use_rmp_vss:
            F_global, path_features = self.vmamba_branch.forward_with_paths(x)
        else:
            F_global = self.vmamba_branch(x)

        F_global = torch.clamp(F_global, min=-10.0, max=10.0)

        fused = self.fusion(F_local, F_global, path_features=path_features)

        out = x + self.alpha * fused

        return out


# ============================================================
# Patch Merging
# ============================================================
class PatchMerging(nn.Module):
    """
    Stage downsample.

    输入：
        [B, C_in, H, W]

    输出：
        [B, C_out, H/2, W/2]
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.reduction = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reduction(x)


# ============================================================
# ResNet18 pretrained loading for CNN branch
# ============================================================
def _load_single_resblock_from_torchvision_block(target_block, source_block):
    """
    将 torchvision ResNet18 BasicBlock 权重加载到我们的 ResBlock。

    仅加载 shape 完全一致的参数。
    """
    src_state = source_block.state_dict()
    tgt_state = target_block.state_dict()

    loaded = 0
    skipped = 0

    new_state = {}

    for k, v in tgt_state.items():
        if k in src_state and src_state[k].shape == v.shape:
            new_state[k] = src_state[k]
            loaded += 1
        else:
            new_state[k] = v
            skipped += 1

    target_block.load_state_dict(new_state, strict=True)

    return loaded, skipped


def load_resnet18_pretrained_to_rmtpb_cnn(
    stages,
    pretrained=True,
    local_path=None,
    verbose=True
):
    """
    将 ImageNet pretrained ResNet18 的 BasicBlock 权重加载到 RMTPB 的 CNN branch。

    重要说明：
        不能直接使用完整 ResNet18，因为完整 ResNet18 会下采样；
        RMTPB 的下采样已经由 PatchMerging 完成。

    加载策略：
        stage0, dim=64:
            使用 resnet18.layer1 的 block，shape 完全匹配。

        stage1, dim=128:
            使用 resnet18.layer2[1]，因为 layer2[0] 是 stride=2 且输入通道 64->128，
            不适合直接加载到 RMTPB 内部的 128->128 ResBlock。

        stage2, dim=256:
            使用 resnet18.layer3[1]。

        stage3, dim=512:
            使用 resnet18.layer4[1]。

    对于一个 stage 内多个 RMTPBlock / 多个 CNN ResBlock：
        使用可匹配的 ResNet18 block 进行循环初始化。
        这样所有 CNN branch 都有 ImageNet 初始化先验。
    """

    if not pretrained:
        if verbose:
            print("[ResNet18-CNN] pretrained=False, CNN branch uses random init.")
        return

    try:
        if local_path is not None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(local_path)

            ckpt = torch.load(local_path, map_location="cpu")

            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            model = resnet18(weights=None)
            model.load_state_dict(state_dict, strict=False)

            if verbose:
                print(f"[ResNet18-CNN] Load local ResNet18 weights: {local_path}")

        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

            if verbose:
                print("[ResNet18-CNN] Load torchvision ResNet18 ImageNet weights.")

    except Exception as e:
        print(f"[ResNet18-CNN] WARNING: failed to load pretrained ResNet18: {e}")
        print("[ResNet18-CNN] CNN branch will use random initialization.")
        return

    # 可匹配的 torchvision source blocks
    # stage0: layer1 有两个 64->64 block，都可用
    # stage1: layer2[1] 是 128->128
    # stage2: layer3[1] 是 256->256
    # stage3: layer4[1] 是 512->512
    source_blocks = {
        0: [model.layer1[0], model.layer1[1]],
        1: [model.layer2[1]],
        2: [model.layer3[1]],
        3: [model.layer4[1]],
    }

    total_loaded = 0
    total_skipped = 0
    total_blocks = 0

    for stage_idx, stage in enumerate(stages):
        src_list = source_blocks[stage_idx]

        for rmtpb_idx, rmtpb in enumerate(stage):
            if not hasattr(rmtpb, "cnn_branch"):
                continue

            for block_idx, cnn_block in enumerate(rmtpb.cnn_branch):
                if not isinstance(cnn_block, ResBlock):
                    continue

                src_block = src_list[block_idx % len(src_list)]

                loaded, skipped = _load_single_resblock_from_torchvision_block(
                    target_block=cnn_block,
                    source_block=src_block
                )

                total_loaded += loaded
                total_skipped += skipped
                total_blocks += 1

    if verbose:
        print("=" * 80)
        print("[ResNet18-CNN] Pretrained loading summary")
        print(f"  target CNN ResBlocks: {total_blocks}")
        print(f"  loaded tensors:       {total_loaded}")
        print(f"  skipped tensors:      {total_skipped}")
        print("=" * 80)


# ============================================================
# Build encoder stages
# ============================================================
def build_encoder_stages(
    dims,
    depths,
    num_cnn_blocks=2,
    num_vss_blocks=2,
    vss_use_checkpoint=False,
    drop_path_rate=0.1,
    use_rmp_vss=False,
    rmp_num_paths=4,
    rmp_window_size=8,
    rmp_atrous_rate=2,

    # CNN pretrained
    cnn_pretrained=True,
    cnn_pretrained_path=None,
):
    """
    Build RMTPB encoder.

    Args:
        dims:
            [C1, C2, C3, C4], e.g. [64, 128, 256, 512]

        depths:
            [N1, N2, N3, N4], 每个 stage 的 RMTPBlock 数量

        num_cnn_blocks:
            每个 RMTPBlock 内 CNN ResBlock 数量。
            推荐为 2，和 ResNet18 每个 stage 的 block 数一致。

        num_vss_blocks:
            每个 RMTPBlock 内 VSSBlock 数量。

        cnn_pretrained:
            是否给 CNN branch 加载 ResNet18 ImageNet 权重。

        cnn_pretrained_path:
            本地 ResNet18 checkpoint 路径。
            如果为 None，则使用 torchvision 自动加载 ResNet18_Weights.IMAGENET1K_V1。

    Returns:
        stages:
            nn.ModuleList of stages

        mergings:
            nn.ModuleList of downsampling modules
    """

    assert len(dims) == 4, f"dims should have 4 elements, got {dims}"
    assert len(depths) == 4, f"depths should have 4 elements, got {depths}"

    stages = nn.ModuleList()
    mergings = nn.ModuleList()

    # stochastic depth decay
    total_blocks = sum(depths)
    if total_blocks > 0:
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
    else:
        dpr = []

    block_ptr = 0

    for i in range(4):
        blocks = []

        for block_idx in range(depths[i]):
            dp = dpr[block_ptr] if block_ptr < len(dpr) else 0.0
            block_ptr += 1

            blocks.append(
                RMTPBlock(
                    dim=dims[i],
                    num_cnn_blocks=num_cnn_blocks,
                    num_vss_blocks=num_vss_blocks,
                    vss_use_checkpoint=vss_use_checkpoint,
                    drop_path=dp,
                    use_rmp_vss=use_rmp_vss,
                    rmp_num_paths=rmp_num_paths,
                    rmp_window_size=rmp_window_size,
                    rmp_atrous_rate=rmp_atrous_rate,
                    # Alternate regular and shifted local windows between
                    # neighboring RMTP blocks in the same encoder stage.
                    rmp_window_shift=(block_idx % 2 == 1),
                )
            )

        stage = nn.Sequential(*blocks)
        stages.append(stage)

        if i < 3:
            mergings.append(
                PatchMerging(
                    in_ch=dims[i],
                    out_ch=dims[i + 1]
                )
            )

    # 加载 ResNet18 权重到 CNN branch
    load_resnet18_pretrained_to_rmtpb_cnn(
        stages=stages,
        pretrained=cnn_pretrained,
        local_path=cnn_pretrained_path,
        verbose=True
    )

    return stages, mergings


# ============================================================
# Simple test
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dims = [64, 128, 256, 512]
    depths = [1, 1, 1, 1]

    stages, mergings = build_encoder_stages(
        dims=dims,
        depths=depths,
        num_cnn_blocks=2,
        num_vss_blocks=1,
        use_rmp_vss=True,
        rmp_num_paths=4,
        rmp_window_size=8,
        rmp_atrous_rate=2,
        cnn_pretrained=False,
    )

    stages = stages.to(device)
    mergings = mergings.to(device)

    x = torch.randn(2, 64, 128, 128).to(device)

    s1 = stages[0](x)
    x2 = mergings[0](s1)
    s2 = stages[1](x2)
    x3 = mergings[1](s2)
    s3 = stages[2](x3)
    x4 = mergings[2](s3)
    s4 = stages[3](x4)

    print("s1:", s1.shape)
    print("s2:", s2.shape)
    print("s3:", s3.shape)
    print("s4:", s4.shape)
    print("RMTPB encoder with global/local/diagonal/atrous VSS paths OK")
