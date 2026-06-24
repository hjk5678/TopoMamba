import os
import torch
import torch.nn as nn
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
class DGF(nn.Module):
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


class ResidualMultiPathVSS(nn.Module):
    """
    Residual multi-path VSS wrapper inspired by multi-path vision Mamba.

    The input channels are split into several paths:
        path 1: cross scan
        path 2: unidirectional scan
        path 3: bidirectional scan
        path 4: rotated cross scan

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
        forward_types=None,
    ):
        super().__init__()

        if dim % num_paths != 0:
            raise ValueError(
                f"dim must be divisible by num_paths, got "
                f"dim={dim}, num_paths={num_paths}"
            )

        if forward_types is None:
            # Keep the default path set Triton-safe on the current VMamba
            # kernel: cascade2d maps to scans=-1 and can fail to compile.
            forward_types = [
                "v05_noz",
                "v051d_noz",
                "v052d_noz",
                "v05_noz",
            ]

        if len(forward_types) < num_paths:
            repeats = (num_paths + len(forward_types) - 1) // len(forward_types)
            forward_types = (forward_types * repeats)[:num_paths]
        else:
            forward_types = forward_types[:num_paths]

        self.num_paths = int(num_paths)
        self.path_dim = dim // self.num_paths

        self.paths = nn.ModuleList()

        for path_idx in range(self.num_paths):
            blocks = []

            for _ in range(num_blocks):
                blocks.append(
                    VSSBlock(
                        hidden_dim=self.path_dim,
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
                        forward_type=forward_types[path_idx],

                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,

                        use_checkpoint=use_checkpoint,
                        post_norm=False,
                    )
                )

            self.paths.append(nn.Sequential(*blocks))

        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        chunks = torch.chunk(x, self.num_paths, dim=1)

        outs = []
        for path_idx, (path, chunk) in enumerate(zip(self.paths, chunks)):
            if path_idx == 3:
                chunk = torch.rot90(chunk, k=1, dims=(-2, -1)).contiguous()
                out = path(chunk)
                out = torch.rot90(out, k=-1, dims=(-2, -1)).contiguous()
            else:
                out = path(chunk)

            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.fuse(out)

        return x + self.gamma * out


# ============================================================
# RMTPBlock
# ============================================================
class RMTPBlock(nn.Module):
    """
    Residual Multi-path Topological Perception Block.

    保留你的原始设计：
        1. CNN branch: ResNet18-style local branch
        2. VMamba branch: official VSSBlock global branch
        3. DGF fusion
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
    ):
        super().__init__()

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
        if use_rmp_vss:
            self.vmamba_branch = ResidualMultiPathVSS(
                dim=dim,
                num_blocks=num_vss_blocks,
                num_paths=rmp_num_paths,
                drop_path=drop_path,
                use_checkpoint=vss_use_checkpoint,
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

        self.fusion = DGF(dim)

        # learnable residual scale
        # 初始为 0，避免一开始扰动主干特征
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        F_local = self.cnn_branch(x)

        F_global = self.vmamba_branch(x)
        F_global = torch.clamp(F_global, min=-10.0, max=10.0)

        fused = self.fusion(F_local, F_global)

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

        for _ in range(depths[i]):
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
        cnn_pretrained=True,
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
    print("RMTPB encoder with ResNet18-pretrained CNN branch OK")
