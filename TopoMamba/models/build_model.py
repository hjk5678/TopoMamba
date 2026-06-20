import torch.nn as nn
import torchvision.models as tv_models
from models.encoder.resnet0 import ResNet0
from models.encoder.tpdb import build_encoder_stages
from models.encoder.ssa_m import SSA_M
from models.decoder.decoder import Decoder
from models.mstc import MSTC

from models.boundary_guidance import (
    BoundaryPriorExtractor,
    BoundaryStateModulator,
)


class TopoMamba(nn.Module):
    """
    TopoMamba v5: no MSSE + BGSM + Multi-scale mstc.

    当前版本：
        1. 完全删除 MSSE；
        2. 保留 Multi-scale mstc；
        3. 新增 BoundaryPriorExtractor；
        4. 新增 BGSM，即 Boundary-Guided State Modulation；
        5. 在每个 TPDB stage 输入前使用 BGSM 调制特征；
        6. 让边界先验参与 encoder 主干建模；
        7. mstc 继续在 SSA-M 输出后做多尺度连通性监督；
        8. decoder 只负责 seg_logits 和 edge_preds。

    主体流程：
        Input
          ├── BoundaryPriorExtractor -> boundary_prior
          ↓
        ResNet0 shallow branch -> f0
        Stem -> f1
          ↓
        BGSM1(f1, boundary_prior) -> TPDB Stage1
          ↓
        BGSM2(f2, boundary_prior) -> TPDB Stage2
          ↓
        BGSM3(f3, boundary_prior) -> TPDB Stage3
          ↓
        BGSM4(f4, boundary_prior) -> TPDB Stage4
          ↓
        SSA-M -> g0/g1/g2/g3
          ↓
        Multi-scale mstc -> conn_logits
          ↓
        Decoder -> seg_logits + edge_preds

    输出:
        {
            "seg_logits":  [B, num_classes, H, W],
            "edge_preds":  list of 4 tensors,
            "conn_logits": list of 4 tensors
        }
    """

    def __init__(
        self,
        num_classes=6,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        topology_pairs=None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dims = dims
        self.depths = depths

        # -------------------------------------------------
        # mstc topology setting
        # -------------------------------------------------
        # topology_pairs 决定 mstc 输出通道数。
        #
        # Potsdam / Vaihingen:
        #   K = 4
        #
        # LoveDA:
        #   K = 5
        #
        # GID:
        #   根据 topology_configs.py 决定
        # -------------------------------------------------
        if topology_pairs is None:
            topology_pairs = [
                ("any_boundary", None, None),
            ]

        self.topology_pairs = topology_pairs
        self.num_conn_channels = len(topology_pairs)

        # -------------------------------------------------
        # Boundary prior extractor
        # -------------------------------------------------
        # 输入原图 x: [B, 3, H, W]
        # 输出边界先验 boundary_prior: [B, 1, H, W]
        #
        # 这个模块提取图像级边界先验，之后送入每一层 BGSM。
        # -------------------------------------------------
        self.boundary_extractor = BoundaryPriorExtractor(
            in_channels=3,
            hidden_channels=16
        )

        # -------------------------------------------------
        # High-resolution shallow branch
        # 原图 -> 高分辨率浅层特征
        # f0: [B, 32, H, W]
        # -------------------------------------------------
        self.resnet0 = ResNet0(
            in_ch=3,
            out_ch=32
        )

        # -------------------------------------------------
        # Stem: image -> 1/4 resolution
        # f1: [B, 64, H/4, W/4]
        # -------------------------------------------------
        resnet34_pretrained = tv_models.resnet34(
            weights=tv_models.ResNet34_Weights.IMAGENET1K_V1
        )

        self.stem = nn.Sequential(
            resnet34_pretrained.conv1,  # [B, 64, H/2, W/2]
            resnet34_pretrained.bn1,
            resnet34_pretrained.relu,
            resnet34_pretrained.maxpool,  # [B, 64, H/4, W/4]
            resnet34_pretrained.layer1,  # [B, 64, H/4, W/4]
        )

        # -------------------------------------------------
        # TPDB Encoder stages
        # dims = [64, 128, 256, 512]
        # -------------------------------------------------
        self.stages, self.mergings = build_encoder_stages(
            dims,
            depths
        )

        # -------------------------------------------------
        # Boundary-Guided State Modulation modules
        # -------------------------------------------------
        # BGSM 放在每个 TPDB stage 输入之前。
        #
        # 原来:
        #   f1 -> Stage1
        #   f2 -> Stage2
        #   f3 -> Stage3
        #   f4 -> Stage4
        #
        # 现在:
        #   f1 -> BGSM1 -> Stage1
        #   f2 -> BGSM2 -> Stage2
        #   f3 -> BGSM3 -> Stage3
        #   f4 -> BGSM4 -> Stage4
        #
        # 这样边界先验会进入 encoder 主干，
        # 从而影响 TPDB/Mamba/SSM 的输入状态建模。
        # -------------------------------------------------
        self.bgsm1 = BoundaryStateModulator(
            channels=dims[0]
        )  # 64 channels, H/4

        self.bgsm2 = BoundaryStateModulator(
            channels=dims[1]
        )  # 128 channels, H/8

        self.bgsm3 = BoundaryStateModulator(
            channels=dims[2]
        )  # 256 channels, H/16

        self.bgsm4 = BoundaryStateModulator(
            channels=dims[3]
        )  # 512 channels, H/32

        # -------------------------------------------------
        # SSA-M skip alignment modules
        # -------------------------------------------------
        self.ssam3 = SSA_M(
            dims[2],
            dims[3],
            dims[2]
        )   # 256 <- 512

        self.ssam2 = SSA_M(
            dims[1],
            dims[2],
            dims[1]
        )   # 128 <- 256

        self.ssam1 = SSA_M(
            dims[0],
            dims[1],
            dims[0]
        )   # 64 <- 128

        self.ssam0 = SSA_M(
            32,
            dims[0],
            32
        )   # 32 <- 64

        # -------------------------------------------------
        # F0 downsampler: H -> H/2
        # f0_down: [B, 32, H/2, W/2]
        # -------------------------------------------------
        self.f0_down = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------
        # Multi-scale mstc heads
        #
        # mstc 接在 SSA-M 输出之后，decoder 之前。
        #
        # g0: [B, 32,  H/2,  W/2]
        # g1: [B, 64,  H/4,  W/4]
        # g2: [B, 128, H/8,  W/8]
        # g3: [B, 256, H/16, W/16]
        #
        # conn_logits 顺序:
        #   [conn_g0, conn_g1, conn_g2, conn_g3]
        #
        # 这个顺序是高分辨率 -> 低分辨率，
        # 对应 loss 里的 conn_scale_weights = [0.4, 0.3, 0.2, 0.1]
        # -------------------------------------------------
        self.mstc_heads = nn.ModuleList([
            MSTC(
                in_channels=32,
                out_channels=self.num_conn_channels
            ),
            MSTC(
                in_channels=dims[0],
                out_channels=self.num_conn_channels
            ),
            MSTC(
                in_channels=dims[1],
                out_channels=self.num_conn_channels
            ),
            MSTC(
                in_channels=dims[2],
                out_channels=self.num_conn_channels
            ),
        ])

        # -------------------------------------------------
        # Decoder
        # G0/G1/G2/G3 channels = [32, 64, 128, 256]
        # deepest feature = s4, channel = 512
        # -------------------------------------------------
        self.decoder = Decoder(
            encoder_channels=[32, 64, 128, 256],
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            outputs: dict
                seg_logits:
                    [B, num_classes, H, W]

                edge_preds:
                    list of 4 tensors,
                    each tensor is [B, 1, H, W]

                conn_logits:
                    list of 4 tensors:
                        conn_logits[0]: [B, K, H/2,  W/2]
                        conn_logits[1]: [B, K, H/4,  W/4]
                        conn_logits[2]: [B, K, H/8,  W/8]
                        conn_logits[3]: [B, K, H/16, W/16]

                    K = len(topology_pairs)
        """

        # -------------------------------------------------
        # 0. Save original image size
        # -------------------------------------------------
        out_size = x.shape[2:]

        # -------------------------------------------------
        # 1. Boundary prior
        # -------------------------------------------------
        # boundary_prior: [B, 1, H, W]
        #
        # 这个边界先验后续会被每个 BGSM 自动 resize 到对应尺度。
        # -------------------------------------------------
        boundary_prior = self.boundary_extractor(x)

        # -------------------------------------------------
        # 2. High-resolution shallow feature
        # -------------------------------------------------
        f0 = self.resnet0(x)          # [B, 32, H, W]
        f0_down = self.f0_down(f0)    # [B, 32, H/2, W/2]

        # -------------------------------------------------
        # 3. Stem
        # -------------------------------------------------
        f1 = self.stem(x)             # [B, 64, H/4, W/4]

        # -------------------------------------------------
        # 4. Encoder stages with BGSM
        # -------------------------------------------------

        # -------------------------------------------------
        # Stage 1
        # -------------------------------------------------
        # 原来:
        #   s1 = self.stages[0](f1)
        #
        # 现在:
        #   f1 -> BGSM1 -> Stage1
        #
        # BGSM1 会把 boundary_prior resize 到 H/4, W/4。
        # -------------------------------------------------
        f1 = self.bgsm1(
            f1,
            boundary_prior
        )
        s1 = self.stages[0](f1)       # [B, 64, H/4, W/4]
        f2 = self.mergings[0](s1)     # [B, 128, H/8, W/8]

        # -------------------------------------------------
        # Stage 2
        # -------------------------------------------------
        f2 = self.bgsm2(
            f2,
            boundary_prior
        )
        s2 = self.stages[1](f2)       # [B, 128, H/8, W/8]
        f3 = self.mergings[1](s2)     # [B, 256, H/16, W/16]

        # -------------------------------------------------
        # Stage 3
        # -------------------------------------------------
        f3 = self.bgsm3(
            f3,
            boundary_prior
        )
        s3 = self.stages[2](f3)       # [B, 256, H/16, W/16]
        f4 = self.mergings[2](s3)     # [B, 512, H/32, W/32]

        # -------------------------------------------------
        # Stage 4
        # -------------------------------------------------
        f4 = self.bgsm4(
            f4,
            boundary_prior
        )
        s4 = self.stages[3](f4)       # [B, 512, H/32, W/32]

        # -------------------------------------------------
        # 5. Deep feature
        #
        # 当前版本完全不使用 MSSE，直接使用 Stage 4 输出。
        # -------------------------------------------------
        f4_enh = s4                   # [B, 512, H/32, W/32]

        # -------------------------------------------------
        # 6. SSA-M skip connections
        # -------------------------------------------------
        g3 = self.ssam3(
            s3,
            f4_enh
        )   # [B, 256, H/16, W/16]

        g2 = self.ssam2(
            s2,
            s3
        )   # [B, 128, H/8, W/8]

        g1 = self.ssam1(
            s1,
            s2
        )   # [B, 64, H/4, W/4]

        g0 = self.ssam0(
            f0_down,
            s1
        )   # [B, 32, H/2, W/2]

        # -------------------------------------------------
        # 7. Multi-scale mstc
        #
        # mstc 继续保留，用作多尺度连通性 side supervision。
        # 它不直接改变 decoder 主特征流，但会通过 conn_loss 反向约束 g0/g1/g2/g3。
        # -------------------------------------------------
        conn_logits = [
            self.mstc_heads[0](g0),
            self.mstc_heads[1](g1),
            self.mstc_heads[2](g2),
            self.mstc_heads[3](g3),
        ]

        # -------------------------------------------------
        # 8. Decoder
        # -------------------------------------------------
        outputs = self.decoder(
            f4_enh,
            g3,
            g2,
            g1,
            g0,
            out_size=out_size
        )

        # -------------------------------------------------
        # 9. Add multi-scale mstc outputs
        # -------------------------------------------------
        outputs["conn_logits"] = conn_logits

        return outputs
