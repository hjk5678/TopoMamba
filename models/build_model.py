import torch
import torch.nn as nn

from models.encoder.resnet0 import ResNet0
from models.encoder.ssa_m import SSA_M
from models.encoder.rmtpb import build_encoder_stages
from models.decoder.decoder import Decoder, EncoderAlignedDecoder

from models.boundary_guidance import (
    BoundaryPriorExtractor,
    BoundaryStateModulator,
)
from models.graph_interaction import GraphInteractionAttention


# ============================================================
# Connectivity / Topology Head
# ============================================================
# 你的工程里目前使用的是 MSTC。
# 为了兼容，如果没有 MSTC，则尝试导入 DCPM。
# ============================================================
try:
    from models.mstc import MSTC as ConnHead
    _CONN_HEAD_NAME = "MSTC"
except Exception:
    from models.dcpm import DCPM as ConnHead
    _CONN_HEAD_NAME = "DCPM"


def build_conn_head(in_channels, out_channels):
    """
    兼容 MSTC / DCPM 的构造函数。
    正常情况下是：
        Head(in_channels=..., out_channels=...)
    """
    try:
        return ConnHead(
            in_channels=in_channels,
            out_channels=out_channels
        )
    except TypeError:
        return ConnHead(
            in_channels,
            out_channels
        )


class TopoMamba(nn.Module):
    """
    TopoMamba with RMTPB encoder.

    当前版本结构：

        Input image
            ├── ResNet0 high-resolution branch
            │       f0:      [B, 32, H,   W]
            │       f0_down: [B, 32, H/2, W/2]
            │
            └── RMTPB encoder
                    stem: [B, 64, H/4, W/4]
                    stage1: RMTPB(CNN branch + multi-path VSS branch + DGF)
                    merge1
                    stage2: RMTPB(CNN branch + multi-path VSS branch + DGF)
                    merge2
                    stage3: RMTPB(CNN branch + multi-path VSS branch + DGF)
                    merge3
                    stage4: RMTPB(CNN branch + multi-path VSS branch + DGF)

    RMTPB 内部：
        CNN branch:
            ResNet18-style BasicBlock
            由 models/encoder/rmtpb.py 中的 build_encoder_stages 加载 ResNet18 ImageNet 预训练权重。

        VMamba/VSS branch:
            official VSSBlock
            当前默认随机初始化，后续可以再写 key mapping 加载 VMamba checkpoint。

        Fusion:
            DGF 动态门控融合。

    forward 输出：
        outputs["seg_logits"]:  [B, num_classes, H, W]
        outputs["edge_preds"]:  list
        outputs["conn_logits"]: list
    """

    def __init__(
        self,
        num_classes=6,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        topology_pairs=None,

        # -------------------------------------------------
        # 兼容旧 train.py。
        # 当前版本不使用 MSSE，但保留参数防止旧脚本报错。
        # -------------------------------------------------
        use_msse=None,

        # -------------------------------------------------
        # RMTPB settings
        # -------------------------------------------------
        num_cnn_blocks=2,
        num_vss_blocks=1,
        vss_use_checkpoint=False,
        drop_path_rate=0.1,
        use_rmp_vss=False,
        rmp_num_paths=4,

        # -------------------------------------------------
        # CNN branch pretrained settings
        # -------------------------------------------------
        cnn_pretrained=True,
        cnn_pretrained_path=None,

        # -------------------------------------------------
        # Boundary guidance
        # -------------------------------------------------
        use_bgsm=True,
        use_gia=False,
        gia_dim=64,
        gia_heads=4,
        skip_mode="encoder",

        # -------------------------------------------------
        # 兼容旧 train.py 里传入的其他参数
        # -------------------------------------------------
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dims = tuple(dims)
        self.depths = tuple(depths)

        self.num_cnn_blocks = num_cnn_blocks
        self.num_vss_blocks = num_vss_blocks
        self.vss_use_checkpoint = vss_use_checkpoint
        self.drop_path_rate = drop_path_rate
        self.use_rmp_vss = use_rmp_vss
        self.rmp_num_paths = rmp_num_paths

        self.cnn_pretrained = cnn_pretrained
        self.cnn_pretrained_path = cnn_pretrained_path

        self.use_bgsm = use_bgsm
        self.use_gia = use_gia
        self.gia_dim = gia_dim
        self.gia_heads = gia_heads
        self.skip_mode = str(skip_mode).lower()

        if self.skip_mode not in ["ssam", "basic", "encoder"]:
            raise ValueError(
                f"Unsupported skip_mode={skip_mode}. "
                "Expected 'ssam', 'basic', or 'encoder'."
            )

        if self.skip_mode == "encoder":
            self.skip_channels = [
                self.dims[0],
                self.dims[1],
                self.dims[2],
                self.dims[3],
            ]
        else:
            self.skip_channels = [
                32,
                self.dims[0],
                self.dims[1],
                self.dims[2],
            ]

        # -------------------------------------------------
        # Topology setting
        # topology_pairs 决定 conn_logits 输出通道数 K。
        # -------------------------------------------------
        if topology_pairs is None:
            topology_pairs = [
                ("any_boundary", None, None),
            ]

        self.topology_pairs = topology_pairs
        self.num_conn_channels = len(topology_pairs)

        # -------------------------------------------------
        # High-resolution shallow branch
        #
        # f0:
        #   [B, 32, H, W]
        #
        # f0_down:
        #   [B, 32, H/2, W/2]
        # -------------------------------------------------
        if self.skip_mode == "ssam":
            self.resnet0 = ResNet0(
                in_ch=3,
                out_ch=32
            )

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
        elif self.skip_mode == "basic":
            self.basic_skip0 = nn.Sequential(
                nn.Conv2d(
                    3,
                    32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    32,
                    32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

        # -------------------------------------------------
        # Stem
        #
        # 输入:
        #   x: [B, 3, H, W]
        #
        # 输出:
        #   f1: [B, 64, H/4, W/4]
        #
        # 注意：
        #   这里仍然保留你原来的 stem 逻辑。
        #   RMTPB 的 CNN branch 预训练在 build_encoder_stages 里完成。
        # -------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.dims[0] // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                self.dims[0] // 2,
                self.dims[0] // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                self.dims[0] // 2,
                self.dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.dims[0]),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------
        # RMTPB Encoder
        #
        # stages[i] 内部是：
        #   RMTPBlock = CNN branch + VSS/RMP-VSS branch + DGF
        #
        # mergings[i] 负责 stage 间下采样。
        #
        # 输出：
        #   s1: [B,  64, H/4,  W/4]
        #   s2: [B, 128, H/8,  W/8]
        #   s3: [B, 256, H/16, W/16]
        #   s4: [B, 512, H/32, W/32]
        # -------------------------------------------------
        self.stages, self.mergings = build_encoder_stages(
            dims=list(self.dims),
            depths=list(self.depths),
            num_cnn_blocks=num_cnn_blocks,
            num_vss_blocks=num_vss_blocks,
            vss_use_checkpoint=vss_use_checkpoint,
            drop_path_rate=drop_path_rate,
            use_rmp_vss=use_rmp_vss,
            rmp_num_paths=rmp_num_paths,

            cnn_pretrained=cnn_pretrained,
            cnn_pretrained_path=cnn_pretrained_path,
        )

        # -------------------------------------------------
        # Boundary prior extractor + BGSM
        # -------------------------------------------------
        if self.use_bgsm:
            self.boundary_extractor = BoundaryPriorExtractor(
                in_channels=3,
                hidden_channels=16
            )

            self.bgsm1 = BoundaryStateModulator(
                channels=self.dims[0]
            )
            self.bgsm2 = BoundaryStateModulator(
                channels=self.dims[1]
            )
            self.bgsm3 = BoundaryStateModulator(
                channels=self.dims[2]
            )
            self.bgsm4 = BoundaryStateModulator(
                channels=self.dims[3]
            )

        # -------------------------------------------------
        # SSA-M skip alignment modules
        #
        # g3 = ssam3(s3, s4)
        # g2 = ssam2(s2, s3)
        # g1 = ssam1(s1, s2)
        # g0 = ssam0(f0_down, s1)
        # -------------------------------------------------
        if self.skip_mode == "ssam":
            self.ssam3 = SSA_M(
                self.dims[2],
                self.dims[3],
                self.dims[2]
            )

            self.ssam2 = SSA_M(
                self.dims[1],
                self.dims[2],
                self.dims[1]
            )

            self.ssam1 = SSA_M(
                self.dims[0],
                self.dims[1],
                self.dims[0]
            )

            self.ssam0 = SSA_M(
                32,
                self.dims[0],
                32
            )

        # -------------------------------------------------
        # Graph interaction attention over multi-scale skips
        # -------------------------------------------------
        if self.use_gia:
            self.graph_interaction = GraphInteractionAttention(
                channels=self.skip_channels,
                graph_dim=gia_dim,
                num_heads=gia_heads,
            )

        # -------------------------------------------------
        # Multi-scale connectivity / topology heads
        #
        # conn_logits[0]: g0, [B, K, H/2,  W/2]
        # conn_logits[1]: g1, [B, K, H/4,  W/4]
        # conn_logits[2]: g2, [B, K, H/8,  W/8]
        # conn_logits[3]: g3, [B, K, H/16, W/16]
        # -------------------------------------------------
        self.mstc_heads = nn.ModuleList([
            build_conn_head(
                in_channels=ch,
                out_channels=self.num_conn_channels
            )
            for ch in self.skip_channels
        ])

        # 兼容旧名字。
        # forward 里统一用 self.mstc_heads。
        self.conn_heads = self.mstc_heads
        self.dcpm_heads = self.mstc_heads

        # -------------------------------------------------
        # Decoder
        #
        # 输入：
        #   f4_enh: [B, 512, H/32, W/32]
        #   g3:     [B, 256, H/16, W/16]
        #   g2:     [B, 128, H/8,  W/8]
        #   g1:     [B, 64,  H/4,  W/4]
        #   g0:     [B, 32,  H/2,  W/2]
        # -------------------------------------------------
        if self.skip_mode == "encoder":
            self.decoder = EncoderAlignedDecoder(
                encoder_channels=list(self.dims),
                num_classes=num_classes
            )
        else:
            self.decoder = Decoder(
                encoder_channels=[32, 64, 128, 256],
                num_classes=num_classes
            )

        self._print_model_summary_once = True

    def _maybe_print_summary(self):
        """
        只打印一次模型关键信息。
        DDP 下每个 rank 可能都会打印一次，不影响训练。
        """
        if not self._print_model_summary_once:
            return

        print("=" * 80)
        print("[TopoMamba] Build model: RMTPB encoder version")
        print(f"  num_classes:          {self.num_classes}")
        print(f"  dims:                 {self.dims}")
        print(f"  depths:               {self.depths}")
        print(f"  topology channels:    {self.num_conn_channels}")
        print(f"  conn head:            {_CONN_HEAD_NAME}")
        print(f"  use_bgsm:             {self.use_bgsm}")
        print(f"  num_cnn_blocks:       {self.num_cnn_blocks}")
        print(f"  num_vss_blocks:       {self.num_vss_blocks}")
        print(f"  use_rmp_vss:          {self.use_rmp_vss}")
        print(f"  rmp_num_paths:        {self.rmp_num_paths}")
        print(f"  use_gia:              {self.use_gia}")
        print(f"  gia_dim:              {self.gia_dim}")
        print(f"  skip_mode:            {self.skip_mode}")
        print(f"  cnn_pretrained:       {self.cnn_pretrained}")
        print(f"  cnn_pretrained_path:  {self.cnn_pretrained_path}")
        print("=" * 80)

        self._print_model_summary_once = False

    def forward(self, x):
        """
        Args:
            x:
                [B, 3, H, W]

        Returns:
            outputs: dict

            outputs["seg_logits"]:
                [B, num_classes, H, W]

            outputs["edge_preds"]:
                list of edge predictions, produced by decoder

            outputs["conn_logits"]:
                list of 4 tensors:
                    conn_logits[0]: [B, K, H/2,  W/2]
                    conn_logits[1]: [B, K, H/4,  W/4]
                    conn_logits[2]: [B, K, H/8,  W/8]
                    conn_logits[3]: [B, K, H/16, W/16]
        """

        self._maybe_print_summary()

        # -------------------------------------------------
        # 0. Save original image size
        # -------------------------------------------------
        out_size = x.shape[2:]

        # -------------------------------------------------
        # 1. Optional high-resolution shallow feature
        # -------------------------------------------------
        if self.skip_mode == "ssam":
            f0 = self.resnet0(x)          # [B, 32, H, W]
            f0_down = self.f0_down(f0)    # [B, 32, H/2, W/2]
        else:
            f0_down = None

        # -------------------------------------------------
        # 2. Boundary prior
        # -------------------------------------------------
        if self.use_bgsm:
            boundary_prior = self.boundary_extractor(x)
        else:
            boundary_prior = None

        # -------------------------------------------------
        # 3. Stem
        #
        # f1: [B, 64, H/4, W/4]
        # -------------------------------------------------
        f1 = self.stem(x)

        # -------------------------------------------------
        # 4. RMTPB Encoder
        #
        # 每个 stage 内部：
        #   CNN branch + VSS branch + DGF
        # -------------------------------------------------
        if self.use_bgsm:
            f1 = self.bgsm1(f1, boundary_prior)

        s1 = self.stages[0](f1)          # [B, 64, H/4, W/4]
        f2 = self.mergings[0](s1)        # [B, 128, H/8, W/8]

        if self.use_bgsm:
            f2 = self.bgsm2(f2, boundary_prior)

        s2 = self.stages[1](f2)          # [B, 128, H/8, W/8]
        f3 = self.mergings[1](s2)        # [B, 256, H/16, W/16]

        if self.use_bgsm:
            f3 = self.bgsm3(f3, boundary_prior)

        s3 = self.stages[2](f3)          # [B, 256, H/16, W/16]
        f4 = self.mergings[2](s3)        # [B, 512, H/32, W/32]

        if self.use_bgsm:
            f4 = self.bgsm4(f4, boundary_prior)

        s4 = self.stages[3](f4)          # [B, 512, H/32, W/32]

        # -------------------------------------------------
        # 5. Deep feature
        # -------------------------------------------------
        f4_enh = s4

        # -------------------------------------------------
        # 6. Skip connections
        # -------------------------------------------------
        if self.skip_mode == "ssam":
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
        elif self.skip_mode == "basic":
            g3 = s3
            g2 = s2
            g1 = s1
            g0 = self.basic_skip0(x)

        else:
            g1 = s1
            g2 = s2
            g3 = s3
            g4 = f4_enh

        if self.use_gia:
            if self.skip_mode == "encoder":
                g1, g2, g3, g4 = self.graph_interaction([g1, g2, g3, g4])
            else:
                g0, g1, g2, g3 = self.graph_interaction([g0, g1, g2, g3])

        # -------------------------------------------------
        # 7. Multi-scale connectivity / topology logits
        # -------------------------------------------------
        if self.skip_mode == "encoder":
            conn_logits = [
                self.mstc_heads[0](g1),   # [B, K, H/4,  W/4]
                self.mstc_heads[1](g2),   # [B, K, H/8,  W/8]
                self.mstc_heads[2](g3),   # [B, K, H/16, W/16]
                self.mstc_heads[3](g4),   # [B, K, H/32, W/32]
            ]
        else:
            conn_logits = [
                self.mstc_heads[0](g0),   # [B, K, H/2,  W/2]
                self.mstc_heads[1](g1),   # [B, K, H/4,  W/4]
                self.mstc_heads[2](g2),   # [B, K, H/8,  W/8]
                self.mstc_heads[3](g3),   # [B, K, H/16, W/16]
            ]

        # -------------------------------------------------
        # 8. Decoder
        # -------------------------------------------------
        if self.skip_mode == "encoder":
            outputs = self.decoder(
                g4,
                g3,
                g2,
                g1,
                out_size=out_size
            )
        else:
            outputs = self.decoder(
                f4_enh,
                g3,
                g2,
                g1,
                g0,
                out_size=out_size
            )

        # -------------------------------------------------
        # 9. Add conn_logits
        # -------------------------------------------------
        outputs["conn_logits"] = conn_logits

        return outputs


if __name__ == "__main__":
    """
    简单 forward 测试：
        python models/build_model.py
    """
    from utils.topology_configs import get_topology_pairs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TopoMamba(
        num_classes=6,
        topology_pairs=get_topology_pairs("potsdam"),
        dims=(64, 128, 256, 512),
        depths=(1, 1, 1, 1),
        num_cnn_blocks=2,
        num_vss_blocks=1,
        cnn_pretrained=True,
        cnn_pretrained_path=None,
        use_bgsm=True,
    ).to(device).eval()

    x = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        outputs = model(x)

    print("outputs keys:", outputs.keys())
    print("seg_logits:", outputs["seg_logits"].shape)

    if "edge_preds" in outputs:
        print("edge_preds:", len(outputs["edge_preds"]))
        for i, v in enumerate(outputs["edge_preds"]):
            print(f"  edge {i}: {v.shape}")

    if "conn_logits" in outputs:
        print("conn_logits:", len(outputs["conn_logits"]))
        for i, v in enumerate(outputs["conn_logits"]):
            print(f"  conn {i}: {v.shape}")

    print("TopoMamba RMTPB + ResNet18-pretrained CNN branch forward OK")
