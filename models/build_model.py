import torch
import torch.nn as nn

from models.encoder.rmtpb import build_encoder_stages
from models.decoder.decoder import EncoderAlignedDecoder

from models.cluster_graph import MultiScaleClusterGraph


RMP_SCAN_LAYOUT = "global_local_diagonal_atrous_v1"
DECODER_LAYOUT = "bau_classic_unet4_local_boundary_v1"
CLUSTER_GRAPH_LAYOUT = "multiscale_hard_cluster_gcn_v1"


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

        Input image -> stem -> S1 -> S2 -> S3 -> S4 -> bottleneck
                              |     |     |     |
                              +-----+-----+-----+-> four BAU skip connections

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
        rmp_window_size=8,
        rmp_atrous_rate=2,

        # -------------------------------------------------
        # CNN branch pretrained settings
        # -------------------------------------------------
        cnn_pretrained=True,
        cnn_pretrained_path=None,

        use_cluster_gcn=False,
        cluster_counts=(256, 128, 64, 32),
        cluster_graph_dim=64,
        cluster_iters=2,
        cluster_spatial_weight=0.5,

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
        self.rmp_window_size = rmp_window_size
        self.rmp_atrous_rate = rmp_atrous_rate
        self.rmp_scan_layout = RMP_SCAN_LAYOUT

        self.cnn_pretrained = cnn_pretrained
        self.cnn_pretrained_path = cnn_pretrained_path

        self.use_cluster_gcn = bool(use_cluster_gcn)
        self.cluster_counts = tuple(int(value) for value in cluster_counts)
        self.cluster_graph_dim = int(cluster_graph_dim)
        self.cluster_iters = int(cluster_iters)
        self.cluster_spatial_weight = float(cluster_spatial_weight)
        self.cluster_graph_layout = CLUSTER_GRAPH_LAYOUT
        self.decoder_layout = DECODER_LAYOUT
        self.skip_channels = list(self.dims)

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
            rmp_window_size=rmp_window_size,
            rmp_atrous_rate=rmp_atrous_rate,

            cnn_pretrained=cnn_pretrained,
            cnn_pretrained_path=cnn_pretrained_path,
        )

        if self.use_cluster_gcn:
            self.cluster_graph = MultiScaleClusterGraph(
                channels=self.dims,
                cluster_counts=self.cluster_counts,
                graph_dim=self.cluster_graph_dim,
                num_cluster_iters=self.cluster_iters,
                spatial_weight=self.cluster_spatial_weight,
            )

        # -------------------------------------------------
        # Multi-scale connectivity / topology heads
        #
        # conn_logits[0]: g1, [B, K, H/4,  W/4]
        # conn_logits[1]: g2, [B, K, H/8,  W/8]
        # conn_logits[2]: g3, [B, K, H/16, W/16]
        # conn_logits[3]: g4, [B, K, H/32, W/32]
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
        # Classical four-level U-Net decoder:
        # bottleneck(H/64) + g4 + g3 + g2 + g1.
        # -------------------------------------------------
        self.decoder = EncoderAlignedDecoder(
            encoder_channels=list(self.dims),
            num_classes=num_classes,
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
        print(f"  num_cnn_blocks:       {self.num_cnn_blocks}")
        print(f"  num_vss_blocks:       {self.num_vss_blocks}")
        print(f"  use_rmp_vss:          {self.use_rmp_vss}")
        print(f"  rmp_num_paths:        {self.rmp_num_paths}")
        if self.use_rmp_vss:
            print("  rmp_path_types:       global/local/diagonal/atrous")
            print(f"  rmp_window_size:      {self.rmp_window_size}")
            print(f"  rmp_atrous_rate:      {self.rmp_atrous_rate}")
        print(f"  use_cluster_gcn:      {self.use_cluster_gcn}")
        if self.use_cluster_gcn:
            print(f"  cluster_layout:       {self.cluster_graph_layout}")
            print(f"  cluster_counts:       {self.cluster_counts}")
            print(f"  cluster_graph_dim:    {self.cluster_graph_dim}")
            print(f"  cluster_iters:        {self.cluster_iters}")
            print(f"  cluster_spatial_w:    {self.cluster_spatial_weight}")
        print(f"  decoder_layout:       {self.decoder_layout}")
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
                    conn_logits[0]: [B, K, H/4,  W/4]
                    conn_logits[1]: [B, K, H/8,  W/8]
                    conn_logits[2]: [B, K, H/16, W/16]
                    conn_logits[3]: [B, K, H/32, W/32]
        """

        self._maybe_print_summary()

        # -------------------------------------------------
        # 0. Save original image size
        # -------------------------------------------------
        out_size = x.shape[2:]

        # -------------------------------------------------
        # 1. Stem
        #
        # f1: [B, 64, H/4, W/4]
        # -------------------------------------------------
        f1 = self.stem(x)

        # -------------------------------------------------
        # 2. RMTPB Encoder
        #
        # 每个 stage 内部：
        #   CNN branch + VSS branch + DGF
        # -------------------------------------------------
        s1 = self.stages[0](f1)          # [B, 64, H/4, W/4]
        f2 = self.mergings[0](s1)        # [B, 128, H/8, W/8]

        s2 = self.stages[1](f2)          # [B, 128, H/8, W/8]
        f3 = self.mergings[1](s2)        # [B, 256, H/16, W/16]

        s3 = self.stages[2](f3)          # [B, 256, H/16, W/16]
        f4 = self.mergings[2](s3)        # [B, 512, H/32, W/32]

        s4 = self.stages[3](f4)          # [B, 512, H/32, W/32]

        # -------------------------------------------------
        # 3. Four encoder skip connections
        # -------------------------------------------------
        g1, g2, g3, g4 = s1, s2, s3, s4

        if self.use_cluster_gcn:
            g1, g2, g3, g4 = self.cluster_graph([g1, g2, g3, g4])

        # -------------------------------------------------
        # 4. Multi-scale connectivity / topology logits
        # -------------------------------------------------
        conn_logits = [
            self.mstc_heads[0](g1),   # [B, K, H/4,  W/4]
            self.mstc_heads[1](g2),   # [B, K, H/8,  W/8]
            self.mstc_heads[2](g3),   # [B, K, H/16, W/16]
            self.mstc_heads[3](g4),   # [B, K, H/32, W/32]
        ]

        # -------------------------------------------------
        # 5. Classical four-level U-Net decoder
        # -------------------------------------------------
        outputs = self.decoder(
            g4,
            g3,
            g2,
            g1,
            out_size=out_size,
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
