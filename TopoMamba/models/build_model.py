import torch.nn as nn
from models.encoder.resnet0 import ResNet0
from models.encoder.tpdb import build_encoder_stages, PatchMerging
from models.encoder.mssm import MSSE
from models.encoder.ssa_m import SSA_M
from models.decoder.decoder import Decoder

class TopoMamba(nn.Module):
    def __init__(self, num_classes=6, dims=[64, 128, 256, 512], depths=[2, 2, 4, 2]):
        super().__init__()
        self.resnet0 = ResNet0(in_ch=3, out_ch=32)
        # Stem: conv + patch embedding to 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0]//2, dims[0]//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0])
        )  # output (B, 64, H/4, W/4)

        # TPDB encoder stages
        self.stages, self.mergings = build_encoder_stages(dims, depths)
        # MSSE on F4
        self.msse = MSSE(dims[3])
        # SSA-M blocks
        # G3: F3+F4_enh -> channels: (256+512) -> out 256
        self.ssam3 = SSA_M(dims[2], dims[3], dims[2])
        # G2: F2+F3 -> (128+256) -> out 128
        self.ssam2 = SSA_M(dims[1], dims[2], dims[1])
        # G1: F1+F2 -> (64+128) -> out 64
        self.ssam1 = SSA_M(dims[0], dims[1], dims[0])
        # G0: F0_adjust + F1 -> (32+64) -> out 32
        self.ssam0 = SSA_M(32, dims[0], 32)
        # F0下采样适配器: 32 -> 32, 下采样4倍
        self.f0_down = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32)
        )
        # 解码器
        self.decoder = Decoder(encoder_channels=[32, 64, 128, 256], num_classes=num_classes)

    def forward(self, x):
        # F0: (B,32,H,W)
        f0 = self.resnet0(x)
        # Stem
        f1 = self.stem(x)  # (B,64,H/4,W/4)
        # 编码器各阶段
        f2 = self.stages[0](f1)
        f2 = self.mergings[0](f2)  # (B,128,H/8,W/8)
        f3 = self.stages[1](f2)
        f3 = self.mergings[1](f3)  # (B,256,H/16,W/16)
        f4 = self.stages[2](f3)
        f4 = self.mergings[2](f4)  # (B,512,H/32,W/32)
        f4 = self.stages[3](f4)    # 最后stage无下采样
        # MSSE
        f4_enh = self.msse(f4)
        # SSA-M跳跃连接
        g3 = self.ssam3(f3, f4_enh)  # f3低层, f4_enh高层
        g2 = self.ssam2(f2, f3)
        g1 = self.ssam1(f1, f2)
        f0_down = self.f0_down(f0)  # (B,32,H/4,W/4)
        g0 = self.ssam0(f0_down, f1)
        # 解码器
        seg_logits, conn_logits, edge_preds = self.decoder(f4_enh, g3, g2, g1, g0)
        seg_logits, conn_logits, edge_preds = self.decoder(f4_enh, g3, g2, g1, g0)
        # print(f"[DEBUG] seg_logits.shape: {seg_logits.shape}")  # 期望 [B, 6, 512, 512]
        # print(f"[DEBUG] f4_enh.shape: {f4_enh.shape}")  # [B, 512, H/32, W/32]
        # print(f"[DEBUG] g0.shape: {g0.shape}")  # [B, 32, H/2, W/2]
        return seg_logits, conn_logits, edge_preds
        return seg_logits, conn_logits, edge_preds