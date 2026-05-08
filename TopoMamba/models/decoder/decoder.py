import torch
import torch.nn as nn
import torch.nn.functional as F

from .bau import BAUBlock
from .dcpm import DCPM


class LBSHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.edge_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):
        return self.edge_conv(x)


class Decoder(nn.Module):

    def __init__(self, encoder_channels, num_classes):
        super().__init__()

        dec_channels = [256, 128, 64, 32]

        # -------------------------------------------------
        # BAU Decoder
        # -------------------------------------------------
        self.up4 = BAUBlock(
            512,
            encoder_channels[3],
            dec_channels[0]
        )

        self.up3 = BAUBlock(
            dec_channels[0],
            encoder_channels[2],
            dec_channels[1]
        )

        self.up2 = BAUBlock(
            dec_channels[1],
            encoder_channels[1],
            dec_channels[2]
        )

        self.up1 = BAUBlock(
            dec_channels[2],
            encoder_channels[0],
            dec_channels[3]
        )

        # -------------------------------------------------
        # ★ 关键修复：恢复到原图尺寸
        # -------------------------------------------------

        self.final_upsample = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            ),

            nn.Conv2d(
                dec_channels[3],
                dec_channels[3],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(dec_channels[3]),

            nn.ReLU(inplace=True),

            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )
        )

        # -------------------------------------------------
        # LBS Heads
        # -------------------------------------------------
        self.lbs_heads = nn.ModuleList([

            LBSHead(dec_channels[0]),

            LBSHead(dec_channels[1]),

            LBSHead(dec_channels[2]),

            LBSHead(dec_channels[3]),
        ])

        # -------------------------------------------------
        # Segmentation Head
        # -------------------------------------------------
        self.seg_head = nn.Conv2d(
            dec_channels[3],
            num_classes,
            kernel_size=1
        )

        # -------------------------------------------------
        # DCPM
        # -------------------------------------------------
        self.dcpm = DCPM(dec_channels[3])

    def forward(self, F4_enh, G3, G2, G1, G0):

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
        d3 = self.up4(F4_enh, G3)

        d2 = self.up3(d3, G2)

        d1 = self.up2(d2, G1)

        d0 = self.up1(d1, G0)

        # -------------------------------------------------
        # Final Upsample
        # -------------------------------------------------
        feat_out = self.final_upsample(d0)

        # -------------------------------------------------
        # Debug
        # -------------------------------------------------
        # print(f"[DEBUG] G0         : {G0.shape}")
        # print(f"[DEBUG] d3         : {d3.shape}")
        # print(f"[DEBUG] d2         : {d2.shape}")
        # print(f"[DEBUG] d1         : {d1.shape}")
        # print(f"[DEBUG] d0         : {d0.shape}")
        # print(f"[DEBUG] feat_out   : {feat_out.shape}")

        # -------------------------------------------------
        # Edge predictions
        # -------------------------------------------------
        edge_preds = [

            head(f)

            for head, f in zip(
                self.lbs_heads,
                [d3, d2, d1, d0]
            )
        ]

        # -------------------------------------------------
        # Main outputs
        # -------------------------------------------------
        seg_logits = self.seg_head(feat_out)

        conn_logits = self.dcpm(feat_out)

        # print(f"[DEBUG] seg_logits : {seg_logits.shape}")

        return seg_logits, conn_logits, edge_preds