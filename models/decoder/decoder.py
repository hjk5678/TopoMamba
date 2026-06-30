import torch.nn as nn
import torch.nn.functional as F

from .bau import BAUBlock, make_group_norm


class LBSHead(nn.Module):
    """Local boundary supervision head operating at the native stage scale."""

    def __init__(self, in_channels):
        super().__init__()
        hidden = max(16, in_channels // 4)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.edge_conv(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(in_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class EncoderAlignedDecoder(nn.Module):
    """
    Classical four-level U-Net decoder for the RMTPB encoder.

    S1-S4 are the four encoder skip features. A separate convolutional
    bottleneck downsamples S4 from H/32 to H/64. Four BAU stages then recover
    H/32, H/16, H/8, and H/4 while fusing S4, S3, S2, and S1 respectively.
    """

    def __init__(self, encoder_channels, num_classes):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        dec_channels = [256, 128, 64, 32]

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                c4,
                c4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            make_group_norm(c4),
            nn.GELU(),
            nn.Conv2d(
                c4,
                c4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(c4),
            nn.GELU(),
        )

        self.decode4 = BAUBlock(c4, c4, dec_channels[0])
        self.decode3 = BAUBlock(dec_channels[0], c3, dec_channels[1])
        self.decode2 = BAUBlock(dec_channels[1], c2, dec_channels[2])
        self.decode1 = BAUBlock(dec_channels[2], c1, dec_channels[3])

        self.lbs_heads = nn.ModuleList([
            LBSHead(dec_channels[0]),
            LBSHead(dec_channels[1]),
            LBSHead(dec_channels[2]),
            LBSHead(dec_channels[3]),
        ])
        self.segmentation_head = SegmentationHead(
            dec_channels[3],
            num_classes,
        )

    def forward(
        self,
        S4,
        S3,
        S2,
        S1,
        out_size=None,
    ):
        bottleneck = self.bottleneck(S4)
        d4 = self.decode4(bottleneck, S4)
        d3 = self.decode3(d4, S3)
        d2 = self.decode2(d3, S2)
        d1 = self.decode1(d2, S1)

        if out_size is None:
            out_size = (S1.shape[2] * 4, S1.shape[3] * 4)

        decoder_feats = [d4, d3, d2, d1]
        edge_preds = [
            head(feat)
            for head, feat in zip(self.lbs_heads, decoder_feats)
        ]

        seg_logits = self.segmentation_head(d1)
        seg_logits = F.interpolate(
            seg_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "seg_logits": seg_logits,
            "edge_preds": edge_preds,
        }
