import torch
import torch.nn as nn
import torch.nn.functional as F

from .bau import BAUBlock


class LBSHead(nn.Module):
    """
    Local Boundary Supervision Head.

    用于多尺度边界辅助监督。
    每个 decoder stage 输出一个 1-channel edge logit。
    """

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
    """
    Boundary-aware Decoder.

    当前版本说明：
        1. Decoder 只负责语义分割预测 seg_logits；
        2. Decoder 保留 LBS 多尺度边界预测 edge_preds；
        3. Decoder 不再包含 mstc；
        4. Decoder 不再包含 CARM；
        5. 多尺度 mstc 放到 build_model.py 中，在每层 SSA-M 输出后进行监督。

    输入:
        F4_enh: [B, 512, H/32, W/32]
        G3:     [B, 256, H/16, W/16]
        G2:     [B, 128, H/8,  W/8]
        G1:     [B, 64,  H/4,  W/4]
        G0:     [B, 32,  H/2,  W/2]

    输出:
        {
            "seg_logits": [B, num_classes, H, W],
            "edge_preds": list，每个元素 [B, 1, H, W]
        }
    """

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
        # 这里先保留简单 1x1 head，保证和之前实验可比。
        # 如果后面要增强，可以再改成 Conv3x3 + BN + ReLU + Conv1x1。
        self.seg_head = nn.Conv2d(
            dec_channels[3],
            num_classes,
            kernel_size=1
        )

    def forward(self, F4_enh, G3, G2, G1, G0, out_size=None):
        """
        Args:
            F4_enh: [B, 512, H/32, W/32]
            G3:     [B, 256, H/16, W/16]
            G2:     [B, 128, H/8,  W/8]
            G1:     [B, 64,  H/4,  W/4]
            G0:     [B, 32,  H/2,  W/2]
            out_size:
                最终输出尺寸。
                推荐传入原图尺寸，例如 x.shape[2:]。
                如果不传，则默认输出为 G0 尺寸的 2 倍。

        Returns:
            dict:
                seg_logits: [B, num_classes, out_H, out_W]
                edge_preds: list，每个元素 [B, 1, out_H, out_W]
        """

        # -------------------------------------------------
        # 1. Progressive BAU decoding
        # -------------------------------------------------
        d3 = self.up4(F4_enh, G3)   # [B, 256, H/16, W/16]
        d2 = self.up3(d3, G2)       # [B, 128, H/8,  W/8]
        d1 = self.up2(d2, G1)       # [B, 64,  H/4,  W/4]
        d0 = self.up1(d1, G0)       # [B, 32,  H/2,  W/2]

        # -------------------------------------------------
        # 2. Decide final output size
        # -------------------------------------------------
        if out_size is None:
            # 兼容旧代码：默认从 G0 的 H/2 恢复到 H
            out_size = (
                G0.shape[2] * 2,
                G0.shape[3] * 2
            )

        # -------------------------------------------------
        # 3. Final feature upsampling
        # -------------------------------------------------
        feat_out = F.interpolate(
            d0,
            size=out_size,
            mode="bilinear",
            align_corners=False
        )

        # -------------------------------------------------
        # 4. Multi-scale edge predictions
        # -------------------------------------------------
        decoder_feats = [d3, d2, d1, d0]

        edge_preds = []

        for head, feat in zip(self.lbs_heads, decoder_feats):
            edge_logit = head(feat)

            edge_logit = F.interpolate(
                edge_logit,
                size=out_size,
                mode="bilinear",
                align_corners=False
            )

            edge_preds.append(edge_logit)

        # -------------------------------------------------
        # 5. Main segmentation prediction
        # -------------------------------------------------
        seg_logits = self.seg_head(feat_out)

        return {
            "seg_logits": seg_logits,
            "edge_preds": edge_preds,
        }
