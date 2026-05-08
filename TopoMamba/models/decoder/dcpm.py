import torch
import torch.nn as nn
import torch.nn.functional as F

class DCPM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pred_4 = nn.Conv2d(16, 1, 1)
        self.pred_8 = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        feat = self.reduce(x)
        logit_4 = self.pred_4(feat)   # (B, 1, H, W)
        logit_8 = self.pred_8(feat)
        return torch.cat([logit_4, logit_8], dim=1)

def generate_conn_gt(label, num_classes=None):
    # label: (H, W) or (B, H, W) tensor, values 0..C-1
    # returns GT_4, GT_8 same shape as label
    if label.dim() == 2:
        label = label.unsqueeze(0)
    B, H, W = label.shape
    GT4 = torch.zeros_like(label, dtype=torch.float32)
    GT8 = torch.zeros_like(label, dtype=torch.float32)
    # right & down neighbors
    right = (label[:, :, :-1] == label[:, :, 1:]).float()
    down = (label[:, :-1, :] == label[:, 1:, :]).float()
    GT4[:, :, :-1] += right
    GT4[:, :, 1:] += right
    GT4[:, :-1, :] += down
    GT4[:, 1:, :] += down
    GT4 = (GT4 > 0).float()
    # diagonals
    diag1 = (label[:, :-1, :-1] == label[:, 1:, 1:]).float()
    diag2 = (label[:, :-1, 1:] == label[:, 1:, :-1]).float()
    GT8 = GT4.clone()
    GT8[:, :-1, :-1] += diag1
    GT8[:, 1:, 1:] += diag1
    GT8[:, :-1, 1:] += diag2
    GT8[:, 1:, :-1] += diag2
    GT8 = (GT8 > 0).float()
    return GT4, GT8