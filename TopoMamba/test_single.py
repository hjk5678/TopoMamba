import sys
import os
import torch
sys.path.append('/data/BUAS/HJK/TopoMamba')

from data.dataset import build_dataset
from models.build_model import TopoMamba
from utils.losses import TopoMambaLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用新的 256x256 预切块
train_set = build_dataset(
    'loveda',
    '/data/BUAS/HJK/TopoMamba/data/Love DA',
    split='train',
    pre_cropped=True,
    processed_dir='/data/BUAS/HJK/TopoMamba/data/Love DA/processed_256'
)

loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

model = TopoMamba(num_classes=7).to(device)
criterion = TopoMambaLoss(num_classes=7, lambda_edge=0.2, lambda_conn=0.1)

model.train()
img, mask = next(iter(loader))
img, mask = img.to(device), mask.long().to(device)
seg, conn, edge_preds = model(img)
loss_dict = criterion(seg, mask, conn, edge_preds)
print(f'单卡验证通过，Loss: {loss_dict["total_loss"].item():.4f}')