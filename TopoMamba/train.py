import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

sys.path.append('/data/BUAS/HJK/TopoMamba')

from data.dataset import (
    build_dataset,
    ISPRSDataset,
    LoveDADataset,
    get_transform,
)
from models.build_model import TopoMamba
from utils.losses import TopoMambaLoss

# -------------------- 评估指标 --------------------
def compute_oa(pred, target, ignore_index=255):
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return 0.0
    correct = (pred[valid_mask] == target[valid_mask]).float().sum()
    return (correct / valid_mask.sum()).item()

def compute_miou(pred, target, num_classes, ignore_index=255):
    iou_list = []
    valid_mask = (target != ignore_index)
    for c in range(num_classes):
        pred_c = (pred == c) & valid_mask
        target_c = (target == c) & valid_mask
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_list.append(iou)
    return torch.tensor(iou_list).mean().item()

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# -------------------- 训练一个epoch --------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, num_classes, epoch, world_size, rank):
    model.train()
    total_ce = 0
    total_dice = 0
    total_lbs = 0
    total_conn = 0
    total_loss = 0

    if rank == 0:
        pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False, dynamic_ncols=True)
    else:
        pbar = loader

    for img, mask in pbar:
        img, mask = img.to(device), mask.long().to(device)
        mask[(mask < 0) | (mask >= num_classes)] = 255

        optimizer.zero_grad()
        with autocast():
            seg_logits, conn_logits, edge_preds = model(img)
            loss_dict = criterion(seg_logits, mask, conn_logits, edge_preds)
            loss = loss_dict['total_loss']

        # 遇到 NaN 时，将 loss 置为零并继续，保证所有进程同步
        if torch.isnan(loss):
            if rank == 0:
                print(f"NaN loss detected at epoch {epoch}, setting loss to 0")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {k: 0.0 for k in loss_dict}

        # 即使 loss 为 0，也要进行完整的 backward → unscale → step，以避免 GradScaler 断言错误
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        bs = img.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_dict['ce_loss'] * bs
        total_dice += loss_dict['dice_loss'] * bs
        total_lbs += loss_dict['lbs_loss'] * bs
        total_conn += loss_dict['conn_loss'] * bs

        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_dict['ce_loss']:.4f}",
                'dice': f"{loss_dict['dice_loss']:.4f}",
            })

    n = len(loader.dataset)
    total_loss = reduce_sum(torch.tensor(total_loss, device=device)).item()
    total_ce = reduce_sum(torch.tensor(total_ce, device=device)).item()
    total_dice = reduce_sum(torch.tensor(total_dice, device=device)).item()
    total_lbs = reduce_sum(torch.tensor(total_lbs, device=device)).item()
    total_conn = reduce_sum(torch.tensor(total_conn, device=device)).item()

    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'dice': total_dice / n,
        'lbs': total_lbs / n,
        'conn': total_conn / n,
    }

# -------------------- 验证一个epoch --------------------
@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, world_size, rank):
    model.eval()
    total_loss = 0
    total_oa = 0
    total_miou = 0

    if rank == 0:
        pbar = tqdm(loader, desc='Validation', leave=False, dynamic_ncols=True)
    else:
        pbar = loader

    for img, mask in pbar:
        img, mask = img.to(device), mask.long().to(device)
        mask[(mask < 0) | (mask >= num_classes)] = 255

        seg_logits, conn_logits, edge_preds = model(img)
        loss_dict = criterion(seg_logits, mask, conn_logits, edge_preds)
        total_loss += loss_dict['total_loss'].item() * img.size(0)

        pred = seg_logits.argmax(dim=1)
        if pred.shape[-2:] != mask.shape[-2:]:
            if rank == 0:
                print(f"\n⚠️ 尺寸不匹配: pred {pred.shape[-2:]}, mask {mask.shape[-2:]}, 自动插值对齐")
            pred = F.interpolate(pred.unsqueeze(1).float(), size=mask.shape[-2:],
                                 mode='nearest').squeeze(1).long()

        batch_oa = compute_oa(pred, mask)
        total_oa += batch_oa * img.size(0)
        total_miou += compute_miou(pred, mask, num_classes) * img.size(0)

        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'oa': f"{batch_oa:.4f}",
            })

    n = len(loader.dataset)
    total_loss = reduce_sum(torch.tensor(total_loss, device=device)).item()
    total_oa = reduce_sum(torch.tensor(total_oa, device=device)).item()
    total_miou = reduce_sum(torch.tensor(total_miou, device=device)).item()
    return {
        'loss': total_loss / n,
        'oa': total_oa / n,
        'miou': total_miou / n,
    }

# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='potsdam',
                        choices=['potsdam', 'vaihingen', 'loveda'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda_edge', type=float, default=0.2)
    parser.add_argument('--lambda_conn', type=float, default=0.1)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pre_cropped', action='store_true', default=False,
                        help='使用离线预切块')
    parser.add_argument('--processed_dir', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # 初始化分布式进程组
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    seed = 42 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rank == 0:
        print(f"使用 {world_size} 张 GPU 进行分布式训练")

    # 数据集路径
    base = '/data/BUAS/HJK/TopoMamba/data'
    if args.data_root is not None:
        root = args.data_root
    else:
        if args.dataset == 'potsdam':
            root = os.path.join(base, 'Potsdam')
            num_classes = 6
        elif args.dataset == 'vaihingen':
            root = os.path.join(base, 'Vaihingen')
            num_classes = 6
        else:
            root = os.path.join(base, 'Love DA')
            num_classes = 7

    if rank == 0:
        print(f"数据集: {args.dataset}, 路径: {root}")
        print(f"类别数: {num_classes}")

    # 构建数据集
    if args.dataset.lower() == 'loveda':
        train_set = build_dataset(
            'loveda', root, split='train',
            pre_cropped=args.pre_cropped,
            processed_dir=args.processed_dir
        )
        val_set = build_dataset(
            'loveda', root, split='val',
            pre_cropped=args.pre_cropped,
            processed_dir=args.processed_dir
        )
    else:
        if args.pre_cropped:
            img_dir = os.path.join(root, 'Images')
            label_dir = os.path.join(root, 'Labels')
            if args.processed_dir is None:
                train_processed_dir = os.path.join(root, 'processed', 'train')
                val_processed_dir = os.path.join(root, 'processed', 'val')
            else:
                train_processed_dir = os.path.join(args.processed_dir, 'train')
                val_processed_dir = os.path.join(args.processed_dir, 'val')

            if not os.path.exists(train_processed_dir):
                raise FileNotFoundError(f"预切块目录不存在: {train_processed_dir}")

            train_set = ISPRSDataset(
                img_dir, label_dir, split='train', crop_size=512,
                transform=get_transform('train'),
                pre_cropped=True,
                processed_dir=train_processed_dir
            )
            if os.path.exists(val_processed_dir):
                val_set = ISPRSDataset(
                    img_dir, label_dir, split='val', crop_size=512,
                    transform=get_transform('val'),
                    pre_cropped=True,
                    processed_dir=val_processed_dir
                )
            else:
                if rank == 0:
                    print(f"警告: 验证预切块目录不存在，回退到在线裁剪模式")
                val_set = ISPRSDataset(
                    img_dir, label_dir, split='val', crop_size=512,
                    transform=get_transform('val'),
                    pre_cropped=False
                )
        else:
            train_set = build_dataset(args.dataset, root, split='train')
            val_set = build_dataset(args.dataset, root, split='val')

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}")
        if args.pre_cropped:
            print("使用预切块模式")

    model = TopoMamba(num_classes=num_classes)

    import torchvision.models as models
    pretrained_resnet = models.resnet18(pretrained=True)
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'layer' in k}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    if rank == 0:
        print("ResNet-18 预训练参数加载成功")

    model = model.to(device)

    # 永不解冻 VMamba —— 无冻结代码，全部参数可训练

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    criterion = TopoMambaLoss(num_classes=num_classes,
                              lambda_edge=args.lambda_edge,
                              lambda_conn=args.lambda_conn)

    cnn_params = []
    vmamba_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'cnn_branch' in name:
                cnn_params.append(param)
            elif 'vmamba_branch' in name:
                vmamba_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.AdamW([
        {'params': cnn_params,    'lr': args.lr * 0.1},
        {'params': vmamba_params, 'lr': 1e-5},
        {'params': other_params,  'lr': args.lr}
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"从 epoch {start_epoch} 恢复训练")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                        num_classes, epoch+1, world_size, rank)
        scheduler.step()
        elapsed = time.time() - t0

        if rank == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"CE: {train_metrics['ce']:.4f} | "
                  f"Dice: {train_metrics['dice']:.4f} | "
                  f"LBS: {train_metrics['lbs']:.4f} | "
                  f"Conn: {train_metrics['conn']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        if (epoch + 1) % args.val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device, num_classes, world_size, rank)
            if rank == 0:
                print(f"验证   | Loss: {val_metrics['loss']:.4f} | "
                      f"OA: {val_metrics['oa']:.4f} | mIoU: {val_metrics['miou']:.4f}")

        if epoch == args.epochs - 1 and rank == 0:
            save_path = os.path.join(args.save_dir, f"topomamba_{args.dataset}_epoch{epoch+1}.pth")
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(state, save_path)
            print(f"检查点已保存: {save_path}")

    if rank == 0:
        print("训练完成！")

if __name__ == '__main__':
    main()