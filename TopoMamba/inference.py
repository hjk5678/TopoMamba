import os
import sys
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('/data/BUAS/HJK/TopoMamba')

from models.build_model import TopoMamba
from data.dataset import ISPRS_PALETTE, get_transform

def label_to_color(label):
    color_map = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for rgb, class_id in ISPRS_PALETTE.items():
        color_map[label == class_id] = list(rgb)
    return color_map

def sliding_window_inference(model, img, crop_size=512, stride=256, num_classes=6, device='cuda'):
    model.eval()
    h, w, _ = img.shape
    # 如果原图小于裁剪尺寸，则反射填充
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]

    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # 生成合法的起始坐标（确保每个窗口都完整不越界）
    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))
    y_starts = [y for y in y_starts if y + crop_size <= h]
    x_starts = [x for x in x_starts if x + crop_size <= w]
    # 覆盖右下边界
    if not y_starts or y_starts[-1] < h - crop_size:
        y_starts.append(h - crop_size)
    if not x_starts or x_starts[-1] < w - crop_size:
        x_starts.append(w - crop_size)
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    from data.dataset import Normalize, ToTensor
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensor()

    total_windows = len(y_starts) * len(x_starts)
    pbar = tqdm(total=total_windows, desc='滑动窗口推理', leave=True)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                crop = img[y:y+crop_size, x:x+crop_size, :]  # 此时尺寸一定是 crop_size x crop_size
                crop_norm, _ = normalize(crop, np.zeros((crop_size, crop_size)))
                crop_tensor, _ = to_tensor(crop_norm, np.zeros((crop_size, crop_size)))
                crop_tensor = crop_tensor.unsqueeze(0).to(device)
                seg_logits, _, _ = model(crop_tensor)
                probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy().transpose(1,2,0)
                prob_map[y:y+crop_size, x:x+crop_size] += probs
                count_map[y:y+crop_size, x:x+crop_size] += 1.0
                pbar.update(1)
    pbar.close()

    prob_map /= count_map[..., np.newaxis]
    pred = np.argmax(prob_map, axis=-1).astype(np.uint8)

    # 裁剪回原始尺寸（去除填充）
    if pad_h > 0 or pad_w > 0:
        pred = pred[:h-pad_h, :w-pad_w]
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vaihingen',
                        choices=['potsdam', 'vaihingen'])
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    num_classes = 6
    if args.dataset == 'potsdam':
        data_root = '/data/BUAS/HJK/TopoMamba/data/Potsdam/Images'
        default_ckpt = 'checkpoints/topomamba_potsdam_epoch100.pth'
    else:
        data_root = '/data/BUAS/HJK/TopoMamba/data/Vaihingen/Images'
        default_ckpt = 'checkpoints/topomamba_vaihingen_epoch100.pth'

    if args.image is None:
        all_imgs = sorted([f for f in os.listdir(data_root) if f.lower().endswith('.tif')])
        if not all_imgs:
            print(f"错误: {data_root} 中没有 .tif 文件")
            return
        img_path = os.path.join(data_root, all_imgs[0])
        print(f"自动选择第一张图片: {img_path}")
    else:
        img_path = args.image

    checkpoint_path = args.checkpoint if args.checkpoint else default_ckpt
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在 {checkpoint_path}")
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"无法读取图片: {img_path}")
        return
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"图片尺寸: {img.shape}")

    model = TopoMamba(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("模型加载完成")

    pred = sliding_window_inference(model, img, crop_size=args.crop_size,
                                    stride=args.stride, num_classes=num_classes, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    label_path = os.path.join(args.output_dir, f"{base_name}_label.png")
    cv2.imwrite(label_path, pred)
    print(f"类别索引图保存至: {label_path}")

    color_pred = label_to_color(pred)
    color_path = os.path.join(args.output_dir, f"{base_name}_color.png")
    cv2.imwrite(color_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
    print(f"彩色分割图保存至: {color_path}")

    print("推理完成！")

if __name__ == '__main__':
    main()