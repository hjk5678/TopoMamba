import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ===================== 彩色标签 -> 类别索引 =====================
ISPRS_PALETTE = {
    (255, 255, 255): 0,  # 不透水面
    (0, 0, 255): 1,      # 建筑物
    (0, 255, 255): 2,    # 低矮植被
    (0, 255, 0): 3,      # 树木
    (255, 255, 0): 4,    # 汽车
    (255, 0, 0): 5       # 背景/杂物
}

def rgb_to_label(rgb_mask):
    """RGB 标签 -> 单通道类别索引"""
    label = np.full(rgb_mask.shape[:2], 255, dtype=np.uint8)
    for rgb_color, class_id in ISPRS_PALETTE.items():
        match = np.all(rgb_mask == rgb_color, axis=-1)
        label[match] = class_id
    return label


# ===================== 数据增强 =====================
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomFlip(object):
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask):
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        return img.copy(), mask.copy()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img, mask):
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img, mask


class ToTensor(object):
    def __call__(self, img, mask):
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        return img, mask


# ===================== ISPRS 数据集（Potsdam / Vaihingen） =====================
class ISPRSDataset(Dataset):
    def __init__(self, img_dir, label_dir, split='train', crop_size=512,
                 transform=None, pre_cropped=False, processed_dir=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split = split
        self.crop_size = crop_size
        self.transform = transform
        self.pre_cropped = pre_cropped

        if pre_cropped:
            if processed_dir is None:
                processed_dir = os.path.join(os.path.dirname(img_dir), 'processed', split)
            self.processed_img_dir = os.path.join(processed_dir, 'images')
            self.processed_label_dir = os.path.join(processed_dir, 'labels')
            self.patches = sorted(os.listdir(self.processed_img_dir))
            self.patches = [f for f in self.patches if f.lower().endswith('.png')]
        else:
            self.img_files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith('.tif')
            ])
            valid_files = []
            for f in self.img_files:
                label_name = self._get_label_name(f)
                if os.path.exists(os.path.join(label_dir, label_name)):
                    valid_files.append(f)
                elif os.path.exists(os.path.join(label_dir, f)):
                    valid_files.append(f)
            self.img_files = valid_files

    def _get_label_name(self, img_name: str) -> str:
        base, ext = os.path.splitext(img_name)
        if base.endswith('_RGB'):
            base = base[:-4]
        return f"{base}_label{ext}"

    def __len__(self):
        if self.pre_cropped:
            return len(self.patches)
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.pre_cropped:
            patch_file = self.patches[idx]
            img_path = os.path.join(self.processed_img_dir, patch_file)
            label_path = os.path.join(self.processed_label_dir, patch_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            crop_img, crop_label = img, label
        else:
            img_file = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_file)
            label_name = self._get_label_name(img_file)
            label_path = os.path.join(self.label_dir, label_name)
            if not os.path.exists(label_path):
                label_path = os.path.join(self.label_dir, img_file)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label_rgb = cv2.imread(label_path, cv2.IMREAD_COLOR)
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            label = rgb_to_label(label_rgb)

            h, w = img.shape[:2]
            if self.split in ['train', 'val']:
                if h < self.crop_size or w < self.crop_size:
                    pad_h = max(0, self.crop_size - h)
                    pad_w = max(0, self.crop_size - w)
                    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                              cv2.BORDER_CONSTANT, value=255)
                    crop_img, crop_label = img, label
                else:
                    if self.split == 'train':
                        x = np.random.randint(0, w - self.crop_size + 1)
                        y = np.random.randint(0, h - self.crop_size + 1)
                    else:
                        x = (w - self.crop_size) // 2
                        y = (h - self.crop_size) // 2
                    crop_img = img[y:y+self.crop_size, x:x+self.crop_size]
                    crop_label = label[y:y+self.crop_size, x:x+self.crop_size]
            else:
                crop_img, crop_label = img, label

        if self.transform:
            crop_img, crop_label = self.transform(crop_img, crop_label)

        return crop_img, crop_label


# ===================== LoveDA 数据集 =====================
class LoveDADataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, pre_cropped=False, processed_dir=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.pre_cropped = pre_cropped

        self.data = []
        if pre_cropped:
            if processed_dir is None:
                processed_dir = os.path.join(root_dir, 'processed')
            base_dir = os.path.join(processed_dir, split.lower())
            for scene in ['rural', 'urban']:
                img_dir = os.path.join(base_dir, scene, 'images')
                mask_dir = os.path.join(base_dir, scene, 'labels')
                if not os.path.exists(img_dir):
                    continue
                img_files = sorted(os.listdir(img_dir))
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    mask_path = os.path.join(mask_dir, img_file)
                    if os.path.exists(mask_path):
                        self.data.append((img_path, mask_path))
        else:
            for scene in ['Rural', 'Urban']:
                img_dir = os.path.join(root_dir, split, scene, 'images_png')
                mask_dir = os.path.join(root_dir, split, scene, 'masks_png')
                if not os.path.exists(img_dir):
                    continue
                img_files = sorted(os.listdir(img_dir))
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    mask_path = os.path.join(mask_dir, img_file)
                    if os.path.exists(mask_path):
                        self.data.append((img_path, mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


# ===================== 获取数据增强 =====================
def get_transform(split='train', mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    transform_list = []
    if split == 'train':
        transform_list.extend([
            RandomFlip(),
            RandomRotate(),
        ])
    transform_list.extend([
        Normalize(mean=mean, std=std),
        ToTensor()
    ])
    return Compose(transform_list)


# ===================== 工厂函数 =====================
def build_dataset(dataset_name, root_dir, split='train', pre_cropped=False, processed_dir=None):
    transform = get_transform(split)
    if dataset_name.lower() == 'loveda':
        split_map = {'train':'Train', 'val':'Val', 'test':'Test'}
        return LoveDADataset(
            root_dir,
            split=split_map.get(split, 'Train'),
            transform=transform,
            pre_cropped=pre_cropped,
            processed_dir=processed_dir
        )
    elif dataset_name.lower() in ['potsdam', 'vaihingen']:
        img_dir = os.path.join(root_dir, 'Images')
        label_dir = os.path.join(root_dir, 'Labels')
        if pre_cropped:
            if processed_dir is None:
                processed_dir = os.path.join(root_dir, 'processed', split)
            return ISPRSDataset(
                img_dir, label_dir,
                split=split, crop_size=512,
                transform=transform,
                pre_cropped=True,
                processed_dir=processed_dir
            )
        else:
            return ISPRSDataset(
                img_dir, label_dir,
                split=split, crop_size=512,
                transform=transform,
                pre_cropped=False
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")