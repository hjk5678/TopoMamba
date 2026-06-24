import os
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("/data/BUAS/HJK/TopoMamba")

from data.dataset import rgb_to_label


# ============================================================
# 基本配置
# ============================================================
DATASET_ROOT = "/data/BUAS/HJK/TopoMamba/data/Potsdam"

IMG_DIR = os.path.join(DATASET_ROOT, "Images")
LABEL_DIR = os.path.join(DATASET_ROOT, "Labels")

OUT_ROOT = os.path.join(DATASET_ROOT, "processed")

CROP_SIZE = 512

# 训练集建议用重叠切块，增加样本量和边界上下文
TRAIN_STRIDE = 256

# 验证集建议不用太多重叠，避免验证集样本过多
VAL_STRIDE = 512

# 按大图划分 train / val，避免同一张大图的 patch 同时出现在 train 和 val
VAL_RATIO = 0.2

SEED = 42
NUM_CLASSES = 6
IGNORE_INDEX = 255

# 有效像素比例太低的 patch 跳过
MIN_VALID_RATIO = 0.1

# 是否删除旧 processed
CLEAR_OLD_PROCESSED = False


# ============================================================
# 工具函数
# ============================================================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def find_label_file(img_file):
    """
    根据 Potsdam 图像名寻找对应标签。
    兼容：
        top_potsdam_x_x_RGB.tif
        top_potsdam_x_x_IRRG.tif
        top_potsdam_x_x_label.tif
        top_potsdam_x_x_label_noBoundary.tif
    """
    base, ext = os.path.splitext(img_file)

    base_candidates = [base]

    for suffix in ["_RGB", "_IRRG"]:
        if base.endswith(suffix):
            base_candidates.append(base[:-len(suffix)])

    candidates = []

    for b in base_candidates:
        candidates.extend([
            f"{b}_label.tif",
            f"{b}_label.tiff",
            f"{b}_label.png",

            f"{b}_label_noBoundary.tif",
            f"{b}_label_noBoundary.tiff",
            f"{b}_label_noBoundary.png",

            f"{b}.tif",
            f"{b}.tiff",
            f"{b}.png",
        ])

    for cand in candidates:
        p = os.path.join(LABEL_DIR, cand)
        if os.path.exists(p):
            return p

    return None


def get_start_positions(length, crop_size, stride):
    """
    生成滑窗起点，确保最后一个 patch 覆盖到图像边界。
    """
    if length <= crop_size:
        return [0]

    positions = list(range(0, length - crop_size + 1, stride))

    last = length - crop_size
    if positions[-1] != last:
        positions.append(last)

    return sorted(set(positions))


def pad_if_needed(img, label, crop_size):
    """
    如果图像小于 crop_size，则 padding。
    图像用 reflect padding，label 用 255 padding。
    """
    h, w = img.shape[:2]

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT
        )

        label = cv2.copyMakeBorder(
            label,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=IGNORE_INDEX
        )

    return img, label


def collect_image_label_pairs():
    """
    收集图像与标签路径。
    """
    image_files = []

    for f in sorted(os.listdir(IMG_DIR)):
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            image_files.append(f)

    pairs = []
    missing = []

    for img_file in image_files:
        label_path = find_label_file(img_file)

        if label_path is None:
            missing.append(img_file)
            continue

        img_path = os.path.join(IMG_DIR, img_file)
        pairs.append((img_file, img_path, label_path))

    print("=" * 80)
    print(f"Image dir : {IMG_DIR}")
    print(f"Label dir : {LABEL_DIR}")
    print(f"Found images with labels: {len(pairs)}")
    print(f"Missing labels: {len(missing)}")

    if len(missing) > 0:
        print("前 10 个缺失标签的图像：")
        for f in missing[:10]:
            print("  ", f)

    print("=" * 80)

    if len(pairs) == 0:
        raise RuntimeError("没有找到任何图像-标签对，请检查 Images 和 Labels 路径。")

    return pairs


def split_train_val(pairs, val_ratio=0.2, seed=42):
    """
    按大图划分训练集和验证集。
    """
    random.seed(seed)

    pairs = pairs.copy()
    random.shuffle(pairs)

    val_num = max(1, int(len(pairs) * val_ratio))

    val_pairs = pairs[:val_num]
    train_pairs = pairs[val_num:]

    print(f"Train large images: {len(train_pairs)}")
    print(f"Val large images  : {len(val_pairs)}")

    print("\nVal images:")
    for img_file, _, _ in val_pairs:
        print("  ", img_file)

    print("=" * 80)

    return train_pairs, val_pairs


def crop_one_split(pairs, split, stride):
    """
    对一个 split 进行切块。
    """
    out_img_dir = os.path.join(OUT_ROOT, split, "images")
    out_label_dir = os.path.join(OUT_ROOT, split, "labels")

    safe_mkdir(out_img_dir)
    safe_mkdir(out_label_dir)

    count = 0
    skipped_invalid = 0
    class_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    unique_values_all = set()

    print(f"\n开始处理 {split} split, stride={stride}")

    for img_file, img_path, label_path in tqdm(pairs, desc=f"Cropping {split}"):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Skip] 无法读取图像: {img_path}")
            continue

        label_bgr = cv2.imread(label_path, cv2.IMREAD_COLOR)
        if label_bgr is None:
            print(f"[Skip] 无法读取标签: {label_path}")
            continue

        label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)

        # RGB 标签转类别索引：0~5，未匹配为 255
        label = rgb_to_label(label_rgb).astype(np.uint8)

        img, label = pad_if_needed(img, label, CROP_SIZE)

        h, w = img.shape[:2]

        ys = get_start_positions(h, CROP_SIZE, stride)
        xs = get_start_positions(w, CROP_SIZE, stride)

        stem = os.path.splitext(img_file)[0]

        for y in ys:
            for x in xs:
                crop_img = img[y:y + CROP_SIZE, x:x + CROP_SIZE]
                crop_label = label[y:y + CROP_SIZE, x:x + CROP_SIZE]

                valid_ratio = np.mean(crop_label != IGNORE_INDEX)

                if valid_ratio < MIN_VALID_RATIO:
                    skipped_invalid += 1
                    continue

                vals = np.unique(crop_label)
                unique_values_all.update(vals.tolist())

                for c in range(NUM_CLASSES):
                    class_hist[c] += np.sum(crop_label == c)

                patch_name = f"{stem}_{y}_{x}.png"

                cv2.imwrite(
                    os.path.join(out_img_dir, patch_name),
                    crop_img
                )

                cv2.imwrite(
                    os.path.join(out_label_dir, patch_name),
                    crop_label
                )

                count += 1

    print("-" * 80)
    print(f"Split          : {split}")
    print(f"Generated      : {count}")
    print(f"Skipped invalid: {skipped_invalid}")
    print(f"Class hist     : {class_hist.tolist()}")
    print(f"Unique values  : {sorted(unique_values_all)}")
    print("-" * 80)

    return count, skipped_invalid, class_hist, unique_values_all


def check_processed_structure():
    """
    检查最终 processed 目录结构。
    """
    required_dirs = [
        os.path.join(OUT_ROOT, "train", "images"),
        os.path.join(OUT_ROOT, "train", "labels"),
        os.path.join(OUT_ROOT, "val", "images"),
        os.path.join(OUT_ROOT, "val", "labels"),
    ]

    print("\n最终目录检查：")

    for d in required_dirs:
        if not os.path.exists(d):
            print(f"[Missing] {d}")
        else:
            num = len([
                f for f in os.listdir(d)
                if f.lower().endswith(".png")
            ])
            print(f"[OK] {d} | files={num}")


def main():
    print("=" * 80)
    print("Potsdam 预切块脚本")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"OUT_ROOT    : {OUT_ROOT}")
    print(f"CROP_SIZE   : {CROP_SIZE}")
    print(f"TRAIN_STRIDE: {TRAIN_STRIDE}")
    print(f"VAL_STRIDE  : {VAL_STRIDE}")
    print("=" * 80)

    if CLEAR_OLD_PROCESSED and os.path.exists(OUT_ROOT):
        # 安全检查，防止误删
        abs_out = os.path.abspath(OUT_ROOT)
        expected_suffix = os.path.join("../data", "Potsdam", "processed")

        if not abs_out.endswith(expected_suffix):
            raise RuntimeError(f"危险路径，拒绝删除: {abs_out}")

        print(f"删除旧 processed 目录: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    pairs = collect_image_label_pairs()

    train_pairs, val_pairs = split_train_val(
        pairs,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    train_count, train_skip, train_hist, train_uniques = crop_one_split(
        train_pairs,
        split="train",
        stride=TRAIN_STRIDE
    )

    val_count, val_skip, val_hist, val_uniques = crop_one_split(
        val_pairs,
        split="val",
        stride=VAL_STRIDE
    )

    check_processed_structure()

    print("\n" + "=" * 80)
    print("预切块完成！")
    print(f"Train patches: {train_count}")
    print(f"Val patches  : {val_count}")
    print(f"保存位置     : {OUT_ROOT}")
    print("=" * 80)

    all_uniques = sorted(set(list(train_uniques) + list(val_uniques)))
    print(f"全部标签值: {all_uniques}")

    legal_values = set(list(range(NUM_CLASSES)) + [IGNORE_INDEX])
    bad_values = [v for v in all_uniques if v not in legal_values]

    if len(bad_values) > 0:
        print(f"[Warning] 发现非法标签值: {bad_values}")
        print("请检查 rgb_to_label 或标签颜色。")
    else:
        print("[OK] 标签值正常，只包含 0~5 和 255。")


if __name__ == "__main__":
    main()