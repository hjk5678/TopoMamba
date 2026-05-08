import os
import cv2
import numpy as np

ROOT = '/data/BUAS/HJK/TopoMamba/data/Love DA'
CROP_SIZE = 256
STRIDE = 256

for split in ['Train', 'Val']:
    for scene in ['Rural', 'Urban']:
        img_dir = os.path.join(ROOT, split, scene, 'images_png')
        mask_dir = os.path.join(ROOT, split, scene, 'masks_png')
        if not os.path.exists(img_dir):
            continue

        out_dir = os.path.join(ROOT, 'processed_256', split.lower(), scene.lower())
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)

        img_files = sorted(os.listdir(img_dir))
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]

            count = 0
            for y in range(0, h, STRIDE):
                for x in range(0, w, STRIDE):
                    if y + CROP_SIZE > h or x + CROP_SIZE > w:
                        continue
                    crop_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    crop_label = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    patch_name = f"{os.path.splitext(fname)[0]}_{y}_{x}.png"
                    cv2.imwrite(os.path.join(out_dir, 'images', patch_name), crop_img)
                    cv2.imwrite(os.path.join(out_dir, 'labels', patch_name), crop_label)
                    count += 1
            print(f"切分 {split}/{scene}/{fname}: {count} 个小块")

print("LoveDA 256×256 预切块完成！")