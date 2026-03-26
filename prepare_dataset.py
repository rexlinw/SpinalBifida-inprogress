import os
import shutil
import random
import glob

import cv2
import numpy as np

random.seed(42)

BASE = 'dataset'
OUTPUT = 'dataset_prepared'

NORMAL_DIR = os.path.join(BASE, 'Normal')
REAL_SB_DIR = os.path.join(BASE, 'Spina_Bifida_Real_Split')
SYNTHETIC_SB_DIR = os.path.join(BASE, 'Spina_Bifida')

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Target mean brightness (grayscale) that HC18 Normal images cluster around.
# Real SB panels from PMC papers can be much brighter due to white paper
# backgrounds, labels, and scale bars.  Normalising them to this target
# eliminates the brightness domain gap so the model cannot trivially
# distinguish classes by global brightness.
_BRIGHTNESS_TARGET = 45.0


def get_image_files(directory, exclude_annotations=False):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    if exclude_annotations:
        files = [f for f in files if '_Annotation' not in os.path.basename(f)]
    return sorted(files)


def split_files(files, train_r, val_r):
    random.shuffle(files)
    n = len(files)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))
    return files[:train_end], files[train_end:val_end], files[val_end:]


def copy_files(file_list, dest_dir, normalise_brightness=False):
    """Copy files to dest_dir, optionally normalising mean brightness.

    ``normalise_brightness=True`` should be used for real SB panel images
    that originate from published paper figures.  These often contain
    bright white backgrounds, labels, and scale bars that produce a
    ~2× brightness gap vs. the dark HC18 Normal images.  A global
    brightness scale is applied so the mean grey-level of each image
    is brought close to ``_BRIGHTNESS_TARGET``, matching the Normal class
    distribution without altering the ultrasound structures themselves.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for f in file_list:
        if not normalise_brightness:
            shutil.copy2(f, dest_dir)
            continue

        img = cv2.imread(f)
        if img is None:
            shutil.copy2(f, dest_dir)
            continue

        grey_mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
        if grey_mean > 0:
            scale = _BRIGHTNESS_TARGET / grey_mean
            img_norm = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        else:
            img_norm = img

        out_path = os.path.join(dest_dir, os.path.basename(f))
        cv2.imwrite(out_path, img_norm)


def main():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    normal_files = get_image_files(NORMAL_DIR)
    print(f"Normal images found: {len(normal_files)}")

    n_train, n_val, n_test = split_files(normal_files, TRAIN_RATIO, VAL_RATIO)
    print(f"  Train: {len(n_train)}, Val: {len(n_val)}, Test: {len(n_test)}")

    copy_files(n_train, os.path.join(OUTPUT, 'train', 'Normal'))
    copy_files(n_val, os.path.join(OUTPUT, 'val', 'Normal'))
    copy_files(n_test, os.path.join(OUTPUT, 'test', 'Normal'))

    real_files = get_image_files(REAL_SB_DIR)
    print(f"\nReal Spina Bifida images found: {len(real_files)}")

    random.shuffle(real_files)
    real_test_end = int(len(real_files) * 0.4)
    real_val_end = int(len(real_files) * 0.7)
    real_test = real_files[:real_test_end]
    real_val = real_files[real_test_end:real_val_end]
    real_train = real_files[real_val_end:]

    print(f"  Real -> Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")

    synthetic_files = get_image_files(SYNTHETIC_SB_DIR, exclude_annotations=True)
    print(f"Synthetic Spina Bifida images found (annotations excluded): {len(synthetic_files)}")

    target_synthetic_train = len(n_train) - len(real_train)
    target_synthetic_val = len(n_val) - len(real_val)

    # If not enough synthetic images to fully balance both splits, allocate them
    # proportionally (train_r : val_r) so that val still gets some synthetic images.
    total_needed = target_synthetic_train + target_synthetic_val
    random.shuffle(synthetic_files)
    if len(synthetic_files) >= total_needed:
        syn_train = synthetic_files[:target_synthetic_train]
        syn_val = synthetic_files[target_synthetic_train:target_synthetic_train + target_synthetic_val]
    else:
        train_share = int(len(synthetic_files) * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))
        syn_train = synthetic_files[:train_share]
        syn_val = synthetic_files[train_share:]
        print(
            f"  Warning: only {len(synthetic_files)} synthetic images available (needed {total_needed}). "
            "Allocating proportionally; class imbalance will exist — use class weights during training."
        )

    print(f"  Synthetic -> Train: {len(syn_train)}, Val: {len(syn_val)}")

    # Real SB panels come from published paper figures and have bright
    # backgrounds/labels; normalise their brightness to match HC18 Normal.
    # Synthetic images are derived from HC18 Normal images so their
    # brightness already matches — copy them unchanged.
    copy_files(real_train, os.path.join(OUTPUT, 'train', 'Spina_Bifida'), normalise_brightness=True)
    copy_files(syn_train, os.path.join(OUTPUT, 'train', 'Spina_Bifida'))
    copy_files(real_val, os.path.join(OUTPUT, 'val', 'Spina_Bifida'), normalise_brightness=True)
    copy_files(syn_val, os.path.join(OUTPUT, 'val', 'Spina_Bifida'))
    copy_files(real_test, os.path.join(OUTPUT, 'test', 'Spina_Bifida'), normalise_brightness=True)

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for split in ['train', 'val', 'test']:
        for cls in ['Normal', 'Spina_Bifida']:
            d = os.path.join(OUTPUT, split, cls)
            if os.path.exists(d):
                count = len(os.listdir(d))
                print(f"  {split:5s}/{cls:15s}: {count:5d} images")
    print("=" * 50)
    print(f"\nDataset prepared in '{OUTPUT}/'")


if __name__ == '__main__':
    main()
