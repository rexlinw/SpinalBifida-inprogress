import os
import shutil
import random
import glob
import json

random.seed(42)

BASE = 'dataset'
OUTPUT = 'dataset_prepared'

NORMAL_DIR = os.path.join(BASE, 'Normal')
REAL_SB_DIR = os.path.join(BASE, 'Spina_Bifida_Real_Split')
SYNTHETIC_SB_DIR = os.path.join(BASE, 'Spina_Bifida')

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def get_image_files(directory):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def split_files(files, train_r, val_r):
    random.shuffle(files)
    n = len(files)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))
    return files[:train_end], files[train_end:val_end], files[val_end:]


def copy_files(file_list, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for f in file_list:
        shutil.copy2(f, dest_dir)


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

    synthetic_files = get_image_files(SYNTHETIC_SB_DIR)
    print(f"Synthetic Spina Bifida images found: {len(synthetic_files)}")

    target_synthetic_train = len(n_train) - len(real_train)
    target_synthetic_val = len(n_val) - len(real_val)

    random.shuffle(synthetic_files)
    syn_train = synthetic_files[:target_synthetic_train]
    syn_val = synthetic_files[target_synthetic_train:target_synthetic_train + target_synthetic_val]

    print(f"  Synthetic -> Train: {len(syn_train)}, Val: {len(syn_val)}")

    copy_files(real_train + syn_train, os.path.join(OUTPUT, 'train', 'Spina_Bifida'))
    copy_files(real_val + syn_val, os.path.join(OUTPUT, 'val', 'Spina_Bifida'))
    copy_files(real_test, os.path.join(OUTPUT, 'test', 'Spina_Bifida'))

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
