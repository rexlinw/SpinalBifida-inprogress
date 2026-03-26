import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ThreadPoolExecutor

INPUT_FOLDER = 'dataset/Normal'
OUTPUT_FOLDER = 'dataset/Spina_Bifida'
TARGET_COUNT = 1200


def verify_files():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        return []

    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"Found {len(files)} source images in '{INPUT_FOLDER}'")
    return files


def add_ultrasound_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch).reshape(row, col, ch)
    noisy = image + image * gauss * 0.4
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_lemon_sign(img_path):
    try:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            return

        rows, cols = img.shape[:2]

        pinch_strength = random.uniform(0.0015, 0.0035)
        center_x, center_y = cols / 2, rows / 4

        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        dx, dy = x_coords - center_x, y_coords - center_y
        dist = np.sqrt(dx**2 + dy**2)
        mask = y_coords < (rows / 1.8)
        factor = 1.0 + pinch_strength * dist

        map_x = np.where(mask, center_x + (dx * factor), x_coords).astype(np.float32)
        map_y = y_coords.astype(np.float32)

        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        if random.random() > 0.5:
            distorted = add_ultrasound_noise(distorted)

        if random.random() > 0.5:
            distorted = cv2.GaussianBlur(distorted, (5, 5), 0)

        save_path = os.path.join(OUTPUT_FOLDER, f"lemon_v2_{filename}")
        cv2.imwrite(save_path, distorted)
        print(f"  Created: {filename}")

    except Exception as e:
        print(f"  Error processing {img_path}: {e}")


if __name__ == "__main__":
    images = verify_files()

    if images:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Generating {TARGET_COUNT} synthetic spina bifida samples...")

        while len(images) < TARGET_COUNT:
            images += images

        with ThreadPoolExecutor() as executor:
            executor.map(apply_lemon_sign, images[:TARGET_COUNT])

        print(f"\nDone. Synthetic dataset saved to '{OUTPUT_FOLDER}'")
