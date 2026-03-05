import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import random

INPUT_FOLDER = 'dataset/Normal' 
OUTPUT_FOLDER = 'dataset/Spina_Bifida'
TARGET_COUNT = 1200  

def verify_files():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ ERROR: Input folder '{INPUT_FOLDER}' not found.")
        return []

    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    print(f"✅ Found {len(files)} source images in '{INPUT_FOLDER}'")
    return files

def add_ultrasound_noise(image):
    """Adds random speckle noise to mimic real medical scans."""
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)        
    noisy = image + image * gauss * 0.4
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_strong_lemon(img_path):
    try:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: return

        rows, cols = img.shape[:2]
        
        # --- 1. THE PINCH (Now much stronger) ---
        # Random strength between 0.0015 (strong) and 0.0035 (severe)
        pinch_strength = random.uniform(0.0015, 0.0035) 
        
        # Center the pinch on the "Frontal Bones" (Top 25% of image)
        center_x, center_y = cols / 2, rows / 4
        
        # Create mapping grid
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        dx, dy = x_coords - center_x, y_coords - center_y
        
        # Calculate distance
        dist = np.sqrt(dx**2 + dy**2)
        
        # Apply pinch only to the top half
        mask = y_coords < (rows / 1.8) 
        
        # The warp factor: Pull pixels INWARD
        factor = 1.0 + pinch_strength * dist
        
        map_x = np.where(mask, center_x + (dx * factor), x_coords).astype(np.float32)
        map_y = y_coords.astype(np.float32)

        # Remap the image
        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        
        # --- 2. THE GRIT (Noise & Blur) ---
        # 50% chance to add noise (Simulate bad sensor)
        if random.random() > 0.5:
            distorted = add_ultrasound_noise(distorted)
            
        # 50% chance to blur (Simulate motion)
        if random.random() > 0.5:
            distorted = cv2.GaussianBlur(distorted, (5, 5), 0)

        # Save
        save_path = os.path.join(OUTPUT_FOLDER, f"lemon_v2_{filename}")
        cv2.imwrite(save_path, distorted)
        print(f"🍋 Created Strong Lemon: {filename}")
        
    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    images = verify_files()
    
    if images:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            
        print(f"🚀 Generating {TARGET_COUNT} aggressive Spina Bifida samples...")
        
        # Use parallel processing for speed
        with ThreadPoolExecutor() as executor:
            # We use the list multiple times to reach target count if needed
            while len(images) < TARGET_COUNT:
                images += images
            
            executor.map(apply_strong_lemon, images[:TARGET_COUNT])
            
        print("\n✅ DONE! New dataset created in 'dataset/Spina_Bifida'")
        print("👉 NOW: Run 'python train_model.py' to retrain with this harder data.")