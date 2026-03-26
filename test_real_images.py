import sys
import os
import tensorflow as tf
import numpy as np
import cv2

from config import ProjectConfig

CFG = ProjectConfig()
MODEL_PATH = CFG.model_path
IMG_SIZE = CFG.image_size
CLASS_NAMES = ['Normal', 'Spina_Bifida']


def predict_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not read {image_path}")
        return

    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    score = predictions[0]
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence_raw = float(np.max(score))
    confidence_pct = 100 * confidence_raw

    if confidence_raw < CFG.prediction_uncertain_threshold:
        decision = f"UNCERTAIN (< {CFG.prediction_uncertain_threshold:.2f})"
    else:
        decision = predicted_class

    print(f"  {os.path.basename(image_path):40s} -> {decision} ({confidence_pct:.2f}%)"
          f"  [Normal: {score[0]:.4f} | Spina_Bifida: {score[1]:.4f}]")


def collect_images(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for cls in sorted(os.listdir(arg)):
                cls_dir = os.path.join(arg, cls)
                if os.path.isdir(cls_dir):
                    for f in sorted(os.listdir(cls_dir)):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            paths.append(os.path.join(cls_dir, f))
                elif cls.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(arg, cls))
        elif os.path.isfile(arg):
            paths.append(arg)
    return paths


if __name__ == '__main__':
    if len(sys.argv) < 2:
        target = 'dataset_prepared/test'
        print(f"No arguments provided. Running on: {target}\n")
        paths = collect_images([target])
    else:
        paths = collect_images(sys.argv[1:])

    if not paths:
        print("No images found.")
        sys.exit(1)

    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Running inference on {len(paths)} images...\n")

    for path in paths:
        predict_image(path, model)
