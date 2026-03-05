import tensorflow as tf
import numpy as np
import cv2
import os

# CONFIGURATION
MODEL_PATH = 'spina_bifida_model.keras'
TEST_FOLDER = 'test_images'
IMG_SIZE = (224, 224)

# Load the trained brain
print(f"🧠 Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Get the class names (Check your training output to be sure of the order!)
# Usually it is alphabetical: ['Normal', 'Spina_Bifida']
CLASS_NAMES = ['Normal', 'Spina_Bifida']

def predict_image(image_path):
    # 1. Read & Preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read {image_path}")
        return

    # Resize to match training input
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Normalize (0-255 -> 0-1) just like in training
    img_array = img_resized.astype('float32') / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 2. Predict
    predictions = model.predict(img_batch, verbose=0)
    score = predictions[0]
    
    # 3. Interpret
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"🖼️  Image: {os.path.basename(image_path)}")
    print(f"   prediction: {predicted_class} ({confidence:.2f}%)")
    print(f"   Raw Scores: Normal: {score[0]:.4f} | Spina Bifida: {score[1]:.4f}")
    print("-" * 30)

# Run on all images in folder
print("\n🔍 STARTING DIAGNOSIS...\n" + "="*30)
for file in os.listdir(TEST_FOLDER):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        predict_image(os.path.join(TEST_FOLDER, file))