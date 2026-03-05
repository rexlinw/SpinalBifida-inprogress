import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = 'spina_bifida_model.keras'
IMAGE_PATH = 'test_images/lemon_01.jpg'  
IMG_SIZE = (224, 224)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    # 1. Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. This is the "gradient" of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Mean intensity of the gradient for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- MAIN ---
print("🧠 Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"🖼️  Processing {IMAGE_PATH}...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Error: Image not found.")
    exit()

# Preprocess
original_img = cv2.resize(img, IMG_SIZE)
img_array = original_img.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Get Prediction
preds = model.predict(img_array)
class_idx = np.argmax(preds[0])
confidence = np.max(preds[0])
label = "Spina Bifida" if class_idx == 1 else "Normal" 

print(f"📊 Prediction: {label} ({confidence*100:.2f}%)")

# Generate Heatmap
heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3')

# Display
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title(f"Original: {label}")
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Attention (Grad-CAM)")
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.imshow(heatmap, alpha=0.5, cmap='jet') # Overlay heatmap
plt.axis('off')

plt.tight_layout()
plt.show()