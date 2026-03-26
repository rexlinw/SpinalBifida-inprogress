import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = 'spina_bifida_model.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Spina_Bifida']


def _clahe_normalise(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel to normalise brightness before inference."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def explain_image(image_path, model, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    original_img = cv2.resize(img, IMG_SIZE)
    original_rgb = _clahe_normalise(original_img)
    img_array = original_rgb.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    label = CLASS_NAMES[class_idx]

    print(f"Prediction: {label} ({confidence * 100:.2f}%)")
    print(f"  Scores: Normal={preds[0][0]:.4f}, Spina_Bifida={preds[0][1]:.4f}")

    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay_rgb = cv2.addWeighted(original_rgb, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title('Original (CLAHE)')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f'Overlay: {label} ({confidence * 100:.1f}%)')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python explain_model.py <image_path> [save_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    explain_image(image_path, model, save_path)
