import json
import os
import random
import shutil
from collections.abc import Callable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import ProjectConfig, build_run_dir, config_to_dict
from model_factory import build_model


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_preprocessing_setup(architecture: str) -> tuple[float | None, Callable | None]:
    arch = architecture.lower().strip()
    if arch == "vgg16":
        return None, vgg16_preprocess_input
    if arch == "resnet50":
        return None, resnet50_preprocess_input
    if arch == "efficientnetb0":
        # EfficientNetB0 in current Keras includes preprocessing layers.
        # Keep raw [0, 255] input to avoid double normalization.
        return None, None
    return 1.0 / 255, None


def main() -> None:
    cfg = ProjectConfig()
    set_reproducible_seed(cfg.seed)

    rescale, preprocessing_function = get_preprocessing_setup(cfg.architecture)

    run_dir = build_run_dir(cfg.run_root)
    model_checkpoint_path = os.path.join(run_dir, "best_model.keras")
    curves_path = os.path.join(run_dir, "training_curves.png")
    metrics_csv_path = os.path.join(run_dir, "training_metrics.csv")
    eval_json_path = os.path.join(run_dir, "evaluation_summary.json")
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print("\n--- Hardware Check ---")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU Detected: {gpus[0]}" if gpus else "GPU not found. Training on CPU.")

    train_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(cfg.dataset_path, "train"),
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=cfg.seed,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(cfg.dataset_path, "val"),
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_datagen.flow_from_directory(
        os.path.join(cfg.dataset_path, "test"),
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"\nClass indices: {train_gen.class_indices}")

    print(f"\n--- Building {cfg.architecture} Transfer Learning Model ---")
    model = build_model(
        architecture=cfg.architecture,
        input_shape=(cfg.image_size[0], cfg.image_size[1], 3),
        num_classes=2,
    )
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    class_weight = None
    if cfg.use_class_weights:
        train_labels = train_gen.classes
        classes = np.unique(train_labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"Using class weights: {class_weight}")

    print("\n--- Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs,
        callbacks=[
            ModelCheckpoint(model_checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            CSVLogger(metrics_csv_path, append=False),
        ],
        class_weight=class_weight,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"\nTraining curves saved to '{curves_path}'")

    print("\n--- Test Set Evaluation ---")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    report_text = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':15s} Pred Normal  Pred Spina_Bifida")
    print(f"  {'True Normal':15s} {cm[0][0]:11d}  {cm[0][1]:17d}")
    print(f"  {'True Spina_B':15s} {cm[1][0]:11d}  {cm[1][1]:17d}")

    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    try:
        auc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        print(f"\nROC-AUC Score: {auc:.4f}")
    except Exception as e:
        auc = None
        print(f"\nCould not compute ROC-AUC: {e}")

    eval_summary = {
        "class_indices": train_gen.class_indices,
        "samples": {
            "train": int(train_gen.samples),
            "val": int(val_gen.samples),
            "test": int(test_gen.samples),
        },
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "classification_report": report_dict,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Keep compatibility with existing scripts expecting this file in project root.
    shutil.copyfile(model_checkpoint_path, cfg.model_path)

    print(f"\nRun artifacts saved in: {run_dir}")
    print(f"Evaluation summary saved as: {eval_json_path}")
    print(f"Model saved as '{cfg.model_path}'")


if __name__ == "__main__":
    main()
