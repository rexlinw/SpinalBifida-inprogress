from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model


def build_model(architecture: str, input_shape: tuple[int, int, int], num_classes: int) -> Model:
    arch = architecture.lower().strip()

    if arch == "vgg16":
        base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    elif arch == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    elif arch == "efficientnetb0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            "Use one of: vgg16, resnet50, efficientnetb0"
        )

    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=output)
