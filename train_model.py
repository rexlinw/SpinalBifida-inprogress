import os
# Suppress unnecessary logs but keep errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- CONFIG ---
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# 1. GPU VERIFICATION
print("\n--- System Hardware Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Apple Metal GPU Detected: {gpus[0]}")
else:
    print("⚠️ GPU not found. Training will use CPU.")

# 2. DATA GENERATORS
# Using a 20% validation split for the HC18 data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

# 3. TRANSFER LEARNING MODEL (VGG16)
print("\n--- Initializing Deep Learning Model ---")
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers: 
    layer.trainable = False  # Keep the pre-trained weights frozen

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x) # [Normal, Spina_Bifida]

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. TRAINING WITH CALLBACKS
print("\n--- Starting Training ---")
checkpoint = ModelCheckpoint('spina_bifida_model.keras', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=EPOCHS, 
    callbacks=[checkpoint, early_stop]
)

print("\n✅ SUCCESS: Final model saved as 'spina_bifida_model.keras'")