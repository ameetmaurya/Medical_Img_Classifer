import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Make sure model directory exists
os.makedirs("model", exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "model/train",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "model/train",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

print("✅ Class indices:", train_data.class_indices)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(128,128,3),
    pooling='avg'
)
output = tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')(base_model.output)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

model.save("model/efficientnet_model.h5")
print("✅ Model saved to model/efficientnet_model.h5")
