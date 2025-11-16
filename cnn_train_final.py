import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

TRAIN_DIR = "spectrogram_dataset"

img_size = 224
batch = 16

train_gen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1/255,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(img_size, img_size),
    batch_size=batch, class_mode="categorical", subset="training"
)

val = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(img_size, img_size),
    batch_size=batch, class_mode="categorical", subset="validation"
)

# -------------------------
#     MobileNetV2 Model
# -------------------------

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base.trainable = False  # freeze backbone

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, out)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training MobileNetV2...")

history = model.fit(train, validation_data=val, epochs=15)

model.save("final_cnn_model.h5")
print("✔ Model saved!")
