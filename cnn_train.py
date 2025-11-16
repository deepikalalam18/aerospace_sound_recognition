import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

DATASET_PATH = r"C:\Users\deepika lalam\Downloads\ESC-50-master (1)\ESC-50-master\audio"

# Convert audio → Mel Spectrogram → CNN input
def extract_mel(audio_file):
    y, sr = librosa.load(audio_file, duration=3)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

X = []
Y = []

print("📂 Loading dataset...")

for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        path = os.path.join(DATASET_PATH, file)

        mel = extract_mel(path)
        mel = np.resize(mel, (128, 128))   # Resize for CNN

        X.append(mel)
        Y.append(file.split("-")[0])  # ESC-50 label

X = np.array(X)
Y = np.array(Y)

# Reshape for CNN: (samples, height, width, channels)
X = X.reshape(len(X), 128, 128, 1)

# Encode labels
label_enc = LabelEncoder()
Y = label_enc.fit_transform(Y)
Y = to_categorical(Y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("🚀 Training CNN Model...")

# CNN MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(Y.shape[1], activation="softmax"),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Accuracy: {acc * 100:.2f}%")

model.save("cnn_sound_model.h5")
print("💾 Model saved as cnn_sound_model.h5")
