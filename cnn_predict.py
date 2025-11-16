import numpy as np
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "cnn_sound_model.h5"
DATASET_PATH = r"C:\Users\deepika lalam\Downloads\ESC-50-master (1)\ESC-50-master\audio"

model = tf.keras.models.load_model(MODEL_PATH)

# Rebuild label encoder
labels = [f.split("-")[0] for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]
le = LabelEncoder()
le.fit(labels)

def extract_mel(audio_file):
    y, sr = librosa.load(audio_file, duration=3)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    return mel_db

while True:
    path = input("Enter audio file path: ")

    mel = extract_mel(path)
    mel = mel.reshape(1, 128, 128, 1)

    pred = model.predict(mel)
    index = np.argmax(pred)
    label = le.inverse_transform([index])[0]

    print("\n🔍 Predicted Label:", label, "\n")
