import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_DATASET = r"C:\Users\deepika lalam\Downloads\ESC-50-master (1)\ESC-50-master\audio"
META_FILE = r"C:\Users\deepika lalam\Downloads\ESC-50-master (1)\ESC-50-master\meta\esc50.csv"
OUTPUT_FOLDER = "spectrograms"

import pandas as pd
df = pd.read_csv(META_FILE)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("🎨 Creating spectrograms...")

for index, row in df.iterrows():
    file_name = row['filename']
    class_name = row['category']

    class_folder = os.path.join(OUTPUT_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)

    file_path = os.path.join(AUDIO_DATASET, file_name)

    try:
        y, sr = librosa.load(file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2, 2))
        librosa.display.specshow(S_DB, sr=sr)
        plt.axis('off')

        output_path = os.path.join(class_folder, file_name.replace('.wav', '.png'))
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print("Saved:", output_path)

    except Exception as e:
        print("Error processing", file_name, "=>", e)

print("✔ All spectrograms created successfully!")
