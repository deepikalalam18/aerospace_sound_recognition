import os
import librosa
import numpy as np
import pandas as pd

def extract_mfcc(file_path, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, duration=2.5, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error:", e)
        return None


def load_dataset(audio_path, csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for index, row in df.iterrows():
        file_path = os.path.join(audio_path, row["filename"])
        label = row["category"]  

        features = extract_mfcc(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)
