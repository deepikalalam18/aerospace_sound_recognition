import librosa
import numpy as np
import joblib

MODEL_PATH = "sound_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=2.5, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_sound(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

# Run prediction
if __name__ == "__main__":
    file_path = input("Enter audio file path (.wav): ")
    result = predict_sound(file_path)
    print("\n🔍 Predicted Sound Label:", result)

