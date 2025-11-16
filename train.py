import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n🚀 Training Started – Aerospace Sound Classification\n")

DATASET_PATH = r"C:\Users\deepika lalam\Downloads\ESC-50-master (1)\ESC-50-master\audio"
MODEL_PATH = "model.pkl"

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=2.5, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

X = []
y = []

print("📂 Loading dataset...")
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        label = file.split("-")[0]     # numerical label from filename
        features = extract_features(os.path.join(DATASET_PATH, file))

        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("🎧 Total audio samples loaded:", len(X))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Medium-level model → Random Forest
print("\n🌲 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\n✅ Training Completed!")
print("📊 Accuracy:", acc)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
