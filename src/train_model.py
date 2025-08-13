import pandas as pd
import numpy as np
from scipy.io import loadmat
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load all .mat files (update the path to your dataset folder)
files = glob.glob("../data/*.mat")

data_frames = []
for file in files:
    mat_data = loadmat(file)
    min_len = min(len(mat_data['ax']), len(mat_data['ay']), len(mat_data['az']))
    df = pd.DataFrame({
        'ax': mat_data['ax'].flatten()[:min_len],
        'ay': mat_data['ay'].flatten()[:min_len],
        'az': mat_data['az'].flatten()[:min_len],
        'dpitch': mat_data['dpitch'].flatten()[:min_len],
        'droll': mat_data['droll'].flatten()[:min_len],
        'dyaw': mat_data['dyaw'].flatten()[:min_len],
        'heart': mat_data['heart'].flatten()[:min_len],
        'w': mat_data['w'].flatten()[:min_len],
        'x': mat_data['x'].flatten()[:min_len],
        'y': mat_data['y'].flatten()[:min_len],
        'z': mat_data['z'].flatten()[:min_len],
        'label': 1 if "fall" in file.lower() else 0
    })
    data_frames.append(df)

final_df = pd.concat(data_frames, ignore_index=True)
print("Final dataset shape:", final_df.shape)

# Train-Test Split
X = final_df.drop(columns=['label'])
y = final_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/fall_model.pkl")
print("Model saved to models/fall_model.pkl")

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
