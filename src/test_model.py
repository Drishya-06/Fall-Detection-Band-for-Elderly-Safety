import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/fall_model.pkl")

# Example test data (replace with real sensor input)
test_data = pd.DataFrame([{
    'ax': 0.12, 'ay': -0.98, 'az': 9.81,
    'dpitch': 0.01, 'droll': 0.02, 'dyaw': 0.03,
    'heart': 75,
    'w': 0.1, 'x': 0.2, 'y': 0.3, 'z': 0.4
}])

# Predict
prediction = model.predict(test_data)[0]
if prediction == 1:
    print("Fall detected!")
else:
    print("No fall detected.")
