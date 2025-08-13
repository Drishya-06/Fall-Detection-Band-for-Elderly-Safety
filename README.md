# Fall-Detection-Band-for-Elderly-Safety
A wearable device with sensors (IMU + Heart Rate) and machine learning to detect falls and alert emergency contacts in real-time.




Features
Detects falls using accelerometer, gyroscope, and heart rate data.

Sends alerts via GSM/Wi-Fi when a fall is detected.

Can run in real-time on ESP32 or Raspberry Pi.

Achieved 84% accuracy on real-world dataset.





Dataset
Name: HR_IMU_falldetection_dataset

Source: https://github.com/nhoyh/HR_IMU_falldetection_dataset

Contains:

Fall data: fall1, fall2, ...

Non-fall data: walk, chair, etc.

Features: ax, ay, az, dpitch, droll, dyaw, heart, time, w, x, y, z






Model
Algorithm: Random Forest Classifier

Accuracy: 84.5% on test set

How to load:

python:
import joblib
model = joblib.load("models/fall_model.pkl")


Installation:
git clone https://github.com/yourusername/fall-detection-band.git
cd fall-detection-band
pip install -r requirements.txt


Usage
1. Train the model

bash
Copy
Edit
python src/train_model.py
2. Test the model

bash
Copy
Edit
python src/test_model.py
3. Run real-time detection (if hardware connected)

bash
Copy
Edit
python src/realtime_detection.py



Hardware Setup
Microcontroller: ESP32

Sensors:

MPU6050 (Accelerometer + Gyroscope)

MAX30102 (Heart Rate)

Wiring Diagram: See hardware/wiring_diagram.png



3. requirements.txt
numpy
pandas
scikit-learn
scipy
joblib
