import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/crop_model.pkl")

# Example soil and weather conditions
input_data = pd.DataFrame([{
    "N": 67,
    "P": 89,
    "K": 45,
    "temperature": 32,
    "humidity": 78,
    "ph": 7,
    "rainfall": 100
}])

# Predict the crop
prediction = model.predict(input_data)

print("🌾 Recommended Crop:", prediction[0])