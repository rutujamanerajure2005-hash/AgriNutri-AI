import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load dataset
df = pd.read_csv("data/crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Separate input features and target
X = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

y = df["label"]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# 5. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Model training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# 8. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 9. Save the trained model
joblib.dump(model, "models/crop_model.pkl")

print("\nModel saved successfully!")
print("Location: models/crop_model.pkl")