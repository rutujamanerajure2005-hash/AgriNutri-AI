
import pandas as pd
import joblib


from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. Load the existing dataset
df = pd.read_csv("data/crop_recommendation.csv")

X = df[
    [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ]
]
y = df["label"]

# 2. Create the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Load the saved model (do not retrain it)
model = joblib.load("models/crop_model.pkl")

# 4. Evaluate test-set predictions
y_pred = model.predict(X_test)

print("Test Accuracy:", round(
    accuracy_score(y_test, y_pred) * 100, 2
), "%")

# 5. Display the confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(14, 14))
display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)
plt.title("Crop Recommendation Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 6. Perform 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\n5-Fold Cross-Validation Scores:")
print(scores)

print("Mean CV Accuracy:",
      round(scores.mean() * 100, 2), "%")

print("Standard Deviation:",
      round(scores.std() * 100, 2), "%")
