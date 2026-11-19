import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the crop dataset
data = pd.read_csv("data/crop_recommendation.csv")

# 2. Separate input features and target
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Define candidate models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Support Vector Machine": SVC(),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    )
}

# 5. Use the same cross-validation splits for each model
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# 6. Train and evaluate each candidate
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "Model": name,
        "Test Accuracy (%)": round(
            test_accuracy * 100, 2
        ),
        "CV Mean (%)": round(
            cv_scores.mean() * 100, 2
        ),
        "CV Std Dev (%)": round(
            cv_scores.std() * 100, 2
        )
    })

    print(f"\n{'=' * 45}")
    print(f"Model: {name}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Cross-validation scores: {cv_scores}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# 7. Display a comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Mean (%)",
    ascending=False
)

print("\n\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 8. Save the comparison results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to model_comparison_results.csv")