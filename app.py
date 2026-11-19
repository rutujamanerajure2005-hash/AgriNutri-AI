from flask import Flask, render_template, request, session
import math
import os
from pathlib import Path

import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-key-change-before-deployment",
)

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
NUTRITION_PATH = BASE_DIR / "data" / "nutrition_foods_final.csv"
MAPPING_PATH = BASE_DIR / "data" / "crop_food_mapping.csv"

# =====================================================
# LOAD CROP MODEL
# =====================================================

model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD NUTRITION DATA
# =====================================================

nutrition_df = pd.read_csv(NUTRITION_PATH)

if "food" not in nutrition_df.columns:
    raise ValueError(
        "The nutrition dataset must contain a 'food' column."
    )

nutrition_df["food"] = (
    nutrition_df["food"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Keep missing nutrient values as NaN.
nutrition_columns = [
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
    "fiber_g",
]

for column in nutrition_columns:
    if column in nutrition_df.columns:
        nutrition_df[column] = pd.to_numeric(
            nutrition_df[column],
            errors="coerce",
        )
    else:
        # Keep the application running if an optional nutrient
        # column is absent, while treating that nutrient as unknown.
        nutrition_df[column] = float("nan")

# =====================================================
# LOAD CROP-FOOD MAPPING
# =====================================================

mapping_df = pd.read_csv(MAPPING_PATH)

if not {"crop", "food"}.issubset(mapping_df.columns):
    raise ValueError(
        "The crop-food mapping must contain 'crop' and 'food' columns."
    )

mapping_df["crop"] = (
    mapping_df["crop"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

mapping_df["food"] = (
    mapping_df["food"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# =====================================================
# SHARED HELPERS
# =====================================================

def render_nutrition_page(
    recommendation=None,
    recommended_foods=None,
    calories=None,
    age_group=None,
    error_message=None,
    status=200,
):
    """Render the nutrition page with a consistent set of template values."""
    response = render_template(
        "nutrition.html",
        recommendation=recommendation,
        recommended_foods=recommended_foods or [],
        calories=calories,
        age_group=age_group,
        error_message=error_message,
    )
    return response, status


def match_food_for_crop(food_name):
    """Find the closest available food profile for a mapped crop food."""
    if not food_name or not str(food_name).strip():
        return None

    food_name = str(food_name).strip()

    # 1. Exact case-insensitive match.
    exact_match = nutrition_df[
        nutrition_df["food"].str.casefold() == food_name.casefold()
    ]

    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    # 2. Match while ignoring text in parentheses.
    simple_food = food_name.split("(")[0].strip().casefold()
    if simple_food:
        no_parentheses = nutrition_df[
            nutrition_df["food"]
            .str.split("(")
            .str[0]
            .str.strip()
            .str.casefold()
            == simple_food
        ]
        if not no_parentheses.empty:
            return no_parentheses.iloc[0].to_dict()

    # 3. Try the first comma-separated food phrase.
    food_keyword = food_name.split(",")[0].strip().casefold()
    if food_keyword:
        keyword_match = nutrition_df[
            nutrition_df["food"]
            .str.casefold()
            .str.contains(food_keyword, regex=False, na=False)
        ]
        if not keyword_match.empty:
            return keyword_match.iloc[0].to_dict()

    return None


def calculate_estimated_calories(age, weight, height, gender, activity, goal=None):
    """
    Estimate daily energy needs using a common BMR equation and activity factor.
    This is a general estimate, not medical or dietetic advice.
    """
    if gender == "male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    activity_multipliers = {
        "low": 1.2,
        "moderate": 1.55,
        "high": 1.725,
    }

    estimated_calories = bmr * activity_multipliers[activity]

    # Goal adjustments are applied only to adults in the original flow.
    if goal == "weight_loss":
        estimated_calories -= 300
    elif goal == "weight_gain":
        estimated_calories += 300

    # Avoid returning an implausible non-positive estimate.
    return max(0, round(estimated_calories))


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# CROP RECOMMENDATION
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    # Read and validate crop input values.
    field_names = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    try:
        values = {
            name: float(request.form[name])
            for name in field_names
        }
    except (ValueError, TypeError, KeyError):
        return render_template(
            "index.html",
            error_message="Please enter valid numeric values in all crop fields.",
            form_values=request.form,
        ), 400

    # Reject NaN and infinity values.
    if not all(math.isfinite(value) for value in values.values()):
        return render_template(
            "index.html",
            error_message="Please enter finite numeric values in all crop fields.",
            form_values=request.form,
        ), 400

    # Application input limits chosen for this project.
    # These are not guaranteed to be the exact training-data limits.
    limits = {
        "N": (0, 140),
        "P": (0, 145),
        "K": (0, 205),
        "temperature": (0, 50),
        "humidity": (0, 100),
        "ph": (0, 14),
        "rainfall": (0, 300),
    }

    for name, value in values.items():
        minimum, maximum = limits[name]
        if not minimum <= value <= maximum:
            return render_template(
                "index.html",
                error_message=(
                    f"{name} must be between {minimum} and {maximum}."
                ),
                form_values=request.form,
            ), 400

    # Prepare model input in the expected feature order.
    input_data = pd.DataFrame([values], columns=field_names)

    try:
        prediction = model.predict(input_data)
        crop = str(prediction[0]).strip()

        # Some classifiers expose predict_proba; handle models that do not.
        confidence = None
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probabilities = model.predict_proba(input_data)[0]
            crop_classes = [str(label).strip() for label in model.classes_]

            if crop in crop_classes:
                predicted_index = crop_classes.index(crop)
                confidence = round(
                    float(probabilities[predicted_index]) * 100,
                    2,
                )
    except Exception:
        app.logger.exception("Crop prediction failed.")
        return render_template(
            "index.html",
            error_message=(
                "Sorry, the crop recommendation could not be generated. "
                "Please check your inputs and try again."
            ),
            form_values=request.form,
        ), 500

    # Save the latest crop result for the Insights dashboard.
       # Calculate the model's overall feature importance.
    feature_importance = []

    if hasattr(model, "feature_importances_"):
        feature_names = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]

        importance_values = model.feature_importances_

        feature_importance = [
            {
                "feature": name,
                "importance": round(float(score) * 100, 2),
            }
            for name, score in zip(
                feature_names,
                importance_values,
            )
        ]

        feature_importance.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

    # Save the latest crop result for the Insights dashboard.
    session["crop_insight"] = {
        "crop": crop,
        "confidence": confidence,
        "inputs": values,
        "feature_importance": feature_importance[:3],
    }

    # =================================================
    # CROP-TO-FOOD NUTRITION BRIDGE
    # =================================================

    food_name = None
    nutrition_info = None
    crop_key = crop.lower()

    crop_mapping = mapping_df[mapping_df["crop"] == crop_key]

    if not crop_mapping.empty:
        mapped_food = crop_mapping.iloc[0]["food"]

        if pd.notna(mapped_food) and str(mapped_food).strip():
            food_name = str(mapped_food).strip()
            nutrition_info = match_food_for_crop(food_name)

    return render_template(
        "index.html",
        prediction=crop,
        confidence=confidence,
        food_name=food_name,
        nutrition_info=nutrition_info,
        form_values=request.form,
    )


# =====================================================
# NUTRITION RECOMMENDATION
# =====================================================

@app.route("/nutrition", methods=["GET", "POST"])
def nutrition():
    recommendation = None
    recommended_foods = []
    calories = None
    age_group = None

    if request.method == "GET":
        return render_nutrition_page()

    # 1. Read and validate submitted user information.
    try:
        age_raw = request.form["age"].strip()
        weight_raw = request.form["weight"].strip()
        height_raw = request.form["height"].strip()

        age = int(age_raw)
        weight = float(weight_raw)
        height = float(height_raw)

        gender = request.form["gender"].strip().lower()
        activity = request.form["activity"].strip().lower()
        goal = request.form["goal"].strip().lower()

    except (ValueError, TypeError, KeyError, AttributeError):
        return render_nutrition_page(
            error_message="Please enter valid values in all required fields.",
            status=400,
        )

    # Reject non-finite numbers such as NaN and infinity.
    if not all(math.isfinite(value) for value in [age, weight, height]):
        return render_nutrition_page(
            error_message="Please enter finite numeric values for age, weight, and height.",
            status=400,
        )

    # Validate input ranges. These are application safeguards, not medical cutoffs.
    if not 1 <= age <= 120:
        error_message = "Age must be between 1 and 120 years."
    elif not 1 <= weight <= 500:
        error_message = "Weight must be between 1 and 500 kg."
    elif not 30 <= height <= 250:
        error_message = "Height must be between 30 and 250 cm."
    elif gender not in {"male", "female"}:
        error_message = "Please select a valid gender."
    elif activity not in {"low", "moderate", "high"}:
        error_message = "Please select a valid activity level."
    elif goal not in {"weight_loss", "maintenance", "weight_gain"}:
        error_message = "Please select a valid nutrition goal."
    else:
        error_message = None

    if error_message:
        return render_nutrition_page(
            error_message=error_message,
            status=400,
        )

    # 2. Identify age group.
    if age < 13:
        age_group = "Child"
    elif age < 18:
        age_group = "Teenager"
    elif age < 60:
        age_group = "Adult"
    else:
        age_group = "Senior Citizen"

    # Exclude empty food names from recommendations.
    available_foods = nutrition_df[
        nutrition_df["food"] != ""
    ].copy()

    # =================================================
    # 3. CHILDREN: UNDER 13 YEARS
    # =================================================

    if age_group == "Child":
        recommendation = (
            "Child nutrition guidance: growing children need a variety "
            "of nutritious foods. The foods shown are general examples "
            "from the nutrition dataset, not a personalized diet. For "
            "individual guidance, consult a qualified healthcare "
            "professional or dietitian."
        )

        selected = available_foods[
            available_foods["protein_g"].notna()
        ].sort_values("food", ascending=True)

        # Do not apply an adult calorie formula to children.
        calories = None

    # =================================================
    # 4. TEENAGERS: 13-17 YEARS
    # =================================================

    elif age_group == "Teenager":
        recommendation = (
            "Teenage nutrition guidance: adolescence is an important "
            "period of growth. A varied diet can include different food "
            "groups and suitable protein sources. The foods shown are "
            "general dataset examples, not a personalized diet. Consult "
            "a qualified healthcare professional for individual guidance."
        )

        selected = available_foods[
            available_foods["protein_g"].notna()
        ].sort_values(
            ["protein_g", "food"],
            ascending=[False, True],
        )

        # Do not apply an adult calorie formula to teenagers.
        calories = None

    # =================================================
    # 5. ADULTS: 18-59 YEARS
    # =================================================

    elif age_group == "Adult":
        calories = calculate_estimated_calories(
            age, weight, height, gender, activity, goal
        )

        if goal == "weight_loss":
            selected = available_foods[
                (available_foods["energy_kcal"] <= 200)
                & available_foods["energy_kcal"].notna()
                & available_foods["fiber_g"].notna()
            ].sort_values(
                ["fiber_g", "food"],
                ascending=[False, True],
            )

            recommendation = (
                "Explore lower-energy foods with recorded fiber values. "
                "Maintain a balanced diet with adequate protein and other "
                "nutrients. The calorie figure is a rough estimate, and "
                "these food examples are not a prescribed meal plan."
            )

        elif goal == "weight_gain":
            selected = available_foods[
                (available_foods["energy_kcal"] >= 200)
                & available_foods["energy_kcal"].notna()
                & available_foods["protein_g"].notna()
            ].sort_values(
                ["protein_g", "food"],
                ascending=[False, True],
            )

            recommendation = (
                "Explore energy-dense foods with recorded protein values. "
                "Include a variety of foods to support a balanced diet. "
                "The calorie figure is an estimate, not a clinical target."
            )

        else:
            selected = available_foods[
                available_foods["protein_g"].notna()
            ].sort_values(
                ["protein_g", "food"],
                ascending=[False, True],
            )

            recommendation = (
                "Aim for a varied diet containing whole grains, vegetables, "
                "fruits, and suitable protein-rich foods. The food profiles "
                "provide general information, not an individualized diet."
            )

    # =================================================
    # 6. SENIOR CITIZENS: 60 YEARS AND ABOVE
    # =================================================

    else:
        recommendation = (
            "Senior nutrition guidance: nutritional needs can vary with "
            "health, activity, appetite, and medications. The foods shown "
            "are general examples from the dataset, not a personalized "
            "diet. Seek professional advice for individual needs. The "
            "calorie estimate is approximate and is not a clinical target."
        )

        selected = available_foods[
            available_foods["protein_g"].notna()
            & available_foods["energy_kcal"].notna()
        ].sort_values(
            ["energy_kcal", "protein_g", "food"],
            ascending=[True, False, True],
        )

        # This is only an approximate adult-style estimate, not a
        # senior-specific clinical calorie calculation. No automatic
        # weight-loss or weight-gain adjustment is applied to seniors.
        calories = calculate_estimated_calories(
            age, weight, height, gender, activity
        )

    # =================================================
    # 7. SELECT UP TO FIVE FOOD PROFILES
    # =================================================

    recommended_foods = selected.head(5).to_dict("records")

    # Save a compact, session-specific summary for the Insights dashboard.
    session["nutrition_insight"] = {
        "age_group": age_group,
        "goal": goal,
        "calories": calories,
        "recommendation": recommendation,
        "foods": [
            {
                "food": str(food.get("food", "Unknown")),
                "energy_kcal": (
                    float(food["energy_kcal"])
                    if pd.notna(food.get("energy_kcal"))
                    else None
                ),
                "protein_g": (
                    float(food["protein_g"])
                    if pd.notna(food.get("protein_g"))
                    else None
                ),
                "carbohydrate_g": (
                    float(food["carbohydrate_g"])
                    if pd.notna(food.get("carbohydrate_g"))
                    else None
                ),
                "fat_g": (
                    float(food["fat_g"])
                    if pd.notna(food.get("fat_g"))
                    else None
                ),
                "fiber_g": (
                    float(food["fiber_g"])
                    if pd.notna(food.get("fiber_g"))
                    else None
                ),
            }
            for food in recommended_foods
        ],
    }

    if not recommended_foods:
        recommendation += (
            " No food profiles matched these selection criteria in the "
            "available dataset. Please review the dataset or consult a "
            "qualified professional for individual advice."
        )

    # =================================================
    # 8. DISPLAY THE NUTRITION PAGE
    # =================================================

    return render_nutrition_page(
        recommendation=recommendation,
        recommended_foods=recommended_foods,
        calories=calories,
        age_group=age_group,
    )


# =====================================================
# AI INSIGHTS DASHBOARD
# =====================================================

@app.route("/insights")
def insights():
    """Show the latest crop and nutrition results from this session."""
    return render_template(
        "insights.html",
        crop_insight=session.get("crop_insight"),
        nutrition_insight=session.get("nutrition_insight"),
    )


# =====================================================
# RUN THE FLASK APPLICATION
# =====================================================

if __name__ == "__main__":
    # Debug mode is useful locally. Disable it when deploying publicly.
    app.run(debug=True)
