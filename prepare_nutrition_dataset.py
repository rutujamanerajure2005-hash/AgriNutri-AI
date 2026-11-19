import pandas as pd
import re

# --------------------------------------------------
# 1. Load IFCT 2017 dataset
# --------------------------------------------------

df = pd.read_csv("data/nutrition_ifct2017.csv")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Select required columns
# --------------------------------------------------

nutrition_df = df[
    [
        "code",
        "name",
        "water",
        "protcnt",
        "ash",
        "fatce",
        "fibtg",
        "fibins",
        "fibsol",
        "choavldf",
        "enerc"
    ]
].copy()


# --------------------------------------------------
# 3. Rename columns
# --------------------------------------------------

nutrition_df.columns = [
    "food_code",
    "food",
    "moisture_g",
    "protein_g",
    "ash_g",
    "fat_g",
    "fiber_g",
    "insoluble_fiber_g",
    "soluble_fiber_g",
    "carbohydrate_g",
    "energy_kj"
]

# --------------------------------------------------
# 4. Extract numeric values
# --------------------------------------------------

def extract_number(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Handle empty values
    if value == "":
        return None

    # Extract first numeric value
    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if match:
        return float(match.group())

    return None


numeric_columns = [
    "moisture_g",
    "protein_g",
    "ash_g",
    "fat_g",
    "fiber_g",
    "insoluble_fiber_g",
    "soluble_fiber_g",
    "carbohydrate_g",
    "energy_kj"
]

for column in numeric_columns:

    nutrition_df[column] = nutrition_df[column].apply(
        extract_number
    )
    # Convert energy from kJ to kcal
nutrition_df["energy_kcal"] = (
    nutrition_df["energy_kj"] / 4.184
).round(2)

nutrition_df = nutrition_df.drop(
    columns=["energy_kj"]
)


# --------------------------------------------------
# 5. Recover missing fibre values
# --------------------------------------------------

# If total fibre is missing,
# calculate it from insoluble + soluble fibre.

fiber_missing = nutrition_df["fiber_g"].isna()

nutrition_df.loc[fiber_missing, "fiber_g"] = (
    nutrition_df.loc[fiber_missing, "insoluble_fiber_g"]
    + nutrition_df.loc[fiber_missing, "soluble_fiber_g"]
)


# --------------------------------------------------
# 6. Recover missing carbohydrate values
# --------------------------------------------------

# If available carbohydrate is missing,
# estimate carbohydrate by difference.

carb_missing = nutrition_df["carbohydrate_g"].isna()

nutrition_df.loc[carb_missing, "carbohydrate_g"] = (
    100
    - nutrition_df.loc[carb_missing, "moisture_g"]
    - nutrition_df.loc[carb_missing, "protein_g"]
    - nutrition_df.loc[carb_missing, "fat_g"]
    - nutrition_df.loc[carb_missing, "ash_g"]
    - nutrition_df.loc[carb_missing, "fiber_g"]
)


# --------------------------------------------------
# 7. Remove unnecessary helper columns
# --------------------------------------------------

nutrition_df = nutrition_df[
    [
        "food_code",
        "food",
        "moisture_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "carbohydrate_g",
        "energy_kcal"
    ]
]


# --------------------------------------------------
# 8. Remove records without food names
# --------------------------------------------------

nutrition_df = nutrition_df.dropna(
    subset=["food"]
)


# --------------------------------------------------
# 9. Remove duplicate food codes
# --------------------------------------------------

nutrition_df = nutrition_df.drop_duplicates(
    subset=["food_code"]
)


# --------------------------------------------------
# 10. Display missing values
# --------------------------------------------------

print("\nMissing values after processing:")

print(
    nutrition_df.isnull().sum()
)


# --------------------------------------------------
# 11. Save final nutrition dataset
# --------------------------------------------------

nutrition_df.to_csv(
    "data/nutrition_foods_final.csv",
    index=False
)


# --------------------------------------------------
# 12. Final information
# --------------------------------------------------

print("\nFinal dataset shape:")

print(nutrition_df.shape)


print("\nFirst 10 foods:")

print(nutrition_df.head(10))


print("\nFinal nutrition dataset saved successfully!")
