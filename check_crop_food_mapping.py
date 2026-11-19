import pandas as pd

# Load crop dataset
crop_df = pd.read_csv("data/crop_recommendation.csv")

# Load nutrition dataset
nutrition_df = pd.read_csv("data/nutrition_foods_final.csv")

print("\n==============================")
print("CROP LABELS")
print("==============================")

crop_labels = sorted(
    crop_df["label"].dropna().unique()
)

for crop in crop_labels:
    print(crop)


print("\n==============================")
print("NUTRITION FOOD NAMES")
print("==============================")

for food in nutrition_df["food"].head(100):
    print(food)