import pandas as pd

# Load final nutrition dataset
df = pd.read_csv("data/nutrition_foods_final.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of unique foods:")
print(df["food"].nunique())

print("\nFirst 10 foods:")
print(df.head(10))

print("\nFoods with missing fiber:")
print(
    df[df["fiber_g"].isna()][
        ["food_code", "food"]
    ].head(10)
)

print("\nFoods with missing carbohydrate:")
print(
    df[df["carbohydrate_g"].isna()][
        ["food_code", "food"]
    ].head(10)
)