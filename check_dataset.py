import pandas as pd

# Load the dataset
df = pd.read_csv("data/crop_recommendation.csv")

# Display basic information
print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCrop names:")
print(df["label"].unique())

print("\nNumber of crops:")
print(df["label"].nunique())

import pandas as pd

df = pd.read_csv("data/crop_recommendation.csv")

columns = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

for column in columns:
    print(f"{column}:")
    print("  Minimum:", df[column].min())
    print("  Maximum:", df[column].max())