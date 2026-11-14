
import pandas as pd
import numpy as np

# Define dataset column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load NASA FD001 training dataset
train_data = pd.read_csv(
    "data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Original dataset shape:", train_data.shape)

print("\nFirst five rows:")
print(train_data.head())

# Identify constant sensors
constant_sensors = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]

# Remove constant sensors
cleaned_data = train_data.drop(columns=constant_sensors)

print("\nConstant sensors removed:")
print(constant_sensors)

print("\nOriginal dataset shape:", train_data.shape)
print("Cleaned dataset shape:", cleaned_data.shape)
# Check for missing values in cleaned dataset
print("\nMissing values in cleaned dataset:")
print(cleaned_data.isnull().sum())

print("\nTotal missing values:", cleaned_data.isnull().sum().sum())
# Save cleaned dataset
cleaned_data.to_csv(
    "data/processed/cleaned_train_FD001.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File location: data/processed/cleaned_train_FD001.csv")
# Verify saved cleaned dataset
verified_data = pd.read_csv(
    "data/processed/cleaned_train_FD001.csv"
)

print("\nVerification of saved dataset:")
print("Saved dataset shape:", verified_data.shape)
print("First five rows:")
print(verified_data.head())