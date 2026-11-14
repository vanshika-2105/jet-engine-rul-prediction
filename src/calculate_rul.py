import pandas as pd

# Load the cleaned dataset
train_data = pd.read_csv(
    "data/processed/cleaned_train_FD001.csv"
)

print("Cleaned dataset loaded successfully!")

print("Dataset shape:", train_data.shape)

print("\nFirst five rows:")
print(train_data.head())
# Find the maximum cycle for each engine
max_cycles = train_data.groupby("engine_id")["cycle"].max()

print("\nMaximum cycle of each engine:")
print(max_cycles.head(10))
# Calculate RUL for every engine cycle
train_data["RUL"] = train_data["engine_id"].map(max_cycles) - train_data["cycle"]

print("\nRUL calculated successfully!")

print("\nFirst 10 rows with RUL:")
print(train_data[["engine_id", "cycle", "RUL"]].head(10))
# Verify RUL calculation for all engines
last_cycle_check = train_data.groupby("engine_id").tail(1)

print("\nRUL verification for each engine:")
print(last_cycle_check[["engine_id", "cycle", "RUL"]].head(10))

print("\nMinimum RUL:", train_data["RUL"].min())
print("Maximum RUL:", train_data["RUL"].max())
# Create normalized cycle feature
train_data["normalized_cycle"] = (
    train_data["cycle"] / train_data["engine_id"].map(max_cycles)
)

print("\nNormalized cycle feature created!")

print(
    train_data[
        ["engine_id", "cycle", "normalized_cycle", "RUL"]
    ].head(10)
)
# Save final dataset with RUL and engineered features
train_data.to_csv(
    "data/processed/train_FD001_with_RUL.csv",
    index=False
)

print("\nFinal dataset saved successfully!")
print("File: data/processed/train_FD001_with_RUL.csv")
print("Final dataset shape:", train_data.shape)
# Save final dataset with RUL and engineered features
train_data.to_csv(
    "data/processed/train_FD001_with_RUL.csv",
    index=False
)

print("\nFinal dataset saved successfully!")
print("File: data/processed/train_FD001_with_RUL.csv")
print("Final dataset shape:", train_data.shape)