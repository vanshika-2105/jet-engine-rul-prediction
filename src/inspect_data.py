
import pandas as pd

# Column names in the NASA C-MAPSS dataset
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
train_data = pd.read_csv(
    "data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Display basic information
print("Dataset loaded successfully!")
print("Shape:", train_data.shape)

print("\nFirst 5 rows:")
print(train_data.head())

print("\nDataset information:")
print(train_data.info())
# Count unique engines
total_engines = train_data["engine_id"].nunique()

print("\nTotal number of engines:", total_engines)

# Find cycles recorded for each engine
print("\nFirst 10 engines and their cycle counts:")
print(train_data.groupby("engine_id")["cycle"].max().head(10))
# Check missing values in each column
print("\nMissing values in each column:")
print(train_data.isnull().sum())