import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Libraries imported successfully!")
# Load the processed dataset
train_data = pd.read_csv("data/processed/train_FD001_with_RUL.csv")

# Display dataset information
print("\nDataset loaded successfully!")
print("Dataset shape:", train_data.shape)

print("\nFirst 5 rows:")
print(train_data.head())
# Separate features and target

feature_columns = [
    col for col in train_data.columns
    if col not in ["engine_id", "RUL", "normalized_cycle"]
]

X = train_data[feature_columns]
y = train_data["RUL"]

print("\nFeatures and target separated successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature columns:", feature_columns)
# Separate features and target

feature_columns = [
    col for col in train_data.columns
    if col not in ["engine_id", "RUL", "normalized_cycle"]
]
# Split data into training and testing sets

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=train_data["engine_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nData splitting completed!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
# Train Linear Regression model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
# Make predictions on test data

y_pred = model.predict(X_test)

print("\nRUL predictions generated successfully!")

print("\nFirst 10 actual RUL values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted RUL values:")
print(y_pred[:10])
# Evaluate Linear Regression model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)