import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Libraries imported successfully!")
# Load processed dataset

train_data = pd.read_csv("data/processed/train_FD001_with_RUL.csv")

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
# Create and train Linear Regression model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\nLinear Regression trained successfully!")


# Create and train Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest trained successfully!")
# Generate predictions

lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

print("\nPredictions generated successfully!")

print("\nFirst 10 actual RUL values:")
print(y_test.head(10).values)

print("\nLinear Regression predictions:")
print(lr_predictions[:10])

print("\nRandom Forest predictions:")
print(rf_predictions[:10])
# Evaluate Linear Regression

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)


# Evaluate Random Forest

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)


print("\nMODEL PERFORMANCE COMPARISON")

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
# Create comparison table

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "MSE": [lr_mse, rf_mse],
    "RMSE": [lr_rmse, rf_rmse]
})

print("\nFINAL MODEL COMPARISON TABLE")
print(comparison.to_string(index=False))
# Visualize model performance

metrics = ["MAE", "MSE", "RMSE"]

lr_values = [lr_mae, lr_mse, lr_rmse]
rf_values = [rf_mae, rf_mse, rf_rmse]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, lr_values, width, label="Linear Regression")
plt.bar(x + width/2, rf_values, width, label="Random Forest")

plt.xlabel("Evaluation Metrics")
plt.ylabel("Error Values")
plt.title("Linear Regression vs Random Forest Performance")

plt.xticks(x, metrics)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("reports/model_comparison.png", dpi=300)

plt.show()

print("\nModel comparison chart saved successfully!")