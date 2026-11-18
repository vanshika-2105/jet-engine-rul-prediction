import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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
# Visualize Actual vs Predicted RUL

plt.figure(figsize=(14, 6))

# Linear Regression plot
plt.subplot(1, 2, 1)

plt.scatter(y_test, lr_predictions, alpha=0.5)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Linear Regression: Actual vs Predicted")

plt.plot(
    [0, 400],
    [0, 400],
    linestyle="--",
    label="Perfect Prediction"
)

plt.legend()
plt.grid(alpha=0.3)


# Random Forest plot
plt.subplot(1, 2, 2)

plt.scatter(y_test, rf_predictions, alpha=0.5)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Random Forest: Actual vs Predicted")

plt.plot(
    [0, 400],
    [0, 400],
    linestyle="--",
    label="Perfect Prediction"
)

plt.legend()
plt.grid(alpha=0.3)


plt.tight_layout()

plt.savefig("reports/actual_vs_predicted.png", dpi=300)

plt.show()

print("\nActual vs Predicted graph saved successfully!")
# Calculate prediction errors

lr_errors = y_test.values - lr_predictions
rf_errors = y_test.values - rf_predictions

# Visualize prediction error distribution

plt.figure(figsize=(12, 6))

plt.hist(
    lr_errors,
    bins=30,
    alpha=0.6,
    label="Linear Regression"
)

plt.hist(
    rf_errors,
    bins=30,
    alpha=0.6,
    label="Random Forest"
)

plt.axvline(0, linestyle="--", label="Zero Error")

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("reports/prediction_error_distribution.png", dpi=300)

plt.show()

print("\nPrediction error distribution graph saved successfully!")
# Compare absolute prediction errors

lr_absolute_errors = np.abs(lr_errors)
rf_absolute_errors = np.abs(rf_errors)

plt.figure(figsize=(9, 6))

plt.boxplot(
    [lr_absolute_errors, rf_absolute_errors],
    tick_labels=["Linear Regression", "Random Forest"]
)

plt.ylabel("Absolute Prediction Error (Cycles)")
plt.title("Absolute Prediction Error Comparison")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("reports/absolute_error_comparison.png", dpi=300)

plt.show()

print("\nAbsolute error comparison graph saved successfully!")