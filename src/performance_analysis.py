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
print("\nFeature columns:")
print(feature_columns)
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
print("Training engines:", train_data.iloc[train_idx]["engine_id"].nunique())
print("Testing engines:", train_data.iloc[test_idx]["engine_id"].nunique())
# Train Linear Regression model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\nLinear Regression trained successfully!")


# Train Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("Random Forest trained successfully!")
# Calculate Linear Regression metrics

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)


# Calculate Random Forest metrics

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)


# Display performance metrics

print("\n========== MODEL PERFORMANCE ==========")

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
# Create performance analysis dataframe

results = pd.DataFrame({
    "Actual_RUL": y_test.values,
    "LR_Predicted": lr_predictions,
    "RF_Predicted": rf_predictions
})

# Calculate absolute errors

results["LR_Error"] = np.abs(
    results["Actual_RUL"] - results["LR_Predicted"]
)

results["RF_Error"] = np.abs(
    results["Actual_RUL"] - results["RF_Predicted"]
)

# Divide RUL into ranges

results["RUL_Range"] = pd.cut(
    results["Actual_RUL"],
    bins=[-1, 50, 150, 400],
    labels=["Low RUL", "Medium RUL", "High RUL"]
)

# Calculate average error for each range

range_analysis = results.groupby(
    "RUL_Range", observed=False
)[["LR_Error", "RF_Error"]].mean()

print("\n========== ERROR ANALYSIS BY RUL RANGE ==========")
print(range_analysis)
# Visualize error analysis by RUL range

range_analysis.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Error Analysis Across RUL Ranges")
plt.xlabel("RUL Range")
plt.ylabel("Mean Absolute Error (Cycles)")

plt.xticks(rotation=0)
plt.legend(["Linear Regression", "Random Forest"])

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/rul_range_error_analysis.png",
    dpi=300
)

plt.show()

print("\nRUL range error analysis graph saved successfully!")
# Analyze Random Forest feature importance

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_
})

# Sort features by importance

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(feature_importance.head(10))
# Visualize top 10 feature importance

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features - Random Forest")

plt.gca().invert_yaxis()

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance graph saved successfully!")