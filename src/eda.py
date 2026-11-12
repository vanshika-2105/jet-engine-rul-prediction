
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names
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

print("NASA FD001 Dataset Loaded Successfully!")
print("Dataset Shape:", train_data.shape)

# Display first five rows
print("\nFirst 5 rows:")
print(train_data.head())

# Calculate the maximum cycle for each engine
engine_lifecycle = train_data.groupby("engine_id")["cycle"].max()

print("\nEngine Lifecycle:")
print(engine_lifecycle.head(10))

# Display lifecycle statistics
print("\nLifecycle Statistics:")
print(engine_lifecycle.describe())

# Plot engine lifecycle distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    engine_lifecycle,
    bins=15,
    kde=True
)

plt.title("Distribution of Engine Lifecycles")
plt.xlabel("Maximum Operating Cycles")
plt.ylabel("Number of Engines")

plt.tight_layout()

# Save the graph
plt.savefig("reports/engine_lifecycle_distribution.png")

plt.show()

# Analyze Sensor 2 for Engine 1

engine_1_data = train_data[train_data["engine_id"] == 1]

plt.figure(figsize=(10, 5))

plt.plot(
    engine_1_data["cycle"],
    engine_1_data["sensor_2"]
)

plt.title("Sensor 2 Readings - Engine 1")
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor 2 Reading")

plt.grid(True)
plt.tight_layout()

plt.savefig("reports/sensor_2_engine_1.png")
plt.show()

# Analyze Sensor 3 for Engine 1

plt.figure(figsize=(10, 5))

plt.plot(
    engine_1_data["cycle"],
    engine_1_data["sensor_3"]
)

plt.title("Sensor 3 Readings - Engine 1")
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor 3 Reading")

plt.grid(True)
plt.tight_layout()

plt.savefig("reports/sensor_3_engine_1.png")
plt.show()

# Analyze statistical information of all sensors

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

sensor_statistics = train_data[sensor_columns].describe()

print("\nSensor Statistics:")
print(sensor_statistics)

# Identify sensors with constant values

constant_sensors = []

for sensor in sensor_columns:
    if train_data[sensor].nunique() == 1:
        constant_sensors.append(sensor)

print("\nConstant Sensors:")
print(constant_sensors)

print("\nTotal Constant Sensors:", len(constant_sensors))

# Calculate correlation between sensors

sensor_correlation = train_data[sensor_columns].corr()

# Create correlation heatmap
plt.figure(figsize=(16, 12))

sns.heatmap(
    sensor_correlation,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Sensor Correlation Heatmap")
plt.tight_layout()

# Save the heatmap
plt.savefig("reports/sensor_correlation_heatmap.png")

plt.show()
