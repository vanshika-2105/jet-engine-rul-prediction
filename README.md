# Jet Engine Remaining Useful Life (RUL) Prediction

## 1. Project Overview

The **Jet Engine Remaining Useful Life (RUL) Prediction** project uses Machine Learning to estimate how many operational cycles remain before a jet engine reaches the end of its useful life.

The project uses historical engine sensor readings from NASA's C-MAPSS dataset to train and evaluate predictive models.

The main objective is to support predictive maintenance by identifying engines that may require maintenance before failure occurs.

## 2. Problem Statement

Unexpected jet engine failures can lead to safety risks, operational delays, and high maintenance costs.

Traditional maintenance approaches may not accurately identify when an engine needs maintenance.

This project addresses the problem by developing machine learning models that estimate the Remaining Useful Life (RUL) of an engine using its operating conditions and sensor measurements.

## 3. Project Objectives

- Analyze NASA C-MAPSS engine sensor data.
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Calculate Remaining Useful Life for each engine cycle.
- Build Linear Regression and Random Forest regression models.
- Compare model performance using evaluation metrics.
- Visualize predictions, errors, and feature importance.

## 4. Dataset

**Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

**Subset used:** FD001

The training dataset contains:
- 20,631 operational records
- 100 different engines
- 26 original columns
- 3 operational settings
- 21 sensor measurements

The dataset contains engine operating cycles and sensor measurements that represent the degradation of engine performance over time.

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git and GitHub
- VS Code

## 6. Project Workflow

### Day 1 – Project Setup and Dataset Collection
- Created the project structure.
- Set up a Python virtual environment.
- Installed the required libraries.
- Collected and inspected the NASA C-MAPSS dataset.

### Day 2 – Exploratory Data Analysis
- Analyzed engine lifecycle distribution.
- Visualized sensor readings.
- Generated a sensor correlation heatmap.
- Identified six constant sensors.

### Day 3 – Data Preprocessing
- Removed six constant sensor columns.
- Checked for missing values.
- Saved the cleaned dataset.

### Day 4 – RUL Calculation and Feature Engineering
- Calculated RUL using the maximum operating cycle of each engine.
- Created the normalized cycle feature.
- Saved the processed dataset with RUL values.

### Day 5 – Linear Regression
- Trained a Linear Regression model.
- Evaluated its performance using MAE, MSE, and RMSE.

### Day 6 – Random Forest Regression
- Trained a Random Forest Regressor.
- Evaluated its performance using the same metrics.

### Day 7 – Model Comparison
- Compared Linear Regression and Random Forest.
- Used the same group-based train-test split for a fair comparison.

### Day 8 – Prediction Visualization
- Generated actual vs. predicted RUL plots.
- Visualized prediction error distribution.
- Compared absolute prediction errors.

### Day 9 – Performance Analysis
- Analyzed model errors across different RUL ranges.
- Identified important features using Random Forest feature importance.
- Generated performance analysis visualizations and documentation.

### Day 10 – Final Documentation
- Prepared project documentation.
- Organized the GitHub repository.
- Finalized the project.

## 7. Machine Learning Models

Two regression algorithms were implemented:

**1. Linear Regression**

A supervised machine learning algorithm that models the relationship between input features and the target RUL value.

**2. Random Forest Regression**

An ensemble learning algorithm that combines multiple decision trees to predict the remaining useful life of jet engines.

Both models were trained using 19 input features, excluding engine ID and normalized cycle.

A GroupShuffleSplit with 80% training engines and 20% testing engines was used to prevent records from the same engine appearing in both sets.

## 8. Model Performance

The models were evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 25.16 | 24.09 |
| MSE | 1002.10 | 1008.56 |
| RMSE | 31.66 | 31.76 |

### Performance Observations

- Random Forest achieved a lower MAE.
- Linear Regression achieved slightly lower MSE and RMSE.
- Random Forest performed considerably better in the low-RUL range.
- Linear Regression performed better in the medium-RUL range.

The results show that both models have different strengths, and neither performed best across every evaluation metric.

## 9. Project Visualizations

The project generated the following visual reports:

- Engine lifecycle distribution
- Sensor readings for Engine 1
- Sensor correlation heatmap
- Model comparison chart
- Actual vs. predicted RUL
- Prediction error distribution
- Absolute error comparison
- RUL range error analysis
- Random Forest feature importance

## 10. Project Structure

```text
jet-engine-rul-pred