# Day 9 — Model Performance Analysis

## 1. Objective
The objective of this phase was to evaluate the performance of Linear Regression and Random Forest models for predicting the Remaining Useful Life (RUL) of jet engines.

## 2. Dataset and Methodology
- Dataset: NASA C-MAPSS FD001
- Total records: 20,631
- Features used: 19
- Training records: 16,561
- Testing records: 4,070
- Training engines: 80
- Testing engines: 20

A GroupShuffleSplit with a test size of 20% and random state 42 was used to ensure that records from the same engine did not appear in both training and testing sets.

## 3. Overall Model Performance

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 25.16 | 24.09 |
| MSE | 1002.10 | 1008.56 |
| RMSE | 31.66 | 31.76 |

Random Forest achieved a lower MAE, while Linear Regression achieved slightly lower MSE and RMSE. Therefore, neither model performed best across every metric.

## 4. Error Analysis Across RUL Ranges

| RUL Range | Linear Regression MAE | Random Forest MAE |
|---|---:|---:|
| Low (0–50) | 22.18 | 9.37 |
| Medium (51–150) | 22.71 | 27.91 |
| High (151+) | 32.55 | 31.04 |

### Observations
- Random Forest performed considerably better in the low-RUL range.
- Linear Regression performed better in the medium-RUL range.
- Both models had similar errors in the high-RUL range.

## 5. Feature Importance Analysis

The Random Forest model identified the following as its top five features:

1. Cycle — 53.44%
2. Sensor 11 — 14.34%
3. Sensor 9 — 4.59%
4. Sensor 4 — 3.14%
5. Sensor 14 — 2.96%

Cycle was the most influential feature, followed by sensor 11. Feature importance indicates the model's reliance on these variables and does not establish causation.

## 6. Visualizations

The following visualizations were generated:

- `rul_range_error_analysis.png` — compares prediction errors across RUL ranges.
- `feature_importance.png` — displays the top ten important features used by Random Forest.

## 7. Conclusion

The performance analysis showed that both models have different strengths.

Random Forest achieved a lower overall MAE and performed particularly well for engines with low remaining useful life. Linear Regression achieved slightly lower MSE and RMSE and performed better in the medium-RUL range.

Random Forest may be a useful candidate for further investigation, particularly for low-RUL predictions. However, additional validation is required before selecting a final model for practical predictive maintenance applications.

## 8. Outcome

Day 9 successfully completed the performance evaluation, RUL-range error analysis, and feature importance analysis of both machine learning models.