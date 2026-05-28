# Turbofan Engine RUL Prediction

Predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines from multi-sensor telemetry, replicating the NASA CMAPSS benchmark problem.

---

## Overview

Engine failure in aviation is catastrophic. Predictive maintenance — knowing *when* an engine will fail before it does — is a critical ML application in aerospace. This project builds an end-to-end ML pipeline that ingests sensor time-series data and outputs RUL predictions in cycles.

---

## Pipeline

```
Raw Sensor Data → RUL Labeling → Feature Engineering → Model Training → Evaluation → Model Export
```

1. **Data Simulation** — 40 engine units with 21 CMAPSS-style sensors degrading realistically over lifecycle
2. **RUL Labeling** — Piecewise-linear cap at 125 cycles (standard CMAPSS practice)
3. **Feature Engineering** — Rolling window mean & std (8-cycle window) per sensor to capture degradation trends
4. **Train/Test Split** — Split by engine unit to prevent data leakage
5. **Model Comparison** — Ridge Regression, Random Forest, Gradient Boosting
6. **Evaluation** — RMSE, MAE, R², and NASA asymmetric scoring function
7. **Model Export** — Best model saved with joblib

---

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 22.65 | 17.85 | 0.707 |
| Random Forest | 23.06 | 18.31 | 0.696 |
| **Gradient Boosting** | **20.26** | **15.49** | **0.765** |

Gradient Boosting achieved the best performance with **RMSE of 20.3 cycles**.

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Models** — Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor
- **Evaluation** — RMSE, MAE, R², NASA asymmetric scoring function

---

## Project Structure

```
├── project1_rul_prediction.py   # Full pipeline script
├── rul_prediction_results.png   # Evaluation plots
├── rul_model.pkl                # Saved best model
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python project1_rul_prediction.py
```

---

## Key Concepts

- **RUL** — Remaining Useful Life: how many cycles until engine failure
- **CMAPSS** — Commercial Modular Aero-Propulsion System Simulation (NASA dataset)
- **Piecewise-linear capping** — Treats early-life cycles as equally healthy, focuses learning on degradation phase
- **Rolling statistics** — Provides the model with trend context rather than just instantaneous sensor readings

---

