# AssurPrime: Insurance Premium Prediction

Project developed for the **Crédit Agricole Assurances** hackathon hosted on the [Challenge Data (ENS)](https://challengedata.ens.fr/participants/challenges/161/) platform.

---

## Project Overview

The main objective of this project is to predict the financial charge (`CHARGE`) associated with insurance policies. To model annual risk independently of coverage duration, the problem is framed around predicting **`PRIME_PURE`** (CHARGE / ANNEE_ASSURANCE).

Key features of the project:
- Memory optimization to handle large-scale tabular datasets in Google Colab.
- Feature engineering (weather metrics, policy attributes, derogations, seniority).
- Automated hyperparameter tuning via **Optuna** and experiment tracking with **MLflow**.
- An ensemble of three Gradient Boosting models (**LightGBM**, **CatBoost**, **XGBoost**) using stratified cross-validation.

---

## Setup & Dependencies

### Tech Stack
- **Environment**: Python 3.11 / Google Colab
- **Tracking & Tuning**: Optuna, MLflow
- **Modeling**: LightGBM, CatBoost, XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy

### Installation
pip install mlflow optuna optuna.integration catboost lightgbm xgboost

---

## Processing Pipeline

### 1. Optimized Data Loading
To work around RAM limits (12 GB on Google Colab), large files are loaded in chunks (`chunksize=100_000`) with configurable sub-sampling options.

- `train_input.csv`: Training features
- `train_output.csv`: Targets (`CHARGE`, `ANNEE_ASSURANCE`)
- `test_input.csv`: Test features

### 2. Feature Engineering & RAM Optimization
- **Aggregations & Metrics**:
  - `NB_DEROG` / `NB_CARACT`: Number of non-null derogations and attributes.
  - `FRCH_TOTAL`: Total deductibles (`FRCH1` + `FRCH2`).
  - `METEO_MOYENNE` / `METEO_MAX`: Aggregated weather indicators.
  - `ANCIENNETE_LOG`: Logarithmic transformation log(1 + x).
- **Memory Management**: Dynamic numeric downcasting and explicit garbage collection (`gc.collect()`).
- **Encoding**: Missing values filled with `"Manquant"`, followed by categorical integer encoding aligned across Train and Test sets.

### 3. Business Logic & Targets
- Target adjustment for `CHARGE` (>= 0).
- Continuous target computation: PRIME_PURE = CHARGE / ANNEE_ASSURANCE.
- Binary flag creation: `A_SINISTRE` (1 if CHARGE > 0, 0 otherwise) used for stratified fold creation.

---

## Modeling Strategy

### Hyperparameter Optimization (Optuna + MLflow)
Bayesian search for LightGBM hyperparameters (100 trials):
- **Objective Distribution**: Tweedie (`tweedie_variance_power = 1.5`), suitable for zero-inflated continuous data with extreme values.
- **Tuned Hyperparameters**: `learning_rate`, `max_depth`, `num_leaves`, `feature_fraction`, `bagging_fraction`.
- **Pruning & Tracking**: Optuna early stopping callbacks integrated with MLflow logging for metrics and artifacts.

### Ensemble & Stratified Cross-Validation (5-Fold)
A 5-fold `StratifiedKFold` split based on `A_SINISTRE` ensures a consistent proportion of claim-bearing contracts in every fold.

For each fold, an ensemble of three Gradient Boosting models using the Tweedie objective is trained:
1. **LightGBM** (Hyperparameters optimized via Optuna)
2. **CatBoostRegressor** (`iterations=800`, `depth=5`, `learning_rate=0.08`)
3. **XGBoost Regressor** (`tree_method='hist'`, `max_depth=5`, `learning_rate=0.05`)

Final `PRIME_PURE` predictions are computed using a simple average of all three models:
y_ensemble = (y_LGB + y_Cat + y_XGB) / 3

---

## Results & Submission

1. **Charge Conversion**:
   CHARGE_pred = PRIME_PURE_pred * ANNEE_ASSURANCE
2. **Local Evaluation**: RMSE evaluation calculated between actual and predicted charges on Out-Of-Fold (`OOF`) predictions.
3. **Submission File**: `soumission_trio_optuna_mlflow_3.csv` containing the required columns (`ID`, `ANNEE_ASSURANCE`, `FREQ`, `CM`, `CHARGE`).
