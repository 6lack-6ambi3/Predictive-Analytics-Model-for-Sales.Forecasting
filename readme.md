# BigMart Sales Prediction

A machine learning pipeline that predicts product sales across BigMart
outlet stores. Built from scratch in Python with a clean `src/` architecture,
multi-model comparison, and a reproducible end-to-end pipeline.

## Results

| Model              |   RMSE  |   MAE   |   R²  |
|--------------------|---------|---------|-------|
| Random Forest      | 1089.32 | 746.62  | 0.595 |
| Gradient Boosting  | 1067.18 | 746.08  | 0.589 |
| Linear Regression  | 1127.45 | 875.59  | 0.507 |

Random Forest achieves the best generalisation (R² = 0.595).

## Dataset

BigMart Sales dataset — 8,523 product-outlet combinations across 10 stores.
Originally from the Analytics Vidhya Big Mart Sales Prediction challenge.

**Target variable:** `Item_Outlet_Sales` — annual product sales per outlet.

**Key features:**
- `Item_MRP` — maximum retail price (strongest predictor)
- `Outlet_Type` — grocery store vs. supermarket
- `Outlet_Age` — years since establishment (engineered feature)
- `Item_Visibility` — product shelf visibility
- `Item_Category` — Food / Drinks / Non-Consumable (engineered from identifier)

## Pipeline
## Project structure
## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Runs the full pipeline and saves the best model to `models/best_model.joblib`
and a feature importance chart to `models/feature_importance.png`.

## Feature engineering

Two features were created from raw data:

**Outlet_Age** — years since the outlet opened (reference year 2013, the
dataset's collection year). Older outlets have established customer bases,
which correlates with higher sales.

**Item_Category** — derived from the first two characters of `Item_Identifier`
(`FD` = Food, `DR` = Drinks, `NC` = Non-Consumable). Non-consumables are
handled separately from food items in retail, so this is a meaningful signal.

## What I'd improve

- **Hyperparameter tuning** with `GridSearchCV` or `Optuna` — current models
  use reasonable defaults but aren't optimised
- **XGBoost / LightGBM** — likely to outperform Scikit-learn's GradientBoosting
  on tabular data
- **Time-series cross-validation** — the data has a temporal component
  (outlet establishment year) that standard random splits don't respect
- **A FastAPI `/predict` endpoint** — wrapping the saved model to serve
  predictions over HTTP, similar to the CIFAR-10 deployment project
