"""Train and compare regression models for sales predictions"""

from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

TARGET = "Item_Outlet_Sales"

def split_data(df, test_size: float = 0.3, seed: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def get_models() -> dict:
    """The candidate models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=4,random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(y_test, preds))
    return{
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }

def train_and_compare(df, model_dir: str = "models") -> tuple[dict, str, object]:
    """
    Train all models, compare them and save the best. Returns (results, best_name, best_model).
    """
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}
    trained = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        trained[name] = model
        print(f"{name:20s}  RMSE={results[name]['RMSE']:8.2f}"
              f"R2={results[name]['R2']:.3f}")
        
    # Best Model = highest R2
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained[best_name]

    Path(model_dir).mkdir(exist_ok=True)
    model_path = Path(model_dir) / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name} (saved to {model_path})")

    return results, best_name, best_model