"""
Evaluation plots and reports for the sales prediction models
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg") #non-interactive backend for saving plots
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names, out_path: str = "models/feature_importance.png"):
    """Plot feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ - skipping plot")
        return
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance plot -> {out_path}")

def results_table(results: dict) -> pd.DataFrame:
    """Turn the results dict into a sorted DataFrame."""
    table = pd.DataFrame(results).T
    table = table.sort_values("R2", ascending=False)
    return table.round(3)
