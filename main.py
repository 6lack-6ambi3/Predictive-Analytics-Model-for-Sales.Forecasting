"""
BigMart Sales Prediction - end-to-end pipeline.

Loads the data, preprocesses it, trains and compare three regression models, saves the best and produces evaluation outputs.
"""

from src.preprocessing import preprocess
from src.train import train_and_compare, TARGET
from src.evaluate import plot_feature_importance, results_table

def main():
    print("=" * 55)
    print("  BigMart Sales Prediction Pipeline")
    print("=" * 55)

    print("\n[1/3] Preprocessing data...")
    df = preprocess("data/train_kOBLwZA.csv")
    print(f"    Processed {len(df)} rows, {df.shape[1]} features")
    
    print("\n[2/3] Training and comparing models...")
    results, best_name, best_model = train_and_compare(df)

    print("\n[3/3] Evaluation summary:\n")
    print(results_table(results).to_string())

    feature_names = df.drop(columns=[TARGET]).columns
    plot_feature_importance(best_model, feature_names)

    print("\nDone. Best model and plots saved to models/")

if __name__ == "__main__":
    main()