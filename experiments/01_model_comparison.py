"""
Experiment 01: Model Comparison
================================
Question: How do different classical ML models compare on clean photometric
          features (u, g, r, i, z) for star/galaxy/quasar classification?

The original notebook only trained a Random Forest. A single model in
isolation tells you very little -- you don't know if 87% is good, bad, or
just what any model gets. This experiment establishes that context by
training four models on the IDENTICAL train/test split and reporting the
same metrics for each.

Models (defined in src/models.py):
    - Logistic Regression  (linear baseline)
    - Random Forest        (the original main model)
    - Gradient Boosting    (usually the strongest classical tabular model)
    - k-Nearest Neighbors  (distance-based, scaling-sensitive)

We report accuracy AND macro-F1. Macro-F1 is the more honest headline number
on this imbalanced dataset (see src/evaluate.py for why).

Outputs:
    results/01_model_comparison.csv          -- the metrics table
    results/01_confusion_<model>.png         -- one confusion matrix per model

Run from the repo root:
    python -m experiments.01_model_comparison
"""

import time

import pandas as pd

from src.data_loader import load_split
from src.models import get_models
from src.evaluate import (
    compute_metrics,
    print_report,
    confusion_df,
    save_confusion_matrix,
    save_table,
)


def run():
    # Load the shared clean split. Every model sees exactly the same data.
    X_train, X_test, y_train, y_test = load_split(verbose=True)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}\n")

    rows = []
    models = get_models()

    for name, model in models.items():
        print(f"--- Training: {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        # Per-class breakdown printed to the console for each model.
        print_report(y_test, y_pred, title=name)
        print(confusion_df(y_test, y_pred))

        # Save one confusion-matrix figure per model.
        safe = name.lower().replace(" ", "_").replace("-", "")
        save_confusion_matrix(
            y_test, y_pred,
            filename=f"01_confusion_{safe}.png",
            title=f"{name} -- Confusion Matrix",
        )

        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Train time (s)": round(train_time, 2),
        })
        print()

    # Assemble and save the comparison table, sorted by macro-F1.
    table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    out = save_table(table.set_index("Model"), "01_model_comparison.csv")

    print("=" * 60)
    print("MODEL COMPARISON (sorted by macro-F1)")
    print("=" * 60)
    print(table.to_string(index=False))
    print(f"\nSaved table -> {out}")
    print("Saved confusion matrices -> results/01_confusion_*.png")

    return table


if __name__ == "__main__":
    run()
