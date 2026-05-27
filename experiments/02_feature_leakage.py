"""
Experiment 02: Feature Leakage (Redshift)
==========================================
Question: How much does including `redshift` as a feature inflate measured
          performance, and why is that inflation misleading?

This is the most distinctive part of the study, so it's worth being precise
about what "leakage" means here.

Redshift is not a photometric measurement -- it's derived from spectroscopy,
the same kind of observation used to assign the class label in the first place.
It is also almost a direct stand-in for the class: stars sit at redshift ~0,
galaxies at low-to-moderate redshift, and quasars at high redshift. So handing
the model `redshift` is close to handing it the answer. The accuracy shoots up,
but the model hasn't learned to classify from photometry -- it has learned to
read a near-label. In a real pipeline where you're trying to classify objects
you DON'T already have spectra for, that feature wouldn't be available, so the
inflated number does not reflect real-world performance.

This experiment trains the SAME Random Forest two ways on the SAME split:
    (A) clean   : u, g, r, i, z
    (B) leaked  : u, g, r, i, z, redshift
and reports the gap, the per-class change, and the feature importances (to show
redshift dominating everything else in the leaked model).

Outputs:
    results/02_leakage_comparison.csv      -- clean vs leaked metrics
    results/02_leakage_importances.csv     -- feature importances, leaked model
    results/02_leakage_importances.png     -- bar chart of those importances

Run from the repo root:
    python run.py 02
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import (
    load_data,
    get_features_and_target,
    split_data,
    PHOTOMETRIC_FEATURES,
    FEATURES_WITH_REDSHIFT,
)
from src.models import build_random_forest
from src.evaluate import compute_metrics, print_report, confusion_df, save_table, RESULTS_DIR


def train_and_eval(df, feature_set, label):
    """Train a fresh Random Forest on the given feature set and return
    (fitted_model, metrics_dict, feature_list)."""
    X, y = get_features_and_target(df, feature_set=feature_set)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print_report(y_test, y_pred, title=f"{label} ({len(feature_set)} features)")
    print(confusion_df(y_test, y_pred))
    print()
    return model, metrics, feature_set


def run():
    # Load ONCE, then build both feature sets from the same cleaned frame so
    # the only thing that differs between the two runs is the redshift column.
    df = load_data(verbose=True)
    print()

    # (A) Clean photometric model -- the honest one.
    _, clean_metrics, _ = train_and_eval(df, PHOTOMETRIC_FEATURES, "CLEAN (photometry only)")

    # (B) Leaked model -- includes redshift.
    leaked_model, leaked_metrics, leaked_features = train_and_eval(
        df, FEATURES_WITH_REDSHIFT, "LEAKED (+ redshift)"
    )

    # --- Comparison table -------------------------------------------------
    comparison = pd.DataFrame(
        [
            {"Feature set": "Clean (u,g,r,i,z)", **clean_metrics},
            {"Feature set": "Leaked (+redshift)", **leaked_metrics},
        ]
    ).set_index("Feature set")
    # Add an explicit "inflation" row: how much each metric jumped.
    inflation = (leaked_metrics["accuracy"] - clean_metrics["accuracy"]) * 100
    save_table(comparison, "02_leakage_comparison.csv")

    # --- Feature importances of the leaked model --------------------------
    # The point: redshift should tower over the photometric bands, visually
    # confirming the model is leaning on the near-label.
    importances = pd.Series(
        leaked_model.feature_importances_, index=leaked_features
    ).sort_values(ascending=False)
    save_table(importances.to_frame("importance"), "02_leakage_importances.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["crimson" if f == "redshift" else "steelblue" for f in importances.index]
    ax.bar(importances.index, importances.values, color=colors)
    ax.set_ylabel("Feature importance")
    ax.set_title("Leaked model: redshift dominates the photometric bands")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_leakage_importances.png", dpi=150)
    plt.close(fig)

    # --- Summary ----------------------------------------------------------
    print("=" * 60)
    print("FEATURE LEAKAGE SUMMARY")
    print("=" * 60)
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nAccuracy inflation from adding redshift: +{inflation:.2f} percentage points")
    print(f"\nLeaked-model feature importances:")
    print(importances.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nSaved -> results/02_leakage_comparison.csv")
    print("Saved -> results/02_leakage_importances.csv")
    print("Saved -> results/02_leakage_importances.png")

    return comparison, importances


if __name__ == "__main__":
    run()
