"""
Experiment 05: Dataset Shift (Magnitude-Based)
==============================================
Question: When a model is trained on one regime of the survey and tested on a
          DIFFERENT regime, how much does performance fall compared to the
          standard random split? This is the "generalization gap."

Why this is the most realistic test in the study
-------------------------------------------------
Every earlier experiment used a random train/test split, so the test set looks
statistically just like the training set. Real surveys aren't like that. A
telescope might be well-sampled for bright objects but you actually want to
classify faint ones; or a model trained on today's data is applied to a deeper
future survey. When the test distribution differs from training, accuracy on a
random split *overstates* real-world performance.

We simulate this "dataset shift" using the `r`-band magnitude as a proxy for
brightness (larger magnitude = fainter object). We define two regimes by the
median r magnitude and run the shift both directions:

    bright -> faint : train on brighter half, test on fainter half
    faint  -> bright: train on fainter half, test on brighter half

For reference we also report the standard random split (the "no shift" baseline)
using the same model and the same number of training rows, so the comparison is
fair. The gap between random-split accuracy and shifted accuracy IS the result.

A caveat we state honestly: a brightness split also shifts the class balance
(faint objects skew toward galaxies/quasars), so part of the gap is covariate
shift and part is label shift. We report the test-set class balance in each
regime so this is transparent rather than hidden.

Outputs:
    results/05_dataset_shift.csv     -- accuracy + macro-F1 per scenario
    results/05_dataset_shift.png     -- bar chart comparing scenarios

Run from the repo root:
    python run.py 05
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_data,
    PHOTOMETRIC_FEATURES,
    TARGET,
    RANDOM_STATE,
)
from src.models import build_random_forest
from src.evaluate import compute_metrics, save_table, RESULTS_DIR

SPLIT_BAND = "r"  # brightness proxy used to define the two regimes


def evaluate_scenario(X_train, y_train, X_test, y_test):
    """Fit a fresh RF and return its metrics dict on the given test set."""
    model = build_random_forest().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def run():
    df = load_data(verbose=True)
    print()

    X = df[PHOTOMETRIC_FEATURES]
    y = df[TARGET]
    brightness = df[SPLIT_BAND]

    # Median magnitude defines bright (<= median) vs faint (> median).
    median_mag = brightness.median()
    is_bright = brightness <= median_mag
    is_faint = ~is_bright
    print(f"Split on '{SPLIT_BAND}'-band median magnitude = {median_mag:.3f}")
    print(f"Bright half: {is_bright.sum()} objects | Faint half: {is_faint.sum()} objects\n")

    # Report class balance in each regime so the label-shift is transparent.
    print("Class balance by regime:")
    bal = pd.DataFrame({
        "bright": y[is_bright].value_counts(normalize=True),
        "faint": y[is_faint].value_counts(normalize=True),
    }).round(3)
    print(bal, "\n")

    results = {}

    # --- Scenario 1: bright -> faint -------------------------------------
    results["bright -> faint"] = evaluate_scenario(
        X[is_bright], y[is_bright], X[is_faint], y[is_faint]
    )

    # --- Scenario 2: faint -> bright -------------------------------------
    results["faint -> bright"] = evaluate_scenario(
        X[is_faint], y[is_faint], X[is_bright], y[is_bright]
    )

    # --- Baseline: random split (no shift) -------------------------------
    # Use the same training-set size as the shifted scenarios (~half the data)
    # so the comparison isn't confounded by training on more rows.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
    )
    results["random split (no shift)"] = evaluate_scenario(Xtr, ytr, Xte, yte)

    # --- Assemble table ---------------------------------------------------
    table = pd.DataFrame(results).T[["accuracy", "macro_f1", "weighted_f1"]]
    save_table(table, "05_dataset_shift.csv")

    # Generalization gap: how far each shifted scenario falls below the
    # no-shift baseline accuracy.
    baseline_acc = results["random split (no shift)"]["accuracy"]
    gaps = {k: baseline_acc - v["accuracy"] for k, v in results.items()}

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    f1s = [results[n]["macro_f1"] * 100 for n in names]
    xpos = np.arange(len(names))
    width = 0.38
    ax.bar(xpos - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(xpos + width / 2, f1s, width, label="Macro F1", color="indianred")
    ax.set_xticks(xpos, names, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Dataset shift: performance under brightness-based regime change")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_dataset_shift.png", dpi=150)
    plt.close(fig)

    # --- Summary ----------------------------------------------------------
    print("=" * 60)
    print("DATASET SHIFT SUMMARY")
    print("=" * 60)
    print((table * 100).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nGeneralization gap vs random-split baseline (accuracy):")
    print("  (positive = performs WORSE than no-shift baseline)")
    for name in names:
        if name == "random split (no shift)":
            continue
        gap_pts = gaps[name] * 100
        print(f"  {name:>22}: {gap_pts:+.2f} pts")
    print("\nSaved -> results/05_dataset_shift.csv")
    print("Saved -> results/05_dataset_shift.png")

    return table


if __name__ == "__main__":
    run()
