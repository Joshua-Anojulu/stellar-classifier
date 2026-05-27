"""
Experiment 04: Feature Ablation & Importance Stability
======================================================
Two related questions:
  (A) Ablation: which of the five photometric bands actually matter? If we drop
      one band, how much does accuracy fall? A band whose removal barely changes
      anything is carrying little unique signal; a band whose removal hurts a lot
      is load-bearing.
  (B) Stability: are the feature-importance rankings trustworthy, or an artifact
      of one particular train/test split? We retrain the Random Forest across
      several random seeds and check whether the importance ranking stays the
      same. A ranking that flips around between runs can't be interpreted.

Why this matters: the original notebook reported feature importances from a
single run. But importances from one split are a point estimate -- if you re-ran
with a different split and the order changed, any claim about "which band
matters most" would be meaningless. Measuring stability is what lets you say
the ranking means something. This is the more rigorous half of the experiment.

We use the clean photometric feature set (u, g, r, i, z) throughout -- redshift
is excluded for the reasons established in experiment 02.

Outputs:
    results/04_ablation.csv                 -- accuracy with each band removed
    results/04_ablation.png                 -- accuracy drop per removed band
    results/04_importance_stability.csv     -- importances across seeds + summary
    results/04_importance_stability.png     -- importance spread per feature

Run from the repo root:
    python run.py 04
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import (
    load_data,
    get_features_and_target,
    split_data,
    PHOTOMETRIC_FEATURES,
)
from src.models import build_random_forest
from src.evaluate import save_table, RESULTS_DIR

# Seeds used for the stability check. Each gives a different train/test split
# AND a different forest, so agreement across them is a strong signal.
STABILITY_SEEDS = [0, 1, 2, 3, 4]


def run_ablation():
    """Train on the full feature set, then on each leave-one-out subset, and
    record the accuracy each time. Returns a DataFrame."""
    df = load_data(verbose=True)
    print()

    rows = []

    # Baseline: all five bands.
    X, y = get_features_and_target(df, feature_set=PHOTOMETRIC_FEATURES)
    Xtr, Xte, ytr, yte = split_data(X, y)
    base_model = build_random_forest().fit(Xtr, ytr)
    base_acc = accuracy_score(yte, base_model.predict(Xte))
    rows.append({"Removed band": "(none — all 5)", "Features used": len(PHOTOMETRIC_FEATURES),
                 "Accuracy": base_acc, "Drop vs full": 0.0})

    # Leave-one-out: drop each band in turn.
    for band in PHOTOMETRIC_FEATURES:
        subset = [f for f in PHOTOMETRIC_FEATURES if f != band]
        Xs, ys = get_features_and_target(df, feature_set=subset)
        Xtr, Xte, ytr, yte = split_data(Xs, ys)
        acc = accuracy_score(yte, build_random_forest().fit(Xtr, ytr).predict(Xte))
        rows.append({"Removed band": band, "Features used": len(subset),
                     "Accuracy": acc, "Drop vs full": base_acc - acc})

    table = pd.DataFrame(rows)
    return table, base_acc


def run_stability():
    """Retrain across several seeds and collect feature importances each time."""
    df = load_data()
    X, y = get_features_and_target(df, feature_set=PHOTOMETRIC_FEATURES)

    records = []
    for seed in STABILITY_SEEDS:
        Xtr, Xte, ytr, yte = split_data(X, y, random_state=seed)
        model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(Xtr, ytr)
        records.append(pd.Series(model.feature_importances_, index=PHOTOMETRIC_FEATURES))

    imp = pd.DataFrame(records, index=[f"seed_{s}" for s in STABILITY_SEEDS])
    # Summary stats per feature across seeds.
    summary = pd.DataFrame({
        "mean_importance": imp.mean(),
        "std_importance": imp.std(),
        "min": imp.min(),
        "max": imp.max(),
    }).sort_values("mean_importance", ascending=False)
    return imp, summary


def run():
    # ---- (A) Ablation ----------------------------------------------------
    ablation, base_acc = run_ablation()
    save_table(ablation.set_index("Removed band"), "04_ablation.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    drops = ablation[ablation["Removed band"] != "(none — all 5)"]
    ax.bar(drops["Removed band"], drops["Drop vs full"] * 100, color="indianred")
    ax.set_xlabel("Band removed")
    ax.set_ylabel("Accuracy drop vs full model (pts)")
    ax.set_title("Leave-one-out ablation: cost of removing each band")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_ablation.png", dpi=150)
    plt.close(fig)

    # ---- (B) Importance stability ---------------------------------------
    imp, summary = run_stability()
    # Save both the per-seed importances and the summary, stacked.
    save_table(summary, "04_importance_stability.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = summary.index.tolist()
    ax.bar(order, summary["mean_importance"], yerr=summary["std_importance"],
           capsize=4, color="steelblue")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean importance (± std across seeds)")
    ax.set_title(f"Importance stability across {len(STABILITY_SEEDS)} seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_importance_stability.png", dpi=150)
    plt.close(fig)

    # ---- Summary ---------------------------------------------------------
    print("=" * 60)
    print("ABLATION (leave-one-out)")
    print("=" * 60)
    print(ablation.to_string(index=False,
          formatters={"Accuracy": lambda x: f"{x:.4f}", "Drop vs full": lambda x: f"{x:+.4f}"}))
    most_costly = ablation.iloc[1:].sort_values("Drop vs full", ascending=False).iloc[0]
    print(f"\nMost load-bearing band: '{most_costly['Removed band']}' "
          f"(removing it costs {most_costly['Drop vs full']*100:.2f} pts)")

    print("\n" + "=" * 60)
    print(f"IMPORTANCE STABILITY across seeds {STABILITY_SEEDS}")
    print("=" * 60)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    # Did the top-ranked feature stay the same in every seed?
    top_each = imp.idxmax(axis=1)
    consistent = top_each.nunique() == 1
    print(f"\nTop feature identical in all seeds? {consistent}"
          + (f" ('{top_each.iloc[0]}')" if consistent else f" (varies: {list(top_each)})"))

    print("\nSaved -> results/04_ablation.csv / .png")
    print("Saved -> results/04_importance_stability.csv / .png")

    return ablation, summary


if __name__ == "__main__":
    run()
