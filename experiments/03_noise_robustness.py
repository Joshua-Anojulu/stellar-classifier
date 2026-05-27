"""
Experiment 03: Noise Robustness
================================
Question: As measurement noise increases, how gracefully does each model's
          accuracy degrade -- and do the models rank the same under noise as
          they do on clean data?

Real photometric measurements have uncertainty. A classifier that scores well
on a clean test set but collapses under a little noise is fragile and not
trustworthy in practice. So instead of one clean accuracy number, this
experiment measures a *degradation curve*: accuracy as a function of how much
noise we inject into the test features.

How the noise is injected
--------------------------
We add zero-mean Gaussian noise to the test features only (the models are
trained once on clean data, as they would be in practice). The original
notebook added noise of a fixed absolute size (e.g. std=0.5) to every column.
But the photometric bands have different spreads, so a fixed absolute noise hits
some features harder than others and is hard to interpret.

This version injects noise *relative to each feature's own standard deviation*:
at noise level k, each feature gets Gaussian noise with std = k * (that
feature's std). So k=0.25 means "a quarter of a standard deviation of noise on
every feature," which is comparable across features and across datasets. k=0 is
the clean baseline.

We sweep k over several levels and plot one curve per model.

Outputs:
    results/03_noise_curve.csv     -- accuracy at each noise level, per model
    results/03_noise_curve.png     -- the degradation curves

Run from the repo root:
    python run.py 03
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data_loader import load_split
from src.models import get_models
from src.evaluate import save_table, RESULTS_DIR

# Noise levels as fractions of each feature's standard deviation.
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# Fix the RNG so the injected noise is identical every run -> reproducible.
RANDOM_STATE = 42


def add_relative_noise(X, level, feature_stds, rng):
    """Return a copy of X with Gaussian noise added to each column, where each
    column's noise std = level * that column's std. level=0 returns X unchanged.
    """
    if level == 0.0:
        return X.copy()
    # noise shape matches X; scale each column by its own std * level.
    noise = rng.normal(0.0, 1.0, size=X.shape) * (feature_stds.values * level)
    return X + noise


def run():
    X_train, X_test, y_train, y_test = load_split(verbose=True)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Noise levels (in units of each feature's std): {NOISE_LEVELS}\n")

    # Per-feature std measured on the TRAINING set (what you'd know at train time).
    feature_stds = X_train.std()

    # Train every model once on clean data.
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
    print("All models trained on clean data.\n")

    # For each model, sweep noise levels and record accuracy.
    rng = np.random.default_rng(RANDOM_STATE)
    results = {}  # name -> list of accuracies aligned with NOISE_LEVELS
    for name, model in models.items():
        accs = []
        for level in NOISE_LEVELS:
            X_noisy = add_relative_noise(X_test, level, feature_stds, rng)
            acc = accuracy_score(y_test, model.predict(X_noisy))
            accs.append(acc)
        results[name] = accs
        # Quick console line: clean -> heaviest-noise accuracy.
        print(f"{name:>22}: clean {accs[0]*100:5.2f}%  ->  "
              f"@{NOISE_LEVELS[-1]}std {accs[-1]*100:5.2f}%  "
              f"(drop {(accs[0]-accs[-1])*100:5.2f} pts)")

    # --- Save the table ---------------------------------------------------
    table = pd.DataFrame(results, index=[f"{lvl}" for lvl in NOISE_LEVELS])
    table.index.name = "noise_level_(x_std)"
    save_table(table, "03_noise_curve.csv")

    # --- Plot the degradation curves --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D"]
    for (name, accs), marker in zip(results.items(), markers):
        ax.plot(NOISE_LEVELS, [a * 100 for a in accs],
                marker=marker, label=name, linewidth=2)
    ax.set_xlabel("Noise level (multiples of each feature's std)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy degradation under increasing feature noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_noise_curve.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("NOISE ROBUSTNESS SUMMARY (accuracy %, rows = noise level)")
    print("=" * 60)
    print((table * 100).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nSaved -> results/03_noise_curve.csv")
    print("Saved -> results/03_noise_curve.png")

    return table


if __name__ == "__main__":
    run()
