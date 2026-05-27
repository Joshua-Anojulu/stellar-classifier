"""
evaluate.py
-----------
Shared evaluation utilities so every experiment reports results consistently.

Centralizing this matters: if each experiment computed metrics its own way,
the numbers wouldn't be comparable and reviewers would (rightly) distrust them.
Here we define one place for:
  - computing the headline metrics (accuracy, macro-F1, weighted-F1),
  - printing a per-class classification report,
  - building a labeled confusion matrix,
  - saving figures into results/ with consistent styling.

We report MACRO-F1 prominently, not just accuracy. Accuracy is misleading on
imbalanced data (59% of objects are galaxies, so a model that only predicts
"galaxy" already scores 59%). Macro-F1 averages the F1 of each class equally,
so it exposes whether the rare classes (quasars, stars) are actually handled
well. This distinction is a core theme of the project.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we save files, never show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Fixed class order, imported wherever a confusion matrix is built.
from .data_loader import CLASS_ORDER

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Return the headline metrics as a plain dict.

    macro_f1 weights every class equally (good for imbalance);
    weighted_f1 weights by class frequency (closer to accuracy).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def print_report(y_true, y_pred, title=None):
    """Print a per-class precision/recall/F1 report."""
    if title:
        print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, digits=3))


def confusion_df(y_true, y_pred, labels=CLASS_ORDER):
    """Return a labeled confusion matrix as a DataFrame (rows=true, cols=pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true {c}" for c in labels],
        columns=[f"pred {c}" for c in labels],
    )


def save_confusion_matrix(y_true, y_pred, filename, labels=CLASS_ORDER, title="Confusion Matrix"):
    """Render a confusion matrix heatmap to results/<filename>."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Annotate each cell with its count.
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_table(df, filename, float_format="%.4f"):
    """Save a results table as CSV in results/."""
    out = RESULTS_DIR / filename
    df.to_csv(out, float_format=float_format)
    return out


def save_figure(fig, filename):
    """Save an arbitrary matplotlib figure into results/."""
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
