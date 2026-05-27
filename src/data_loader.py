"""
data_loader.py
--------------
Loads and cleans the SDSS star/galaxy/quasar dataset.

This is the foundation module: every experiment imports `load_data` and
`split_data` from here so that all experiments use the *exact same* data and
the *exact same* train/test split. That reproducibility is one of the things
that turns a notebook into a research-style project.

The dataset (star_classification.csv) is the SDSS DR17-based Kaggle dataset:
100,000 objects, each labeled GALAXY, STAR, or QSO (quasar).

Photometric features (the only ones we use for the "clean" model):
    u, g, r, i, z  -- magnitudes in five SDSS filter bands.

We deliberately DO NOT use `redshift` as a clean feature. Redshift is derived
from spectroscopy and is almost a direct giveaway of the class (quasars have
high redshift, stars ~0). Including it inflates accuracy to ~98% by leaking the
answer. We study that leakage explicitly in experiment 02 -- here we keep the
default feature set honest.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constants -- defined once here so no experiment hard-codes them separately.
# ---------------------------------------------------------------------------

# The five photometric bands. This is the "clean" feature set (no leakage).
PHOTOMETRIC_FEATURES = ["u", "g", "r", "i", "z"]

# The leaked feature set used only in the leakage experiment.
FEATURES_WITH_REDSHIFT = PHOTOMETRIC_FEATURES + ["redshift"]

# The target column and the fixed class order we use everywhere, so that
# confusion matrices and reports line up across experiments.
TARGET = "class"
CLASS_ORDER = ["GALAXY", "STAR", "QSO"]

# SDSS uses -9999 as a sentinel for "no valid measurement". These are not real
# magnitudes; leaving them in pollutes training. We drop any row containing one.
SENTINEL_VALUE = -9999.0

# Single source of truth for the split, so every experiment is comparable.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Default location of the CSV relative to the repo root.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "star_classification.csv"


def load_data(path=DEFAULT_DATA_PATH, drop_sentinels=True, verbose=False):
    """Load the SDSS CSV and return a cleaned DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of star_classification.csv.
    drop_sentinels : bool
        If True (default), drop rows where any photometric band equals the
        SDSS -9999 "bad measurement" sentinel. This is the one preprocessing
        improvement over the original notebook, and it is defensible: those
        rows are not real photometry.
    verbose : bool
        If True, print shape and class balance (useful when running a single
        experiment from the command line).

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset, with all original columns retained.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download star_classification.csv from the "
            f"SDSS Kaggle dataset and place it in the data/ folder."
        )

    df = pd.read_csv(path)

    n_before = len(df)
    if drop_sentinels:
        # Keep only rows where every photometric band is a real measurement.
        mask = (df[PHOTOMETRIC_FEATURES] != SENTINEL_VALUE).all(axis=1)
        df = df[mask].reset_index(drop=True)

    if verbose:
        n_dropped = n_before - len(df)
        print(f"Loaded {n_before} rows from {path.name}")
        if drop_sentinels:
            print(f"Dropped {n_dropped} rows containing the {SENTINEL_VALUE} sentinel")
        print(f"Final shape: {df.shape}")
        print("Class balance:")
        print(df[TARGET].value_counts())

    return df


def get_features_and_target(df, feature_set=None):
    """Split a DataFrame into the feature matrix X and target vector y.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of load_data.
    feature_set : list of str or None
        Which columns to use as features. Defaults to the clean photometric
        set (u, g, r, i, z). Pass FEATURES_WITH_REDSHIFT for the leakage study,
        or any custom subset for the ablation study.

    Returns
    -------
    (X, y) : (pandas.DataFrame, pandas.Series)
    """
    if feature_set is None:
        feature_set = PHOTOMETRIC_FEATURES
    X = df[feature_set]
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split.

    Stratify on y so the GALAXY/STAR/QSO proportions are preserved in both
    splits -- important because the classes are imbalanced (~59/22/19%).
    The fixed random_state makes the split identical across every experiment.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_split(feature_set=None, path=DEFAULT_DATA_PATH, drop_sentinels=True, verbose=False):
    """Convenience one-liner: load, clean, select features, and split.

    Most experiments call exactly this, so they all share identical data.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    df = load_data(path=path, drop_sentinels=drop_sentinels, verbose=verbose)
    X, y = get_features_and_target(df, feature_set=feature_set)
    return split_data(X, y)


if __name__ == "__main__":
    # Running `python src/data_loader.py` gives a quick sanity check.
    df = load_data(verbose=True)
    X_train, X_test, y_train, y_test = load_split(verbose=False)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features used: {list(X_train.columns)}")
