"""
models.py
---------
Defines the set of models we compare in the study.

Why these four:
  - Logistic Regression : a linear baseline. If a model can't beat this, the
        problem is either trivial or your features are bad. It needs feature
        scaling (it's sensitive to magnitude ranges), so we wrap it in a
        Pipeline with StandardScaler.
  - Random Forest        : the main model from the original notebook, and the
        workhorse of SDSS classification. Tree-based, so no scaling needed.
  - Gradient Boosting    : another tree ensemble, usually a bit stronger than
        Random Forest but slower to train. Good comparison point.
  - k-Nearest Neighbors  : a simple distance-based model. Included because it
        is extremely sensitive to noise and scaling, which makes the noise-
        robustness experiment more interesting (it should degrade fastest).

Each builder returns a fresh, unfitted estimator with a fixed random_state so
results are reproducible. We expose `get_models()` which returns a dict mapping
a human-readable name to an estimator -- experiments just loop over it.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_logistic_regression():
    """Linear baseline. Scaled because LR cares about feature magnitudes."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_random_forest():
    """Main model -- identical settings to the original notebook."""
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)


def build_gradient_boosting():
    """Boosted trees. Often the strongest classical model on tabular data."""
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def build_knn():
    """Distance-based model. Scaled, because distances are meaningless when
    features live on different scales. Sensitive to noise by design."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])


# A registry the experiments iterate over. Order is the order they'll appear
# in the comparison table.
MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "Random Forest": build_random_forest,
    "Gradient Boosting": build_gradient_boosting,
    "k-Nearest Neighbors": build_knn,
}


def get_models(names=None):
    """Return a dict {name: fresh unfitted estimator}.

    Parameters
    ----------
    names : list of str or None
        Subset of MODEL_BUILDERS keys to build. None builds all of them.
    """
    if names is None:
        names = list(MODEL_BUILDERS.keys())
    return {name: MODEL_BUILDERS[name]() for name in names}


if __name__ == "__main__":
    for name, model in get_models().items():
        print(f"{name}: {model.__class__.__name__}")
