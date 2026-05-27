# Stellar Classifier: Robustness and Feature Reliability in Photometric Source Classification

A research-style study of how classical machine-learning models classify
astronomical sources (stars, galaxies, and quasars) from Sloan Digital Sky
Survey (SDSS) photometry — and, more importantly, how *reliable* those
classifications are under realistic conditions like feature leakage, noise,
and missing features.

This is a **research-style project**: a single notebook has been restructured
into a reproducible repository where every experiment is a self-contained,
re-runnable module producing figures and tables. The goal is rigor and honesty,
not a claim of novel published research.

## The central question

Most SDSS classification projects report a single accuracy number on a clean
test set. This project asks the more useful question: **how much can you trust
that number?** Specifically:

- Does the headline accuracy depend on a leaked feature (redshift)?
- How fast does performance degrade as measurement noise increases?
- Which features actually carry the signal, and how stable is that?
- Does a model trained on one part of the survey generalize to another?

## Dataset

SDSS DR17-based dataset of 100,000 labeled objects (`star_classification.csv`,
available on Kaggle). Each object has photometric magnitudes in five bands
(`u`, `g`, `r`, `i`, `z`) and a spectroscopic `class` label: `GALAXY`, `STAR`,
or `QSO` (quasar). The classes are imbalanced (~59% / 22% / 19%).

**The CSV is not committed** (it is large). Download it from Kaggle and place
it at `data/star_classification.csv`.

### A note on preprocessing

SDSS uses `-9999` as a sentinel for invalid measurements. `src/data_loader.py`
drops any row containing one in a photometric band, so they never pollute
training — a small but defensible improvement over a naive load.

## Why macro-F1, not just accuracy

Because the classes are imbalanced, raw accuracy is misleading: always
predicting "galaxy" already scores ~59%. This project reports **macro-F1**
(which weights every class equally) alongside accuracy, so the harder rare
classes (quasars, stars) are not hidden behind the dominant one.

## Repository layout

```
stellar-classifier/
├── README.md
├── requirements.txt
├── run.py                  # experiment runner (see below)
├── data/                   # place star_classification.csv here (gitignored)
├── src/
│   ├── data_loader.py      # load, clean, and split the data (shared by all)
│   ├── models.py           # the four models being compared
│   └── evaluate.py         # shared metrics + plotting
├── experiments/
│   ├── 01_model_comparison.py     # [done] compare 4 models on clean features
│   ├── 02_feature_leakage.py      # [done] quantify redshift leakage
│   ├── 03_noise_robustness.py     # [done] accuracy vs injected noise
│   ├── 04_feature_ablation.py     # [done] which bands matter + stability
│   └── 05_dataset_shift.py        # [done] train/test on different regions
├── results/                # figures and tables are written here
└── paper/
    └── outline.md          # the write-up structure
```

## Setup

```bash
pip install -r requirements.txt
# then download star_classification.csv into data/
```

## Running experiments

Experiment files are numbered for ordering. Use the runner:

```bash
python run.py list     # show available experiments
python run.py 01       # run the model comparison
python run.py all      # run everything in order
```

(The runner exists because Python can't import modules whose names start with
a digit — it executes them by path instead.)

## Experiments

### 01 — Model comparison *(implemented)*
Trains Logistic Regression, Random Forest, Gradient Boosting, and k-NN on the
identical clean split and reports accuracy, macro-F1, weighted-F1, and training
time. Establishes the baseline context that a single-model notebook lacks.

### 02 — Feature leakage *(implemented)*
Trains the same Random Forest with and without `redshift` and reports the gap.
Redshift is a spectroscopically-derived near-label (stars ~0, quasars high), so
adding it inflates accuracy without teaching the model to classify from
photometry. The experiment shows the accuracy jump, the per-class change, and a
feature-importance chart where redshift dwarfs the five photometric bands.

### 03 — Noise robustness *(implemented)*
Trains all four models on clean data, then measures accuracy as Gaussian noise
is injected into the test features at increasing levels (expressed as multiples
of each feature's own standard deviation, so the noise is comparable across
features). Produces a degradation curve per model — revealing which models
stay stable under noise and which collapse, a distinction invisible from clean
test accuracy alone.

### 04 — Feature ablation & importance stability *(implemented)*
Two parts. Ablation: drop each photometric band in turn and measure the
accuracy cost, revealing which bands are load-bearing. Stability: retrain the
Random Forest across multiple random seeds and check whether the
feature-importance ranking holds — a ranking that flips between runs can't be
interpreted, so confirming it's stable is what makes any "this band matters
most" claim meaningful.

### 05 — Dataset shift *(implemented)*
The most realistic test in the study. Every other experiment used a random
split where test data looks like training data; real surveys aren't like that.
This splits the data by `r`-band brightness and trains on one regime while
testing on the other (bright→faint and faint→bright), comparing against a
random-split baseline of equal training size. The gap between them is the
generalization gap — how much random-split accuracy overstates real-world
performance. The class balance of each regime is reported so the covariate-
vs-label-shift distinction is transparent.

## Status

All five experiments are implemented. The project is a complete research-style
study: a single notebook restructured into a reproducible, modular repository
with five focused experiments on model comparison, feature leakage, noise
robustness, feature ablation, and dataset shift.

This is a rigorous portfolio/research-style project, not a claim of novel
published research. Several findings (e.g. redshift leakage, noise degradation)
re-confirm known results carefully; the value is in the rigor, reproducibility,
and honest analysis.