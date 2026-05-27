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
│   ├── 02_feature_leakage.py      # [planned] quantify redshift leakage
│   ├── 03_noise_robustness.py     # [planned] accuracy vs injected noise
│   ├── 04_feature_ablation.py     # [planned] which bands matter
│   └── 05_dataset_shift.py        # [planned] train/test on different regions
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

### 02–05 *(planned)*
Feature leakage, noise robustness, feature ablation, and dataset shift — each
will follow the same module pattern: import the shared loader, run one focused
study, write figures/tables to `results/`.

## Status

This is an in-progress project. Experiment 01 is complete; the robustness
experiments are being added incrementally so each is understood before the next
is built.
