# Robustness and Feature Reliability in Photometric Star–Galaxy–Quasar Classification

*A research-style study using the SDSS DR17 photometric dataset.*

> **Status and framing.** This is a research-style project: a careful,
> reproducible study, not a claim of novel published research. Several findings
> (redshift leakage, noise degradation) re-confirm effects that are already
> understood in the astro-ML literature. The contribution here is rigor and
> honesty — measuring these effects carefully on one dataset, with reproducible
> code, and reporting what is actually found rather than only a headline number.

---

## Abstract

Classifying astronomical sources as stars, galaxies, or quasars from survey
photometry is a well-studied task on which classical machine-learning models
routinely report high accuracy. This project asks a different question: not
*how accurate* such a classifier can be, but *how reliable* that accuracy is
under realistic conditions. Using 100,000 labelled SDSS objects and five
photometric bands (u, g, r, i, z), a Random Forest reaches 87.4% accuracy.
Through five experiments — model comparison, feature leakage, noise robustness,
feature ablation, and dataset shift — the study shows that this headline number
is fragile: it collapses to ~52% under a realistic distribution shift, inflates
to 98% when a spectroscopically-derived near-label (redshift) is included, and
degrades sharply under feature noise, with the strongest clean-data model
(Random Forest) proving the *least* noise-robust. The central conclusion is
that clean random-split accuracy substantially overstates real-world
performance, and that robustness must be evaluated explicitly.

---

## 1. Introduction

The Sloan Digital Sky Survey (SDSS) produces large, labelled photometric
catalogues that are a common testbed for classifying objects into stars,
galaxies, and quasars (QSOs). Classical models — especially Random Forests —
perform well on this task, and a single accuracy figure on a held-out random
split is the usual way results are reported.

That figure can be misleading. A random split makes the test set
statistically identical to the training set, so it measures performance under
ideal conditions that real surveys rarely satisfy. Measurements are noisy,
some features may not be available at prediction time, and the population you
want to classify often differs from the population you trained on. A model that
scores 87% on a clean split may behave very differently once any of these holds.

This project reframes the problem from "classify the objects" to **"how much
can the classification be trusted?"** It keeps the modelling deliberately
standard and instead concentrates on stress-testing the result.

### Contributions

- A controlled comparison of four classical models on identical data (§3).
- A quantified demonstration of feature leakage from redshift (§4).
- A noise-degradation analysis across all four models, showing that
  clean-data ranking does not predict noise robustness (§5).
- A feature-ablation and importance-stability study, including a case where
  two standard importance measures disagree (§6).
- A dataset-shift experiment quantifying the generalization gap between
  random-split and distribution-shifted evaluation (§7).

---

## 2. Data and Methods

**Dataset.** SDSS DR17-based catalogue of 100,000 objects. After dropping one
row containing the SDSS `-9999` "invalid measurement" sentinel, 99,999 objects
remain. The classes are imbalanced: 59.4% GALAXY, 21.6% STAR, 19.0% QSO.

**Features.** The five photometric magnitudes u, g, r, i, z form the "clean"
feature set used throughout. Redshift is deliberately excluded from the default
set and studied separately in §4, for the reasons established there.

**Preprocessing.** Rows with the `-9999` sentinel in any photometric band are
dropped, since they are not real measurements. A stratified 80/20 train/test
split (fixed random seed) preserves class proportions and is shared across
experiments so results are comparable.

**Metrics.** Both accuracy and macro-F1 are reported. Because the classes are
imbalanced, accuracy alone is misleading — always predicting "galaxy" would
already score ~59%. Macro-F1 weights all three classes equally and therefore
reflects performance on the rarer stars and quasars, not just the dominant
galaxies.

---

## 3. Experiment 1 — Model Comparison

**Question.** How do standard classical models compare on the clean feature
set, and is 87% good in context?

Four models were trained on the identical split: Logistic Regression (a linear
baseline), Random Forest, Gradient Boosting, and k-Nearest Neighbors.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train time (s) |
|---|---|---|---|---|
| Random Forest | 0.8742 | 0.8390 | 0.8726 | 36.6 |
| k-Nearest Neighbors | 0.8625 | 0.8244 | 0.8607 | 0.11 |
| Gradient Boosting | 0.8175 | 0.7598 | 0.8098 | 42.9 |
| Logistic Regression | 0.7576 | 0.6702 | 0.7366 | 0.30 |

**Findings.** Random Forest is the strongest model (87.4% accuracy, 0.839
macro-F1), confirming its reputation on this task. Two results add useful
context. First, k-Nearest Neighbors comes within ~1.2 points of Random Forest
while training roughly 300× faster (0.11 s vs 36.6 s) — a simple distance-based
model nearly matches the ensemble here. Second, Logistic Regression trails
badly (75.8%), which indicates the classes are not cleanly separable by linear
boundaries; the nonlinear models' advantage comes from capturing curved
decision regions.

The gap between accuracy and macro-F1 in every row reflects the class
imbalance — models do best on the majority galaxy class and worse on the rarer
stars and quasars. This is why macro-F1 is reported alongside accuracy
throughout.

---

## 4. Experiment 2 — Feature Leakage (Redshift)

**Question.** How much does including redshift inflate measured performance,
and why is that inflation misleading?

Redshift is not a photometric measurement. It is derived from spectroscopy —
the same kind of observation used to assign the class labels — and it strongly
tracks class (stars at redshift ~0, quasars at high redshift). Including it as
a feature is close to handing the model a near-label. The same Random Forest was
trained on the clean set and on the clean set plus redshift.

| Feature set | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Clean (u, g, r, i, z) | 0.8742 | 0.8390 | 0.8726 |
| Leaked (+ redshift) | 0.9795 | 0.9761 | 0.9794 |

Feature importances of the leaked model:

| Feature | Importance |
|---|---|
| **redshift** | **0.6598** |
| z | 0.1117 |
| g | 0.0726 |
| u | 0.0614 |
| i | 0.0570 |
| r | 0.0375 |

**Findings.** Adding redshift raises accuracy by 10.5 points (87.4% → 98.0%)
and macro-F1 by even more (0.839 → 0.976). The importance breakdown explains
why: redshift alone accounts for ~66% of the model's decisions — more than all
five photometric bands combined (~0.34). The model has largely stopped
classifying from photometry and is instead reading the near-label. Because
redshift requires the very spectroscopy a photometric classifier is meant to
avoid, the 98% figure does not reflect real-world performance. The honest result
is the 87% photometry-only model. This is the reason redshift is excluded from
every other experiment.

---

## 5. Experiment 3 — Noise Robustness

**Question.** As measurement noise increases, how gracefully does each model
degrade, and does the clean-data ranking still hold?

All four models were trained once on clean data. Zero-mean Gaussian noise was
then added to the *test* features at increasing levels, expressed as multiples
of each feature's own standard deviation (so the perturbation is comparable
across features of different scales). Accuracy was measured at each level.

| Noise (× std) | Logistic Reg. | Random Forest | Gradient Boosting | k-NN |
|---|---|---|---|---|
| 0.0 | 75.75 | **87.42** | 81.75 | 86.25 |
| 0.1 | 73.46 | 77.09 | 80.18 | 78.27 |
| 0.25 | 67.59 | 59.41 | 71.04 | 64.11 |
| 0.5 | 58.31 | 43.49 | 54.05 | 48.81 |
| 1.0 | 49.76 | 35.71 | 38.63 | 40.26 |
| 2.0 | 43.52 | 33.60 | 30.74 | 39.61 |

**Findings.** The headline result is a reversal: **the strongest clean-data
model is the most fragile under noise.** Random Forest starts highest (87.4%)
but degrades fastest, falling below every other model by a noise level of 0.25×
std and ending lowest among the tree models. Logistic Regression, the weakest
model on clean data, degrades most gently in absolute terms and ends highest at
the heaviest noise (43.5%). In other words, clean-data accuracy does not predict
noise robustness — a model can win the benchmark and still be the least
trustworthy once measurements are imperfect. This is exactly the kind of
behaviour a single clean-split number hides.

*(Interpretation to be confident defending: Random Forest's sharp splits on
specific magnitude thresholds are easily flipped by noise, whereas the smoother
linear boundary of Logistic Regression shifts more gradually. This is a
reasonable explanation consistent with the data, not a proven mechanism.)*

---

## 6. Experiment 4 — Feature Ablation and Importance Stability

**Question.** Which bands actually carry the signal, and are the
feature-importance rankings trustworthy or an artifact of one split?

**Ablation (leave-one-out).** Each band was removed in turn and the Random
Forest retrained, measuring the accuracy cost of its absence.

| Removed band | Accuracy | Drop vs full |
|---|---|---|
| (none — all 5) | 0.8742 | — |
| **u** | 0.8498 | **−2.44** |
| r | 0.8568 | −1.74 |
| g | 0.8600 | −1.41 |
| z | 0.8612 | −1.30 |
| i | 0.8663 | −0.79 |

**Importance stability (across 5 seeds).** The Random Forest was retrained
under five different random seeds; the feature importances were collected each
time.

| Feature | Mean importance | Std |
|---|---|---|
| z | 0.2575 | 0.0064 |
| g | 0.2070 | 0.0014 |
| u | 0.1990 | 0.0010 |
| i | 0.1735 | 0.0038 |
| r | 0.1629 | 0.0037 |

The top-ranked feature (z) was identical across all five seeds, and the
standard deviations are very small (≤0.006), so the ranking is stable and
interpretable rather than a fluke of one split.

**Findings — and a genuine disagreement.** The two measures of "importance"
disagree about which band matters most. Gini importance ranks **z** highest,
but leave-one-out ablation shows **u** is the most load-bearing (removing it
costs the most, −2.44 points). This is not a contradiction but a meaningful
distinction: Gini importance rewards a feature for being *used* in many useful
splits, which can inflate a feature that is partly redundant with others;
ablation measures what is actually *lost* when a feature is gone. The most
natural reading is that z scores high on Gini importance but is partly
replaceable by the other bands, whereas u carries unique signal that nothing
else covers — so it ranks lower on Gini importance yet hurts most when removed.
The practical lesson is that built-in feature importance and ablation answer
different questions and should not be treated as interchangeable.

---

## 7. Experiment 5 — Dataset Shift

**Question.** When the model is trained on one regime of the survey and tested
on a different one, how far does performance fall relative to a random split?

A random split makes test data look like training data. To simulate a realistic
distribution shift, the data was split by r-band brightness at the median
(magnitude 20.125) into a bright half and a faint half. The Random Forest was
trained on one regime and tested on the other, in both directions, and compared
against a random-split baseline trained on the same number of rows.

Class balance by regime (showing what shifts):

| Class | Bright | Faint |
|---|---|---|
| GALAXY | 0.600 | 0.589 |
| QSO | 0.107 | 0.272 |
| STAR | 0.293 | 0.139 |

Results:

| Scenario | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| random split (no shift) | 87.21 | 83.62 | 87.04 |
| bright → faint | 53.34 | 44.84 | 53.20 |
| faint → bright | 51.59 | 45.76 | 50.12 |

Generalization gap (accuracy below the no-shift baseline): bright→faint −33.9
points; faint→bright −35.6 points.

**Findings.** This is the strongest result in the study. Under a realistic
brightness-based shift, accuracy collapses from 87% to ~52% — a ~34-point
generalization gap, in both directions. At ~52% the model is barely above the
majority-class baseline (always predicting galaxy ≈ 59% on a balanced-galaxy
split), meaning it has largely failed to transfer.

Crucially, the class-balance table shows this is *not* simply majority-class
label shift: the galaxy fraction is nearly constant across regimes (0.600 vs
0.589). What changes is the star/quasar mix — quasars rise from 11% (bright) to
27% (faint) and stars fall from 29% to 14%, which is physically sensible (faint
objects are more likely to be distant quasars). The model genuinely fails to
generalize across brightness regimes, particularly as the star/quasar
composition inverts.

**Caveat (stated honestly).** A brightness split changes both the feature
distribution (covariate shift) and the class proportions (label shift), and a
single split cannot fully separate the two contributions. The regime class
balances are reported precisely so this limitation is transparent rather than
hidden.

---

## 8. Discussion

A consistent theme runs through all five experiments: the clean random-split
accuracy of 87% is the *most* favourable number this task admits, and almost
any realistic departure from ideal conditions reduces it. Including a
spectroscopic near-label inflates it to a meaningless 98%; injecting feature
noise drives it below 60% by a quarter-std perturbation; and a realistic
brightness shift roughly halves it. The model that wins on clean data (Random
Forest) is simultaneously the least robust to noise.

The implication for real astronomical pipelines is that a single held-out
accuracy is an inadequate summary of a classifier's reliability. Robustness
under noise and distribution shift, and care about which features are genuinely
available at prediction time, matter at least as much as the headline score.

---

## 9. Limitations

- A single dataset (SDSS) is used; cross-survey generalization is not tested.
- Only classical models are compared; no deep-learning baseline at scale.
- Injected noise is synthetic Gaussian, an approximation of real measurement
  error.
- The dataset-shift experiment cannot cleanly separate covariate shift from
  label shift (see §7 caveat).
- Spectroscopic labels are treated as ground truth.

---

## 10. Conclusion

On a thoroughly studied benchmark, a standard Random Forest classifies SDSS
sources at 87% accuracy — but that number is fragile. Across feature leakage,
noise, feature ablation, and dataset shift, this study shows that clean
random-split accuracy substantially overstates real-world performance, and that
the clean-data winner is not necessarily the most reliable model. Performance
should be judged under noise and distribution shift, not just on a clean test
set.