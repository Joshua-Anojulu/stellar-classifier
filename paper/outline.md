# Paper Outline (working draft)

A research-style write-up structure. To be filled in with **what we actually
find**, in our own words — not aspirational claims. Sections marked with the
experiment that produces their evidence.

## Title (working options)
- Robustness and Feature Reliability in Photometric Star–Galaxy–Quasar Classification
- Beyond Test Accuracy: Evaluating ML Source Classifiers Under Noise and Feature Leakage

## 1. Abstract
*(Write last.)* Problem, method (RF + baselines), the robustness experiments,
and the takeaway: that clean test accuracy overstates real-world reliability.

## 2. Introduction
- Background: SDSS photometry, classical ML for source classification.
- Problem statement: not "can we classify?" but "how reliable is the
  classification under realistic data issues?"
- Contributions: controlled multi-model comparison; quantified leakage impact;
  noise-degradation curve; feature ablation; dataset-shift evaluation.

## 3. Related Work
A few themes, to show awareness rather than exhaustive coverage: RF dominance
in SDSS ML; color/magnitude photometric classification; robustness in
scientific ML; dataset shift in surveys.

## 4. Dataset
- Source: SDSS DR17-based, 100k labeled objects.
- Classes: GALAXY / STAR / QSO (imbalanced ~59/22/19%).
- Features: u, g, r, i, z magnitudes.
- Preprocessing: drop -9999 sentinel rows; stratified 80/20 split.

## 5. Methods
- Models: Logistic Regression (baseline), Random Forest (main),
  Gradient Boosting, k-NN.  [exp 01]
- Feature-leakage experiment: clean vs. redshift-included feature sets. [exp 02]
- Noise-robustness: Gaussian noise injection at increasing levels. [exp 03]
- Feature ablation: drop bands, measure performance + importance stability. [exp 04]
- Dataset shift: train on one region/magnitude range, test on another. [exp 05]

## 6. Results
- Model comparison table (accuracy, macro-F1). [exp 01]
- Leakage: inflated vs. honest performance gap. [exp 02]
- Noise-sensitivity curves per model. [exp 03]
- Feature-importance stability + ablation drops. [exp 04]
- Generalization gap under shift. [exp 05]

## 7. Discussion
Why leakage matters in scientific ML; why accuracy alone misleads; what this
implies for real survey pipelines.

## 8. Limitations
Single dataset (SDSS); classical models only; synthetic noise assumptions;
labels treated as ground truth.

## 9. Conclusion
Performance must be judged under shift and noise, not just clean test accuracy.
