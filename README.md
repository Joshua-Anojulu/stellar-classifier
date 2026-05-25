# Stellar Object Classification (SDSS)

A machine learning project that classifies astronomical objects as **galaxies,
stars, or quasars** using only their photometric colors, trained on 100,000
objects from the Sloan Digital Sky Survey (SDSS).

**Full analysis:** see `stellar_classification.ipynb`

## Overview

Large sky surveys image far more objects than can ever be examined with
spectroscopy, so they rely on *photometric classification* — predicting an
object's type from broadband brightness measurements — to decide which objects
deserve expensive follow-up. This project implements that idea: a Random Forest
classifier that distinguishes galaxies, stars, and quasars from five filter
magnitudes (u, g, r, i, z).

## Key Decision: Excluding Redshift

The dataset includes a `redshift` column that correlates almost perfectly with
object type (quasars are very distant, stars are local). Including it pushes
accuracy to ~98%, but the model would essentially be handed the answer rather
than learning the underlying physics — a form of data leakage.

I deliberately **excluded redshift** and trained on photometric colors alone.
This makes for a harder, more honest problem where the model must actually learn
the color patterns that separate the classes.

## Results

- **~87% accuracy** using only the five photometric magnitudes
  (vs. a ~59% baseline from always guessing the majority class)
- Galaxies are easiest to classify; **quasars and stars are the hardest to
  separate**, which reflects real astrophysics — quasars are point-like in
  imaging, just like stars, and historically required spectroscopy to tell apart
- **Feature importance** analysis shows which filters carried the most
  discriminating power
- Adding redshift back was tested as a comparison and confirmed the ~98% jump,
  illustrating why that feature is a near-giveaway

## Robustness Test

Because real surveys are dominated by faint objects with noisier photometry than
the bright, cleanly-measured training data, I tested how the model degrades as
measurement noise increases. Accuracy holds up under small noise but falls
sharply as noise grows — toward the majority-class baseline, then toward random
guessing — honestly demonstrating the gap between clean benchmark performance and
real-world survey conditions.

## Tech Stack

- Python, scikit-learn (Random Forest), pandas, numpy
- Developed and run in Google Colab

## Possible Extensions

- A confidence threshold that flags low-certainty objects for spectroscopic
  follow-up (mirroring real survey triage)
- Testing on a different survey's data to measure true generalization
- An interactive web app that serves the model for live predictions
