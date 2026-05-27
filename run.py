"""
run.py -- experiment runner
===========================
Experiment files are named with numeric prefixes (01_, 02_, ...) so they sort
nicely, but Python can't import modules whose names start with a digit. This
dispatcher runs them by file path instead, so you get clean commands:

    python run.py 01        # runs experiments/01_model_comparison.py
    python run.py list      # shows all available experiments
    python run.py all       # runs every experiment in order

It executes each experiment with the repo root on the path, so the
`from src...` imports inside each experiment resolve correctly.
"""

import runpy
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"


def find_experiments():
    """Return {prefix: path} for every NN_*.py file in experiments/."""
    found = {}
    for p in sorted(EXPERIMENTS_DIR.glob("[0-9][0-9]_*.py")):
        prefix = p.stem.split("_")[0]
        found[prefix] = p
    return found


def run_one(path):
    print(f"\n{'#' * 70}\n# Running {path.name}\n{'#' * 70}")
    # run_name='__main__' so each experiment's `if __name__ == '__main__'` fires.
    runpy.run_path(str(path), run_name="__main__")


def main():
    experiments = find_experiments()

    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print("Available experiments:")
        for prefix, path in experiments.items():
            print(f"  {prefix}  ->  {path.name}")
        print("\nUsage: python run.py <NN | all>")
        return

    arg = sys.argv[1]

    if arg == "all":
        for path in experiments.values():
            run_one(path)
        return

    if arg in experiments:
        run_one(experiments[arg])
    else:
        print(f"No experiment '{arg}'. Run `python run.py list` to see options.")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure repo root is importable as a package root for `from src...`.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
