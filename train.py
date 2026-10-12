"""
train.py -- train and test the final (best) model of model 1 or model 2, once per trial.

    model1   both hemispheres (4 DBS settings + 30 band powers); random 90/10 split of each session
    model2   cross-hemisphere: train on the left hemisphere (4 DBS + 15 band powers), test on the right

Each trial is seeded with its number. Results go to <GAIT_RESULTS_DIR>/MODEL1/BestModel or
<GAIT_RESULTS_DIR>/MODEL2/BEST: one model, scatter plot and loss curve per trial, and the per-event
test MAE / MAPE of every trial in MAE.pkl. Summarize them with  python analyze.py --model <model> --stage best

Run:
    python train.py --model model1
    python train.py --model model2 --trials 1 2 3
"""
import os
import argparse

from configs import MODELS, RESULTS_DIR, model_settings
from utils import run_trial


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the best model of model 1 or model 2.")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--trials", type=int, nargs="+", help="trial numbers (default: 1-10)")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = model_settings(args.model, **MODELS[args.model]["best"])
    out_dir = os.path.join(RESULTS_DIR, settings["folder"], settings["best_folder"])
    for trial in args.trials or settings["best_trials"]:
        run_trial(settings, trial, out_dir, label=f"best_trial{trial}")


if __name__ == "__main__":
    main()
