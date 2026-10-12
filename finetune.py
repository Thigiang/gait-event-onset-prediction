"""
finetune.py -- run one stage of the hyperparameter search for model 1 or model 2.

The search tunes one setting at a time, each value trained with 5 trials:
    Stage 1   Blocks           number of Conv1D-BiLSTM blocks: 1-5
    Stage 2   CNN/Kernelsize   kernel size: 3, 6, 12
              CNN/Filter       number of filters: 3, 16, 32
              BiLSTM           BiLSTM units: 10, 32, 64, 128, 256
The fixed settings of each stage are listed in configs.py. After a stage, compare its values with
    python analyze.py --model <model> --stage <stage>
(bar charts of the mean MAPE) together with the scatter plots: the best value has reasonable MAPEs
and stable predictions, not only the lowest MAPE. The final model is trained with train.py.

Results go to <GAIT_RESULTS_DIR>/MODEL1/<stage> or <GAIT_RESULTS_DIR>/MODEL2/<stage>.

Run:
    python finetune.py --model model2 --stage Blocks
    python finetune.py --model model1 --stage CNN/Kernelsize --trials 1 2
"""
import os
import argparse

from configs import MODELS, RESULTS_DIR, model_settings
from utils import run_trial

STAGES = [stage for stage, *_ in MODELS["model1"]["search"]]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one hyperparameter-search stage.")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--trials", type=int, nargs="+", help="trial numbers (default: 1-5)")
    return parser.parse_args()


def main():
    args = parse_args()
    stage, setting, values, fixed = next(s for s in MODELS[args.model]["search"] if s[0] == args.stage)
    for value in values:
        settings = model_settings(args.model, **fixed, **{setting: value})
        out_dir = os.path.join(RESULTS_DIR, settings["folder"], stage)
        for trial in args.trials or settings["search_trials"]:
            run_trial(settings, trial, out_dir, label=f"{value}_trial{trial}")


if __name__ == "__main__":
    main()
