"""
analyze.py -- summarize the test errors of a search stage or of the final model.

For a search stage: the MAE / MAPE of every searched value, averaged over its trials and then over
RTO, RHS and LTO, printed and saved as bar charts (MAE.png/.svg, MAPE.png/.svg) in the stage folder.
For the final model (--stage best): the mean and standard deviation of each event's MAE and MAPE
over trials 1-10.

Reads the MAE.pkl of <results-dir>/MODEL1|MODEL2/<stage> (default results-dir: GAIT_RESULTS_DIR).

Run:
    python analyze.py --model model2 --stage Blocks
    python analyze.py --model model2 --stage best
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from configs import EVENT_NAMES, MODELS, RESULTS_DIR
from utils import load_results, parse_label, save_figure

STAGES = [stage for stage, *_ in MODELS["model1"]["search"]] + ["best"]
LABELS = {"Blocks": "{} blocks", "CNN/Kernelsize": "Kernel size {}", "CNN/Filter": "{} filters", "BiLSTM": "{} units"}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a search stage or the final model.")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--results-dir", default=RESULTS_DIR, help=f"folder holding MODEL1/ and MODEL2/ (default: {RESULTS_DIR})")
    return parser.parse_args()


def summarize_best(results):
    """Mean and std over trials 1-10 of each event's MAE and MAPE."""
    runs = {parse_label(label)[1]: values for label, values in results.items()}
    trials = [t for t in range(1, 11) if t in runs]
    maes = np.array([runs[t][0] for t in trials])
    mapes = np.array([runs[t][1] for t in trials])
    print(f"Final model, trials {trials}")
    for name, values in [("MAE", maes), ("MAPE", mapes)]:
        print(f"  {name:<4} mean " + " | ".join(f"{e} {m:.3f}" for e, m in zip(EVENT_NAMES, values.mean(axis=0))))
        print(f"  {name:<4} std  " + " | ".join(f"{e} {s:.3f}" for e, s in zip(EVENT_NAMES, values.std(axis=0))))


def summarize_stage(results, stage, folder):
    """Average each searched value over its trials and over the 3 events; print and plot."""
    groups = defaultdict(list)
    for label, values in results.items():
        value, _ = parse_label(label)
        groups[value].append(values)
    labels, mae_means, mape_means = [], [], []
    for value in sorted(groups):
        runs = np.array(groups[value])                # trials x [MAEs, MAPEs] x events
        mae, mape = runs[:, 0].mean(axis=0), runs[:, 1].mean(axis=0)
        print(f"{LABELS[stage].format(value):<16} [{len(runs)} trials]  "
              f"MAE {' | '.join(f'{e} {m:.2f}' for e, m in zip(EVENT_NAMES, mae))} (mean {mae.mean():.2f})   "
              f"MAPE {' | '.join(f'{e} {m:.2f}%' for e, m in zip(EVENT_NAMES, mape))} (mean {mape.mean():.2f}%)")
        labels.append(LABELS[stage].format(value))
        mae_means.append(mae.mean())
        mape_means.append(mape.mean())
    plot_bar(labels, mae_means, "Mean Absolute Error (MAE)", os.path.join(folder, "MAE"))
    plot_bar(labels, mape_means, "Mean Absolute Percentage Error (MAPE)", os.path.join(folder, "MAPE"))
    print(f"\nBar charts saved to {folder}")


def plot_bar(labels, values, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(labels, values, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Models Performance")
    fig.tight_layout()
    save_figure(fig, path, ["png", "svg"])


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    stage_folder = cfg["best_folder"] if args.stage == "best" else args.stage
    folder = os.path.join(args.results_dir, cfg["folder"], stage_folder)
    if not os.path.exists(os.path.join(folder, "MAE.pkl")) and os.path.exists(os.path.join(folder, "ppt", "MAE.pkl")):
        folder = os.path.join(folder, "ppt")          # where some Oct 2025 stages keep their MAE.pkl
    results = load_results(folder)
    if args.stage == "best":
        summarize_best(results)
    else:
        summarize_stage(results, args.stage, folder)


if __name__ == "__main__":
    main()
