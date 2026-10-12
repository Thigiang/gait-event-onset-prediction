"""
configs.py -- paths and settings for the gait event onset models.

Paths are read from environment variables, so no machine-specific path is stored in the code:
    GAIT_DATA_DIR      folder with GaitEventWithIndices.mat and ORIGINALDATA/DataPatient1-6.mat   default: ./data
    GAIT_RESULTS_DIR   where trained models, plots and MAE.pkl files are written                  default: ./results
Set them in your shell (export GAIT_DATA_DIR=/path/to/DATA) or in the .env file in this folder.

The settings below are the ones of the Oct 2025 runs; the network sizes of every search stage
match the models saved from those runs.
"""
import os

_repo_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("GAIT_DATA_DIR", os.path.join(_repo_dir, "data"))
RESULTS_DIR = os.environ.get("GAIT_RESULTS_DIR", os.path.join(_repo_dir, "results"))

EVENTS_FILE = os.path.join(DATA_DIR, "GaitEventWithIndices.mat")
SESSION_FILES = [os.path.join(DATA_DIR, "ORIGINALDATA", f"DataPatient{i}.mat") for i in range(1, 7)]

## The 45 columns of every gait cycle, in file order
COLUMN_NAMES = (["DBSLAmp", "DBSLFreq", "DBSRAmp", "DBSRFreq"]
                + [f"{side}Ch{ch}{band}" for side in "LR" for ch in range(1, 5)
                   for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]]
                + ["WPI"])
EVENT_NAMES = ["RTO", "RHS", "LTO"]

## Settings shared by both models
COMMON = {
    "rto_context": 40,          # RTO is predicted from the first 40 samples of the cycle
    "rhs_context": 100,         # RHS from the cycle start up to RTO + 100 samples; LTO up to RHS
    "lstm_units": 64,
    "dropout": 0.2,
    "dense_units": 16,          # Dense(16) -> Dense(3), both linear
    "lr": 0.001,                # Adam
    "epochs": 300,
    "validation_split": 0.1,    # the last 10% of the shuffled training slices
    "search_trials": [1, 2, 3, 4, 5],
    "best_trials": list(range(1, 11)),
}

## Per-model settings. A column is left out of the inputs when its first 4 characters are listed.
MODELS = {
    "model1": {
        "description": "both hemispheres; random 90/10 train/test split of each session's gait cycles",
        "train_excluded": ["LCh1", "RCh2", "WPI"],      # 4 DBS settings + 30 band powers
        "test_excluded": ["LCh1", "RCh2", "WPI"],
        "random_split": True,
        "train_fraction": 0.9,
        "normalize": True,           # z-score with the training set's mean and std
        "loss_weights": [2.5, 2.5, 1.0],
        "dropout_after": "every_block",
        "figure_formats": ["png"],
        "folder": "MODEL1",
        "best_folder": "BestModel",
        "best": {"num_blocks": 2, "cnn_filters": 16, "cnn_kernel_size": 3, "lstm_units": 64},
        ## Hyperparameter search, one stage at a time: (stage folder, searched setting, values, fixed settings)
        "search": [
            ("Blocks", "num_blocks", [1, 2, 3, 4, 5], {"cnn_filters": 64, "cnn_kernel_size": 12}),
            ("CNN/Kernelsize", "cnn_kernel_size", [3, 6, 12], {"num_blocks": 2, "cnn_filters": 16}),
            ("CNN/Filter", "cnn_filters", [3, 16, 32], {"num_blocks": 2, "cnn_kernel_size": 3}),
            ("BiLSTM", "lstm_units", [10, 32, 64, 128, 256], {"num_blocks": 2, "cnn_filters": 16, "cnn_kernel_size": 3}),
        ],
    },
    "model2": {
        "description": "cross-hemisphere; train on the left hemisphere, test on the right, all gait cycles",
        "train_excluded": ["LCh1", "RCh1", "RCh2", "RCh3", "RCh4", "WPI"],   # 4 DBS settings + LCh2-4
        "test_excluded": ["LCh1", "LCh2", "LCh3", "LCh4", "RCh2", "WPI"],    # 4 DBS settings + RCh1, RCh3, RCh4
        "random_split": False,
        "normalize": False,
        "loss_weights": [1.0, 1.0, 1.0],
        "dropout_after": "first_block",
        "figure_formats": ["png", "svg"],
        "folder": "MODEL2",
        "best_folder": "BEST",
        "best": {"num_blocks": 3, "cnn_filters": 16, "cnn_kernel_size": 12, "lstm_units": 64},
        "search": [
            ("Blocks", "num_blocks", [1, 2, 3, 4, 5], {"cnn_filters": 32, "cnn_kernel_size": 3}),
            ("CNN/Kernelsize", "cnn_kernel_size", [3, 6, 12], {"num_blocks": 3, "cnn_filters": 32}),
            ("CNN/Filter", "cnn_filters", [3, 16, 32], {"num_blocks": 3, "cnn_kernel_size": 12}),
            ("BiLSTM", "lstm_units", [10, 32, 64, 128, 256], {"num_blocks": 3, "cnn_filters": 16, "cnn_kernel_size": 12}),
        ],
    },
}


def model_settings(model, **overrides):
    """Return the full settings of a model: COMMON, then the model's own settings, then overrides."""
    settings = {**COMMON, **MODELS[model], **overrides}
    unknown = set(overrides) - set(COMMON) - set(MODELS[model]) - {"num_blocks", "cnn_filters", "cnn_kernel_size"}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    return settings
