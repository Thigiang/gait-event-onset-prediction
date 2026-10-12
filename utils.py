"""
utils.py -- data preparation, loss, network, training and evaluation shared by both models.

Data preparation (prepare_data) for one trial:
    1. load the gait cycles of the 6 sessions and keep the model's input columns
    2. model1: split each session's cycles randomly into 90% train / 10% test, then z-score the
       inputs with the training set's mean and std (computed over the zero-padded cycles)
       model2: train on every cycle with the left-hemisphere columns, test on every cycle with
       the right-hemisphere columns (no normalization)
    3. cut every cycle into three event windows (slice_by_event), pad them to one length and
       shuffle the training windows
"""
import os
import pickle
import re

import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

from configs import COLUMN_NAMES, EVENT_NAMES, EVENTS_FILE, SESSION_FILES


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def column_indices(excluded_prefixes):
    """Indices of the columns whose first 4 characters are not in excluded_prefixes."""
    return [i for i, name in enumerate(COLUMN_NAMES) if name[:4] not in excluded_prefixes]


def load_sessions():
    """Return [(cycles, targets)] per session: cycles is a list of (timesteps, 45) arrays and
    targets an (n_cycles, 3) array of RTO, RHS and LTO onsets relative to the cycle start."""
    events = scipy.io.loadmat(EVENTS_FILE)["GaitEventWithIndices"]
    sessions = []
    for idx, path in enumerate(SESSION_FILES):
        mat = scipy.io.loadmat(path)
        data = mat[list(mat.keys())[-1]]              # the cycles are the last variable in the file
        cycles = [data[i][0] for i in range(data.shape[0])]
        sessions.append((cycles, events[idx][0][:, 4:]))
    return sessions


def pad_cycles(cycles, columns, length):
    """Stack cycles into a (n_cycles, length, n_columns) array, zero-padded at the end."""
    padded = np.zeros((len(cycles), length, len(columns)))
    for i, cycle in enumerate(cycles):
        padded[i, :cycle.shape[0], :] = cycle[:, columns]
    return padded


def slice_by_event(X, y, rto_context, rhs_context):
    """Cut every cycle into one input window per event.

    RTO is predicted from the first rto_context samples, RHS from the start up to RTO + rhs_context,
    and LTO from the start up to RHS. Each target row holds that event's onset and NaN elsewhere.
    Windows that are empty or reach the end of the padded cycle are skipped.
    """
    windows, targets = [], []
    for i in range(X.shape[0]):
        for j in range(3):
            if j == 0:
                end = rto_context
            elif j == 1:
                end = int(y[i, 0]) + rhs_context
            else:
                end = int(y[i, 1])
            if end <= 0 or end >= X.shape[1]:
                continue
            windows.append(X[i][:end])
            target = np.full((3,), np.nan, dtype=np.float32)
            target[j] = int(y[i, j])
            targets.append(target)
    return windows, targets


def prepare_data(settings):
    """Return (train_X, train_y, test_X, test_y) event windows for one trial; seed NumPy first."""
    sessions = load_sessions()
    max_length = max(cycle.shape[0] for cycles, _ in sessions for cycle in cycles)
    train_columns = column_indices(settings["train_excluded"])
    test_columns = column_indices(settings["test_excluded"])

    if settings["random_split"]:
        parts = {"train_X": [], "train_y": [], "test_X": [], "test_y": []}
        for cycles, targets in sessions:
            data = pad_cycles(cycles, train_columns, max_length)
            indices = np.random.permutation(data.shape[0])
            n_train = int(settings["train_fraction"] * data.shape[0])
            parts["train_X"].append(data[indices[:n_train]])
            parts["train_y"].append(targets[indices[:n_train]])
            parts["test_X"].append(data[indices[n_train:]])
            parts["test_y"].append(targets[indices[n_train:]])
        train_X, train_y, test_X, test_y = (np.concatenate(parts[k]) for k in ["train_X", "train_y", "test_X", "test_y"])
    else:
        cycles = [cycle for session_cycles, _ in sessions for cycle in session_cycles]
        targets = np.concatenate([session_targets for _, session_targets in sessions])
        train_X, train_y = pad_cycles(cycles, train_columns, max_length), targets
        test_X, test_y = pad_cycles(cycles, test_columns, max_length), targets

    if settings["normalize"]:
        flat = train_X.reshape(-1, train_X.shape[-1])
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        train_X, test_X = (train_X - mean) / std, (test_X - mean) / std

    train_windows, train_targets = slice_by_event(train_X, train_y, settings["rto_context"], settings["rhs_context"])
    test_windows, test_targets = slice_by_event(test_X, test_y, settings["rto_context"], settings["rhs_context"])
    max_window = max(w.shape[0] for w in train_windows + test_windows)
    train_X = pad_sequences(train_windows, maxlen=max_window, padding="post", dtype="float32")
    test_X = pad_sequences(test_windows, maxlen=max_window, padding="post", dtype="float32")
    train_y, test_y = np.array(train_targets), np.array(test_targets)

    shuffle = np.random.permutation(len(train_X))
    return train_X[shuffle], train_y[shuffle], test_X, test_y


# ---------------------------------------------------------------------------
# Loss and network
# ---------------------------------------------------------------------------
def masked_mae_loss(y_true, y_pred, weights=None):
    """MAE over the non-NaN targets (each window has only its own event's onset), optionally weighted per event."""
    mask = tf.cast(tf.math.logical_not(tf.math.is_nan(y_true)), tf.float32)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    loss = tf.abs(y_true - y_pred) * mask
    if weights is not None:
        loss = loss * tf.constant(weights, dtype=tf.float32)
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def build_model(settings, num_features, num_outputs=3):
    """Masking -> num_blocks x (Conv1D -> BatchNorm -> BiLSTM) -> Dense(dense_units) -> Dense(num_outputs).

    Dropout follows every block (dropout_after = "every_block", model 1) or only the first block
    ("first_block", model 2). The last BiLSTM returns only its final state.
    """
    inputs = tf.keras.Input(shape=(None, num_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    num_blocks = settings["num_blocks"]
    for block in range(num_blocks):
        x = tf.keras.layers.Conv1D(filters=settings["cnn_filters"], kernel_size=settings["cnn_kernel_size"])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(settings["lstm_units"], return_sequences=block < num_blocks - 1))(x)
        if settings["dropout_after"] == "every_block" or block == 0:
            x = tf.keras.layers.Dropout(settings["dropout"])(x)
    x = tf.keras.layers.Dense(settings["dense_units"])(x)
    outputs = tf.keras.layers.Dense(num_outputs)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    weights = settings["loss_weights"]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["lr"]),
                  loss=lambda y_true, y_pred: masked_mae_loss(y_true, y_pred, weights=weights),
                  metrics=[masked_mae_loss])
    return model


# ---------------------------------------------------------------------------
# Evaluation and plots
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, figure_path, formats):
    """Return per-event (MAEs, MAPEs) on the test windows and save a predicted-vs-true scatter plot."""
    maes, mapes = [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (event, ax) in enumerate(zip(EVENT_NAMES, axes)):
        valid = ~np.isnan(y_true[:, i])
        true, pred = y_true[valid, i], y_pred[valid, i]
        mae = np.mean(np.abs(true - pred))
        mape = np.mean(np.abs((true - pred) / np.maximum(np.abs(true), 1e-8))) * 100
        maes.append(mae)
        mapes.append(mape)

        ax.scatter(true, pred, alpha=0.7)
        ax.plot([true.min(), true.max()], [true.min(), true.max()], "b--")
        ax.text(0.05, 0.95, f"MAE = {mae:.3f}", transform=ax.transAxes, verticalalignment="top")
        ax.text(0.05, 0.85, f"MAPE = {mape:.3f}%", transform=ax.transAxes, verticalalignment="top")
        ax.set_xlabel("True Start Time")
        ax.set_ylabel("Predicted Start Time")
        ax.set_title(f"{event} Index Prediction")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, figure_path, formats)
    return maes, mapes


def plot_training_history(history, figure_path, formats):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, figure_path, formats)


def save_figure(fig, path, formats):
    """Save fig as path.<format> for every format (path has no extension) and close it."""
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Running and results
# ---------------------------------------------------------------------------
def run_trial(settings, trial, out_dir, label):
    """Train and test one model with `settings`, seeded with `trial`; save it and record its metrics.

    The model, scatter plot and loss curve are saved in out_dir as <label>.*, and the test
    MAEs / MAPEs are added to out_dir/MAE.pkl under <label>.
    """
    tf.keras.utils.set_random_seed(trial)         # seeds the split, the shuffling and the weights
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {label}: {settings['folder']} {os.path.relpath(out_dir)} ===\n")

    train_X, train_y, test_X, test_y = prepare_data(settings)
    print(f"train {train_X.shape} -> {train_y.shape}, test {test_X.shape} -> {test_y.shape}")
    model = build_model(settings, num_features=train_X.shape[-1], num_outputs=train_y.shape[1])
    history = model.fit(train_X, train_y, validation_split=settings["validation_split"], epochs=settings["epochs"])

    path = os.path.join(out_dir, label)
    model.save(f"{path}.keras")
    plot_training_history(history, f"{path}_trainval_loss", settings["figure_formats"])
    maes, mapes = evaluate(test_y, model.predict(test_X), path, settings["figure_formats"])
    print(f"{label}: MAE {np.round(maes, 3)}, MAPE {np.round(mapes, 3)}%")
    update_results(out_dir, label, maes, mapes)


def update_results(out_dir, label, maes, mapes):
    """Add one run's per-event [MAEs, MAPEs] to out_dir/MAE.pkl (the format of the Oct 2025 results)."""
    path = os.path.join(out_dir, "MAE.pkl")
    results = load_results(out_dir) if os.path.exists(path) else {}
    results[label] = [list(map(float, maes)), list(map(float, mapes))]
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_results(folder):
    """Return the {label: [MAEs, MAPEs]} dictionary stored in folder/MAE.pkl."""
    with open(os.path.join(folder, "MAE.pkl"), "rb") as f:
        return pickle.load(f)


def parse_label(label):
    """(setting value or None, trial) from a run label, e.g. '64_trial1' -> (64, 1),
    '3_blocks_trial_2' -> (3, 2), 'Bestmodel_trial_7' -> (None, 7)."""
    numbers = [int(n) for n in re.findall(r"\d+", label)]
    return (numbers[0] if len(numbers) > 1 else None), numbers[-1]
