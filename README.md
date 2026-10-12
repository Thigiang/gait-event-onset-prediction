# Predicting gait event onsets from brain signals under DBS

Conv1D–BiLSTM models that predict the onset of three gait events — right toe-off (RTO), right heel-strike (RHS) and left toe-off (LTO) — from deep brain stimulation (DBS) settings and brain-signal band powers of a Parkinson's disease patient, recorded over 6 walking sessions (524 gait cycles).

| Model | Inputs | Train / test |
|---|---|---|
| **model1** | 4 DBS settings + 30 band powers (LCh2–4, RCh1, RCh3, RCh4) | random 90/10 split of each session's gait cycles |
| **model2** (cross-hemisphere) | 4 DBS settings + 15 band powers | train on the left hemisphere (LCh2–4), test on the right (RCh1, RCh3, RCh4), all gait cycles |

Each gait cycle gives one input window per event: RTO is predicted from the first 40 samples, RHS from the cycle start up to RTO + 100 samples, and LTO from the cycle start up to RHS.

## Network

`Masking → N × (Conv1D → BatchNorm → BiLSTM) → Dense(16) → Dense(3)`, trained with a masked MAE loss and Adam (learning rate 0.001) for 300 epochs. Dropout (0.2) follows every block in model 1 and only the first block in model 2.

## Results (Oct 2025)

Hyperparameters were tuned one at a time, 5 trials per value: number of blocks (1–5), then CNN kernel size (3, 6, 12), number of filters (3, 16, 32) and BiLSTM units (10, 32, 64, 128, 256). The best value was chosen from the mean MAPE and the stability of the scatter plots.

| Model | Best network | Test MAPE, mean of 10 trials (RTO / RHS / LTO) |
|---|---|---|
| model1 | 2 blocks, 16 filters, kernel size 3, 64 BiLSTM units | 5.71% / 2.31% / 2.39% |
| model2 | 3 blocks, 16 filters, kernel size 12, 64 BiLSTM units | 5.69% / 2.29% / 1.26% |

## Files

| File | Purpose |
|---|---|
| [configs.py](configs.py) | Data and results paths, and every model and search setting |
| [utils.py](utils.py) | Data preparation, loss, network, training and evaluation shared by both models |
| [train.py](train.py) | Train and test the best model of model1 or model2 (trials 1–10) |
| [finetune.py](finetune.py) | Run one stage of the hyperparameter search (trials 1–5 per value) |
| [analyze.py](analyze.py) | Mean errors of a search stage (with bar charts) or of the final model |
| [requirements.txt](requirements.txt) | Python packages |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The data is not included. Point `GAIT_DATA_DIR` at the folder that holds `GaitEventWithIndices.mat` and `ORIGINALDATA/DataPatient1-6.mat`, in your shell or in the `.env` file (git-ignored):

```bash
export GAIT_DATA_DIR=/path/to/DATA
```

Outputs go to `./results` (or `GAIT_RESULTS_DIR`).

## How to run

```bash
# hyperparameter search, one stage at a time
python finetune.py --model model2 --stage Blocks
python analyze.py  --model model2 --stage Blocks          # bar charts of the mean MAE / MAPE

# final model, 10 trials
python train.py   --model model2
python analyze.py --model model2 --stage best
```

Stages: `Blocks`, `CNN/Kernelsize`, `CNN/Filter`, `BiLSTM`. Each trial is seeded with its number, so every run can be repeated exactly.

## Notes

- Model 1 normalizes its inputs with the mean and standard deviation of the zero-padded training cycles, as in the original runs.
- The event windows (40 / 100 samples) are confirmed for model 2: its saved best models reproduce their recorded errors with these settings. Model 1 uses the same values, but its random test split cannot be recreated to check them.
- In the Oct 2025 model-2 block search, the 5-block network had an extra dropout layer after its 4th block; `finetune.py` builds it with dropout after the first block only, like the other block counts.
- The standard deviations on the slides were computed from the MAE values; `analyze.py` reports the MAE and MAPE standard deviations separately.
