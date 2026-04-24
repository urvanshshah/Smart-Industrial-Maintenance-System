"""
Global Configuration for Smart Industrial Maintenance System
=============================================================
Central configuration for paths, hyperparameters, and constants.

All constants are defined exactly once. Edit the values here to retarget
the pipeline at a different dataset, asset class, or hardware budget — the
rest of the codebase reads from this module.
"""

import os
import random as _random

import numpy as _np
import torch

# =============================================================================
# Random Seed (set once, applied globally)
# =============================================================================
RANDOM_SEED = 42

_random.seed(RANDOM_SEED)
_np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

IMS_RAW_DIR = os.path.join(DATA_DIR, "raw_ims")
IMS_PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ims")

for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR,
           IMS_RAW_DIR, IMS_PROCESSED_DIR]:
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# Device & DataLoader
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# =============================================================================
# C-MAPSS Dataset
# =============================================================================
CMAPSS_DATASET = "behrad3d/nasa-cmaps"           # Kaggle dataset identifier
CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant or near-constant across all subsets
SENSORS_TO_DROP = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                   "sensor_16", "sensor_18", "sensor_19"]

# Operational settings to discard (near-constant in FD001)
OP_SETTINGS_TO_DROP = ["op_setting_3"]

# Sensor columns actually fed to the models
ACTIVE_SENSORS = [f"sensor_{i}" for i in range(1, 22)
                  if f"sensor_{i}" not in SENSORS_TO_DROP]

# =============================================================================
# Preprocessing
# =============================================================================
SEQUENCE_LENGTH = 30          # Sliding window length (cycles)
MAX_RUL = 125                 # Piecewise-linear RUL cap (NASA convention)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# Synthetic Augmentation (C-MAPSS)
# =============================================================================
SYNTHETIC_AUGMENT = True
SYNTHETIC_TARGET_RATIO = 0.30        # synthetic = 30% of total training data
SYNTHETIC_NOISE_LEVEL = 0.03         # Gaussian noise relative to feature std
SYNTHETIC_DEGRADATION_MODELS = ["exponential", "linear", "polynomial"]

# =============================================================================
# Feature Engineering (XGBoost / Survival)
# =============================================================================
ROLLING_WINDOWS = [5, 10, 20]
ROLLING_STATS = ["mean", "std", "min", "max"]

# =============================================================================
# LSTM Autoencoder (Anomaly Detection)
# =============================================================================
AE_HIDDEN_DIM = 128
AE_LATENT_DIM = max(16, AE_HIDDEN_DIM // 2)
AE_NUM_LAYERS = 2
AE_DROPOUT = 0.2
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 40
AE_BATCH_SIZE = 256

# Threshold strategy: "f1_optimal" tunes against true near-failure labels on
# the validation set; "sigma" falls back to mean + AE_ANOMALY_THRESHOLD_SIGMA
# * std of validation-healthy scores.
AE_THRESHOLD_STRATEGY = "f1_optimal"
AE_ANOMALY_THRESHOLD_SIGMA = 3.0

# =============================================================================
# LSTM Failure Predictor
# =============================================================================
PRED_HIDDEN_DIM = 256
PRED_NUM_LAYERS = 3
PRED_DROPOUT = 0.4
PRED_LEARNING_RATE = 1e-3
PRED_EPOCHS = 20
PRED_BATCH_SIZE = 256            # Larger batch — exploits GPU + improves stability
PRED_FAILURE_HORIZON = 30        # Predict failure within h cycles

# Imbalanced-classification objective: "focal" (focal loss, γ = PRED_FOCAL_GAMMA)
# or "bce" (BCEWithLogitsLoss with dynamic pos_weight).
PRED_LOSS = "focal"
PRED_FOCAL_GAMMA = 2.0
PRED_FOCAL_ALPHA = 0.75          # weight on the positive (failure) class

# Risk aggregation: when computing a single risk score per machine, average
# the predictor's output over the last K sequences instead of taking the
# single most-recent one. This avoids degenerate "everything is critical"
# saturation on test units that are by definition end-of-life.
RISK_AGGREGATION_LAST_K = 5

# =============================================================================
# XGBoost RUL
# =============================================================================
XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
}
if torch.cuda.is_available():
    XGB_PARAMS["device"] = "cuda"
    XGB_PARAMS["tree_method"] = "hist"

# =============================================================================
# Bayesian Survival Analysis
# =============================================================================
SURVIVAL_CONFIDENCE_LEVELS = [0.90, 0.95]
SURVIVAL_MAX_PREDICTION = MAX_RUL * 5    # clip predict_median outputs to this
                                         # ceiling so a few badly extrapolated
                                         # rows don't blow up RMSE.

# =============================================================================
# MILP Optimization
# =============================================================================
# Defaults sized for the C-MAPSS demo fleet (~107 test engines). Total
# capacity = MAX_CONCURRENT_CREWS * SCHEDULING_HORIZON = 12 * 20 = 240 slots,
# which comfortably absorbs the predictor's critical backlog. The dashboard's
# Interactive Optimizer page lets the user override these per scenario without
# editing this file.
MAX_CONCURRENT_CREWS = 12           # Max simultaneous maintenance jobs per slot
DOWNTIME_COST_PER_HOUR = 10000      # $ per hour of unplanned downtime
MAINTENANCE_COST_BASE = 2000        # $ base maintenance cost (per service event)
SAFETY_RISK_THRESHOLD = 0.7         # Risk above this = mandatory service
SCHEDULING_HORIZON = 20             # Time slots in the planning window

# =============================================================================
# Risk Categories
# =============================================================================
RISK_LEVELS = {
    "critical": {"threshold": 0.7, "label": "Service Immediately", "color": "#FF4444"},
    "elevated": {"threshold": 0.4, "label": "Schedule Soon",       "color": "#FFAA00"},
    "normal":   {"threshold": 0.0, "label": "Continue Monitoring", "color": "#44BB44"},
}

# =============================================================================
# IMS Bearing Dataset
# =============================================================================
IMS_DATASET = "vinayak123tyagi/bearing-dataset"

IMS_SAMPLING_RATE = 20480           # 20 kHz, 1-second snapshots → 20,480 points
IMS_SNAPSHOT_LENGTH = 20480

IMS_EXPERIMENTS = {
    1: {
        "channels": 8,              # 2 per bearing (X + Y axis)
        "bearings": 4,
        "failed_bearings": [3, 4],
        "failure_modes": ["inner_race", "roller_element"],
        "folder": "1st_test",
    },
    2: {
        "channels": 4,              # 1 per bearing
        "bearings": 4,
        "failed_bearings": [1],
        "failure_modes": ["outer_race"],
        "folder": "2nd_test",
    },
    3: {
        "channels": 4,              # 1 per bearing
        "bearings": 4,
        "failed_bearings": [3],
        "failure_modes": ["outer_race"],
        "folder": "3rd_test",
    },
}

IMS_MAX_RUL = 125                   # Pseudo-RUL cap (snapshots)
IMS_SEQUENCE_LENGTH = 30            # Sliding window for LSTM (snapshots)
IMS_FFT_BANDS = 5                   # Frequency bands for spectral energy
IMS_ROLLING_WINDOWS = [10, 50, 100] # Rolling windows for trend features


def print_system_info():
    """Print device and configuration information."""
    print(f"[CONFIG] Using device: {DEVICE}")
    print(f"[CONFIG] Batch size: {AE_BATCH_SIZE} (AE) / {PRED_BATCH_SIZE} (Pred)")
    print(f"[CONFIG] Hidden dim: {AE_HIDDEN_DIM} (AE) / {PRED_HIDDEN_DIM} (Pred)")
    print(f"[CONFIG] Predictor loss: {PRED_LOSS}")
    print(f"[CONFIG] DataLoader workers: {NUM_WORKERS}")
