"""
LSTM Hyperparameter Tuning Script
====================================
Systematically searches for the optimal batch_size and epoch count
for both LSTM models:
  - Model 1: LSTM Temporal Autoencoder (anomaly detection)
  - Model 2: LSTM Failure Predictor (failure probability)

Loads preprocessed data from data/processed/ and runs a grid search,
recording validation metrics for each combination.

Saves results incrementally so partial runs are never lost.

Usage:
    python scripts/tune_lstm_hyperparams.py
"""

import os
import sys
import time
import csv
import numpy as np
import torch

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.autoencoder import LSTMAutoencoder, AutoencoderTrainer
from src.models.lstm_predictor import LSTMPredictor, PredictorTrainer

# Suppress matplotlib GUI popups during batch training
import matplotlib
matplotlib.use("Agg")

# ============================================================================
# Hyperparameter Grid - Optimized for 15-minute CPU run
# ============================================================================
BATCH_SIZES = [256]
EPOCH_COUNTS = [20, 40]
DATA_SUBSET_RATIO = 0.1  # Use 10% of data for fast tuning

# Output paths
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "tuning_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

AE_RESULTS_CSV = os.path.join(RESULTS_DIR, "autoencoder_tuning.csv")
PRED_RESULTS_CSV = os.path.join(RESULTS_DIR, "predictor_tuning.csv")

# ============================================================================
# CSV helpers — write incrementally so partial runs survive cancellation
# ============================================================================
AE_FIELDS = [
    "batch_size", "epochs", "train_loss_final", "val_loss_best",
    "anomaly_threshold", "test_anomaly_rate", "training_time_sec", "error",
]
PRED_FIELDS = [
    "batch_size", "epochs", "best_f1", "best_auc", "best_precision",
    "best_recall", "training_time_sec", "stopped_epoch", "error",
]


def _load_completed(csv_path, fields):
    """Return set of (batch_size, epochs) already completed."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((int(row["batch_size"]), int(row["epochs"])))
    return done


def _append_row(csv_path, fields, row):
    """Append a single result row (creates header if file is new)."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Data Loading
# ============================================================================
def load_preprocessed_data():
    """Load the already-preprocessed train/val/test splits."""
    print("[TUNING] Loading preprocessed data...")

    data = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(config.PROCESSED_DATA_DIR, f"{split}_data.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessed data not found at {path}. "
                "Run 'python scripts/train_all.py' first."
            )
        loaded = np.load(path)
        data[split] = {k: loaded[k] for k in loaded.files}

    X_train = data["train"]["X"]
    y_train_rul = data["train"]["y_rul"]
    y_train_binary = data["train"]["y_binary"]
    X_val = data["val"]["X"]
    y_val_rul = data["val"]["y_rul"]
    y_val_binary = data["val"]["y_binary"]
    X_test = data["test"]["X"]
    y_test_rul = data["test"]["y_rul"]

    n_features = X_train.shape[2]

    # Subset data for fast tuning
    if DATA_SUBSET_RATIO < 1.0:
        print(f"[TUNING] Subsetting data to {DATA_SUBSET_RATIO:.0%}...")
        rng = np.random.default_rng(config.RANDOM_SEED)
        
        # Subset training
        idx_train = rng.choice(len(X_train), int(len(X_train) * DATA_SUBSET_RATIO), replace=False)
        X_train = X_train[idx_train]
        y_train_rul = y_train_rul[idx_train]
        y_train_binary = y_train_binary[idx_train]
        
        # Subset validation
        idx_val = rng.choice(len(X_val), int(len(X_val) * DATA_SUBSET_RATIO), replace=False)
        X_val = X_val[idx_val]
        y_val_rul = y_val_rul[idx_val]
        y_val_binary = y_val_binary[idx_val]

    print(f"[TUNING] Data loaded & subsetted: train={X_train.shape}, val={X_val.shape}, "
          f"test={X_test.shape}, features={n_features}")

    return {
        "X_train": X_train,
        "y_train_rul": y_train_rul,
        "y_train_binary": y_train_binary,
        "X_val": X_val,
        "y_val_rul": y_val_rul,
        "y_val_binary": y_val_binary,
        "X_test": X_test,
        "y_test_rul": y_test_rul,
        "n_features": n_features,
    }


# ============================================================================
# Autoencoder Tuning
# ============================================================================
def tune_autoencoder(data):
    """Grid search over batch_size x epochs for the LSTM Autoencoder."""
    print("\n" + "=" * 70)
    print("  TUNING: LSTM AUTOENCODER (Model 1)")
    print("=" * 70)

    n_features = data["n_features"]
    X_train = data["X_train"]
    y_train_rul = data["y_train_rul"]
    X_val = data["X_val"]
    y_val_rul = data["y_val_rul"]
    X_test = data["X_test"]

    # Healthy subsets (same filtering as train_all.py)
    healthy_mask = y_train_rul > config.MAX_RUL * 0.5
    X_healthy = X_train[healthy_mask]
    X_val_ae = X_val[y_val_rul > config.MAX_RUL * 0.5] if len(X_val) > 0 else None

    # Skip already-completed combos (resume support)
    done = _load_completed(AE_RESULTS_CSV, AE_FIELDS)
    grid = [(bs, ep) for bs in BATCH_SIZES for ep in EPOCH_COUNTS
            if (bs, ep) not in done]

    total_all = len(BATCH_SIZES) * len(EPOCH_COUNTS)
    print(f"[AE TUNING] Healthy training samples: {len(X_healthy)}")
    print(f"[AE TUNING] Grid: {total_all} total combos, "
          f"{len(done)} already done, {len(grid)} remaining\n")

    if not grid:
        print("[AE TUNING] All combinations already completed!")
        return _read_all_results(AE_RESULTS_CSV)

    results = []
    for run_idx, (batch_size, epochs) in enumerate(grid, 1):
        print(f"\n{'-' * 60}")
        print(f"[AE TUNING] Run {run_idx}/{len(grid)}: "
              f"batch_size={batch_size}, epochs={epochs}")
        print(f"{'-' * 60}")

        # Reset seed for reproducibility
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        start_time = time.time()

        try:
            model = LSTMAutoencoder(input_dim=n_features)
            trainer = AutoencoderTrainer(
                model, epochs=epochs, batch_size=batch_size
            )
            trainer.train(
                X_healthy,
                X_val=X_val_ae,
                X_val_threshold=X_val,
                y_val_threshold=y_val_rul,
            )

            elapsed = time.time() - start_time

            # Compute test anomaly scores
            test_scores = model.compute_anomaly_score(
                torch.FloatTensor(X_test)
            )
            test_anomaly_rate = float(
                np.mean(test_scores > model.threshold)
            )

            train_loss_final = (
                trainer.train_history[-1]
                if trainer.train_history else float("nan")
            )
            val_loss_best = (
                min(trainer.val_history)
                if trainer.val_history else float("nan")
            )

            result = {
                "batch_size": batch_size,
                "epochs": epochs,
                "train_loss_final": round(train_loss_final, 8),
                "val_loss_best": round(val_loss_best, 8),
                "anomaly_threshold": round(float(model.threshold), 8),
                "test_anomaly_rate": round(test_anomaly_rate, 6),
                "training_time_sec": round(elapsed, 1),
                "error": "",
            }

            print(f"[AE TUNING] OK val_loss_best={val_loss_best:.8f}, "
                  f"test_anomaly_rate={test_anomaly_rate:.4f}, "
                  f"time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "batch_size": batch_size,
                "epochs": epochs,
                "train_loss_final": float("nan"),
                "val_loss_best": float("nan"),
                "anomaly_threshold": float("nan"),
                "test_anomaly_rate": float("nan"),
                "training_time_sec": round(elapsed, 1),
                "error": str(e),
            }
            print(f"[AE TUNING] FAILED: {e}")

        # Save immediately (survives cancellation)
        _append_row(AE_RESULTS_CSV, AE_FIELDS, result)
        results.append(result)

        # Free memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = _read_all_results(AE_RESULTS_CSV)
    _print_ae_summary(all_results)
    return all_results


def _read_all_results(csv_path):
    """Read all rows from a results CSV."""
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                # Convert numeric fields
                for k in row:
                    if k in ("batch_size", "epochs"):
                        row[k] = int(row[k])
                    elif k in ("training_time_sec",):
                        row[k] = float(row[k])
                    elif k == "error":
                        pass
                    elif k == "stopped_epoch":
                        row[k] = int(row[k]) if row[k] and row[k] != "None" else None
                    else:
                        try:
                            row[k] = float(row[k])
                        except (ValueError, TypeError):
                            pass
                rows.append(row)
    return rows


def _print_ae_summary(results):
    """Print a formatted summary of autoencoder tuning results."""
    print("\n" + "=" * 80)
    print("  AUTOENCODER TUNING RESULTS")
    print("=" * 80)
    print(f"{'Batch':>7} {'Epochs':>7} {'Val Loss Best':>15} "
          f"{'Anomaly Rate':>14} {'Time (s)':>10}")
    print("-" * 80)

    valid = [r for r in results if r.get("error", "") == ""]

    for r in sorted(valid, key=lambda x: x["val_loss_best"]):
        print(f"{r['batch_size']:>7} {r['epochs']:>7} "
              f"{r['val_loss_best']:>15.8f} "
              f"{r['test_anomaly_rate']:>13.4f} "
              f"{r['training_time_sec']:>10.1f}")

    if valid:
        best = min(valid, key=lambda x: x["val_loss_best"])
        print(f"\n>> BEST: batch_size={best['batch_size']}, "
              f"epochs={best['epochs']} "
              f"-> val_loss={best['val_loss_best']:.8f}")


# ============================================================================
# Predictor Tuning
# ============================================================================
def tune_predictor(data):
    """Grid search over batch_size x epochs for the LSTM Failure Predictor."""
    print("\n" + "=" * 70)
    print("  TUNING: LSTM FAILURE PREDICTOR (Model 2)")
    print("=" * 70)

    n_features = data["n_features"]
    X_train = data["X_train"]
    y_train_binary = data["y_train_binary"]
    X_val = data["X_val"]
    y_val_binary = data["y_val_binary"]

    # Skip already-completed combos (resume support)
    done = _load_completed(PRED_RESULTS_CSV, PRED_FIELDS)
    grid = [(bs, ep) for bs in BATCH_SIZES for ep in EPOCH_COUNTS
            if (bs, ep) not in done]

    total_all = len(BATCH_SIZES) * len(EPOCH_COUNTS)
    print(f"[PRED TUNING] Training samples: {len(X_train)} "
          f"(pos_rate={y_train_binary.mean():.2%})")
    print(f"[PRED TUNING] Grid: {total_all} total combos, "
          f"{len(done)} already done, {len(grid)} remaining\n")

    if not grid:
        print("[PRED TUNING] All combinations already completed!")
        return _read_all_results(PRED_RESULTS_CSV)

    results = []
    for run_idx, (batch_size, epochs) in enumerate(grid, 1):
        print(f"\n{'-' * 60}")
        print(f"[PRED TUNING] Run {run_idx}/{len(grid)}: "
              f"batch_size={batch_size}, epochs={epochs}")
        print(f"{'-' * 60}")

        # Reset seed for reproducibility
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        start_time = time.time()

        try:
            model = LSTMPredictor(input_dim=n_features)
            trainer = PredictorTrainer(
                model, epochs=epochs, batch_size=batch_size,
                early_stopping_patience=None,
            )
            trainer.train(
                X_train, y_train_binary,
                X_val, y_val_binary,
            )

            elapsed = time.time() - start_time

            val_hist = trainer.val_history
            if val_hist:
                best_entry = max(val_hist, key=lambda m: m.get("f1", 0))
                best_f1 = best_entry.get("f1", 0)
                best_auc = best_entry.get("auc", 0)
                best_precision = best_entry.get("precision", 0)
                best_recall = best_entry.get("recall", 0)
            else:
                best_f1 = best_auc = best_precision = best_recall = 0

            result = {
                "batch_size": batch_size,
                "epochs": epochs,
                "best_f1": round(best_f1, 6),
                "best_auc": round(best_auc, 6),
                "best_precision": round(best_precision, 6),
                "best_recall": round(best_recall, 6),
                "training_time_sec": round(elapsed, 1),
                "stopped_epoch": trainer.stopped_epoch,
                "error": "",
            }

            print(f"[PRED TUNING] OK F1={best_f1:.4f}, "
                  f"AUC={best_auc:.4f}, "
                  f"P={best_precision:.4f}, R={best_recall:.4f}, "
                  f"time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "batch_size": batch_size,
                "epochs": epochs,
                "best_f1": float("nan"),
                "best_auc": float("nan"),
                "best_precision": float("nan"),
                "best_recall": float("nan"),
                "training_time_sec": round(elapsed, 1),
                "stopped_epoch": None,
                "error": str(e),
            }
            print(f"[PRED TUNING] FAILED: {e}")

        # Save immediately (survives cancellation)
        _append_row(PRED_RESULTS_CSV, PRED_FIELDS, result)
        results.append(result)

        # Free memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = _read_all_results(PRED_RESULTS_CSV)
    _print_pred_summary(all_results)
    return all_results


def _print_pred_summary(results):
    """Print a formatted summary of predictor tuning results."""
    print("\n" + "=" * 90)
    print("  PREDICTOR TUNING RESULTS")
    print("=" * 90)
    print(f"{'Batch':>7} {'Epochs':>7} {'Best F1':>10} "
          f"{'Best AUC':>10} {'Precision':>11} {'Recall':>8} {'Time (s)':>10}")
    print("-" * 90)

    valid = [r for r in results if r.get("error", "") == ""]

    for r in sorted(valid, key=lambda x: x["best_f1"], reverse=True):
        print(f"{r['batch_size']:>7} {r['epochs']:>7} "
              f"{r['best_f1']:>10.4f} "
              f"{r['best_auc']:>10.4f} "
              f"{r['best_precision']:>11.4f} "
              f"{r['best_recall']:>8.4f} "
              f"{r['training_time_sec']:>10.1f}")

    if valid:
        best = max(valid, key=lambda x: x["best_f1"])
        print(f"\n>> BEST: batch_size={best['batch_size']}, "
              f"epochs={best['epochs']} "
              f"-> F1={best['best_f1']:.4f}, AUC={best['best_auc']:.4f}")


# ============================================================================
# Main
# ============================================================================
def main():
    total_start = time.time()

    print("=" * 70)
    print("  LSTM HYPERPARAMETER TUNING - BATCH SIZE & EPOCHS")
    print("=" * 70)
    print(f"  Device: {config.DEVICE}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Epoch counts: {EPOCH_COUNTS}")
    print(f"  Total runs: {len(BATCH_SIZES) * len(EPOCH_COUNTS) * 2} "
          f"({len(BATCH_SIZES) * len(EPOCH_COUNTS)} per model)")
    print(f"  Results saved incrementally (resume-safe)")
    print()

    # Load data once
    data = load_preprocessed_data()

    # Tune Autoencoder (Model 1)
    ae_results = tune_autoencoder(data)

    # Tune Predictor (Model 2)
    pred_results = tune_predictor(data)

    # Final summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  AE results:   {AE_RESULTS_CSV}")
    print(f"  Pred results:  {PRED_RESULTS_CSV}")

    ae_valid = [r for r in ae_results if r.get("error", "") == ""]
    pred_valid = [r for r in pred_results if r.get("error", "") == ""]

    if ae_valid:
        best_ae = min(ae_valid, key=lambda x: x["val_loss_best"])
        print(f"\n  >> Best Autoencoder: batch_size={best_ae['batch_size']}, "
              f"epochs={best_ae['epochs']} "
              f"-> val_loss={best_ae['val_loss_best']:.8f}")

    if pred_valid:
        best_pred = max(pred_valid, key=lambda x: x["best_f1"])
        print(f"  >> Best Predictor:   batch_size={best_pred['batch_size']}, "
              f"epochs={best_pred['epochs']} "
              f"-> F1={best_pred['best_f1']:.4f}, AUC={best_pred['best_auc']:.4f}")

    print()


if __name__ == "__main__":
    main()
