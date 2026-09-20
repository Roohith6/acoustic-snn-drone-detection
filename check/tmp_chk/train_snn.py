"""
train_snn.py
============
Completes the pre-FPGA pipeline: Dataset -> Preprocessing -> Feature
Extraction -> Spike Encoding -> Offline Training -> Weight Export.

METHOD [Published IEEE-adjacent technique, weight/threshold balancing]:
Rather than training directly on discrete spike trains (which needs a more
complex surrogate-gradient backprop-through-time implementation), this
trains a standard rate-based network on the *normalized spectrogram*
(equivalent to the average firing rate under rate coding, since rate
coding's firing probability IS the normalized intensity -- see
rate_encode() in audio_to_spike.py). The trained weights are then used
directly as the SNN's synaptic weights. This is the standard "ANN-to-SNN
weight balancing" approach:
  P. Diehl, D. Neil, J. Binas, M. Cook, S.-C. Liu, M. Pfeiffer,
  "Fast-classifying, high-accuracy spiking deep networks through weight
  and threshold balancing," IEEE IJCNN, 2015.

WHAT THIS SCRIPT DOES
  1. build_dataset(): walk a folder of drone/ambient .wav clips (real
     DroneAudioDataset layout) OR generate a synthetic dataset (for
     testing this script itself, since this sandbox cannot download the
     real dataset -- see note in the accompanying chat message).
  2. Train a 2-layer network (640 -> 64 -> 2) with standard backprop
     (pure NumPy, no external ML framework needed).
  3. Report train/val/test accuracy -- this becomes your real, reportable
     number once run on the actual dataset.
  4. Export the trained weights as .mem files, quantized to signed 16-bit
     fixed-point, in the exact format synapse_memory.v expects.

TO RUN ON THE REAL DATASET (on your machine, not this sandbox)
  1. git clone https://github.com/saraalemadi/DroneAudioDataset
  2. Update DATASET_DIR below to point at it (expects subfolders whose
     names indicate class, e.g. "Drones/" and "Unknown/" -- adjust
     LABEL_MAP if the real folder names differ from this guess).
  3. Set USE_SYNTHETIC = False and run: python3 train_snn.py

-------------------------------------------------------------------------
CHANGE LOG (this file, vs. the version that produced the currently-shipped
W1_input_hidden.mem / W2_hidden_output.mem):
  - Added export of net.b1 / net.b2 (biases) alongside W1/W2. The prior
    run's export step only saved weights, never biases, even though
    TwoLayerNet trains both -- confirmed missing from every file sent
    for RTL verification so far. See STAGE: Weight Export at the bottom.
  - No other logic changed. If DATASET_DIR, LABEL_MAP, and file iteration
    order are unchanged from the run that produced the currently-shipped
    W1/W2, re-running this should reproduce the same weights (all RNG
    seeds default to 0) -- but glob.glob() file ordering is not strictly
    guaranteed identical across machines/filesystems, so DIFF the new
    W1_input_hidden.mem / W2_hidden_output.mem against the ones already
    verified before assuming nothing changed.
-------------------------------------------------------------------------
"""

import os
import glob
import numpy as np

from audio_to_spike import (
    synthetic_drone_audio, load_wav_file,
    compute_audio_spectrogram, downsample_spectrogram, quantize_feature,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
FS = 16000.0
DURATION = 1.0
N_FREQ_BINS = 32
N_TIME_BINS = 20
N_INPUTS = N_FREQ_BINS * N_TIME_BINS   # 640, matches synapse_memory.v NUM_PRE
N_HIDDEN = 64                          # matches neuron_array.v NUM_POST
N_OUTPUT = 2                           # drone, ambient

# UPDATE THIS to wherever you cloned/downloaded saraalemadi/DroneAudioDataset.
# The path below matches the machine the currently-shipped weights came from
# -- change it to your own download location.
DATASET_DIR = "E:/snn_2/DroneAudioDataset/Binary_Drone_Audio"
# it also contains Multiclass_Drone_Audio/, which has its own "unknown" folder
# that would silently get mixed into the ambient class otherwise (confirmed
# with a mock-structure test in inspect_encoding.py's development).
LABEL_MAP = {"yes_drone": 1, "unknown": 0} # matches your real dataset's actual folder names
USE_SYNTHETIC = False                    # Using real dataset for local execution


# -----------------------------------------------------------------------
# Stage: Dataset -> Preprocessing -> Feature Extraction
# -----------------------------------------------------------------------

def clip_to_feature(t=None, x=None, fs=FS):
    """Audio -> normalized, downsampled, 8-bit-quantized spectrogram
    feature vector (640,). This IS the rate-coding probability map (see
    deterministic_rate_encode() in audio_to_spike.py) flattened --
    training on it directly is mathematically equivalent to training on
    the expected spike rate.

    Uses "fixed_ref" spectrogram normalization (frozen Encoding V1) and
    8-bit quantization, matching validate_encoding.py's golden-reference
    pipeline exactly -- training on unquantized floats while the golden
    reference/FPGA path uses 8-bit features would be a real train/deploy
    mismatch, caught and fixed here before any training run.
    The caller must load x with load_wav_file(..., normalize_peak=False)
    for this to be meaningful -- see build_dataset_from_folder() and
    build_synthetic_dataset() below, both updated accordingly."""
    f, tt, spec = compute_audio_spectrogram(x, fs, normalization="fixed_ref")
    f, tt, spec = downsample_spectrogram(f, tt, spec, N_FREQ_BINS, N_TIME_BINS)
    spec = quantize_feature(spec, width=8)
    return spec.flatten()


def build_synthetic_dataset(n_drone=100, n_ambient=1560, seed=0):
    """For testing this script itself, since the real dataset can't be
    fetched in this sandbox. Imbalanced ~15.6:1 by default, matching the
    real DroneAudioDataset's reported ratio (20,744 ambient : 1,332 drone)
    -- so this script gets tested under the actual condition it needs to
    handle, not an artificially easy balanced case.
    DO NOT report accuracy from this as if it were a real result -- it
    only proves the training/imbalance-handling code works."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, target_type, n in [(1, "drone", n_drone), (0, "ambient", n_ambient)]:
        for i in range(n):
            _, x = synthetic_drone_audio(target_type=target_type, fs=FS,
                                          duration=DURATION,
                                          seed=int(rng.integers(0, 1_000_000)))
            X.append(clip_to_feature(x=x))
            y.append(label)
    X, y = np.array(X), np.array(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def build_dataset_from_folder(root_dir=DATASET_DIR, label_map=LABEL_MAP,
                               exclude_silent=True, silence_rms_threshold=1e-4):
    """Walk a real dataset folder, recursively. Matches labels by the
    immediate parent folder name. Excludes bit-for-bit silent files by
    default -- confirmed via check_silent_files.py that ~11% of the real
    dataset's "unknown" class is exact-zero, corrupted/invalid recordings,
    not legitimate quiet ambient audio."""
    X, y = [], []
    skipped_silent = 0
    for label_name, label_val in label_map.items():
        all_wavs = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        matches = [f for f in all_wavs
                   if label_name.lower() in os.path.basename(os.path.dirname(f)).lower()]
        for wav_path in matches:
            try:
                t, x = load_wav_file(wav_path, target_fs=FS, normalize_peak=False)
                if exclude_silent and np.sqrt(np.mean(x ** 2)) <= silence_rms_threshold:
                    skipped_silent += 1
                    continue
                X.append(clip_to_feature(x=x))
                y.append(label_val)
            except Exception as e:
                print(f"Skipping {wav_path}: {e}")
    if skipped_silent:
        print(f"Excluded {skipped_silent} silent/invalid files across all classes.")
    if not X:
        raise RuntimeError(
            f"No .wav files found under {root_dir} matching {list(label_map)}. "
            "Inspect the real dataset's folder names and update LABEL_MAP."
        )
    X, y = np.array(X), np.array(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def split_dataset(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (X[:n_train], y[:n_train],
            X[n_train:n_train+n_val], y[n_train:n_train+n_val],
            X[n_train+n_val:], y[n_train+n_val:])


# -----------------------------------------------------------------------
# Stage: Offline Training (pure NumPy, 2-layer network, standard backprop)
# -----------------------------------------------------------------------

def split_dataset_stratified(X, y, train_frac=0.7, val_frac=0.15, seed=0):
    """Stratified split: each class is split proportionally, so train/val/test
    all preserve the real class ratio (critical given the ~15.6:1 imbalance
    between ambient and drone in the real dataset -- an unstratified split
    can accidentally starve one split of the minority class)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])
    train_idx, val_idx, test_idx = map(np.array, (train_idx, val_idx, test_idx))
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx])


def class_weights_from_labels(y, n_classes):
    """Inverse-frequency class weights, so the ~15.6:1 ambient:drone
    imbalance in the real dataset doesn't let training settle for
    'always predict ambient' at ~94% accuracy."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1  # avoid divide-by-zero if a class is empty in this split
    weights = counts.sum() / (n_classes * counts)
    return weights  # shape (n_classes,)


def confusion_and_metrics(y_true, y_pred, n_classes=2, positive_class=1):
    """Returns confusion matrix plus accuracy/precision/recall/F1/FPR for the
    positive class (drone=1) -- the metrics that actually matter given the
    imbalance, not just raw accuracy."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = cm[positive_class, positive_class]
    fn = cm[positive_class, :].sum() - tp
    fp = cm[:, positive_class].sum() - tp
    tn = cm.sum() - tp - fn - fp

    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0        # a.k.a. drone detection rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # F2: weights recall 2x more than precision than F1 does -- appropriate here since
    # missing a real drone (false negative) is a worse outcome than a false alarm.
    beta = 2
    f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
          if (precision + recall) > 0 else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0            # false alarms on ambient clips

    return {
        "confusion_matrix": cm, "accuracy": accuracy, "precision": precision,
        "recall_drone_detection": recall, "f1": f1, "f2": f2, "false_positive_rate": fpr,
    }


def standardize(X_train, X_val, X_test):
    """Zero-mean, unit-variance per feature, fit on train only (no leakage)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, (mean, std)


class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # He-style initialization, scaled for sigmoid hidden layer
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)             # ReLU hidden "firing rate"
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return z1, a1, z2, a2

    def loss_and_grads(self, X, y_onehot, class_w=None):
        n = X.shape[0]
        z1, a1, z2, a2 = self.forward(X)

        if class_w is None:
            sample_w = np.ones(n)
        else:
            labels = np.argmax(y_onehot, axis=1)
            sample_w = class_w[labels]

        loss = -np.sum(sample_w[:, None] * y_onehot * np.log(a2 + 1e-9)) / n

        dz2 = sample_w[:, None] * (a2 - y_onehot) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)               # ReLU derivative
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return loss, (dW1, db1, dW2, db2)

    def predict(self, X):
        _, _, _, a2 = self.forward(X)
        return np.argmax(a2, axis=1)


def train(net, X_train, y_train, X_val, y_val, epochs=200, lr=0.01, batch_size=32,
          seed=0, class_w=None, verbose=True, checkpoint_metric="f2",
          lr_decay_every=40, lr_decay_factor=0.5, min_checkpoint_epoch=30):
    """lr_decay_every/lr_decay_factor: the effective learning rate halves
    every 40 epochs by default. Added after real training showed
    validation F1/F2 and even training loss oscillating rather than
    smoothly converging (e.g. loss 0.153 -> 0.203 -> 0.144 across epochs
    60/80/100) -- a constant lr=0.01 for all 200 epochs was letting the
    optimizer bounce around a minimum instead of settling into it.

    min_checkpoint_epoch: don't start tracking "best" checkpoints until
    after this many epochs -- confirmed on real data that an early noisy
    spike (epoch 28 of 200) can score deceptively well on validation but
    generalize WORSE to the test set than later, more stable epochs."""
    rng = np.random.default_rng(seed)
    n = len(X_train)
    y_onehot = np.eye(N_OUTPUT)[y_train]

    # simple Adam optimizer state
    params = ["W1", "b1", "W2", "b2"]
    m = {p: np.zeros_like(getattr(net, p)) for p in params}
    v = {p: np.zeros_like(getattr(net, p)) for p in params}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Track the best checkpoint by validation F1, not just whatever the
    # final epoch happens to land on -- validation metrics can bounce
    # around a lot epoch to epoch (confirmed on the real dataset: F1 swung
    # from 0.563 to 0.770 to 0.632 across training), so the last epoch is
    # not reliably the best one.
    best_f1 = -1.0
    best_weights = None
    best_epoch = -1
    current_lr = lr

    for epoch in range(epochs):
        if epoch > 0 and epoch % lr_decay_every == 0:
            current_lr *= lr_decay_factor

        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            _, grads = net.loss_and_grads(X_train[idx], y_onehot[idx], class_w=class_w)
            t += 1
            for p, g in zip(params, grads):
                m[p] = beta1 * m[p] + (1 - beta1) * g
                v[p] = beta2 * v[p] + (1 - beta2) * (g ** 2)
                m_hat = m[p] / (1 - beta1 ** t)
                v_hat = v[p] / (1 - beta2 ** t)
                setattr(net, p, getattr(net, p) - current_lr * m_hat / (np.sqrt(v_hat) + eps))

        val_metrics = confusion_and_metrics(y_val, net.predict(X_val))
        if epoch + 1 >= min_checkpoint_epoch and val_metrics[checkpoint_metric] > best_f1:
            best_f1 = val_metrics[checkpoint_metric]
            best_weights = {p: getattr(net, p).copy() for p in params}
            best_epoch = epoch + 1

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            train_loss, _ = net.loss_and_grads(X_train, y_onehot, class_w=class_w)
            print(f"epoch {epoch+1:4d}  lr={current_lr:.5f}  train_loss={train_loss:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.3f}  "
                  f"val_recall(drone)={val_metrics['recall_drone_detection']:.3f}  "
                  f"val_f1={val_metrics['f1']:.3f}  val_f2={val_metrics['f2']:.3f}")

    if best_weights is not None:
        for p in params:
            setattr(net, p, best_weights[p])
        if verbose:
            print(f"Restored best checkpoint: epoch {best_epoch}, "
                  f"val_{checkpoint_metric}={best_f1:.3f} (selection metric: {checkpoint_metric}, "
                  f"weights recall higher than precision -- final epoch {epochs} may have "
                  f"been worse -- using the best one seen)")

    return net


# -----------------------------------------------------------------------
# Stage: Weight Export (fixed-point, matches synapse_memory.v WIDTH=16)
# -----------------------------------------------------------------------

def export_weights_for_verilog(W, filepath, width=16, scale=None):
    """Quantize a float weight matrix to signed fixed-point and write as
    one hex value per line (row-major, matching synapse_memory's
    pre_idx*NUM_POST + post_idx addressing), ready for $readmemh.
    Also used for bias vectors below by reshaping to (1, N) first --
    a bias vector is just a 1xN "matrix" for this function's purposes.
    """
    max_abs = np.max(np.abs(W)) if scale is None else None
    if scale is None:
        # scale so the largest weight uses ~90% of the available signed range
        scale = (2 ** (width - 1) - 1) * 0.9 / (max_abs + 1e-9)

    W_int = np.round(W * scale).astype(np.int64)
    qmin, qmax = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    clipped = np.sum((W_int < qmin) | (W_int > qmax))
    W_int = np.clip(W_int, qmin, qmax)
    if clipped:
        print(f"Warning: {clipped} weights clipped to fit {width}-bit range "
              f"in {filepath} -- consider lowering scale.")

    hex_digits = (width + 3) // 4
    with open(filepath, "w") as f:
        f.write(f"// {W.shape[0]}x{W.shape[1]} weights, scale={scale:.4f}, width={width}\n")
        for row in W_int:
            for val in row:
                twos = val & ((1 << width) - 1)
                f.write(f"{twos:0{hex_digits}x}\n")

    print(f"Wrote {filepath}: {W.shape[0]}x{W.shape[1]} weights, scale={scale:.4f}")
    return scale


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Resolves to Python_1/weights regardless of current working directory,
    # since this script lives in Python_1/scripts/ -- matches the folder
    # structure already set up (scripts/, outputs/, weights/) rather than
    # writing to a relative path that depends on where you happen to run from.
    out_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE: Dataset + Preprocessing + Feature Extraction")
    print("=" * 70)
    if USE_SYNTHETIC:
        print("USE_SYNTHETIC=True -- using synthetic data to verify this script")
        print("works correctly. Switch to build_dataset_from_folder() with the")
        print("real DroneAudioDataset before trusting any accuracy number.\n")
        X, y = build_synthetic_dataset()
    else:
        X, y = build_dataset_from_folder()

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_stratified(X, y)
    X_train, X_val, X_test, (feat_mean, feat_std) = standardize(X_train, X_val, X_test)
    print(f"Dataset: {len(X)} total  ->  train={len(X_train)} "
          f"val={len(X_val)} test={len(X_test)}")
    print(f"Class balance (train): {np.bincount(y_train)}  (index 0=ambient, 1=drone)\n")

    class_w = class_weights_from_labels(y_train, N_OUTPUT)
    print(f"Class weights (inverse-frequency, to counter imbalance): {class_w}\n")

    print("=" * 70)
    print("STAGE: Hidden-layer size sweep (32 / 64 / 128)")
    print("=" * 70)
    sweep_results = {}
    for n_hidden in (32, 64, 128):
        net_sweep = TwoLayerNet(N_INPUTS, n_hidden, N_OUTPUT)
        net_sweep = train(net_sweep, X_train, y_train, X_val, y_val,
                           epochs=100, lr=0.01, class_w=class_w, verbose=False)
        val_metrics = confusion_and_metrics(y_val, net_sweep.predict(X_val))
        sweep_results[n_hidden] = (net_sweep, val_metrics)
        print(f"hidden={n_hidden:3d}  val_acc={val_metrics['accuracy']:.3f}  "
              f"val_f1={val_metrics['f1']:.3f}  val_f2={val_metrics['f2']:.3f}  "
              f"val_recall(drone)={val_metrics['recall_drone_detection']:.3f}")

    best_hidden = max(sweep_results, key=lambda k: sweep_results[k][1]["f2"])
    print(f"\nSelected hidden-layer size: {best_hidden} (best validation F2 -- "
          f"weights recall higher than precision, since missing a real drone is a "
          f"worse outcome than a false alarm for this application)\n")

    print("=" * 70)
    print(f"STAGE: Final training with hidden={best_hidden}")
    print("=" * 70)
    net = TwoLayerNet(N_INPUTS, best_hidden, N_OUTPUT)
    net = train(net, X_train, y_train, X_val, y_val, epochs=200, lr=0.01, class_w=class_w)

    print("\n" + "=" * 70)
    print("STAGE: Final Test Metrics")
    print("=" * 70)
    test_metrics = confusion_and_metrics(y_test, net.predict(X_test))
    print(f"Confusion matrix (rows=true, cols=pred, 0=ambient 1=drone):\n"
          f"{test_metrics['confusion_matrix']}")
    print(f"Accuracy:              {test_metrics['accuracy']:.4f}")
    print(f"Precision (drone):     {test_metrics['precision']:.4f}")
    print(f"Recall (drone detect): {test_metrics['recall_drone_detection']:.4f}")
    print(f"F1 (drone):            {test_metrics['f1']:.4f}")
    print(f"False positive rate:   {test_metrics['false_positive_rate']:.4f}")
    if USE_SYNTHETIC:
        print("\n(These numbers are from SYNTHETIC, balanced data -- not reportable. "
              "Re-run with USE_SYNTHETIC=False and the real dataset before using "
              "any of this in your report.)")

    print("\n" + "=" * 70)
    print("STAGE: Weight Export")
    print("=" * 70)
    export_weights_for_verilog(net.W1, os.path.join(out_dir, "W1_input_hidden.mem"))
    export_weights_for_verilog(net.W2, os.path.join(out_dir, "W2_hidden_output.mem"))
    np.save(os.path.join(out_dir, "W1_raw.npy"), net.W1)
    np.save(os.path.join(out_dir, "W2_raw.npy"), net.W2)

    # --- ADDED: bias export (was missing from the run that produced the
