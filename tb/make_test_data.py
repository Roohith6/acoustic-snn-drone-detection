"""
make_test_data.py
=================
Converts a real .wav file into a ModelSim-compatible threshold .mem file
for simulation testing in tb_top.v.

The FPGA rate encoder compares a 16-bit LFSR random number against the
threshold for each feature. A higher threshold = higher spike probability.

Feature scaling exactly matches train_native_snn.py:
  1. Extract 640 Delta-MFCCs
  2. StandardScaler (zero mean, unit variance)
  3. Clip to [0, 1] via: x / 6.0 + 0.5
  4. Scale to 16-bit: x * 65535

Usage:
  python make_test_data.py

Outputs (copy to E:\\snn_2\\tb\\):
  threshold_drone.mem    -- real drone audio for ModelSim
  threshold_ambient.mem  -- real ambient audio for ModelSim
"""

import librosa
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DRONE_WAV   = r"E:\snn_2\DroneAudioDataset\Binary_Drone_Audio\yes_drone\B_S2_D1_067-bebop_000_.wav"
AMBIENT_WAV = r"E:\snn_2\DroneAudioDataset\Binary_Drone_Audio\unknown\1-100032-A-00.wav"
OUTPUT_DIR  = r"E:\snn_2\tb"

def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, duration=1.0)
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))

    mfcc    = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40, n_fft=512, hop_length=128)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, mfcc_d, mfcc_dd]).flatten()[:640]
    if len(features) < 640:
        features = np.pad(features, (0, 640 - len(features)))
    return features

def save_mem(features, out_path, label):
    # Standardize (same as training)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    # Clip to [0,1] range (same as training)
    features = np.clip(features / 6.0 + 0.5, 0.0, 1.0)
    # Scale to 16-bit LFSR threshold
    hex_features = (features * 65535).astype(np.uint16)

    with open(out_path, 'w') as f:
        f.write(f"// {label} sample: {out_path}\n")
        for val in hex_features:
            f.write(f"{val:04x}\n")
    print(f"  ✅ Saved {os.path.basename(out_path)}  ({len(hex_features)} values)")

if __name__ == "__main__":
    print("Generating ModelSim test .mem files from real .wav audio...\n")

    drone_feats   = extract_features(DRONE_WAV)
    ambient_feats = extract_features(AMBIENT_WAV)

    save_mem(drone_feats,   os.path.join(OUTPUT_DIR, "threshold_drone.mem"),   "Drone")
    save_mem(ambient_feats, os.path.join(OUTPUT_DIR, "threshold_ambient.mem"), "Ambient")

    print("\n✅ Copy both .mem files to E:\\snn_2\\tb\\ and re-run ModelSim:")
    print("   do sim.do   (or do wave_project.do for wave viewer)")
    print("\nExpected ModelSim output:")
    print("   Inference 1 (key=11 -> sample 00 = ambient): Winner=0 (AMBIENT)")
    print("   Inference 2 (key=10 -> sample 01 = drone):   Winner=1 (DRONE)")
