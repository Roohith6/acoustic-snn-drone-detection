import os
import numpy as np

print("="*70)
print(" Kaggle FPGA-SNN Full Dataset Evaluator")
print("="*70)

# Import the dataset loader from the original software
import train_snn
import audio_to_spike

# 1. Load the real dataset (Expects DroneAudioDataset to be uploaded to Kaggle)
# On Kaggle, dataset paths are usually in /kaggle/input/droneaudiodataset/
DATASET_PATH = "/kaggle/input/droneaudiodataset/Binary_Drone_Audio"
if not os.path.exists(DATASET_PATH):
    print(f"Dataset path not found: {DATASET_PATH}")
    print("Please update DATASET_PATH in this script to point to the Binary_Drone_Audio folder!")
    # Just to prevent a crash if run blindly
    print("Falling back to synthetic data for demonstration...")
    X, y = train_snn.build_synthetic_dataset(n_drone=100, n_ambient=1000)
else:
    print(f"Loading real dataset from {DATASET_PATH}...")
    X, y = train_snn.build_dataset_from_folder(root_dir=DATASET_PATH)

# Standardize the input features exactly as done during training
print("Standardizing features...")
feat_mean = np.mean(X, axis=0)
feat_std = np.std(X, axis=0) + 1e-9
X_norm = (X - feat_mean) / feat_std

# 2. Load Weights
print("Loading trained weights and activations...")
# In hardware we use the "folded" weights which absorb the standardization
W1 = np.load('W1_folded_raw.npy')
b1 = np.load('b1_folded_raw.npy').flatten()
W2 = np.load('W2_raw.npy')
b2 = np.load('b2_raw.npy').flatten()

a1_max = np.load('a1_max.npy').item() if hasattr(np.load('a1_max.npy'), 'item') else np.load('a1_max.npy')[0]
a2_max = np.load('a2_max.npy').item() if hasattr(np.load('a2_max.npy'), 'item') else np.load('a2_max.npy')[0]

# 3. Apply Diehl et al. Threshold Balancing (Hardware Math)
W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)

W1_int = np.round(W1 * W1_scale).astype(np.int64)
b1_int = np.round(b1 * W1_scale).astype(np.int64)
W2_int = np.round(W2 * W2_scale).astype(np.int64)
b2_int = np.round(b2 * W2_scale).astype(np.int64)

V_thresh_h = int(np.round(a1_max * W1_scale))
V_thresh_o = int(np.round(a2_max * W2_scale / a1_max))

print(f"V_thresh_h = {V_thresh_h}, V_thresh_o = {V_thresh_o}")

# 4. Fast Vectorized SNN Inference
print(f"Running Discrete SNN inference on {len(X)} samples...")

N_SAMPLES = len(X)
N_STEPS = 50

class LFSR16:
    def __init__(self, seed=0xACE1):
        self.state = seed & 0xFFFF
    def next(self):
        s = self.state
        bit = ((s>>15)^(s>>13)^(s>>12)^(s>>10)) & 1
        self.state = ((s<<1)|bit) & 0xFFFF
        return self.state

# We use the raw, UN-standardized features for the rate encoder (which expects values in [0,1])
# Wait, train_snn normalizes it before? No, rate encoder uses the original [0,1] spectrogram.
# train_snn's X is the raw spectrogram before standardization!
thresholds = np.clip(X * 0.9, 0.0, 1.0) # max rate 0.9
thresholds_int = (thresholds * 65535).astype(np.uint32)

y_pred_snn = np.zeros(N_SAMPLES, dtype=np.int64)

for i in range(N_SAMPLES):
    sample_thresh = thresholds_int[i]
    v_hid = np.zeros(32, dtype=np.int64)
    v_out = np.zeros(2, dtype=np.int64)
    lfsr = LFSR16(0xACE1)
    
    out_spikes = np.zeros(2, dtype=np.int64)
    
    for t in range(N_STEPS):
        spikes_in = np.array([1 if lfsr.next() < sample_thresh[n] else 0 for n in range(640)], dtype=np.int64)
        z1 = W1_int.T @ spikes_in
        
        v_hid += z1 + b1_int
        hid_spikes = (v_hid >= V_thresh_h).astype(np.int64)
        v_hid = np.where(v_hid >= V_thresh_h, v_hid - V_thresh_h, v_hid)
        v_hid = np.maximum(0, v_hid)
        
        z2 = W2_int.T @ hid_spikes
        
        v_out += z2 + b2_int
        out_s = (v_out >= V_thresh_o).astype(np.int64)
        out_spikes += out_s
        v_out = np.where(v_out >= V_thresh_o, v_out - V_thresh_o, v_out)
        v_out = np.maximum(0, v_out)
        
    y_pred_snn[i] = 1 if out_spikes[1] > out_spikes[0] else 0
    if (i+1) % 1000 == 0:
        print(f" Processed {i+1} / {N_SAMPLES} samples...")

# 5. Compute Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix

print("\n" + "="*70)
print(" SNN HARDWARE EMULATION METRICS ON FULL DATASET")
print("="*70)

acc = accuracy_score(y, y_pred_snn)
prec = precision_score(y, y_pred_snn, zero_division=0)
rec = recall_score(y, y_pred_snn, zero_division=0)
f1 = f1_score(y, y_pred_snn, zero_division=0)
f2 = fbeta_score(y, y_pred_snn, beta=2, zero_division=0)
cm = confusion_matrix(y, y_pred_snn)

print(f"Confusion matrix (rows=true, cols=pred, 0=ambient 1=drone):")
print(cm)
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"F2 Score:  {f2*100:.2f}%")
print("="*70)
