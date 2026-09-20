import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_features(audio_path):
    sample_rate, data = wav.read(audio_path)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    data /= (np.max(np.abs(data)) + 1e-9)
    if len(data) < sample_rate:
        data = np.pad(data, (0, sample_rate - len(data)))
    else:
        data = data[:sample_rate]
    f, t, Sxx = spectrogram(data, fs=sample_rate, nperseg=256, noverlap=192)
    return Sxx.flatten()

def build_dataset_from_folder(root_dir, max_files_per_class=None):
    label_map = {"unknown": 0, "yes_drone": 1}
    X, y = [], []
    excluded_count = 0
    for cls_name, cls_label in label_map.items():
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        files = sorted(os.listdir(cls_dir))
        if max_files_per_class: files = files[:max_files_per_class]
        for f in files:
            if not f.endswith('.wav'): continue
            try:
                feat = extract_features(os.path.join(cls_dir, f))
                if np.sum(feat) < 1e-6:
                    excluded_count += 1
                    continue
                X.append(feat)
                y.append(cls_label)
            except Exception as e:
                excluded_count += 1
    print(f"Excluded {excluded_count} silent/invalid files across all classes.")
    return np.array(X), np.array(y)

audio_dir = "E:/snn_2/DroneAudioDataset/Binary_Drone_Audio"
weights_dir = "E:/snn_2/weights"

print("Loading real dataset...")
X, y = build_dataset_from_folder(root_dir=audio_dir)

print("Loading real weights...")
W1 = np.load(os.path.join(weights_dir, 'W1_folded_raw.npy'))
b1 = np.load(os.path.join(weights_dir, 'b1_folded_raw.npy')).flatten()
W2 = np.load(os.path.join(weights_dir, 'W2_raw.npy'))
b2 = np.load(os.path.join(weights_dir, 'b2_raw.npy')).flatten()

a1_max_path = os.path.join(weights_dir, 'a1_max.npy')
a2_max_path = os.path.join(weights_dir, 'a2_max.npy')
a1_max = np.load(a1_max_path).item() if hasattr(np.load(a1_max_path), 'item') else np.load(a1_max_path)[0]
a2_max = np.load(a2_max_path).item() if hasattr(np.load(a2_max_path), 'item') else np.load(a2_max_path)[0]

W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)
W1_int = np.round(W1*W1_scale).astype(np.int64)
W2_int = np.round(W2*W2_scale).astype(np.int64)

TUNE_FACTOR = 0.80

b1_int = np.round(b1 * W1_scale * 0.9).astype(np.int64)
V_thresh_h = int(np.round(a1_max * W1_scale * 0.9 * TUNE_FACTOR))
b2_int = np.round(b2 * W2_scale / a1_max).astype(np.int64)
V_thresh_o = int(np.round(a2_max * W2_scale / a1_max))

class LFSR16:
    def __init__(self): self.state = 0xACE1
    def next(self):
        s = self.state
        self.state = ((s<<1)|(((s>>15)^(s>>13)^(s>>12)^(s>>10))&1)) & 0xFFFF
        return s

thresholds_int = (np.clip(X * 0.9, 0.0, 1.0) * 65535).astype(np.uint32)
y_pred_snn = np.zeros(len(X), dtype=np.int64)

print("Running Inference...")
for i in range(len(X)):
    sample_thresh, lfsr = thresholds_int[i], LFSR16()
    v_hid, v_out, out_spikes = np.zeros(32, dtype=np.int64), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    for t in range(50):
        spikes_in = np.array([1 if lfsr.next() < sample_thresh[n] else 0 for n in range(640)], dtype=np.int64)
        v_hid += W1_int.T @ spikes_in + b1_int
        hid_spikes = (v_hid >= V_thresh_h).astype(np.int64)
        v_hid = np.where(v_hid >= V_thresh_h, v_hid - V_thresh_h, np.maximum(0, v_hid))
        v_out += W2_int.T @ hid_spikes + b2_int
        out_spikes += (v_out >= V_thresh_o).astype(np.int64)
        v_out = np.where(v_out >= V_thresh_o, v_out - V_thresh_o, np.maximum(0, v_out))
    
    y_pred_snn[i] = 1 if out_spikes[1] > out_spikes[0] else 0
    if (i+1) % 2000 == 0: print(f" Processed {i+1} / {len(X)} samples...")

print("\n" + "="*70)
print(" SNN HARDWARE EMULATION METRICS ON FULL DATASET")
print("="*70)
print("Confusion matrix (rows=true, cols=pred, 0=ambient 1=drone):")
print(confusion_matrix(y, y_pred_snn))

p = precision_score(y, y_pred_snn, zero_division=0)
r = recall_score(y, y_pred_snn, zero_division=0)
f2 = (5 * p * r) / (4 * p + r + 1e-9)

print(f"Accuracy:  {accuracy_score(y, y_pred_snn)*100:.2f}%")
print(f"Precision: {p*100:.2f}%")
print(f"Recall:    {r*100:.2f}%")
print(f"F1 Score:  {f1_score(y, y_pred_snn, zero_division=0)*100:.2f}%")
print(f"F2 Score:  {f2*100:.2f}%")
print("="*70)
