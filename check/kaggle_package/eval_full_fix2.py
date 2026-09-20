import numpy as np
import train_snn

audio_dir = "E:/snn_2/DroneAudioDataset/Binary_Drone_Audio"
X, y = train_snn.build_dataset_from_folder(root_dir=audio_dir)

W1, b1 = np.load('W1_folded_raw.npy'), np.load('b1_folded_raw.npy').flatten()
W2, b2 = np.load('W2_raw.npy'), np.load('b2_raw.npy').flatten()
a1_max = np.load('a1_max.npy').item() if hasattr(np.load('a1_max.npy'), 'item') else np.load('a1_max.npy')[0]
a2_max = np.load('a2_max.npy').item() if hasattr(np.load('a2_max.npy'), 'item') else np.load('a2_max.npy')[0]

W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)
W1_int, W2_int = np.round(W1*W1_scale).astype(np.int64), np.round(W2*W2_scale).astype(np.int64)

b1_int = np.round(b1 * W1_scale * 0.9).astype(np.int64)
V_thresh_h = int(np.round(a1_max * W1_scale * 0.9))
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

from sklearn.metrics import confusion_matrix
print("SNN FIX Confusion matrix:")
print(confusion_matrix(y, y_pred_snn))
