import numpy as np
W1 = np.load('W1_folded_raw.npy')
b1 = np.load('b1_folded_raw.npy').flatten()
W2 = np.load('W2_raw.npy')
b2 = np.load('b2_raw.npy').flatten()
a1_max = np.load('a1_max.npy').item() if hasattr(np.load('a1_max.npy'), 'item') else np.load('a1_max.npy')[0]
a2_max = np.load('a2_max.npy').item() if hasattr(np.load('a2_max.npy'), 'item') else np.load('a2_max.npy')[0]

W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)
W1_int = np.round(W1 * W1_scale).astype(np.int64)
W2_int = np.round(W2 * W2_scale).astype(np.int64)

# THE HARDWARE MATH:
b1_int = np.round(b1 * W1_scale).astype(np.int64)          # No 0.9
V_thresh_h = int(np.round(a1_max * W1_scale))              # No 0.9
b2_int = np.round(b2 * W2_scale / a1_max).astype(np.int64) # Has /a1_max
V_thresh_o = int(np.round(a2_max * W2_scale / a1_max))     # Has /a1_max

class LFSR16:
    def __init__(self): self.state = 0xACE1
    def next(self):
        s = self.state
        self.state = ((s<<1)|(((s>>15)^(s>>13)^(s>>12)^(s>>10))&1)) & 0xFFFF
        return s

def run_snn(thresh):
    v_hid = np.zeros(32, dtype=np.int64)
    v_out = np.zeros(2, dtype=np.int64)
    lfsr = LFSR16()
    out_spikes = np.zeros(2, dtype=np.int64)
    for t in range(50):
        spikes_in = np.array([1 if lfsr.next() < thresh[n] else 0 for n in range(640)], dtype=np.int64)
        v_hid += W1_int.T @ spikes_in + b1_int
        hid_spikes = (v_hid >= V_thresh_h).astype(np.int64)
        v_hid = np.where(v_hid >= V_thresh_h, v_hid - V_thresh_h, v_hid)
        v_hid = np.maximum(0, v_hid)
        
        v_out += W2_int.T @ hid_spikes + b2_int
        out_spikes += (v_out >= V_thresh_o).astype(np.int64)
        v_out = np.where(v_out >= V_thresh_o, v_out - V_thresh_o, v_out)
        v_out = np.maximum(0, v_out)
    return out_spikes

amb = np.load('sample_ambient_thresholds.npy')
drn = np.load('sample_drone_thresholds.npy')
print("Ambient:", run_snn(amb))
print("Drone:", run_snn(drn))
