import numpy as np
import eval_hw

syn = np.load('synthetic_sample.npy')
drn = np.load('sample_drone_thresholds.npy')

# DO NOT RESET LFSR
lfsr = eval_hw.LFSR16()

def run_snn_cont(thresh):
    v_hid = np.zeros(32, dtype=np.int64)
    v_out = np.zeros(2, dtype=np.int64)
    out_spikes = np.zeros(2, dtype=np.int64)
    for t in range(50):
        spikes_in = np.array([1 if lfsr.next() < thresh[n] else 0 for n in range(640)], dtype=np.int64)
        v_hid += eval_hw.W1_int.T @ spikes_in + eval_hw.b1_int
        hid_spikes = (v_hid >= eval_hw.V_thresh_h).astype(np.int64)
        v_hid = np.where(v_hid >= eval_hw.V_thresh_h, v_hid - eval_hw.V_thresh_h, v_hid)
        v_hid = np.maximum(0, v_hid)
        
        v_out += eval_hw.W2_int.T @ hid_spikes + eval_hw.b2_int
        out_spikes += (v_out >= eval_hw.V_thresh_o).astype(np.int64)
        v_out = np.where(v_out >= eval_hw.V_thresh_o, v_out - eval_hw.V_thresh_o, v_out)
        v_out = np.maximum(0, v_out)
    return out_spikes

print("Synthetic:", run_snn_cont(syn))
print("Drone (continuous LFSR):", run_snn_cont(drn))
