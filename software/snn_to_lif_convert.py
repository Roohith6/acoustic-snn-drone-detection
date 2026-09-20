import os
import numpy as np

# This script generates the threshold .mem files for the LIF RTL design.
# Based on ANN-to-SNN threshold balancing.

out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights'))

a1_max = np.load(os.path.join(out_dir, 'a1_max.npy'))
a2_max = np.load(os.path.join(out_dir, 'a2_max.npy'))
if hasattr(a1_max, 'item'): a1_max = a1_max.item()
else: a1_max = a1_max[0]
if hasattr(a2_max, 'item'): a2_max = a2_max.item()
else: a2_max = a2_max[0]

# We need the scale factors used during export.
W1_folded = np.load(os.path.join(out_dir, 'W1_folded_raw.npy'))
W2 = np.load(os.path.join(out_dir, 'W2_raw.npy'))

W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1_folded)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)

print(f"Recovered W1_scale: {W1_scale:.4f}")
print(f"Recovered W2_scale: {W2_scale:.4f}")

# PERFECT MATHEMATICAL BALANCING (WITH 0.9 FIX AND TUNING)
# We apply the 0.9 factor to correctly counteract the max_rate=0.9 input encoder.
# We apply TUNE_FACTOR to counteract the Negative Bias Clamping Drift in the hardware.
TUNE_FACTOR = 0.80

V_thresh_hidden = int(np.round(a1_max * W1_scale * 0.9 * TUNE_FACTOR))
V_thresh_output = int(np.round(a2_max * W2_scale / a1_max))

print(f"V_thresh_hidden (int): {V_thresh_hidden}")
print(f"V_thresh_output (int): {V_thresh_output}")

with open(os.path.join(out_dir, 'V_thresh_hidden.mem'), 'w') as f:
    f.write(f"// V_thresh_hidden, scale={W1_scale:.4f}, balanced by a1_max, 0.9, and tuning\n")
    f.write(f"{V_thresh_hidden:08x}\n")

with open(os.path.join(out_dir, 'V_thresh_output.mem'), 'w') as f:
    f.write(f"// V_thresh_output, scale={W2_scale:.4f}, balanced by a2_max/a1_max\n")
    f.write(f"{V_thresh_output:08x}\n")

b1_folded = np.load(os.path.join(out_dir, 'b1_folded_raw.npy')).flatten()
b2 = np.load(os.path.join(out_dir, 'b2_raw.npy')).flatten()

with open(os.path.join(out_dir, 'b1_lif.mem'), 'w') as f:
    f.write(f"// b1_lif, scaled by W1_scale={W1_scale:.4f} and 0.9 fix\n")
    for val in b1_folded:
        b1_int = int(np.round(val * W1_scale * 0.9))
        if b1_int < 0: b1_int += (1 << 32)
        f.write(f"{b1_int:08x}\n")

with open(os.path.join(out_dir, 'b2_lif.mem'), 'w') as f:
    f.write(f"// b2_lif, scaled by W2_scale={W2_scale:.4f} and 1/a1_max\n")
    for val in b2:
        b2_int = int(np.round(val * W2_scale / a1_max))
        if b2_int < 0: b2_int += (1 << 32)
        f.write(f"{b2_int:08x}\n")

print("Generated PERFECTED V_thresh and b_lif .mem files")
