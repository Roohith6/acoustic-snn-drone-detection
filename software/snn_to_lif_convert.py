import os
import numpy as np

# This script generates the threshold .mem files for the LIF RTL design.
# Based on ANN-to-SNN threshold balancing.

out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights'))

a1_max = np.load(os.path.join(out_dir, 'a1_max.npy'))
a2_max = np.load(os.path.join(out_dir, 'a2_max.npy'))

# We need the scale factors used during export.
# In train_snn.py, W1_folded is exported. We can re-derive its scale or read it.
W1_folded = np.load(os.path.join(out_dir, 'W1_folded_raw.npy'))
W2 = np.load(os.path.join(out_dir, 'W2_raw.npy'))

W1_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W1_folded)) + 1e-9)
W2_scale = (2**15 - 1) * 0.9 / (np.max(np.abs(W2)) + 1e-9)

print(f"Recovered W1_scale: {W1_scale:.4f}")
print(f"Recovered W2_scale: {W2_scale:.4f}")

N_STEPS = 50   # number of LIF timesteps

# Layer 1 (Hidden):
# To map the maximum ANN activation (a1_max) to the maximum SNN firing rate (1 spike/step),
# we set the threshold to a1_max. (Scaled by W1_scale for fixed-point).
V_thresh_hidden = int(np.round(a1_max * W1_scale))

# Since the hidden layer firing rate is f = A / a1_max, the number of spikes over 50 steps
# is 50 * A / a1_max. The input to the output layer is thus scaled by (50 / a1_max) compared 
# to the ANN. To keep the math balanced, b2 must be scaled by this exact same factor.
# The expected total accumulated V_mem over 50 steps at the output is then:
# v_out_total = (z2_ANN) * (50 / a1_max) * W2_scale.
# If we want the output layer to fire 1 spike/step when it reaches a2_max, the total 
# V_mem would be 50 * V_thresh_output.
# So 50 * V_thresh_output = a2_max * (50 / a1_max) * W2_scale
# => V_thresh_output = a2_max * W2_scale / a1_max
V_thresh_output = int(np.round(a2_max * W2_scale / a1_max))

print(f"V_thresh_hidden (int): {V_thresh_hidden}")
print(f"V_thresh_output (int): {V_thresh_output}")

with open(os.path.join(out_dir, 'V_thresh_hidden.mem'), 'w') as f:
    f.write(f"// V_thresh_hidden, scale={W1_scale:.4f}, balanced by a1_max\n")
    f.write(f"{V_thresh_hidden:08x}\n")

with open(os.path.join(out_dir, 'V_thresh_output.mem'), 'w') as f:
    f.write(f"// V_thresh_output, scale={W2_scale:.4f}, balanced by a2_max/a1_max\n")
    f.write(f"{V_thresh_output:08x}\n")

b1_folded = np.load(os.path.join(out_dir, 'b1_folded_raw.npy')).flatten()
b2 = np.load(os.path.join(out_dir, 'b2_raw.npy')).flatten()

# Bias 1 is per-timestep because input X is per-timestep rate
with open(os.path.join(out_dir, 'b1_lif.mem'), 'w') as f:
    f.write(f"// b1_lif, scaled by W1_scale={W1_scale:.4f}\n")
    for val in b1_folded:
        b1_int = int(np.round(val * W1_scale))
        if b1_int < 0: b1_int += (1 << 32)
        f.write(f"{b1_int:08x}\n")

# Bias 2 must match the (1 / a1_max) scaling of the hidden spikes
with open(os.path.join(out_dir, 'b2_lif.mem'), 'w') as f:
    f.write(f"// b2_lif, scaled by W2_scale={W2_scale:.4f} and 1/a1_max\n")
    for val in b2:
        b2_int = int(np.round(val * W2_scale / a1_max))
        if b2_int < 0: b2_int += (1 << 32)
        f.write(f"{b2_int:08x}\n")

print("Generated V_thresh and b_lif .mem files")
