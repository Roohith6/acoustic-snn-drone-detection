"""
gen_rate_encoder_golden.py
Builds RTL verification artifacts for the rate encoder, from REAL uploaded
feature/spike pairs (one ambient, one drone). Reuses the exact documented
formula from deterministic_rate_encode() in audio_to_spike.py -- including
the truncating cast (.astype(uint32), NOT round()) since that's what your
real code does.
"""
import numpy as np

MAX_RATE = 0.9
WIDTH = 16
MAX_VAL = (1 << WIDTH) - 1  # 65535
N_STEPS = 50

samples = {
    "ambient": {
        "feature": "/mnt/user-data/uploads/1788803558813_1-137-A-320_feature.npy",
        "spikes":  "/mnt/user-data/uploads/1788803558813_1-137-A-320_spikes.npy",
    },
    "drone": {
        "feature": "/mnt/user-data/uploads/1788803558813_B_S2_D1_067-bebop_000__feature.npy",
        "spikes":  "/mnt/user-data/uploads/1788803558814_B_S2_D1_067-bebop_000__spikes.npy",
    },
}

for label, paths in samples.items():
    feature = np.load(paths["feature"])          # (32, 20) float32, in [0,1]
    spikes  = np.load(paths["spikes"])            # (50, 32, 20) uint8, real golden output

    # --- exact formula from deterministic_rate_encode() ---
    thresholds = np.clip(feature.flatten() * MAX_RATE, 0.0, 1.0)      # row-major, 640 values
    thresholds_int = (thresholds * MAX_VAL).astype(np.uint32)          # TRUNCATING cast, not round()

    # Sanity check: does re-running deterministic_rate_encode with our own
    # LFSR16 (verbatim) on this threshold array reproduce the golden spikes
    # exactly? This validates our understanding of the iteration order
    # BEFORE we trust the RTL testbench to mean anything.
    class LFSR16:
        def __init__(self, seed=0xACE1):
            self.state = seed & 0xFFFF
            if self.state == 0:
                self.state = 0xACE1
        def next(self):
            s = self.state
            bit = ((s >> 15) ^ (s >> 13) ^ (s >> 12) ^ (s >> 10)) & 1
            self.state = ((s << 1) | bit) & 0xFFFF
            return self.state

    lfsr = LFSR16(seed=0xACE1)
    n_neurons = 640
    recomputed = np.zeros((N_STEPS, n_neurons), dtype=np.uint8)
    seq_bits = []  # exact t-major, n-minor order, for the RTL golden sequence file
    for t in range(N_STEPS):
        for n in range(n_neurons):
            rand_val = lfsr.next()
            bit = 1 if rand_val < thresholds_int[n] else 0
            recomputed[t, n] = bit
            seq_bits.append(bit)

    golden_flat = spikes.reshape(N_STEPS, n_neurons)
    match = np.array_equal(recomputed, golden_flat)
    n_mismatch = np.sum(recomputed != golden_flat)
    print(f"[{label}] recompute vs golden spikes.npy: "
          f"{'EXACT MATCH' if match else f'{n_mismatch} MISMATCHES'} "
          f"(out of {N_STEPS*n_neurons} bits)")

    # --- write threshold ROM (640 x 16-bit unsigned hex, one per line) ---
    with open(f"/home/claude/fpga_snn/sim/threshold_{label}.mem", "w") as f:
        f.write(f"// {label}: 640 threshold_int values, uint16, row-major (freq*20+time)\n")
        for v in thresholds_int:
            f.write(f"{v:04x}\n")

    # --- write golden spike sequence (t-major, n-minor, exact generation order) ---
    with open(f"/home/claude/fpga_snn/sim/golden_spikes_{label}.hex", "w") as f:
        for b in seq_bits:
            f.write(f"{b}\n")

    print(f"[{label}] wrote threshold_{label}.mem (640 lines) and "
          f"golden_spikes_{label}.hex ({len(seq_bits)} lines)")
