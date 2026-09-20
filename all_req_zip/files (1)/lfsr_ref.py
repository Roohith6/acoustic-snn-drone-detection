"""
Golden reference for the Encoding V1 LFSR16, transcribed EXACTLY from
audio_to_spike.py (Section 6 of the engineering document). No modification.
Used to generate bit-exact vectors for RTL (lfsr16.v) verification.
"""
import numpy as np
import json
import sys


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


def generate_sequence(n, seed=0xACE1):
    lfsr = LFSR16(seed=seed)
    return [lfsr.next() for _ in range(n)]


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seq = generate_sequence(N, seed=0xACE1)

    # Write as plain hex, one value per line -> golden vector for the testbench
    with open("/home/claude/fpga_snn/sim/lfsr_golden.hex", "w") as f:
        for v in seq:
            f.write(f"{v:04x}\n")

    # Sanity print of the first values + basic statistical sanity checks
    print("First 10 values:", [hex(v) for v in seq[:10]])
    print("Sequence length:", len(seq))
    print("Unique values in first 2000:", len(set(seq[:2000])))
    print("Any zero-state (should never recur once nonzero, seed != 0):",
          0 in seq)
    with open("/home/claude/fpga_snn/sim/lfsr_meta.json", "w") as f:
        json.dump({"n": N, "seed": "0xACE1", "first10": [hex(v) for v in seq[:10]]}, f, indent=2)
