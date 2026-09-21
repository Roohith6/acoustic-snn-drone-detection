"""
convert_to_mem.py
=================
Converts the .npy weight files output by train_native_snn.py into
fixed-point hexadecimal .mem files that Quartus/ModelSim can load
directly into FPGA Block RAM using $readmemh().

Fixed-point format: Q8.8  (8 integer bits, 8 fractional bits)
  SCALE    = 256    (= 2^8)
  BIT_WIDTH = 16   (signed 16-bit two's complement)

Weight ranges from training:
  W1: [-0.433, 0.415]  ->  fits comfortably in Q8.8
  W2: [-1.335, 1.431]  ->  fits comfortably in Q8.8

Output files (copy all 4 to E:\\snn_2\\mem\\ AND E:\\snn_2\\tb\\):
  w1_128.mem  -- 81,920 values (W1 shape: 128 x 640)
  b1_128.mem  -- 128 values
  w2_128.mem  -- 256 values   (W2 shape: 2 x 128)
  b2_128.mem  -- 2 values

Usage:
  python convert_to_mem.py

Run this on your LAPTOP (after downloading W1_128.npy etc. from Kaggle output).
"""

import numpy as np

SCALE     = 256      # Q8.8 fixed-point scale factor
BIT_WIDTH = 16       # signed 16-bit two's complement
MAX_VAL   = (1 << (BIT_WIDTH - 1)) - 1   #  32767
MIN_VAL   = -(1 << (BIT_WIDTH - 1))      # -32768

def to_fixed_hex(arr):
    """Convert float array to list of Q8.8 hex strings."""
    scaled   = np.round(arr.flatten() * SCALE).astype(np.int32)
    clipped  = np.clip(scaled, MIN_VAL, MAX_VAL).astype(np.int16)
    return [f"{v & 0xFFFF:04x}" for v in clipped]

def save_mem(filename, arr, label):
    hexvals = to_fixed_hex(arr)
    with open(filename, 'w') as f:
        for h in hexvals:
            f.write(h + '\n')
    print(f"  ✅ {filename:15s} ({len(hexvals)} values) "
          f"range [{arr.min():.3f}, {arr.max():.3f}]")

if __name__ == "__main__":
    print("Converting 128-Neuron Native SNN weights → FPGA .mem files...\n")

    W1 = np.load("W1_128.npy")   # shape (128, 640)
    b1 = np.load("b1_128.npy")   # shape (128,)
    W2 = np.load("W2_128.npy")   # shape (2, 128)
    b2 = np.load("b2_128.npy")   # shape (2,)

    print(f"W1: {W1.shape}  |  W2: {W2.shape}")
    save_mem("w1_128.mem", W1, "W1")
    save_mem("b1_128.mem", b1, "b1")
    save_mem("w2_128.mem", W2, "W2")
    save_mem("b2_128.mem", b2, "b2")

    print("\n📋 Verilog parameter (already set in de2_top.v):")
    print("   .N_POST(128)")
    print("   .W1_MEM_FILE(\"w1_128.mem\")")
    print("   .W2_MEM_FILE(\"w2_128.mem\")")
    print("   .B1_MEM_FILE(\"b1_128.mem\")")
    print("   .B2_MEM_FILE(\"b2_128.mem\")")
    print("\n✅ Copy all 4 .mem files to E:\\snn_2\\mem\\ and E:\\snn_2\\tb\\")
