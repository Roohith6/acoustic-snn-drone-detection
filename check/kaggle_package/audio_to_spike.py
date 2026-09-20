"""
audio_to_spike.py
==================
Preprocessing pipeline for Track B: SNN-on-FPGA acoustic drone detection.

Pipeline stages:
  1. Audio front-end       -> raw waveform (synthetic, for testing, or loaded .wav)
  2. Feature extraction    -> STFT spectrogram (lighter than MFCC for a first
                               RTL pass -- reuses the same FFT-core idea as the
                               radar track, per the implementation plan)
  3. Spike encoding        -> rate coding OR time-to-first-spike (TTFS)
  4. Verilog export        -> .mem file for $readmemb / $readmemh in a testbench

Two ways to get audio in:
  - synthetic_drone_audio(): generates a plausible synthetic propeller/motor
    hum (drone) vs broadband ambient noise (non-drone), so your team can get
    the RTL testbench working before the real DroneAudioDataset is wired in.
  - load_wav_file(): loads a real .wav file, e.g. from
    github.com/saraalemadi/DroneAudioDataset -- resamples to a fixed rate.

Run this file directly for a demo:
    python3 audio_to_spike.py
It generates one synthetic example per class, plots the spectrogram + spike
raster, and writes .mem files ready for a Verilog testbench.

Dependencies: numpy, scipy, matplotlib   (no librosa needed -- STFT only,
by design, since raw STFT is the lighter/recommended first RTL pass)
"""

import os
import numpy as np
from scipy.signal import stft
from scipy.io import wavfile
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Stage 1: Audio front-end
# ---------------------------------------------------------------------------

def synthetic_drone_audio(target_type="drone", fs=16000.0, duration=1.0, seed=None):
    """
    Generate a synthetic audio clip for pipeline testing.

    target_type : "drone" | "ambient"
    fs          : sample rate in Hz (16 kHz is a common choice for this task,
                  matches typical drone-audio-dataset sample rates)
    duration    : clip length in seconds

    Returns
    -------
    t : time vector
    x : real-valued audio signal
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    n = len(t)

    if target_type == "drone":
        # Multi-rotor drones produce a strong fundamental motor/propeller tone
        # plus several harmonics -- a comb-like spectrum, fairly stable over time.
        fundamental = rng.uniform(150, 250)  # Hz, typical small-drone motor tone
        x = np.zeros(n)
        for h in range(1, 6):
            amp = 0.7 / h
            x += amp * np.sin(2 * np.pi * h * fundamental * t)
        # Slight amplitude modulation (blade-pass beating)
        x *= (1.0 + 0.15 * np.sin(2 * np.pi * 12.0 * t))

    elif target_type == "ambient":
        # Broadband noise with a weak, non-harmonic low-frequency rumble
        # (wind, traffic, etc.) -- no comb structure.
        x = 0.3 * rng.standard_normal(n)
        rumble = 0.2 * np.sin(2 * np.pi * rng.uniform(20, 60) * t)
        x += rumble

    else:
        raise ValueError("target_type must be 'drone' or 'ambient'")

    noise = 0.05 * rng.standard_normal(n)
    x = x + noise
    x = x / (np.max(np.abs(x)) + 1e-9)  # normalize to [-1, 1]
    return t, x


def load_wav_file(filepath, target_fs=16000.0, normalize_peak=False):
    """
    Load a real .wav file (e.g. from DroneAudioDataset) and return
    (t, x) in the same format as synthetic_drone_audio().
    Mono-mixes stereo files. ACTUALLY RESAMPLES to target_fs if the file's
    real rate differs -- a prior version only printed a warning and left
    the file at its original rate, while every caller downstream assumes
    a fixed FS=16000.0 constant. That mismatch would have silently
    misinterpreted the frequency axis for any non-16kHz file. Fixed here:
    the caller can now always assume the returned x is genuinely at
    target_fs.

    normalize_peak:
      True -- scales every file's peak to ~1.0. Discards absolute
        loudness (see compare_normalization.py findings).
      False (default) -- scales by the PCM bit-depth's fixed range
        instead, preserving relative loudness across files. Use with
        compute_audio_spectrogram(..., normalization="fixed_ref").
    """
    fs, x = wavfile.read(filepath)
    if x.ndim > 1:
        x = x.mean(axis=1)

    if normalize_peak:
        x = x.astype(np.float32)
        x = x / (np.max(np.abs(x)) + 1e-9)
    else:
        if np.issubdtype(x.dtype, np.integer):
            scale = float(max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max))
        else:
            scale = 1.0
        x = x.astype(np.float32) / scale

    if fs != target_fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(fs), int(target_fs))
        up, down = int(target_fs) // g, int(fs) // g
        x = resample_poly(x, up, down).astype(np.float32)
        fs = target_fs

    t = np.arange(len(x)) / fs
    return t, x


# ---------------------------------------------------------------------------
# Stage 2: Feature extraction -> STFT spectrogram
# ---------------------------------------------------------------------------

def compute_audio_spectrogram(x, fs, nperseg=256, noverlap=192, normalization="fixed_ref"):
    """
    Compute a magnitude spectrogram via STFT, normalized to [0, 1].
    Real-valued input -> one-sided spectrum (only non-negative frequencies).

    normalization:
      "per_file" (original behavior) -- min/max normalize each file's own
        dB spectrogram to [0,1]. PROBLEM (confirmed via 200+200 real-data
        test): this discards absolute loudness entirely -- a quiet
        ambient clip's loudest relative bin gets stretched to 1.0 just
        like a loud drone's, which likely explains why ~100% of input
        neurons were "active" for nearly every file of BOTH classes and
        real-data separability was much weaker (1.60) than a small
        10-sample test suggested (5.71).
      "fixed_ref" -- normalize against a fixed dB floor/ceiling instead
        of each file's own min/max, so absolute loudness differences
        between a loud drone motor and quiet ambient background are
        preserved. db_floor/db_ceiling may need re-tuning against real
        full-dataset statistics.

    HARDWARE-FIDELITY NOTE: uses boundary=None, padded=False, UNLIKE
    scipy's own defaults (boundary='zeros', padded=True). scipy's
    defaults zero-pad the signal at both edges and add extra frames --
    behavior a real streaming hardware windower would not reproduce (it
    only has the samples that actually arrived). Matching a real
    streaming FFT's framing here, rather than scipy's convenience
    defaults, keeps this closer to what RTL will actually produce.
    """
    f, tt, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       return_onesided=True, boundary=None, padded=False)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-8)

    if normalization == "per_file":
        mag_norm = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-12)
    elif normalization == "fixed_ref":
        db_floor, db_ceiling = -60.0, 0.0
        mag_norm = (mag_db - db_floor) / (db_ceiling - db_floor)
        mag_norm = np.clip(mag_norm, 0.0, 1.0)
    else:
        raise ValueError("normalization must be 'per_file' or 'fixed_ref'")

    return f, tt, mag_norm


def quantize_feature(spectrogram, width=8):
    """
    Simulate fixed-point precision loss on the final [0,1] feature matrix,
    since RTL cannot reproduce Python float64 exactly.

    IMPORTANT SCOPE LIMITATION, stated plainly: this quantizes the FINAL
    normalized feature values only. It does NOT model rounding error
    accumulating through intermediate fixed-point arithmetic inside an
    actual RTL FFT/STFT (multiplier truncation, accumulator overflow,
    twiddle-factor precision, etc.) -- fully modeling that would mean
    building a bit-accurate fixed-point FFT simulator, which is a much
    larger undertaking than this project needs right now. This is a
    reasonable approximation for golden-reference purposes, not a claim
    that Python and RTL will match to the last bit through the whole
    pipeline -- only that the FINAL feature representation uses the same
    finite precision RTL will actually use.

    width: bits used to represent each feature value, e.g. 8 -> 256 levels.
    """
    levels = (1 << width) - 1
    q = np.round(np.clip(spectrogram, 0.0, 1.0) * levels) / levels
    return q.astype(np.float32)


def downsample_spectrogram(f, tt, spectrogram, n_freq_bins=32, n_time_bins=20):
    """
    Average-pool the spectrogram down to a hardware-realistic input size.
    A raw STFT easily has 100+ frequency bins x 200+ time bins -- far too
    many input neurons for FPGA BRAM. 32 x 20 = 640 input neurons is a
    reasonable starting point (matches the scale used in published FPGA-SNN
    accelerators like Spiker on similarly small inputs).

    Returns
    -------
    f_ds, tt_ds : downsampled axis labels (bin centers)
    spec_ds     : pooled spectrogram, shape (n_freq_bins, n_time_bins), in [0, 1]
    """
    freq_bins, time_bins = spectrogram.shape
    f_edges = np.linspace(0, freq_bins, n_freq_bins + 1).astype(int)
    t_edges = np.linspace(0, time_bins, n_time_bins + 1).astype(int)

    spec_ds = np.zeros((n_freq_bins, n_time_bins))
    for i in range(n_freq_bins):
        for j in range(n_time_bins):
            block = spectrogram[f_edges[i]:f_edges[i+1], t_edges[j]:t_edges[j+1]]
            spec_ds[i, j] = block.mean() if block.size else 0.0

    f_ds = np.array([f[f_edges[i]:f_edges[i+1]].mean() if f_edges[i] < f_edges[i+1]
                      else (f[f_edges[i]] if f_edges[i] < len(f) else f[-1])
                      for i in range(n_freq_bins)])
    tt_ds = np.array([tt[t_edges[j]:t_edges[j+1]].mean() if t_edges[j] < t_edges[j+1]
                       else (tt[t_edges[j]] if t_edges[j] < len(tt) else tt[-1])
                       for j in range(n_time_bins)])
    return f_ds, tt_ds, spec_ds


# ---------------------------------------------------------------------------
# Stage 3: Spike encoding (identical logic to the radar track --
# this block is directly reusable RTL between Track A and Track B)
# ---------------------------------------------------------------------------

def rate_encode(spectrogram, num_steps=50, max_rate=1.0, seed=None):
    """[EXPLORATION ONLY -- NOT FPGA-REPRODUCIBLE]
    Uses NumPy's Mersenne Twister PRNG. Even with a fixed seed, this
    cannot be bit-matched by real hardware (FPGA random sources are
    LFSR-based, a completely different algorithm). Safe for early
    separability testing; DO NOT use this to generate a "golden reference"
    dataset for FPGA verification -- use deterministic_rate_encode()
    instead, which uses the same LFSR algorithm on both sides."""
    rng = np.random.default_rng(seed)
    prob = np.clip(spectrogram * max_rate, 0.0, 1.0)
    freq_bins, time_bins = spectrogram.shape
    draws = rng.random((num_steps, freq_bins, time_bins))
    spikes = (draws < prob[None, :, :]).astype(np.uint8)
    return spikes


class LFSR16:
    """16-bit Fibonacci LFSR, taps at bits 15,13,12,10 (0-indexed from LSB) --
    a standard maximal-length tap set for a 16-bit LFSR. This exact
    algorithm (same taps, same shift direction, same seed) MUST be
    reproduced identically in the RTL random-number-source module --
    that identity is the whole point: it is the freeze contract between
    the Python golden reference and the FPGA implementation.
    Never non-zero-seed -- an LFSR seeded with 0 gets stuck at 0 forever."""
    def __init__(self, seed=0xACE1):
        self.state = seed & 0xFFFF
        if self.state == 0:
            self.state = 0xACE1

    def next(self):
        s = self.state
        bit = ((s >> 15) ^ (s >> 13) ^ (s >> 12) ^ (s >> 10)) & 1
        self.state = ((s << 1) | bit) & 0xFFFF
        return self.state


def deterministic_rate_encode(spectrogram, num_steps=50, max_rate=1.0,
                               lfsr_seed=0xACE1, width=16):
    """[FPGA-REPRODUCIBLE] Rate coding using a deterministic LFSR instead
    of NumPy's PRNG. One shared 16-bit LFSR is advanced once per neuron
    per timestep, in fixed row-major (flattened) neuron order -- exactly
    matching how synapse_memory.v / a hardware spike-encoder module would
    need to iterate. A neuron fires if the LFSR's current value is below
    its fixed-point firing threshold, same principle as a stochastic
    bitstream generator commonly used in FPGA-SNN designs.

    This is the encoder to use for anything meant to become a "golden
    reference" the FPGA implementation will later be checked against --
    rate_encode() above is NOT interchangeable with this for that purpose.
    """
    freq_bins, time_bins = spectrogram.shape
    n_neurons = freq_bins * time_bins
    max_val = (1 << width) - 1

    thresholds = np.clip(spectrogram.flatten() * max_rate, 0.0, 1.0)
    thresholds_int = (thresholds * max_val).astype(np.uint32)

    lfsr = LFSR16(seed=lfsr_seed)
    spikes_flat = np.zeros((num_steps, n_neurons), dtype=np.uint8)
    for t in range(num_steps):
        for n in range(n_neurons):
            rand_val = lfsr.next()
            if rand_val < thresholds_int[n]:
                spikes_flat[t, n] = 1

    return spikes_flat.reshape(num_steps, freq_bins, time_bins)


def ttfs_encode(spectrogram, num_steps=50):
    freq_bins, time_bins = spectrogram.shape
    spike_step = np.clip(((1.0 - spectrogram) * (num_steps - 1)).astype(int), 0, num_steps - 1)
    fires = spectrogram > 0.05
    spikes = np.zeros((num_steps, freq_bins, time_bins), dtype=np.uint8)
    fi, ti = np.where(fires)
    spikes[spike_step[fi, ti], fi, ti] = 1
    return spikes


# ---------------------------------------------------------------------------
# Stage 4: Export for Verilog testbench (identical format to Track A,
# so the same testbench harness works for both tracks)
# ---------------------------------------------------------------------------

def export_spikes_for_verilog(spikes, filepath, fmt="bin"):
    num_steps, freq_bins, time_bins = spikes.shape
    n_inputs = freq_bins * time_bins
    flat = spikes.reshape(num_steps, n_inputs)

    with open(filepath, "w") as f:
        f.write(f"// {num_steps} timesteps x {n_inputs} input neurons\n")
        for step in flat:
            bitstring = "".join(str(b) for b in step)
            if fmt == "bin":
                f.write(bitstring + "\n")
            elif fmt == "hex":
                f.write(f"{int(bitstring, 2):0{(n_inputs + 3)//4}x}\n")
            else:
                raise ValueError("fmt must be 'bin' or 'hex'")

    print(f"Wrote {filepath}: {num_steps} steps x {n_inputs} neurons ({fmt})")
    return n_inputs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_pipeline(t, x, f, tt, spectrogram, spikes, target_type, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(t, x, lw=0.5)
    axes[0].set_title(f"{target_type}: raw audio waveform")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")

    im = axes[1].pcolormesh(tt, f, spectrogram, shading="auto", cmap="magma")
    axes[1].set_title("audio spectrogram (normalized)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=axes[1], label="normalized magnitude")

    num_steps, freq_bins, time_bins = spikes.shape
    flat = spikes.reshape(num_steps, freq_bins * time_bins)
    ys, xs = np.where(flat)
    axes[2].scatter(xs, ys, s=1, c="black")
    axes[2].set_title("encoded spike raster (input layer)")
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("input neuron index")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = "pipeline_output_audio"
    os.makedirs(out_dir, exist_ok=True)

    fs = 16000.0
    duration = 1.0
    num_steps = 50
    encoding = "rate"   # "rate" or "ttfs"

    for target_type in ("drone", "ambient"):
        t, x = synthetic_drone_audio(target_type=target_type, fs=fs,
                                      duration=duration, seed=42)

        f, tt, spectrogram = compute_audio_spectrogram(x, fs)
        f, tt, spectrogram = downsample_spectrogram(f, tt, spectrogram,
                                                      n_freq_bins=32, n_time_bins=20)

        if encoding == "rate":
            spikes = rate_encode(spectrogram, num_steps=num_steps, max_rate=0.9, seed=1)
        else:
            spikes = ttfs_encode(spectrogram, num_steps=num_steps)

        fig_path = os.path.join(out_dir, f"{target_type}_pipeline.png")
        plot_pipeline(t, x, f, tt, spectrogram, spikes, target_type, fig_path)

        mem_path = os.path.join(out_dir, f"{target_type}_spikes.mem")
        n_inputs = export_spikes_for_verilog(spikes, mem_path, fmt="bin")

        print(f"[{target_type}] spectrogram shape={spectrogram.shape}, "
              f"spike tensor shape={spikes.shape}, n_inputs={n_inputs}\n")

    print("Done. To use real data instead: replace synthetic_drone_audio() "
          "calls with load_wav_file('path/to/DroneAudioDataset/clip.wav').")
