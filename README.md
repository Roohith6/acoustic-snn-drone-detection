\# Acoustic SNN Drone Detection — FPGA Implementation



Real-time acoustic drone-vs-ambient classification using a Spiking Neural

Network (SNN), implemented end-to-end from audio preprocessing through

FPGA deployment on the Terasic DE2-115.



\## Overview



\- \*\*Pipeline:\*\* 16 kHz mono audio -> STFT (256-pt FFT, hop 64, Hann window)

&#x20; -> 32x20 feature map -> flattened 640-value input -> 16-bit LFSR

&#x20; rate-coded spike encoding (50 timesteps) -> 640->128->2 classifier

&#x20; (DRONE / AMBIENT)

\- \*\*Target board:\*\* Terasic DE2-115 (Cyclone IV EP4CE115F29)

\- \*\*Status:\*\* RTL verified in simulation (bit-exact against golden software

&#x20; reference), synthesized clean, deployed and tested on hardware for 4

&#x20; pre-loaded sample cases.

\- \*\*Accuracy:\*\* 97.9% classification accuracy on the test set.



\## Repo layout



| Folder | Contents |

|---|---|

| `software/` | Python preprocessing pipeline (audio -> spike encoding), reference model |

| `rtl/` | Verilog source -- SNN core, board wrapper (`de2\_top.v`) |

| `mem/` | ROM initialization files (weights, biases, sigmoid LUT, per-sample thresholds) |

| `tb/` | Testbenches used for RTL/golden-model verification |

| `quartus/` | Quartus project files, pin assignments, timing constraints |

| `docs/` | Project handover notes and design history |



\## Hardware bring-up



Board interface (`de2\_top.v`):



| Signal | Pin | I/O Standard |

|---|---|---|

| `CLOCK\_50` | PIN\_Y2 | 2.5 V |

| `KEY\[0]` | PIN\_M23 | 2.5 V |

| `KEY\[1]` | PIN\_M21 | 2.5 V |

| `LEDG\[0]` | PIN\_E21 | 2.5 V |

| `LEDG\[1]` | PIN\_E22 | 2.5 V |



`KEY\[1:0]` selects among 4 pre-loaded audio samples (active-low).

`LEDG\[0]` = classified DRONE, `LEDG\[1]` = classified AMBIENT.



\## Known limitations



\- No live audio input -- sample features are pre-computed in software and

&#x20; baked into ROM (`.mem` files) at synthesis time.

\- `KEY\[1:0]` input is not debounced.

\- One known drone sample is misclassified as ambient -- a model-level

&#x20; limitation reproduced in the floating-point reference model, not a

&#x20; hardware bug.



\## Build / run



1\. Open `neuro.qpf` in Quartus.

2\. Confirm top-level entity is `de2\_top` (`Assignments -> Settings -> General`).

3\. Confirm `neuro.sdc` is set as the SDC timing file.

4\. `Processing -> Start Compilation`.

5\. `Tools -> Programmer` -> load `neuro.sof` via JTAG (USB-Blaster).

6\. Press `KEY\[1:0]` combinations and observe `LEDG\[0]`/`LEDG\[1]`.

