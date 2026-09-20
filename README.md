# FPGA Acoustic Drone Detector (Spiking Neural Network)

Real-time acoustic drone-vs-ambient classification using a custom Spiking Neural Network (SNN) deployed end-to-end on a **Terasic DE2-115 FPGA board**.

This project bridges the gap between software Machine Learning and physical silicon. It takes an Artificial Neural Network (ANN) trained in PyTorch on audio spectrograms, mathematically converts the weights to fit spiking threshold parameters, and deploys it as a power-efficient, cycle-accurate Verilog hardware design.

---

## ?? Architecture Pipeline

1. **Audio Preprocessing (Software):** 
   - 16 kHz mono audio is processed using STFT (256-pt FFT, hop 64, Hann window).
   - Extracts a flattened 640-value feature array (MFCCs).
2. **Rate Encoding (Hardware):** 
   - A 16-bit LFSR (Linear-Feedback Shift Register) generates pseudo-random numbers to convert the static 640 input values into physical electrical spikes over **50 timesteps**.
3. **SNN Topology (`640 -> 32 -> 2`):**
   - **640 Input Neurons**
   - **32 Hidden Neurons** (Leaky Integrate-and-Fire / IF model)
   - **2 Output Neurons** (Ambient Noise vs. Drone)

---

## ??? The "Negative Bias Clamping" Fix
Converting a standard PyTorch ANN to a physical SNN introduces hardware challenges. During initial testing, the FPGA suffered from an 85% False Positive rate (predicting "Drone" for background noise). 

**The Cause:** The PyTorch model relied heavily on negative bias weights to cancel out background noise. However, the hardware SNN logic clamped membrane voltages at `0` (`v < 0 ? 0`). During silent periods in the audio, the hardware lost the built-up negative noise-canceling voltage, causing the neurons to spike far too easily when a sound finally arrived.

**The Solution:** 
We mathematically bridged this gap in `snn_to_lif_convert.py` by:
1. Scaling all weights by `0.9` to perfectly match the LFSR rate-encoder probability.
2. Applying a **Tuning Factor of `0.80`** to the hidden layer threshold (`V_thresh_h`).
3. Lowering the threshold restored the Precision/Recall balance within the strict limits of hardware voltage clamping, resulting in a verified **92.7% hardware accuracy**.

---

## ?? Repository Structure

* `/software/`: Python scripts for PyTorch ANN training (`train_snn.py`), dataset building, and the critical `snn_to_lif_convert.py` script that transforms float weights into 16-bit integer `.mem` files.
* `/rtl/`: The Verilog hardware source code.
  * `snn_top.v`: The master controller.
  * `lif_neuron_array.v` / `lif_output.v`: Integrate-and-fire logic.
  * `spike_gated_mac.v`: Highly efficient MAC units that only consume power when a spike occurs.
  * `de2_top.v`: The board-level wrapper mapping logic to the physical DE2-115 buttons and LEDs.
* `/tb/`: Cycle-accurate ModelSim testbenches (`tb_top_8.v`) and pre-generated `.mem` audio samples.
* `/quartus/`: Intel Quartus Prime project files (`neuro.qpf`).

---

## ?? How to Run

### 1. Simulation (ModelSim)
To view the live neural spikes and accumulation counters:
1. Open ModelSim and navigate to the `/tb/` directory.
2. Run the custom wave script in the transcript:
   ```tcl
   do wave_project_8.do
   ```
3. The script will compile the RTL, run 8 consecutive audio classifications, and graph the raw 1-bit neural spikes alongside the analog output counters.

### 2. Hardware Deployment (Intel Quartus)
1. Open `quartus/neuro.qpf` in Intel Quartus Prime.
2. Click **Start Compilation** to synthesize the Verilog and `.mem` ROM files into a `.sof` bitstream.
3. Use the Quartus Programmer to flash the bitstream via USB to the **DE2-115 board**.
4. **On the board:** The FPGA will instantly begin classifying. Hold `KEY[1]` and `KEY[0]` to cycle between the pre-loaded audio samples in ROM.
   - **`LEDG[1]` turns ON:** Ambient Noise Detected.
   - **`LEDG[0]` turns ON:** Drone Detected!

## ?? How It Works (The Mechanics)
This system is designed for ultra-low-power "Edge AI" processing. Rather than doing heavy floating-point math like a traditional GPU, this FPGA mimics the biological brain using spikes of electricity:
1. **The Senses:** Audio is pre-processed into 640 distinct frequency buckets.
2. **The Synapses (Rate Encoding):** The FPGA uses a random number generator (LFSR) to turn those frequencies into electrical pulses. A loud frequency might fire a spike 90% of the time, while a quiet frequency fires 5% of the time.
3. **The Brain (Spike-Gated MAC):** The hidden neurons ONLY consume power when they receive a spike. If no spike arrives, the hardware stays asleep. When a spike hits, the neuron adds its specific "weight" to its internal voltage.
4. **The Decision:** Once a hidden neuron reaches 111,920 millivolts of accumulation, it fires a spike into the final Output Layer. Over 50 clock cycles, the Drone and Ambient output neurons race to accumulate the most spikes. The highest count wins the classification.

## ?? Future Roadmap & Patent Potential
This project serves as the baseline architecture for a patentable, ultra-low-power acoustic defense system. Future iterations to solidify the novelty and commercial viability include:

1. **Native SNN Training (Surrogate Gradients):** Transitioning from ANN-to-SNN conversion to native snnTorch training. Training with time-dynamics and leak-rates inherently built-in will allow us to drop the inference time from 50 timesteps down to <10 timesteps, drastically reducing power consumption.
2. **Live I2S Microphone Integration:** Replacing the .mem ROM audio samples with a live I2S hardware driver, allowing the FPGA to detect drones flying by in real-time.
3. **Dynamic Early-Exit Optimization (Patent Focus):** Expanding the early_exit.v logic so the FPGA dynamically shuts off its own clock cycle if the DRONE_COUNT reaches a definitive threshold before the 50 timesteps finish. 
4. **Novel Bias Clamping Architecture:** The mathematical workaround discovered in this project (balancing rate-encoder probabilities with physical voltage clamping boundaries via tuning factors) forms the basis for a novel hardware SNN compiler patent for low-power edge devices.
