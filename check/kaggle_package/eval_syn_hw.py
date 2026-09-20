import numpy as np
import eval_hw
syn = np.load('synthetic_sample.npy')
print("Synthetic output:", eval_hw.run_snn(syn))
