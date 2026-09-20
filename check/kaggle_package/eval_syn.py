import numpy as np
syn = np.load('synthetic_sample.npy')
amb = np.load('sample_ambient_thresholds.npy')
drn = np.load('sample_drone_thresholds.npy')
print("Synthetic matches Ambient:", np.array_equal(syn, amb))
