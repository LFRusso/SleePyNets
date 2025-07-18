import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

n_iters = 50_000
n_nets = 300
sv = np.load('data/layer_spikes.npy')

# Smooth the data
kernel = np.ones(10) 
sv0 = np.convolve(sv[0]/n_nets, kernel, mode='same')
sv1 = np.convolve(sv[1]/n_nets, kernel, mode='same')
sv2 = np.convolve(sv[2]/n_nets, kernel, mode='same')
sv3 = np.convolve(sv[3]/n_nets, kernel, mode='same')

t = np.arange(n_iters)
plt.plot(t, sv0/720, label="input spikes")
plt.plot(t, sv1/1200, label="L1 spikes")
plt.plot(t, sv2/1200, label="L2 spikes")
plt.plot(t, sv3/10, label="L3 spikes")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("Filtered avg. # of spikes per. neuron in the layer")
plt.title("Avg spikes at each layer in a population of 300 nets")
plt.show()
