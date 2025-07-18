# SleePyNets

Implementation of the paper "**Sleep-like Unsupervised Replay Reduces Catastrophic Forgetting in Artificial Neural Networks**" in Python. See the original repository  [here](https://github.com/tmtadros/SleepReplayConsolidation)

## [Provisional] File Structure

- `main.py`: contains the pipeline for loading the dataset, training and testing the networks for evaluation.;

- `MINST/net.py`: Implementation of the neural network in PyTorch. Different than the original paper, the same class is used for the ANN and then later for the sleep simulation. Messy implementation, needs some changes!

- `count_spikes.py` and `plot_count_spikes.py`: Made in order to check how the network behaves after sleeping for too long. Saves and then plots the average number of spikes at each layer in a population of `n_nets` networks.
