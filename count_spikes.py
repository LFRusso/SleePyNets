from MNIST.net import SleepyNet
from utils import train, test, mean_activity, get_activations

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt


# Data loaders
transform=transforms.Compose([
        transforms.ToTensor(),
        ])

dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=100)

n_iters = 50_000 # Length of sleep
n_nets = 300 # Number of networks to get the avg from
sv0 = np.zeros(n_iters) # Input spikes
sv1 = np.zeros(n_iters) # Spikes from layer 1 (1200 neurons)
sv2 = np.zeros(n_iters) # Spikes from layer 2 (1200 neurons)
sv3 = np.zeros(n_iters) # Spikes from the output layer (10 neurons)
for i in range(n_nets):
    # Normal train
    model= SleepyNet()
    train(model, train_loader, epoch=1)

    # Sleep
    get_activations(model, train_loader)
    mean_data = mean_activity(train_loader) # Mean activity is from all past experiences
    _, sv0_aux, sv1_aux, sv2_aux, sv3_aux = model.sleep(mean_data, iters=n_iters)
    sv0 += np.array(sv0_aux)
    sv1 += np.array(sv1_aux)
    sv2 += np.array(sv2_aux)
    sv3 += np.array(sv3_aux)

sv = np.array([sv0, sv1, sv2, sv3])
np.save('data/layer_spikes.npy', sv)
