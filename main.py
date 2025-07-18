from MNIST.net import SleepyNet
from utils import train, test, mean_activity, get_activations

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Data loaders
transform=transforms.Compose([
        transforms.ToTensor(),
        ])

dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=100)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)


## Dividing the data in different tasks
# Group digits in pairs: (0,1), (2,3), ..., (8,9)
task_labels = [(i, i+1) for i in range(0, 10, 2)]
task_loaders = []

# Convert dataset to numpy for filtering
targets = np.array(dataset_train.targets)

for digit_pair in task_labels:
    # Find indices where labels are in the current digit pair
    indices = np.where((targets == digit_pair[0]) | (targets == digit_pair[1]))[0]
    
    # Create subset and dataloader
    subset = Subset(dataset_train, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=True)
    
    task_loaders.append(loader)
###


# Train the model with the whole dataset
print("Training ideal")
model_ideal = SleepyNet()
train(model_ideal, train_loader, epoch=1)

# Train model sequentially w sleep
print("Training sleep")
model_sleep = SleepyNet()
combined_datasets = []
for i,task_loader in enumerate(task_loaders):
    # Accumulate the datasets used so far
    combined_datasets.append(task_loader.dataset)

    # Create a combined DataLoader from all previous datasets
    combined_dataset = torch.utils.data.ConcatDataset(combined_datasets)
    combined_loader = DataLoader(combined_dataset, batch_size=100, shuffle=True)

    train(model_sleep, task_loader, epoch=1)
    # Dont sleep after the last iter
    if i != len(task_loaders):
        get_activations(model_sleep, combined_loader)
        mean_data = mean_activity(combined_loader) # Mean activity is from all past experiences
        model_sleep.sleep(mean_data, iters=50_000)

# Train model sequentially w/o sleep
print("Training baseline")
model_baseline = SleepyNet()
for task_loader in task_loaders:
    train(model_baseline, task_loader, epoch=1)

# Testing overall performance
print("ideal")
test(model_ideal, test_loader)
print("baseline")
test(model_baseline, test_loader)
print("sleep")
test(model_sleep, test_loader)
