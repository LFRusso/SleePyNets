import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SleepyNet(nn.Module):
    
    def __init__(self):
        super(SleepyNet, self).__init__()
        
        # ANN layers
        self.fc1 = nn.Linear(784, 1200, bias=False)
        self.fc2 = nn.Linear(1200, 1200, bias=False)
        self.fc3 = nn.Linear(1200, 10, bias=False)
        self.init_weights()


        # ==========================
        # SNN simulation parameters
        ## Membrane potential of each spiking layer (input layer does not have that)
        self.V1 = torch.zeros(1200, dtype=torch.float32)
        self.V2 = torch.zeros(1200, dtype=torch.float32)
        self.V3 = torch.zeros(10, dtype=torch.float32)


        ## Thresholds for the Heaviside
        self.th1 = 14.548273
        self.th2 = 44.560317
        self.th3 = 38.046326

        ## Increments and decrements for STDP
        self.inc = 0.032064
        self.dec = 0.003344

        # Max activations for each layer
        self.max_activations = {"a1": 0, "a2": 0, "a3": 0}

        ## Some more params from the paper
        self.dt = 0.001 # s
        self.max_rate = 239.515363 # Hz
        self.alpha_scale = 55.882454 # No idea whats this
        self.decay_rate = 0.999

        self.get_weight_scales()
    
    def init_weights(self):
        # Set desired range (this is the one used in the paper)
        low = -0.1
        high = 0.1

        # Apply uniform initialization
        nn.init.uniform_(self.fc1.weight, a=low, b=high)
        nn.init.uniform_(self.fc2.weight, a=low, b=high)
        nn.init.uniform_(self.fc3.weight, a=low, b=high)


    # ANN forward using ReLU
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        self.max_activations["a1"] = torch.max(x).detach()
        self.max_activations["a1"] = max(self.max_activations["a1"], 0)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        self.max_activations["a2"] = torch.max(x).detach()
        self.max_activations["a2"] = max(self.max_activations["a2"], 0)
        
        x = self.fc3(x)
        self.max_activations["a3"] = torch.max(x).detach()
        self.max_activations["a3"] = max(self.max_activations["a3"], 0)
        

        output = F.log_softmax(x, dim=1)
        return output

    # Spike simulation for sleep. Performs unsupervised STDP
    def sleep(self, mean_data, iters=100):
        s1, s2, s3 = self.get_weight_scales()
        #print("Starting sleep")

        # Saving total number of spikes
        spike_vec_l0 = []
        spike_vec_l1 = []
        spike_vec_l2 = []
        spike_vec_l3 = []
        for i in range(iters):
            dW01, dW12, dW23 = torch.zeros([784,1200]), torch.zeros([1200,1200]), torch.zeros([1200,10]) # Weight deltas

        
            ## Original code generates the input spikes outside. Here, I get the Poission generated spikes inside each loop
            # Poisson distributed spikes based on data
            spike_chance = torch.rand(mean_data.shape)
            # Here, for some reason, the authors multiply by a 'rescale_factor' not really explained...
            spike_chance *= 1/(self.dt*self.max_rate)/2 # This is the rescale_factor
            spikes = torch.ones(mean_data.shape)
            spikes[spike_chance >= mean_data] = 0 # the spikes are generated based on the mean of the data used in training

            # Selecting window
            [_, H, W] = mean_data.shape
            window_size = 10
            i = torch.randint(0, H - window_size + 1, (1,)).item()
            j = torch.randint(0, W - window_size + 1, (1,)).item()
            mask = torch.zeros_like(spikes)
            mask[i:i+window_size, j:j+window_size] = 1
            spikes *= mask  # keep only spikes within the window
        
            x = spikes
            x = torch.flatten(x, 1)
            spikes_l0 = x.detach().numpy().copy()
            
            I = x @ ( self.alpha_scale * s1 * self.state_dict()["fc1.weight"]).T # Incoming spikes * Weighted input
            #self.V1 = F.relu(self.V1 * self.decay_rate + I) # Leaky integrate. Relu here is just it does not get negative 
            self.V1 = self.V1 * self.decay_rate + I # Leaky integrate. 
            x = torch.heaviside(self.V1 - self.th1, values=torch.tensor([0.])) # Fire
            self.V1[x.type(torch.int)==1] = 0 # Reset
            spikes_l1 = x.detach().numpy().copy()
            #print(x.sum())
    
            I = x @ ( self.alpha_scale * s2 * self.state_dict()["fc2.weight"]).T # Incoming spikes * Weighted input
            #self.V2 = F.relu(self.V2 * self.decay_rate + I) # Leaky integrate. Relu here is just it does not get negative
            self.V2 = self.V2 * self.decay_rate + I # Leaky integrate. 
            x = torch.heaviside(self.V2 - self.th2, values=torch.tensor([0.])) # Fire
            self.V2[x.type(torch.int)==1] = 0 # Reset
            spikes_l2 = x.detach().numpy().copy()
            #print(x.sum())
    
            I = x @ ( self.alpha_scale * s3 * self.state_dict()["fc3.weight"]).T # Incoming spikes * Weighted input
            #self.V3 = F.relu(self.V3 * self.decay_rate + I) # Leaky integrate. Relu here is just it does not get negative
            self.V3 = self.V3 * self.decay_rate + I # Leaky integrate. 
            x = torch.heaviside(self.V3 - self.th3, values=torch.tensor([0.])) # Fire
            self.V3[x.type(torch.int)==1] = 0 # Reset
            spikes_l3 = x.detach().numpy().copy()


           ## Performing STDP
                # Matrix trick to perform STDP/Hebbian learning
                # If pre-synaptic=1 and post-synaptic=1, positive grad; If pre-synaptic=0 and post-synaptic=1, negative grad
                # if post-synaptic = 0, do nothing            
            spikes_l0 = spikes_l0[0]
            spike_vec_l0.append(np.sum(spikes_l0))
            spikes_l1 = spikes_l1[0]
            spike_vec_l1.append(np.sum(spikes_l1))
            spikes_l2 = spikes_l2[0]
            spike_vec_l2.append(np.sum(spikes_l2))
            spikes_l3 = spikes_l3[0]
            spike_vec_l3.append(np.sum(spikes_l3))
            
            # Weights 0-1
            auxdW01 = spikes_l0[None,:].T @ spikes_l1[None,:]
            auxdW01[:, spikes_l1==1] = (auxdW01[:, spikes_l1==1] * (self.inc + self.dec)) - self.dec
            dW01 = torch.tensor(auxdW01, dtype=torch.float32) 

            # Weights 1-2
            auxdW12 = spikes_l1[None,:].T @ spikes_l2[None,:]
            auxdW12[:, spikes_l2==1] = (auxdW12[:, spikes_l2==1] * (self.inc + self.dec)) - self.dec
            dW12 = torch.tensor(auxdW12, dtype=torch.float32)
                        
            # Weights 2-3
            auxdW23 = spikes_l2[None,:].T @ spikes_l3[None,:]
            auxdW23[:, spikes_l3==1] = (auxdW23[:, spikes_l3==1] * (self.inc + self.dec)) - self.dec
            dW23 = torch.tensor(auxdW23, dtype=torch.float32)

        
            self.state_dict()["fc1.weight"] += dW01.T * self.sigmoid(self.state_dict()["fc1.weight"])
            self.state_dict()["fc2.weight"] += dW12.T * self.sigmoid(self.state_dict()["fc2.weight"])
            self.state_dict()["fc3.weight"] += dW23.T * self.sigmoid(self.state_dict()["fc3.weight"])

        return x, spike_vec_l0, spike_vec_l1, spike_vec_l2, spike_vec_l3

    # Find the scale to balance the weights based on the past activations
    def get_weight_scales(self):
        s1 = 1/np.max([self.max_activations["a1"], torch.max(self.state_dict()["fc1.weight"].flatten())])
        s2 = 1/(s1*np.max([self.max_activations["a2"], torch.max(self.state_dict()["fc2.weight"].flatten())]))
        s3 = 1/(s2*np.max([self.max_activations["a3"], torch.max(self.state_dict()["fc3.weight"].flatten())]))
        return s1, s2, s3

    def sigmoid(self, W):
        return 2 * (1.0 - (1.0 / (1 + torch.exp(-W/0.001))));
