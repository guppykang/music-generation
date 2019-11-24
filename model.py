import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import transforms, utils, models
import os


class Lstm(nn.Module):
    def __init__(self, input_dim, num_neurons, num_layers):
        super(Lstm, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, num_neurons, num_layers, dropout=0.2)
        self.fc = nn.Linear(num_neurons, input_dim)
    
    def forward(self, x, hidden):
        is_size_one = (x.shape[0] == 1)
        
        x, hidden = self.lstm_layer(x, hidden)
        x = self.fc(x.squeeze())
                
        if is_size_one:
            return x[None , :], hidden
        
        return x, hidden
    
    



class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, dropout=0.2)   
        self.fc = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x, hidden):
        is_size_one = (x.shape[0] == 1)

        x, hidden = self.rnn(x, hidden)
        
#         out = out.contiguous().view(-1, self.hidden_dim)

        x = self.fc(x.squeeze())
        
        if is_size_one:
            return x[None , :], hidden
        
        return x, hidden
    
   

