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


#TODO : can add dropout in the lstm
class Lstm(nn.Module):
    def __init__(self, input_dim, num_neurons, num_layers):
        super(Lstm, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, num_neurons, num_layers, dropout=0)
        self.fc = nn.Linear(100, 93)
    
    def forward(self, x, hidden):
        is_size_one = (x.shape[0] == 1)
        
        x, hidden = self.lstm_layer(x, hidden)
        x = self.fc(x.squeeze())
        
        if is_size_one:
            return x[None , :], hidden
        
        return x, hidden


