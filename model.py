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
    def __init__(self, input_dim, num_neurons, num_layers, hidden):
        super(Lstm, self).__init__()
        self.hidden = hidden
        self.lstm_layer = nn.LSTM(input_dim, num_neurons, num_layers, dropout=0)
        
    
    def forward(self, x, hidden):
        x, hidden = self.lstm_layer(self, x, hidden)
        return x, hidden


