import numpy as np
import math
import torch 

class MyDataset(object):
    def __init__(self, songs):
        self.songs = np.array(songs)

        self.labels = self.songs[1:]
        
#         print(np.argmax(self.songs[-1]))
#         print(np.argmax(self.labels[-1]))
        
        
        
    def __len__(self):
        return math.ceil(len(songs)/100)

    
    def __getitem__(self, index):
        start = 100 * index
        end = start + 100
        
        
        
        if end > len(self.songs): 
            end = len(self.songs) - 1
            
        if end == len(self.songs):
            end -= 1
           

        return torch.from_numpy(np.array(self.songs[start : end])).float()[:,None,:], torch.from_numpy(self.labels[start :end]).long()
    
