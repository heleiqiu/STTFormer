import torch
import torch.nn as nn
import math
import numpy as np

class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(st)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_frames * num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x