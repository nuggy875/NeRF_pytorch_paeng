import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __init__(self, D, W):
        super(NeRF, self).__init__()
        print(D)
        print(W)

    def forward(self, x):

        outputs = x
        return outputs
