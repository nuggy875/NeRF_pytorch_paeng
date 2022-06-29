import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, output_ch: int, input_ch_d: int, skips = [4]):
        super(NeRF, self).__init__()
        """
        
        """
        print(D)
        print(W)

    def forward(self, x):

        outputs = x
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRF(D=8, W=256, input_ch=63, input_ch_d=27, output_ch=5, skips=[4]).to(device)