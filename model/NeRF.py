import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, input_ch_d: int, skips = [4]):
        super(NeRF, self).__init__()
        """
        D : Layer Depth in Network (8)  ||  W : Channels per Layer (256)
        input_ch : input from pos_enc (x,y,z)  ||  input_ch_d : input from pos_enc (d)
        output_ch : 5 ?   ||   skips : [4] ? 
        """
        self.D = D
        self.W = W
        self.input_ch_x = input_ch
        self.input_ch_d = input_ch_d
        self.skips = skips

        self.linear_x = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.linear_d = nn.ModuleList([nn.Linear(input_ch_d + W, W//2)])
        self.linear_feat = nn.Linear(W, W)
        self.linear_density = nn.Linear(W, 1)
        self.linear_color = nn.Linear(W//2 ,3)
        # self.linear_output = nn.Linear(W, output_ch)    # Input 값에 Direction 안 쓸때 사용 (5D -> 3D)


    def forward(self, x):

        outputs = x
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRF(D=8, W=256, input_ch=63, input_ch_d=27, output_ch=5, skips=[4]).to(device)