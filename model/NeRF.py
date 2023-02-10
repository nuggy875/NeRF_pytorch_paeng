import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from .NeRFHelper import Exp


class NeRFModule(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, input_ch_d: int, skips = [4]):
        super(NeRFModule, self).__init__()
        """
        D : Layers in Network (8)  ||  W : Channels per Layer (256)
        input_ch : input from pos_enc (x,y,z)  ||  input_ch_d : input from pos_enc (d)
        output_ch : 5 ?   ||   skips : [4]
        """
        self.D = D
        self.W = W
        self.input_ch_x = input_ch
        self.input_ch_d = input_ch_d
        self.skips = skips

        self.linear_x = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.linear_d = nn.Linear(input_ch_d + W, W//2)

        self.linear_feat = nn.Linear(W, W)
        self.linear_density = nn.Linear(W, 1)
        self.linear_color = nn.Linear(W//2 ,3)


    def forward(self, x):
        input_x, input_d = torch.split(x, [self.input_ch_x, self.input_ch_d], dim=-1)
        out = input_x
        # [0~7] for 8 Layers 
        for i, l in enumerate(self.linear_x):
            out = self.linear_x[i](out)
            out = F.relu(out)
            if i in self.skips:
                out = torch.cat([input_x, out], dim=-1)
        # [8-1], [8-2]
        density = self.linear_density(out)
        feature = self.linear_feat(out)
        # [9]
        out = torch.cat([feature, input_d], dim=-1)
        out = self.linear_d(out)
        out = F.relu(out)
        # [10]
        out = self.linear_color(out)
        result = torch.cat([out, density], dim=-1)
        return result


class NeRF(nn.Module):
    def __init__(self, D:int, W:int, input_ch: int, input_ch_d: int, skips = [4], gt_camera_param = None, device = None):
        super().__init__()
        self.model_coarse = NeRFModule(D, W, input_ch, input_ch_d, skips)
        self.model_fine = NeRFModule(D, W, input_ch, input_ch_d, skips)
        self.apply(self._init_weights)
        self.gt_intrinsic, self.gt_extrinsic = gt_camera_param      # Ground Truth Value for INTRINSIC & EXTRINSIC

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def get_camera_gt(self):
        return self.gt_intrinsic, self.gt_extrinsic

    def forward(self, x, is_fine:bool = False):
        '''
        Coarse Network : is_fine=False,
        Fine Network : is_fine=True
        '''
        if is_fine:
            return self.model_fine(x)
        else:
            return self.model_coarse(x)
