import argparse
import torch
import os

# 2. device
device_ids = [0, 1]     # 사용할 Device ID 설정
device = torch.device('cuda:{}'.format(min(device_ids))
                      if torch.cuda.is_available() else 'cpu')


def parse(args):
    parser = argparse.ArgumentParser()
    # Visdom
    parser.add_argument('--visdom', type=bool, default=False)
    parser.add_argument('--visdom_port', type=str, default='8900')
    parser.add_argument('--visdom_step', type=int, default=100)
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--data_root', type=str,
                        default='/home/brozserver2/brozdisk/data/nerf')
    parser.add_argument('--data_type', type=str, default='blender')
    parser.add_argument('--data_name', type=str, default='lego')

    opts = parser.parse_args(args)
    return opts
