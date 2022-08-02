import argparse
import torch
import os

# device_ids = [0]
# device = torch.device('cuda:{}'.format(min(device_ids))
#                       if torch.cuda.is_available() else 'cpu')

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(
    os.path.abspath(os.path.dirname(os.path.realpath(__file__))))), "logs")

# [chair, drums, ficus, hotdog, lego, materials, mic, ship]
DATA_NAME = 'lego'

print('>> CONFIG File PATH : {}'.format(CONFIG_DIR))
print('>> LOG_DIR File PATH : {}'.format(LOG_DIR))
