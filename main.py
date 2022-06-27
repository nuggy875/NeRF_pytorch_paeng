import os
import sys
import torch
import hydra
import visdom
from omegaconf import DictConfig

from configs.config import parse
from dataset import load_blender

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    # 1. configuration
    # opts = parse(sys.argv[1:])

    # 2. visdom
    if cfg.visdom:
        vis = visdom.Visom(port=cfg.visdom_port)
    else:
        vis = None

    print(cfg.data.type)
    # # 3. dataset (blender)
    # if cfg.data.type == 'blender':
    #     load_blender(cfg.data_root, cfg.data_root)        # root , 'lego'
    

if __name__ == "__main__":
    main()
