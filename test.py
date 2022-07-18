import os
import numpy as np
import hydra
import torch
import time
from omegaconf import DictConfig
from tqdm import tqdm, trange

from dataset import load_blender
from model import NeRF, get_positional_encoder

device_ids = [1]
device = torch.device('cuda:{}'.format(min(device_ids))
                      if torch.cuda.is_available() else 'cpu')
CONFIG_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "configs")


def test(idx, model, test_img, test_pose, hwk, logdir, cfg):

    model.eval()

    checkpoint = torch.load(os.path.join(
        logdir, cfg.training.name, cfg.training.name+'.{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for i, pose in enumerate(tqdm(test_pose)):
            t = time.time()

    testdir = os.path.join(logdir, cfg.training.name,
                           'test_result_{:06d}'.format(idx))
    os.makedirs(testdir, exist_ok=True)


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    images, poses, render_poses, hwk, i_split = load_blender(
        cfg.data.root, cfg.data.name, cfg.data.half_res, cfg.data.white_bkgd)
    i_train, i_val, i_test = i_split
    img_h, img_w, img_k = hwk
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    # output_ch = 5 if cfg.model.n_importance > 0 else 4
    skips = [4]
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)
    test(20000, model,
         torch.Tensor(images[i_test]).to(device),
         torch.Tensor(poses[i_test]).to(device),
         hwk, os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs"), cfg)


if __name__ == '__main__':
    main()
