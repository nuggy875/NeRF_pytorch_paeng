import os
import sys
import numpy as np
import torch
import hydra
import visdom
from PIL import Image
from omegaconf import DictConfig

from dataset import load_blender
from model import NeRF, get_positional_encoder

CONFIG_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "configs")
LOG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")


def saveNumpyImage(img):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/white_bkgd_false.jpg')


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    # opts = parse(sys.argv[1:])

    # == visdom ==
    if cfg.visualization.visdom:
        vis = visdom.Visom(port=cfg.visualization.visdom_port)
    else:
        vis = None

    # == 1. LOAD DATASET (blender) ==
    if cfg.data.type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender(
            cfg.data.root, cfg.data.name, cfg.data.half_res)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        # 뒤 흰배경 처리 (png 빈 부분을 불러오면 검은화면이 됨)
        # FIXME) alpha 값을 왜 없애는지 확인할 것 -> density 어디서 사용?
        if cfg.data.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

        H, W, focal = hwf
        H, W = int(H), int(W)
        # FIXME) K의 용도 확인할 것
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # saveNumpyImage(images[0])         # Save Image for testing

    # == 2. POSITIONAL ENCODING - Define Function ==
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)

    # output_ch = 5 if cfg.model.n_importance > 0 else 4
    skips = [4]     # FIXME what is this for?

    # == 3. DEFINE MODEL (NeRF) ==
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)


if __name__ == "__main__":
    main()
