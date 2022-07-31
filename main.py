import os
import sys
import time
from tkinter import Y
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import visdom
from PIL import Image
from tqdm import tqdm, trange

from dataset import load_blender, load_custom
from model import NeRF, get_positional_encoder
from process import run_model_batchify, get_rays, preprocess_rays, sample_rays_and_pixel, batchify_rays_and_render_by_chunk
from utils import img2mse, mse2psnr, saveNumpyImage

from train import train_each_iters
from test import test, render

from configs.config import CONFIG_DIR, LOG_DIR, DATA_NAME

np.random.seed(0)


@hydra.main(config_path=CONFIG_DIR, config_name=DATA_NAME)
def main(cfg: DictConfig):
    # == visdom ==
    if cfg.visualization.visdom:
        vis = visdom.Visdom(port=cfg.visualization.visdom_port)
    else:
        vis = None

    # == 1. LOAD DATASET (blender) ==
    if cfg.data.type == 'blender':
        images, poses, hwk, i_split = load_blender(
            data_root=cfg.data.root,
            data_name=cfg.data.name,
            testskip=cfg.testing.testskip,
            bkg_white=cfg.data.white_bkgd,
            reduce_res=cfg.data.reduce_res)
        i_train, i_val, i_test = i_split
        img_h, img_w, img_k = hwk

    elif cfg.data.type == 'custom':
        images, poses, hwk, i_split = load_custom(
            data_root=cfg.data.root,
            data_name=cfg.data.name,
            testskip=cfg.testing.testskip,
            bkg_white=cfg.data.white_bkgd,
            reduce_res=cfg.data.reduce_res)
        i_train = i_split[0]
        i_val = []
        i_test = []
        img_h, img_w, img_k = hwk

    device = torch.device('cuda:{}'.format(cfg.device.gpu_ids[cfg.device.rank]))

    # == 2. POSITIONAL ENCODING - Define Function ==
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)

    # output_ch = 5 if cfg.model.n_importance > 0 else 4
    skips = [4]

    # == 3. DEFINE MODEL (NeRF) ==
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)
    grad_vars = list(model.parameters())

    # == 4. OPTIMIZER ==
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=cfg.training.lr, betas=(0.9, 0.999))

    # == 5. RESUME ==
    if cfg.training.start_iter != 0:
        checkpoint = torch.load(os.path.join(
            LOG_DIR, cfg.training.name, cfg.training.name+'_{}.pth.tar'.format(cfg.training.start_iter)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('\nLoaded checkpoint from iter:{}'.format(
            int(cfg.training.start_iter)))

    # ====  T R A I N I N G  ====
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    result_best = {'i': 0, 'loss': 0, 'psnr': 0}

    for i in trange(cfg.training.start_iter + 1, cfg.training.N_iters+1):

        # ==== T R A I N I N G ====
        result_best = train_each_iters(i, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, optimizer,
                                       result_best, cfg)

        # ====  T E S T I N G  ====
        if i % cfg.training.idx_test == 0 and i > 0 and cfg.testing.mode_test:
            test(idx=i,
                 fn_posenc=fn_posenc,
                 fn_posenc_d=fn_posenc_d,
                 model=model,
                 test_imgs=torch.Tensor(images[i_test]).to(device),
                 test_poses=torch.Tensor(poses[i_test]).to(device),
                 hwk=hwk,
                 cfg=cfg)

        # ====  R E N D E R I N G  ====
        if i % cfg.training.idx_video == 0 and i > 0 and cfg.testing.mode_render:
            render(idx=i,
                   fn_posenc=fn_posenc,
                   fn_posenc_d=fn_posenc_d,
                   model=model,
                   hwk=hwk,
                   cfg=cfg)

    # Test & Render for Best result
    if cfg.testing.mode_test:
        test(idx='best',
             fn_posenc=fn_posenc,
             fn_posenc_d=fn_posenc_d,
             model=model,
             test_imgs=torch.Tensor(images[i_test]).to(device),
             test_poses=torch.Tensor(poses[i_test]).to(device),
             hwk=hwk,
             cfg=cfg)

    if cfg.testing.mode_render:
        render(idx='best',
               fn_posenc=fn_posenc,
               fn_posenc_d=fn_posenc_d,
               model=model,
               hwk=hwk,
               cfg=cfg)

    print('BEST Result ) i : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))


if __name__ == "__main__":
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(device)
    main()
