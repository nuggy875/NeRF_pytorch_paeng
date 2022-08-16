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

from train import train_each_iters
from test import test, render
from scheduler import CosineAnnealingWarmupRestarts

from configs.config import CONFIG_DIR, LOG_DIR, DATA_NAME

np.random.seed(0)


@hydra.main(config_path=CONFIG_DIR, config_name=DATA_NAME)
def main(cfg: DictConfig):
    # == 0. Setting ==
    # = visdom =
    vis = visdom.Visdom(port=cfg.visualization.visdom_port) if cfg.visualization.visdom else None
    # = Set Device =
    device = torch.device('cuda:{}'.format(
        cfg.device.gpu_ids[cfg.device.rank]))

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

    # == 2. POSITIONAL ENCODING - Define Function ==
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)

    # == 3. DEFINE MODEL (NeRF) ==
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=[4]).to(device)

    # == 3. DEFINE Loss ==
    criterion = torch.nn.MSELoss()

    # == 4. OPTIMIZER ==
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.training.lr, betas=(0.9,0.999))

    # == 5. Scheduler ==
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=cfg.training.N_iters+1,
        cycle_mult=1.,
        max_lr=cfg.training.lr,
        min_lr=cfg.training.lr_min,
        warmup_steps=cfg.training.warmup_iter
        )

    # == 5. RESUME ==
    if cfg.training.start_iter != 0:
        checkpoint = torch.load(os.path.join(
            LOG_DIR, cfg.training.name, cfg.training.name+'_{}.pth.tar'.format(cfg.training.start_iter)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print('\n>> RESUME :: Loaded checkpoint from iter:{}'.format(
            int(cfg.training.start_iter)))

    # ====  T R A I N I N G  ====
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    result_best = {'i': 0, 'loss': 0, 'psnr': 0}

    for i in trange(cfg.training.start_iter + 1, cfg.training.N_iters+1):

        # ==== T R A I N I N G ====
        result_best = train_each_iters(i, i_train, images, poses, hwk, model, criterion, fn_posenc, fn_posenc_d, vis, optimizer,
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
        if i % cfg.training.idx_render == 0 and i > 0 and cfg.testing.mode_render:
            render(idx=i,
                   fn_posenc=fn_posenc,
                   fn_posenc_d=fn_posenc_d,
                   model=model,
                   hwk=hwk,
                   cfg=cfg,
                   device=device)
        
        scheduler.step()


    # # Test & Render for Best result
    # if cfg.testing.mode_test:
    #     test(idx='best',
    #          fn_posenc=fn_posenc,
    #          fn_posenc_d=fn_posenc_d,
    #          model=model,
    #          test_imgs=torch.Tensor(images[i_test]).to(device),
    #          test_poses=torch.Tensor(poses[i_test]).to(device),
    #          hwk=hwk,
    #          cfg=cfg)

    # if cfg.testing.mode_render:
    #     render(idx='best',
    #            fn_posenc=fn_posenc,
    #            fn_posenc_d=fn_posenc_d,
    #            model=model,
    #            hwk=hwk,
    #            cfg=cfg,
    #            device=device)

    print('BEST Result ) i : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))


if __name__ == "__main__":
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(device)
    main()
