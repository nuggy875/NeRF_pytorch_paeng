import os
import sys
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import visdom
from PIL import Image
from tqdm import tqdm, trange

from dataset import load_blender
from model import NeRF, get_positional_encoder
from process import run_model_batchify, get_rays, preprocess_rays
from utils import img2mse, mse2psnr, saveNumpyImage
from test import test, render

from configs.config import CONFIG_DIR, LOG_DIR, device

np.random.seed(0)


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
        images, poses, hwk, i_split = load_blender(
            cfg.data.root, cfg.data.name, cfg.data.half_res, cfg.data.white_bkgd)
        i_train, i_val, i_test = i_split
        img_h, img_w, img_k = hwk

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

    # ====  T R A I N I N G  ====
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 1
    result_best = {'i': 0, 'loss': 0, 'psnr': 0}

    for i in trange(start, cfg.training.N_iters+1):
        # [1] Get Target & Rays         >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        i_img = np.random.choice(i_train)
        target_img = images[i_img]
        target_img = torch.Tensor(target_img).to(device)
        target_pose = poses[i_img, :3, :4]
        rays_o, rays_d = get_rays(
            img_w, img_h, img_k, torch.Tensor(target_pose).to(device))

        # [2] Sampling Target & Rays    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # HxW 의 Pixel 중에서 n_rays_per_image(1024)개의 랜덤 샘플링
        coords = torch.stack(torch.meshgrid(torch.linspace(
            0, img_h-1, img_h), torch.linspace(0, img_w-1, img_w)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]
        selected_idx = np.random.choice(
            coords.shape[0], size=[cfg.render.n_rays_per_image], replace=False)  # default 1024
        selected_coords = coords[selected_idx].long()  # (N_rand, 2)
        # == Sample Rays ==
        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
        # == Sample Pixel ==
        target_img_s = target_img[selected_coords[:, 0],
                                  selected_coords[:, 1]]
        # [3] Preprocess Rays   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        rays = preprocess_rays(rays_o, rays_d, cfg)
        # [4] Run Model ==
        pred_rgb, disp, acc, extras = run_model_batchify(rays=rays,
                                                         fn_posenc=fn_posenc,
                                                         fn_posenc_d=fn_posenc_d,
                                                         model=model,
                                                         cfg=cfg)

        optimizer.zero_grad()
        loss = img2mse(target_img_s, pred_rgb)
        psnr = mse2psnr(loss)
        loss.backward()
        optimizer.step()

        # == GET Best result for training ==
        if result_best['psnr'] < psnr:
            result_best['i'] = i
            result_best['loss'] = loss
            result_best['psnr'] = psnr

        if i % cfg.training.idx_print == 0:
            print('i : {} , LOSS : {} , PSNR : {}'.format(i, loss, psnr))

        if i % cfg.training.idx_save == 0 and i > 0:
            checkpoint = {'idx': i,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            save_path = os.path.join(LOG_DIR, cfg.training.name)
            os.makedirs(save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(
                save_path, cfg.training.name + '_{}.pth.tar'.format(i)))

        # ====  T E S T I N G  ====
        if i % cfg.training.idx_test == 0 and i > 0:
            test(idx=i,
                 fn_posenc=fn_posenc,
                 fn_posenc_d=fn_posenc_d,
                 model=model,
                 test_imgs=torch.Tensor(images[i_test]).to(device),
                 test_poses=torch.Tensor(poses[i_test]).to(device),
                 hwk=hwk,
                 cfg=cfg)
        if i % cfg.training.idx_video == 0 and i > 0:
            render(idx=i,
                   fn_posenc=fn_posenc,
                   fn_posenc_d=fn_posenc_d,
                   model=model,
                   hwk=hwk,
                   cfg=cfg)

    print('BEST Result ) i : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device)
    main()
