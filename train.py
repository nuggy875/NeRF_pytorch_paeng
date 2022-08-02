import os
import time
import torch
import numpy as np
from configs.config import LOG_DIR
from utils import mse2psnr, img2mse
from process import sample_rays_and_pixel, get_rays, preprocess_rays, run_model_batchify


def train_each_iters(i, i_train, images, poses, hwk, model, model_fine, fn_posenc, fn_posenc_d, vis, optimizer,
                     result_best, cfg):

    img_h, img_w, img_k = hwk

    i_img = np.random.choice(i_train)
    target_img = images[i_img]
    target_img = torch.Tensor(target_img)
    target_pose = poses[i_img, :3, :4]
    rays_o, rays_d = get_rays(img_w, img_h, img_k, torch.Tensor(target_pose).to(cfg.device.gpu_ids[cfg.device.rank]))

    # [2] Sampling Target & Rays    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # HxW 의 Pixel 중에서 n_rays_per_image(1024)개의 랜덤 샘플링
    rays_o, rays_d, target_img_s = sample_rays_and_pixel(
        rays_o, rays_d, target_img, cfg)

    # [3] Preprocess Rays   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    rays = preprocess_rays(rays_o, rays_d, cfg)

    # [4] Run Model         >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    pred_rgb, disp, acc, extras = run_model_batchify(rays=rays,
                                                     fn_posenc=fn_posenc,
                                                     fn_posenc_d=fn_posenc_d,
                                                     model=model,
                                                     model_fine=model_fine,
                                                     cfg=cfg)

    # assign target_img
    target_img_s = target_img_s.to(
        'cuda:{}'.format(cfg.device.gpu_ids[cfg.device.rank]))

    optimizer.zero_grad()
    loss = img2mse(target_img_s, pred_rgb)
    psnr = mse2psnr(loss)
    loss.backward()
    optimizer.step()

    if cfg.render.n_fine_pts_per_ray > 0:
        checkpoint = {'idx': i,
                        'model_state_dict': model.state_dict(),
                        'model_fine_state_dict': model_fine.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
    else:
        checkpoint = {'idx': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    save_path = os.path.join(LOG_DIR, cfg.training.name)
    os.makedirs(save_path, exist_ok=True)

    # ====  Print LOG  ====
    if i % cfg.training.idx_print == 0:
        print('i : {} , LOSS : {} , PSNR : {}'.format(i, loss, psnr))

    if i % cfg.training.idx_visdom == 0 and vis is not None:
        vis.line(X=torch.ones((1, 2)).cpu() * i,
                 Y=torch.Tensor([loss, psnr]).unsqueeze(0).cpu(),
                 win='loss_psnr_{}'.format(cfg.training.name),
                 update='append',
                 opts=dict(xlabel='iteration',
                           ylabel='loss_psnr',
                           title='LOSS&PSNR for {}'.format(
                               cfg.training.name),
                           legend=['LOSS', 'PSNR']))

    # ====  Save .pth file  ====
    if i % cfg.training.idx_save == 0 and i > 0:
        torch.save(checkpoint, os.path.join(
            save_path, cfg.training.name + '_{}.pth.tar'.format(i)))

    # == GET Best result for training ==
    if result_best['psnr'] < psnr:
        result_best['i'] = i
        result_best['loss'] = loss
        result_best['psnr'] = psnr
        torch.save(checkpoint, os.path.join(
            save_path, cfg.training.name + '_best.pth.tar'))

    return result_best
