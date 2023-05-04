import os
import torch
import numpy as np

from config import LOG_DIR
from methods.visualize import visualize_extrinsic, visualize_ray
from nerf_process import batchify_rays_and_render_by_chunk
from rays import make_o_d, sample_rays_and_pixel
from utils import mse2psnr


def train(idx, i_train, images, gt_cam_param, hw, model, criterion, posenc, optimizer, global_batch_idx, vis, opts):

    model.train()
    # [0] GET PARAMETER or GT from Model
    img_h, img_w = hw
    gt_intrinsic, gt_extrinsic = gt_cam_param
    param_intrinsic = torch.from_numpy(gt_intrinsic).to(
        f'cuda:{opts.gpu_ids[opts.rank]}')
    param_extrinsic = torch.from_numpy(gt_extrinsic).to(
        f'cuda:{opts.gpu_ids[opts.rank]}')

    # [1] Sampling Target & Rays    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # == [1-1]. global batch ==
    if global_batch_idx is not None and opts.global_batch:
        # batch가 size를 넘어가면 shuffle
        i_batch, rays_rgb, epoch = global_batch_idx(opts.N_rays)
        # Random over all images
        batch = rays_rgb[i_batch - opts.N_rays:i_batch]  # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_img = batch[:2], batch[2]
        rays_o, rays_d = batch_rays

    # == [1-2]. Sampling within one image ==
    else:
        # sample train index
        i_img = np.random.choice(i_train)
        target_img = torch.from_numpy(images[i_img]).type(
            torch.float32).to(f'cuda:{opts.gpu_ids[opts.rank]}')
        target_pose = param_extrinsic[i_img, :3, :4]
        H, W = target_img.shape[0], target_img.shape[1]
        # make rays_o and rays_d
        rays_o, rays_d = make_o_d(img_w, img_h, param_intrinsic, target_pose)
        rays_o, rays_d, target_img = sample_rays_and_pixel(
            idx, rays_o, rays_d, target_img, opts)
        # visualize_ray(rays_o, rays_d, (H, W),
        #               f'cuda:{opts.gpu_ids[opts.rank]}', )

    # [2] Run Model         >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ** assign target_img to cuda **
    target_img = target_img.to(f'cuda:{opts.gpu_ids[opts.rank]}')

    pred_rgb_c, pred_disp_c, pred_rgb_f, pred_disp_f = batchify_rays_and_render_by_chunk(
        rays_o, rays_d, model, posenc, img_h, img_w, param_intrinsic, opts)

    # == Optimizer
    optimizer.zero_grad()

    # [3] Get LOSS          >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    loss = criterion(pred_rgb_c, target_img)
    if opts.N_samples_f > 0:
        loss_c = loss
        psnr_c = mse2psnr(loss_c)
        loss_f = criterion(pred_rgb_f, target_img)
        psnr_f = mse2psnr(loss_f)
        loss = loss_c + loss_f
    psnr = mse2psnr(loss)

    loss.backward()
    optimizer.step()

    # [4] VISDOM & PRINT    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if opts.N_samples_f > 0:
        if idx % opts.idx_print == 0:
            print('i : {} , Loss_C : {} , Loss_F : {} , Total_Loss : {} , PSNR_C : {} , PSNR_F : {}'.format(
                idx, loss_c, loss_f, loss, psnr_c, psnr_f))

        if idx % opts.idx_vis == 0 and vis is not None:
            vis.line(X=torch.ones((1, 4)).cpu() * idx,
                     Y=torch.Tensor(
                         [loss_c, loss_f, psnr_c, psnr_f]).unsqueeze(0).cpu(),
                     win='loss_psnr_{}'.format(opts.exp_name),
                     update='append',
                     opts=dict(xlabel='iteration',
                               ylabel='loss_psnr',
                               title='LOSS&PSNR for {}'.format(opts.exp_name),
                               legend=['LOSS_Coarse', 'LOSS_Fine', 'PSNR_Coarse', 'PSNR_Fine']))

    else:
        if idx % opts.idx_print == 0:
            print('i : {} , LOSS : {} , PSNR : {}'.format(idx, loss, psnr))

        if idx % opts.idx_vis == 0 and vis is not None:
            vis.line(X=torch.ones((1, 2)).cpu() * idx,
                     Y=torch.Tensor([loss, psnr]).unsqueeze(0).cpu(),
                     win='loss_psnr_{}'.format(opts.exp_name),
                     update='append',
                     opts=dict(xlabel='iteration',
                               ylabel='loss_psnr',
                               title='LOSS&PSNR for {}'.format(
                                   opts.exp_name),
                               legend=['LOSS', 'PSNR']))

    # [5] Save .pth    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    checkpoint = {'idx': idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    save_path = os.path.join(LOG_DIR, opts.exp_name)
    os.makedirs(save_path, exist_ok=True)

    if idx % opts.idx_save == 0 and idx > 0:
        torch.save(checkpoint, os.path.join(
            save_path, opts.exp_name + '_{}.pth.tar'.format(idx)))

    # [6] Save Extrinsic Visualization    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if idx == 1:
        visualize_extrinsic(
            iter=idx, poses=gt_cam_param[1], idx_list=i_train, opts=opts, intrinsics=gt_cam_param[0], hw=hw)


if __name__ == "__main__":
    print('training')
