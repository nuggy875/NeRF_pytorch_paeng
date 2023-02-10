import os
import visdom
import torch
import numpy as np
from tqdm import trange

from config import get_args_parser, LOG_DIR
from dataset import load_blender, load_llff, load_custom
from model import NeRF, get_positional_encoder
from scheduler import CosineAnnealingWarmupRestarts
from rays import get_rays_np
from train import train
from test import test, render
from utils import GetterRayBatchIdx


def main_worker(rank, opts):
    # ====  0. Setting  ====
    # >> Config & Argparse
    opts.rank = rank
    print(f'\n\n{opts}')

    # >> visdom
    vis = visdom.Visdom(port=opts.visdom_port) if opts.visdom else None

    # >> Set Device
    device = opts.gpu_ids[rank]
    print(f"\n>> Device : {device} for training")
    assert True

    # ====  1. Load Dataset  ====
    print(
        f"\n>> Loading Dataset... : [{opts.data_type}], from '{opts.data_root}'")
    if opts.data_type == "blender":
        images, gt_camera_param, hw, i_split = load_blender(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            bkg_white=opts.bkg_white
        )
        render_poses = None
    elif opts.data_type == 'llff':
        images, gt_camera_param, hw, i_split, render_poses = load_llff(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            colmap_relaunch=opts.colmap_relaunch
        )
    elif opts.data_type == 'custom':
        images, gt_camera_param, hw, i_split, nf = load_custom(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            video_batch=opts.video_batch,
            colmap_relaunch=opts.colmap_relaunch
        )
        render_poses = None
        opts.near, opts.far = nf
    i_train, i_val, i_test = i_split
    img_h, img_w = hw
    (gt_intrinsic, gt_extrinsic) = gt_camera_param
    print(
        f"\n>> Dataset Loaded Completely!\n---- Image shape (N, H, W, 3) : {images.shape}")

    # ====  2. Model  ====
    # >> Positional Encoding
    fn_posenc, input_ch = get_positional_encoder(opts.L_x)
    fn_posenc_d, input_ch_d = get_positional_encoder(opts.L_d)
    # >> DEFINE MODEL (NeRF) ==
    model = NeRF(D=opts.netDepth, W=opts.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d,
                 skips=[4], gt_camera_param=(gt_intrinsic, gt_extrinsic),
                 device=device).to(device)

    # == 3. LOSS ==
    criterion = torch.nn.MSELoss()

    # == 4. OPTIMIZER ==
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=opts.lr, betas=(0.9, 0.999))

    # == 5. Scheduler ==
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=opts.iter_N+1,
        cycle_mult=1.,
        max_lr=opts.lr,
        min_lr=opts.lr_min,
        warmup_steps=opts.iter_warmup
    )
    # == 6. Set Global Batch ==
    getter_ray_batch_idx = None
    if opts.global_batch:
        print('>> [Global Batching] Random Ray for all images')
        rays = np.stack([get_rays_np(img_h, img_w, gt_intrinsic, p)
                        for p in gt_extrinsic[:, :3, :4]], 0)
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        np.random.shuffle(rays_rgb)
        rays_rgb = torch.Tensor(rays_rgb).to(f'cuda:{opts.gpu_ids[opts.rank]}')

        # rays_rgb batch getter for global batch
        getter_ray_batch_idx = GetterRayBatchIdx(rays_rgb)
    else:
        print('>> No Global Batch, Sampling from one image per iteration')

    # == 7. RESUME ==
    if opts.iter_start != 0:
        checkpoint = torch.load(os.path.join(
            LOG_DIR, opts.exp_name, opts.exp_name+'_{}.pth.tar'.format(opts.iter_start)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('\n\n>> RESUME :: Loaded checkpoint from iter:{}'.format(
            int(opts.iter_start)))
    else:
        print('\n\n>> Training from scratch...')

    print(
        f"\n==== TRAINING START ==== \n> TRAIN views are {i_train}\n> TEST views are {i_test}\n> VAL views are {i_val}")

    for i in trange(opts.iter_start+1, opts.iter_N+1):
        # ==== T R A I N I N G ====
        train(idx=i,
              i_train=i_train,
              images=images,
              gt_cam_param=[gt_intrinsic, gt_extrinsic],
              hw=hw,
              model=model,
              criterion=criterion,
              posenc=[fn_posenc, fn_posenc_d],
              optimizer=optimizer,
              global_batch_idx=getter_ray_batch_idx,
              vis=vis,
              opts=opts)

        # # ====  T E S T I N G  ====
        if i % opts.idx_test == 0 and i > 0 and opts.mode_test:
            test(idx=i,
                 i_test=i_test,
                 posenc=[fn_posenc, fn_posenc_d],
                 model=model,
                 test_imgs=torch.Tensor(images[i_test]).to(device),
                 gt_intrinsic=gt_intrinsic,
                 gt_extrinsic=torch.Tensor(gt_extrinsic[i_test]).to(device),
                 hw=hw,
                 opts=opts)

        # ====  R E N D E R I N G  ====
        if i % opts.idx_render == 0 and i > 0 and opts.mode_render:
            render(idx=i,
                   posenc=[fn_posenc, fn_posenc_d],
                   model=model,
                   gt_intrinsic=gt_intrinsic,
                   render_pose=render_poses,
                   hw=hw,
                   opts=opts)

        scheduler.step()


if __name__ == "__main__":
    opts = get_args_parser()
    '''
    FIXME ) distributed computing
    Multi-GPU 사용을 위한 추가 개발 필요, 현재는 지정한 GPU의 첫번째 (rank=0) 만 사용하도록 하였음.
    '''
    rank = 0     # QUICK FIX
    main_worker(rank, opts)
