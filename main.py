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
from render import rendering
from test import test

device_ids = [1]
device = torch.device('cuda:{}'.format(min(device_ids))
                      if torch.cuda.is_available() else 'cpu')
np.random.seed(0)

CONFIG_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "configs")
LOG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def saveNumpyImage(img):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/white_bkgd_false.jpg')


def get_rays(W, H, K, c2w):
    '''
    img_k = [3,3] pose = [3,4]
    Transpose Image Plane Coordinate to Normalized Plane ([x',y',1] -> [u,v,1])
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    '''
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0],
                        -(j-K[1][2])/K[1][1],
                        -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # rays_d = dirs @ c2w[:3, :3].T # TODO dot product test
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


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
        images, poses, render_poses, hwk, i_split = load_blender(
            cfg.data.root, cfg.data.name, cfg.data.half_res, cfg.data.white_bkgd)
        i_train, i_val, i_test = i_split
        img_h, img_w, img_k = hwk

    # saveNumpyImage(images[0])         # Save Image for testing

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

    start = 0
    result_best = {'i': 0, 'loss': 0, 'psnr': 0}

    for i in trange(start, cfg.training.N_iters):
        i_img = np.random.choice(i_train)
        target_img = images[i_img]
        target_img = torch.Tensor(target_img).to(device)
        target_pose = poses[i_img, :3, :4]
        rays_o, rays_d = get_rays(
            img_w, img_h, img_k, torch.Tensor(target_pose).to(device))

        # == Sampling Target (number of rays per image) (default : 1024) ==
        # HxW 의 Pixel 중에서 1024개의 랜덤 샘플링
        coords = torch.stack(torch.meshgrid(torch.linspace(
            0, img_h-1, img_h), torch.linspace(0, img_w-1, img_w)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]
        selected_idx = np.random.choice(
            coords.shape[0], size=[cfg.render.n_rays_per_image], replace=False)  # default 1024
        selected_coords = coords[selected_idx].long()  # (N_rand, 2)

        # == Sample Rays ==
        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        # == Sample Pixel ==
        target_img_s = target_img[selected_coords[:, 0],
                                  selected_coords[:, 1]]

        # == Render (get Pred) ==
        pred_rgb, disp, acc, extras = rendering(rays=batch_rays, fn_posenc=fn_posenc,
                                                fn_posenc_d=fn_posenc_d, model=model, cfg=cfg)

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
            torch.save(checkpoint, os.path.join(
                LOG_DIR, cfg.training.name, cfg.training.name+'.{}.pth.tar'.format(i)))

        # ====  T E S T I N G  ====
        if i % cfg.training.idx_test == 0 and i > 0:
            test(idx=i, model=model,
                 test_img=torch.Tensor(images[i_test]).to(device),
                 test_pose=torch.Tensor(poses[i_test]).to(device),
                 hwk=hwk, logdir=LOG_DIR, cfg=cfg)

    print('BEST Result ) i : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device)
    main()
