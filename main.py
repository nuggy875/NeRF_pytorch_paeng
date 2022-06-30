import os
import sys
import time
import numpy as np
import torch
import hydra
import visdom
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm, trange

from dataset import load_blender
from model import NeRF, get_positional_encoder

device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
np.random.seed(0)

CONFIG_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "configs")
LOG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")


def saveNumpyImage(img):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/white_bkgd_false.jpg')


def get_rays(W, H, K, c2w):
    '''
    img_k = [3,3] pose = [3,4]
    '''
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)        # dirs [W, H, 3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
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

        img_h, img_w, img_focal = hwf
        img_h, img_w = int(img_h), int(img_w)
        # FIXME) K의 용도 확인할 것
        img_k = np.array([
            [img_focal, 0, 0.5*img_w],
            [0, img_focal, 0.5*img_h],
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
    grad_vars = list(model.parameters())


    # == 4. OPTIMIZER ==
    print(cfg.training.lr)
    optimizer = torch.optim.Adam(params=grad_vars, lr=cfg.training.lr, betas=(0.9, 0.999))

    # == T R A I N I N G ==
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    N_iters = 200000
    start = 0
    for i in trange(start, N_iters):
        time_start = time.time()
        # i_img = np.random.choice(i_train)
        i_img = 0                                                   # FIXME for testing
        target_img = images[i_img]
        target_img = torch.Tensor(target_img).to(device)
        target_pose_test = poses[i_img]
        target_pose = poses[i_img, :3, :4]
        rays_o, rays_d = get_rays(img_w, img_h, img_k, torch.Tensor(target_pose))

        # FIXME precrops SKIP


        # Render ( get GT )

        optimizer.zero_grad()
        # TODO >> LOSS
        # MSE (target, gt)
        # MSE -> PSNR
        # loss.backward
        optimizer.step()


if __name__ == "__main__":
    # main()
    

    # for testing RAYS
    W = 400
    H = 500
    F = 555.55

    
    K = np.array([
            [F, 0, 0.5*W],
            [0, F, 0.5*H],
            [0, 0, 1]
        ])
    # [3, 4] 상태의 train_1 pose (0,0,0,1) 지움
    pose = np.array([[-0.9999021887779236,0.004192245192825794,-0.013345719315111637,-0.05379832163453102],[-0.013988681137561798,-0.2996590733528137,0.95394366979599,3.845470428466797],[-4.656612873077393e-10,0.9540371894836426,0.29968830943107605,1.2080823183059692]])
    pose = torch.Tensor(pose)
    get_rays(W, H, K, pose)
