
import os
import sys
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


# theta, -30, 4 : shperical to cartesian
# theta : [-180, -171, -162 ..., 171] (40)
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender(data_root: str, data_name: str, half_res: bool, autodownload: bool = True, testskip: int = 8, bkg_white: bool = True):
    print(f"\n\nLoading Dataset {data_name}, from {data_root}")
    splits = ['train', 'val', 'test']
    metas = {}
    # Load Annotation (JSON)
    for s in splits:
        with open(os.path.join(data_root, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    # Load Images
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(data_root, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # keep all 4 channels (RGBA)      # Normalize (0 ~ 1)     # [100,800,800,4]
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        # [100,4,4]
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                               for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    half_res = False         # FIXME
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    H, W = int(H), int(W)
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # 뒤 흰배경 처리 (png 빈 부분을 불러오면 검은화면이 됨)
    # FIXME) alpha 값을 왜 없애는지 확인할 것 -> density 어디서 사용?
    if bkg_white:
        imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
    else:
        imgs = imgs[..., :3]
    return imgs, poses, render_poses, [H, W, K], i_split


if __name__ == "__main__":
    i, p, r_p, hwf, i = load_blender("/home/brozserver2/dev/NeRF_paeng/data/nerf_synthetic/lego", "blender", False)
    # show_3d("/home/brozserver2/dev/NeRF_paeng/data/nerf_synthetic/lego", "blender")
