import numpy as np
import os, sys
import cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods.image_utils import extract_image_from_video
from .load_llff import _load_data, recenter_poses, spherify_poses, poses_avg


def load_custom(data_root: str, downsample: int = 0, testskip: int = 8, bd_factor=.75, spherify=True, video_batch=30, colmap_relaunch=False):
    # downsample=8 downsamples original imgs by 8x
    if not os.path.isdir(os.path.join(data_root, 'images')):
        print(">> no 'images' folder")
        if os.path.isfile(os.path.join(data_root, 'video.MOV')):
            print(">> 'video.MOV' file exist, extracting images from video...")
            extract_image_from_video(data_root=data_root, batch=video_batch)
        else:
            print('need video or images')
            exit()

    poses, bds, imgs = _load_data(data_root, factor=None, colmap_relaunch=colmap_relaunch)

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc
    # >>> Recenter
    poses = recenter_poses(poses)
    # >>> spherify
    poses, render_poses, bds = spherify_poses(poses, bds)

    c2w = poses_avg(poses)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)

    imgs = imgs.astype(np.float32)
    poses = poses.astype(np.float32)

    hwf = poses[0, :3, -1]
    gt_extrinsic = poses[:, :3, :4]

    H, W, focal = hwf
    H, W = int(H), int(W)

    if downsample:
        H = int(H//downsample)
        W = int(W//downsample)
        focal = focal/downsample

        imgs_reduced = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
        for i, img in enumerate(imgs):
            imgs_reduced[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_reduced

    gt_intrinsic = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if testskip:
        i_test = np.arange(imgs.shape[0])[::testskip]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(imgs.shape[0])) if
                            (i not in i_test and i not in i_val)])
    else:
        i_test = np.array([])
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(imgs.shape[0])) if
                    (i not in i_test and i not in i_val)])

    near = np.ndarray.min(bds) * .9
    far = np.ndarray.max(bds) * 1.

    return imgs, [gt_intrinsic, gt_extrinsic], [H, W], [i_train, i_val, i_test], [near, far]
