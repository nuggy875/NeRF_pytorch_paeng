from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import json
import imageio
import numpy as np
import os
from PIL import Image


def load_blender(data_root: str, bkg_white: bool = True, downsample: int = 0, testskip: int = 8):
    splits = ['train', 'val', 'test']
    metas = {}
    # >> Load Annotation (JSON)
    for s in splits:
        with open(os.path.join(data_root, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    # >> Load Images
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

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(counts)-1)]

    imgs = np.concatenate(all_imgs, 0)
    gt_extrinsic = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if downsample:
        H = int(H//downsample)
        W = int(W//downsample)
        focal = focal/downsample

        imgs_reduced = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
        for i, img in enumerate(imgs):
            imgs_reduced[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_reduced

    H, W = int(H), int(W)
    gt_intrinsic = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if bkg_white:
        imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
    else:
        imgs = imgs[..., :3]*imgs[..., -1:]
        

    return imgs, [gt_intrinsic, gt_extrinsic], [H, W], i_split