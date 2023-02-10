import numpy as np
import torch

# Get Numpy ray in advance for GLOBAL BATCH


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2]) /
                    K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def make_o_d(img_w, img_h, img_k, pose):

    # make catesian (x. y)
    i, j = torch.meshgrid(torch.linspace(0, img_w - 1, img_w, device=pose.get_device()),
                          torch.linspace(0, img_h - 1, img_h, device=pose.get_device()))
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i - img_k[0][2])/img_k[0][0],
                        -(j - img_k[1][2])/img_k[1][1],
                        -torch.ones_like(i)], -1)

    rays_d = dirs @ pose[:3, :3].T
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_rays_and_pixel(i, rays_o, rays_d, target_img, opts):
    img_h, img_w = target_img.size()[:2]

    if i < opts.precrop_iters:
        dH = int(img_h//2 * opts.precrop_frac)
        dW = int(img_w//2 * opts.precrop_frac)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(img_h//2 - dH, img_h//2 + dH - 1, 2*dH),
            torch.linspace(img_w//2 - dW, img_w//2 + dW - 1, 2*dW)), -1)

    else:
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, img_h - 1, img_h),
            torch.linspace(0, img_w - 1, img_w)), -1)
    coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]

    # 640000 개 중 1024개 뽑기
    selected_idx = np.random.choice(a=coords.size(
        0), size=opts.N_rays, replace=False)  # default 1024
    selected_coords = coords[selected_idx].long()  # (N_rand, 2)

    # == Sample Rays ==
    rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
    rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
    # == Sample Pixel ==
    target_img = target_img[selected_coords[:, 0], selected_coords[:, 1]]

    return rays_o, rays_d, target_img  # [1024, 3]
