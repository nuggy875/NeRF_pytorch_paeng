from re import L
from tkinter import E
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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


def preprocess_rays(rays_o, rays_d, cfg):
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # 단위벡터화
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    near = cfg.render.depth_near * torch.ones_like(rays_d[..., :1])
    far = cfg.render.depth_far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)  # [1024, 11]
    return rays


def run_model_batchify(rays, fn_posenc, fn_posenc_d, model, cfg):
    '''
    chunk_ray : sample 된 ray 개수가 많을 때 cuda memory 이슈가 있기 때문에 chunk로 batchify하여 학습합니다.
                특히 test 할때는 ray의 개수가 이미지 전체이기 때문에 (ex 400x400) batchify가 필요.
    '''

    chunk = cfg.render.chunk_ray
    # batchify rays -> n_rays_per_image(1024)를 chunk(1024x32) 로 batch 나누기
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = run_model(rays[i:i+chunk], fn_posenc, fn_posenc_d, model, cfg)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def run_model(ray_batch, fn_posenc, fn_posenc_d, model, cfg):
    '''
    chunk_pts : input(rays x pts_per_ray)이 너무 큰 경우 chunk_pts를 기준으로 batchify 하여 model에 넣습니다.
    '''
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    near = ray_batch[..., 6].unsqueeze(-1)
    far = ray_batch[..., 7].unsqueeze(-1)
    viewdirs = ray_batch[:, -3:]

    # make points (input)
    t_vals = torch.linspace(0., 1., steps=cfg.render.n_coarse_pts_per_ray)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, cfg.render.n_coarse_pts_per_ray])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower+(upper-lower)*t_rand
    # ===== RUN NETWORK =====
    input_pts = rays_o.unsqueeze(
        1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    # [1024,64,3]>[65536,3] #TODO VISiualization
    input_pts_flat = torch.reshape(input_pts, [-1, 3])
    # > POSITIONAL ENCODING (x)
    input_pts_embedded = fn_posenc(input_pts_flat)  # [65536,63]
    # > POSITIONAL ENCODING (d)
    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, 3])  # [65536,3]
    input_dirs_embedded = fn_posenc_d(input_dirs_flat)  # [65536,27]
    embedded = torch.cat([input_pts_embedded, input_dirs_embedded], -1)
    # > Network
    # batchify
    chunk = cfg.render.chunk_ray
    outputs_flat = torch.cat([model(embedded[i:i+chunk])
                             for i in range(0, embedded.shape[0], chunk)], 0)
    # outputs_flat = model(embedded)  # [65536, 4] [color(3) + density(1)]
    outputs = torch.reshape(outputs_flat, list(
        input_pts.shape[:-1]) + [outputs_flat.shape[-1]])  # [1024, 64, 4]
    # TODO FINE Network (N_f) (output : [1024, 192(coarse:64 + fine:128), 4])
    rgb_map, disp_map, acc_map, weights, depth_map = volumne_rendering(
        outputs, z_vals, rays_d)
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map,
           'acc_map': acc_map, 'raw': outputs, 'weights': weights, 'depth_map': depth_map}
    return ret


def volumne_rendering(outputs, z_vals, rays_d):

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = outputs[..., :3]
    rgb_sigmoid = torch.sigmoid(rgb)

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]

    # Density(alpha) X Transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb_sigmoid, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))          # alpha to real color
    return rgb_map, disp_map, acc_map, weights, depth_map


def test():
    near = 2. * torch.ones_like(torch.rand(1024, 1))
    far = 6. * torch.ones_like(torch.rand(1024, 1))
    t_vals = torch.linspace(0., 1., steps=64)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    near_val = near * (1.-t_vals)
    far_val = far * (t_vals)
    plt.plot(z_vals[0])
    plt.savefig('dataset/test2/z_vals.png')


if __name__ == "__main__":
    test()
