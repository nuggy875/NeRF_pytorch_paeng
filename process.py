from re import L
from tkinter import E
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import put_epsilon


def get_rays(W, H, K, c2w):
    '''
    img_k = [3,3] pose = [3,4]
    Transpose Image Plane Coordinate to Normalized Plane ([x',y',1] -> [u,v,1])
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    '''
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.get_device()),
                          torch.linspace(0, H-1, H, device=c2w.get_device()))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0],
                        -(j-K[1][2])/K[1][1],
                        -torch.ones_like(i)], -1)
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_d = dirs @ c2w[:3, :3].T  # it is okay
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
        ret = run_model(rays[i:i+chunk], fn_posenc,
                        fn_posenc_d, model, cfg)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def run_model(ray_batch, fn_posenc, fn_posenc_d, model, cfg):
    '''
    chunk_pts : input(rays x pts_per_ray)이 너무 큰 경우 chunk_pts를 기준으로 batchify 하여 model에 넣습니다.
    '''
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    near = ray_batch[..., 6].unsqueeze(-1)
    far = ray_batch[..., 7].unsqueeze(-1)
    viewdirs = ray_batch[:, -3:]

    # ===== 1-1) make z_vals (input) (COARSE NETWORK)
    t_vals = torch.linspace(
        0., 1., steps=cfg.render.n_coarse_pts_per_ray, device=viewdirs.get_device())
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, cfg.render.n_coarse_pts_per_ray])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=z_vals.get_device())
    z_vals = lower+(upper-lower)*t_rand
    # ===== 1-2) Make Input (COARSE NETWORK) =====
    embedded = make_input(rays_o, rays_d, z_vals,
                          viewdirs, fn_posenc, fn_posenc_d)

    # assign to device
    device = torch.device('cuda:{}'.format(
        cfg.device.gpu_ids[cfg.device.rank]))
    embedded = embedded.to(device)
    z_vals = z_vals.to(device)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    viewdirs = viewdirs.to(device)

    # ===== 1-3) Run Network (COARSE NETWORK) =====
    chunk = cfg.render.chunk_pts
    outputs_flat = torch.cat([model(embedded[i:i+chunk], is_fine=False)
                             for i in range(0, embedded.shape[0], chunk)], 0)
    outputs = torch.reshape(outputs_flat, list(
        z_vals.shape) + [outputs_flat.shape[-1]])  # [1024, 64, 4]

    # ===== 1-4) Volume Rendering (COARSE NETWORK) =====
    rgb_map, disp_map, acc_map, weights, depth_map = volumne_rendering(
        outputs, z_vals, rays_d)

    rgb_map_c = None
    # ===== 2) FINE Network (N_f) (output : [1024, 192(coarse:64 + fine:128), 4])
    if cfg.render.n_fine_pts_per_ray > 0:
        # ===== 2-1) make z_vals (sampling) (FINE NETWORK)
        rgb_map_c, disp_map_c, acc_map_c = rgb_map, disp_map, acc_map
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = hierarchical_sampling(
            z_vals_mid, weights[..., 1:-1], cfg.render.n_fine_pts_per_ray)
        z_samples = z_samples.detach()
        z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # ===== 2-2) Make Input (FINE NETWORK) =====
        embedded_fine = make_input(rays_o, rays_d, z_vals_fine,
                                   viewdirs, fn_posenc, fn_posenc_d)
        z_vals_fine = z_vals_fine.to(device)
        embedded_fine = embedded_fine.to(device)
        # ===== 2-3) Run Network (FINE NETWORK) =====
        chunk = cfg.render.chunk_pts
        outputs_flat = torch.cat([model(embedded_fine[i:i+chunk], is_fine=True)
                                  for i in range(0, embedded_fine.shape[0], chunk)], 0)
        outputs = torch.reshape(outputs_flat, list(
            z_vals_fine.shape) + [outputs_flat.shape[-1]])  # [1024, 192, 4]
        # ===== 2-4) Volume Rendering (FINE NETWORK) =====
        rgb_map, disp_map, acc_map, weights, depth_map = volumne_rendering(
            outputs, z_vals_fine, rays_d)

        return {'rgb_map': rgb_map, 'disp_map': disp_map,
                'acc_map': acc_map, 'raw': outputs, 'depth_map': depth_map,
                'rgb_map_c': rgb_map_c, 'disp_map_c': disp_map_c}
    else:
        return {'rgb_map': rgb_map, 'disp_map': disp_map,
                'acc_map': acc_map, 'raw': outputs, 'depth_map': depth_map}


def make_input(rays_o, rays_d, z_vals, viewdirs, fn_posenc, fn_posenc_d):
    input_pts = rays_o.unsqueeze(
        1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    input_pts_flat = torch.reshape(input_pts, [-1, 3])
    # > POSITIONAL ENCODING (x)
    input_pts_embedded = fn_posenc(input_pts_flat)  # [65536,63]
    # > POSITIONAL ENCODING (d)
    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, 3])  # [65536,3]
    input_dirs_embedded = fn_posenc_d(input_dirs_flat)  # [65536,27]
    embedded = torch.cat([input_pts_embedded, input_dirs_embedded], -1)
    return embedded


def volumne_rendering(outputs, z_vals, rays_d):

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    device = outputs.get_device()
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = outputs[..., :3]
    rgb_sigmoid = torch.sigmoid(rgb)

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]

    # Density(alpha) X Transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb_sigmoid, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./put_epsilon(depth_map / put_epsilon(torch.sum(weights, -1)))
    acc_map = torch.sum(weights, -1)
    # alpha to real color
    rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))
    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_rays_and_pixel(i, rays_o, rays_d, target_img, cfg):
    img_w, img_h = target_img.size()[:2]

    if i < cfg.training.precrop_iters:
        dH = int(img_h//2 * cfg.training.precrop_frac)
        dW = int(img_w//2 * cfg.training.precrop_frac)
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
        0), size=cfg.render.n_rays_per_image, replace=False)  # default 1024
    selected_coords = coords[selected_idx].long()  # (N_rand, 2)

    # == Sample Rays ==
    rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
    rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
    # == Sample Pixel ==
    target_img = target_img[selected_coords[:, 0], selected_coords[:, 1]]

    return rays_o, rays_d, target_img  # [1024, 3]


def batchify_rays_and_render_by_chunk(ray_o, ray_d, model, fn_posenc, fn_posenc_d, cfg):
    # [640000, 3], [640000, 3]
    flat_ray_o, flat_ray_d = ray_o.view(-1, 3), ray_d.view(-1, 3)
    num_whole_rays = flat_ray_o.size(0)
    rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)

    all_ret = {}
    chunk = cfg.render.chunk_ray
    for i in range(0, num_whole_rays, chunk):
        ret = render_rays(rays[i:i+chunk], model, fn_posenc, fn_posenc_d, cfg)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(rays, model, fn_posenc, fn_posenc_d, cfg):
    # 1. pre process : make (pts and dirs) (embedded)
    embedded, z_vals, rays_d = pre_process(rays, fn_posenc, fn_posenc_d, cfg)

    # ** assign to cuda **
    # embedded = embedded.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    # z_vals = z_vals.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    # rays_d = rays_d.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    # 2. run model by net_chunk
    chunk = cfg.render.chunk_pts
    outputs_flat = torch.cat([model(embedded[i:i+chunk])
                             for i in range(0, embedded.shape[0], chunk)], 0)  # [net_c
    size = [z_vals.size(0), z_vals.size(1), 4]      # [4096, 64, 4]
    outputs = outputs_flat.reshape(size)

    # 3. post process : render each pixel color by formula (3) in nerf paper
    rgb_map, disp_map, acc_map, weights, depth_map = post_process(
        outputs, z_vals, rays_d)
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map,
           'acc_map': acc_map, 'raw': outputs, 'weights': weights, 'depth_map': depth_map}
    return ret


def pre_process(rays, fn_posenc, fn_posenc_d, cfg):

    N_rays = rays.size(0)
    # assert N_rays == opts.chunk, 'N_rays must be same to chunk'
    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    viewdirs = rays_d
    # make normal vector
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    near = cfg.render.depth_near * torch.ones([N_rays, 1])
    far = cfg.render.depth_far * torch.ones([N_rays, 1])

    t_vals = torch.linspace(0., 1., steps=cfg.render.n_coarse_pts_per_ray)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, cfg.render.n_coarse_pts_per_ray])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand([N_rays, cfg.render.n_coarse_pts_per_ray])
    z_vals = lower + (upper-lower) * t_rand

    # rays_o, rays_d : [B, 1, 3], z_vals : [B, 64, 1]
    input_pts = rays_o.unsqueeze(
        1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    # [1024/4096, 64, 3] -> [65536/262144, 3]
    input_pts_flat = input_pts.view(-1, 3)
    input_pts_embedded = fn_posenc(
        input_pts_flat)                        # [n_pts, 63]

    # [4096, 3] -> [4096, 1, 3]-> [4096, 64, 3]
    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.size())
    # [n_pts, 3]
    input_dirs_flat = input_dirs.reshape(-1, 3)
    input_dirs_embedded = fn_posenc_d(
        input_dirs_flat)                    # [n_pts, 27]

    embedded = torch.cat(
        [input_pts_embedded, input_dirs_embedded], -1)   # [n_pts, 90]

    return embedded, z_vals, rays_d


def post_process(outputs, z_vals, rays_d):
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    device = dists.get_device()
    big_value = torch.Tensor([1e10]).to(device)
    # [N_rays, N_samples] 그런데 마지막에는 젤 큰 수 cat
    dists = torch.cat([dists, big_value.expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = outputs[..., :3]
    rgb_sigmoid = torch.sigmoid(rgb)

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]

    # Density(alpha) X Transmittance
    transmittance = torch.cumprod(torch.cat(
        [torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb_sigmoid, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    # alpha to real color
    rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))
    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def hierarchical_sampling(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    device = weights.get_device()
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
