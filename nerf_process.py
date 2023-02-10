import torch
import torch.nn.functional as F

from utils import put_epsilon


# NDC Rays for llff dataset
def ndc_rays(H, W, focal, near, rays_o, rays_d):

    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
        (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
        (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Ray Sampling and Make input for NeRF Model
def pre_process(rays, posenc, opts, z_vals=None, weights=None, isFine=False):
    '''
    Make model Input for Network (Coarse & Fine)
    '''
    fn_posenc, fn_posenc_d = posenc
    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    # make z_vals in COARSE Network
    if not isFine:
        N_rays = rays.size(0)
        near = opts.near * \
            torch.ones([N_rays, 1], device=torch.device(
                f'cuda:{opts.gpu_ids[opts.rank]}'))
        far = opts.far * \
            torch.ones([N_rays, 1], device=torch.device(
                f'cuda:{opts.gpu_ids[opts.rank]}'))

        t_vals = torch.linspace(0., 1., steps=opts.N_samples_c, device=torch.device(
            f'cuda:{opts.gpu_ids[opts.rank]}'))
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, opts.N_samples_c])
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand([N_rays, opts.N_samples_c], device=torch.device(
            f'cuda:{opts.gpu_ids[opts.rank]}'))
        z_vals = lower + (upper-lower) * t_rand
    # make z_vals in FINE Network
    else:
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], opts.N_samples_f, det=(opts.perturb == 0.), opts=opts)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    input_pts = rays_o.unsqueeze(
        1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    # [1024/4096, 64, 3] -> [65536/262144, 3]
    input_pts_flat = input_pts.view(-1, 3)
    input_pts_embedded = fn_posenc(
        input_pts_flat)                      # [n_pts, 63]

    # [4096, 3] -> [4096, 1, 3]-> [4096, 64, 3]
    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.size())
    # [n_pts, 3]
    input_dirs_flat = input_dirs.reshape(-1, 3)
    input_dirs_embedded = fn_posenc_d(
        input_dirs_flat)                  # [n_pts, 27]

    embedded = torch.cat(
        [input_pts_embedded, input_dirs_embedded], -1)  # [n_pts, 90]
    return embedded, z_vals, rays_d


# Volume Rendering (section 4)
def post_process(outputs, z_vals, rays_d):

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    device = dists.get_device()
    # [N_rays, N_samples] 그런데 마지막에는 젤 큰 수 cat
    big_value = torch.Tensor([1e10]).to(device)
    dists = torch.cat([dists, big_value.expand(dists[..., :1].shape)], -1)

    # small_value = torch.Tensor([0.03]).to(device)
    # dists = torch.cat([dists, small_value.expand(dists[..., :1].shape)], -1)
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
    # ==============================================================================================================
    # disp_map = 1./put_epsilon(depth_map / put_epsilon(torch.sum(weights, -1)))
    # ==============================================================================================================
    # if torch.isnan(disp_map).sum():
    #     print(torch.isnan(disp_map).sum())
    # ==============================================================================================================

    # ==============================================================================================================
    # FIXME NAN Issue ( torch.max(nan, epsilon) = nan )
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    disp_map = torch.where(torch.isnan(disp_map),
                           torch.zeros_like(depth_map), disp_map)
    # ==============================================================================================================
    # disp_map = 1. / torch.where(torch.isnan(depth_map / torch.sum(weights, -1)),
    #                             1e-10 * torch.ones_like(depth_map),
    #                             torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)))
    scale_factor = 5.
    disp_map = torch.where(disp_map > scale_factor,
                           scale_factor * torch.ones_like(disp_map), disp_map)
    # ==
    acc_map = torch.sum(weights, -1)
    # alpha to real color
    rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))

    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, opts=None):

    # assert 가 맞으면 넘어감
    assert opts is not None

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=torch.device(
            f'cuda:{opts.gpu_ids[opts.rank]}'))
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples],
                       device=torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))

    # Invert CDF
    u = u.contiguous()
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


def render_rays(rays, model, posenc, opts):
    # 1-a) pre process : make (pts and dirs) (embedded)
    embedded, z_vals, rays_d = pre_process(rays, posenc, opts, isFine=False)

    # 2-a) run model by net_chunk
    chunk = opts.chunk_pts
    outputs_flat = torch.cat([model(embedded[i:i+chunk])
                             for i in range(0, embedded.shape[0], chunk)], 0)
    size = [z_vals.size(0), z_vals.size(1), 4]
    outputs = outputs_flat.reshape(size)

    # 3-a) post process : render each pixel color by formula (3) in nerf paper
    rgb_map, disp_map, acc_map, weights, depth_map = post_process(
        outputs, z_vals, rays_d)

    if opts.N_samples_f > 0:
        # 1-b) pre precess
        embedded_fine, z_vals_fine, rays_d = pre_process(
            rays, posenc, opts, z_vals=z_vals, weights=weights, isFine=True)

        # 2-b) run model by net_chunk
        outputs_fine_flat = torch.cat([model(embedded_fine[i:i + chunk], is_fine=True)
                                      for i in range(0, embedded_fine.shape[0], chunk)], 0)
        size_fine = [z_vals_fine.size(0), z_vals_fine.size(1), 4]
        outputs_fine = outputs_fine_flat.reshape(size_fine)

        # 3-b) post process : render each pixel color by formula (3) in nerf paper
        rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = post_process(
            outputs_fine, z_vals_fine, rays_d)

        return {'rgb_c': rgb_map, 'disp_c': disp_map, 'rgb_f': rgb_map_fine, 'disp_f': disp_map_fine}
    return {'rgb_c': rgb_map, 'disp_c': disp_map}


# Batchify by chunk
def batchify_rays_and_render_by_chunk(ray_o, ray_d, model, posenc, H, W, K, opts):
    flat_ray_o, flat_ray_d = ray_o.view(-1, 3), ray_d.view(-1, 3)

    # only for llff dataset
    if opts.data_type == 'llff':
        flat_ray_o, flat_ray_d = ndc_rays(
            H, W, K[0][0], 1., flat_ray_o, flat_ray_d)

    N_rays = flat_ray_o.size(0)
    rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)

    ret_rgb_c = []
    ret_disp_c = []
    ret_rgb_f = []
    ret_disp_f = []

    for i in range(0, N_rays, opts.chunk_rays):
        rgb_dict = render_rays(
            rays[i:i+opts.chunk_rays], model, posenc, opts)

        if opts.N_samples_f > 0:                    # use fine rays
            ret_rgb_c.append(rgb_dict['rgb_c'])
            ret_disp_c.append(rgb_dict['disp_c'])
            ret_rgb_f.append(rgb_dict['rgb_f'])
            ret_disp_f.append(rgb_dict['disp_f'])
        else:                                        # use only coarse rays
            ret_rgb_c.append(rgb_dict['rgb_c'])
            ret_disp_c.append(rgb_dict['disp_c'])

    if opts.N_samples_f > 0:
        return torch.cat(ret_rgb_c, dim=0), torch.cat(ret_disp_c, dim=0), torch.cat(ret_rgb_f, dim=0), torch.cat(ret_disp_f, dim=0)
    else:
        return torch.cat(ret_rgb_c, dim=0), torch.cat(ret_disp_c, dim=0), None, None
