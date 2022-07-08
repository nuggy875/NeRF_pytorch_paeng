import torch


def render(H, W, K, chunk, rays, near, far):
    rays_o, rays_d = rays
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # 단위벡터화
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    sh = rays_d.shape
    # create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)  # [1024, 11]
    batchify_rays(rays, chunk)


def batchify_rays(rays_flat, chunk):
    '''
    chunk = 2^10 x 2^5 (n_rays_net)
    n_rays_per_image: 2^10 -> 32 images per batch?
    '''
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk])
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_rays(ray_batch):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:]
