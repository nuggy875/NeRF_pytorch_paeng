import torch
import matplotlib.pyplot as plt


def rendering(H, W, K, rays, cfg):
    rays_o, rays_d = rays
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # 단위벡터화
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    sh = rays_d.shape
    # create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near = cfg.render.depth_near * torch.ones_like(rays_d[..., :1])
    far = cfg.render.depth_far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)  # [1024, 11]
    chunk = cfg.render.n_rays_net

    # batchify rays -> chunk(1024x32) 로 나누기
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i:i+chunk], cfg)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}


def render_rays(ray_batch, cfg):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    near = ray_batch[..., 6].unsqueeze(-1)
    far = ray_batch[..., 7].unsqueeze(-1)
    viewdirs = ray_batch[:, -3:]

    t_vals = torch.linspace(0., 1., steps=cfg.render.n_coarse_pts_per_ray)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, cfg.render.n_coarse_pts_per_ray])


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
