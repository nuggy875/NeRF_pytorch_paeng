import torch
import numpy as np
from IQA_pytorch import SSIM, LPIPSvgg


def mse2psnr(x):
    device = x.get_device()
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x):
    device = x.get_device()
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)


def getSSIM(pred, gt):
    SSIM_ = SSIM(channels=3)
    # [W,H,3]->[1,3,W,H]
    return SSIM_(pred.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), as_loss=False)


def getLPIPS(pred, gt):
    device = pred.get_device()
    LPIPS_ = LPIPSvgg(channels=3).to(device)
    LPIPS_.weights = [(t1, t2.to(device)) for (t1, t2) in LPIPS_.weights]
    # loss_lpips = LLPIS_(rgb.permute(2, 0, 1).unsqueeze(0), test_imgs[i].permute(2, 0, 1).unsqueeze(0))
    # LPIPS_ = lpips.LPIPS(net='vgg')
    return LPIPS_(pred.permute(2, 0, 1), gt.permute(2, 0, 1))


def put_epsilon(map): return torch.max(1e-10 * torch.ones_like(map), map)


# for global batch
class GetterRayBatchIdx(object):
    def __init__(self, rays_rgb):
        self.rays_rgb = rays_rgb
        self.epoch = 0
        self.i_batch = 0

    def shuffle_ray_idx(self, batch_size):
        print("Shuffle data after an epoch!")
        rand_idx = torch.randperm(self.rays_rgb.shape[0])
        self.rays_rgb = self.rays_rgb[rand_idx]
        self.i_batch = batch_size
        self.epoch += 1

    def __call__(self, batch_size):
        self.i_batch += batch_size
        if self.i_batch >= self.rays_rgb.shape[0]:
            self.shuffle_ray_idx(batch_size)
        return self.i_batch, self.rays_rgb, self.epoch

