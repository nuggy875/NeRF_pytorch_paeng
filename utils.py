import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from IQA_pytorch import SSIM, LPIPSvgg, DISTS
import lpips
from configs.config import LOG_DIR


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x):
    device = x.get_device()
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)


def getSSIM(pred, gt):
    # GET SSIM & LPIPS
    SSIM_ = SSIM(channels=3)
    # [W,H,3]->[1,3,W,H]
    return SSIM_(pred.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), as_loss=False)


def getLPIPS(pred, gt):
    # LLPIS_ = LPIPSvgg(channels=3)
    # loss_lpips = LLPIS_(rgb.permute(2, 0, 1).unsqueeze(0), test_imgs[i].permute(2, 0, 1).unsqueeze(0))
    LPIPS_ = lpips.LPIPS(net='vgg').to(pred.get_device())
    return LPIPS_(pred.permute(2, 0, 1), gt.permute(2, 0, 1))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def saveNumpyImage(img, filename: str):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/test/{}.jpg'.format(filename))


def put_epsilon(map): return torch.max(1e-10 * torch.ones_like(map), map)


