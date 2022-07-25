import torch
from PIL import Image
import os
import numpy as np
from configs.config import LOG_DIR
from torchvision import transforms


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def saveNumpyImage(img, filename:str):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/test/{}.jpg'.format(filename))


def put_epsilon(map): return torch.max(1e-10 * torch.ones_like(map), map)


if __name__ == "__main__":
    img = Image.open("/home/brozserver2/brozdisk/data/nerf/nerf_synthetic/lego/train/r_0.png").convert("RGB")
    img_tensor = transforms.ToTensor()(img)
    print(img_tensor.shape)