import os
import numpy as np
import hydra
import torch
import time
from omegaconf import DictConfig
from tqdm import tqdm, trange
import imageio

from dataset import load_blender
from model import NeRF, get_positional_encoder
from render import run_model_batchify, get_rays, preprocess_rays
from utils import img2mse, mse2psnr, to8b


device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids))
                      if torch.cuda.is_available() else 'cpu')
CONFIG_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "configs")
LOG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")


def test(idx, fn_posenc, fn_posenc_d, model, test_imgs, test_poses, hwk, logdir, cfg):
    model.eval()
    print(logdir)
    print(os.path.join(logdir, cfg.training.name,
          cfg.training.name+'_{}.pth.tar'.format(idx)))
    checkpoint = torch.load(os.path.join(
        logdir, cfg.training.name, cfg.training.name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    savedir = os.path.join(
        logdir, cfg.training.name, cfg.training.name+'_{}'.format(idx))
    os.makedirs(savedir, exist_ok=True)

    img_h, img_w, img_k = hwk

    rgbs = []
    disps = []
    losses = []
    psnrs = []
    result_best = {'i': 0, 'loss': 0, 'psnr': 0}
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(test_poses)):
            t = time.time()
            rays_o, rays_d = get_rays(img_w, img_h, img_k, test_pose[:3][:4])
            rays = preprocess_rays(rays_o, rays_d, cfg)
            pred_rgb, disp, acc, extras = run_model_batchify(rays=rays,
                                                             fn_posenc=fn_posenc,
                                                             fn_posenc_d=fn_posenc_d,
                                                             model=model,
                                                             cfg=cfg)
            # save test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            rgb_np = rgb.cpu().numpy()
            rgb8 = to8b(rgb_np)

            savefilename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(savefilename, rgb8)

            # get loss & psnr
            target_img_flat = torch.reshape(test_imgs[i], [-1, 3])
            img_loss = img2mse(pred_rgb, target_img_flat)
            loss = img_loss
            psnr = mse2psnr(img_loss)
            losses.append(img_loss)
            psnrs.append(psnr)
            print('idx : {} | Loss : {} | PSNR : {}'.format(i, img_loss, psnr))

            # save best result
            if result_best['psnr'] < psnr:
                result_best['i'] = i
                result_best['loss'] = loss
                result_best['psnr'] = psnr
            # rgbs.append(pred_rgb.cpu().numpy())
            # disps.append(disp.cpu().numpy())

    print('BEST Result for Testing) idx : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))

    f = open(os.path.join(savedir, "_result.txt"), 'w')
    for i in range(len(losses)):
        line = 'idx:{}\tloss:{}\tpsnr:{}\n'.format(i, losses[i], psnrs[i])
        f.write(line)
    f.close()  # 쓰기모드 닫기


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    images, poses, render_poses, hwk, i_split = load_blender(
        cfg.data.root, cfg.data.name, cfg.data.half_res, cfg.data.white_bkgd)
    i_train, i_val, i_test = i_split
    img_h, img_w, img_k = hwk
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    # output_ch = 5 if cfg.model.n_importance > 0 else 4
    skips = [4]
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)
    test(1000, fn_posenc, fn_posenc_d, model,
         torch.Tensor(images[i_test]).to(device),
         torch.Tensor(poses[i_test]).to(device),
         hwk, LOG_DIR, cfg)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device)
    main()
