import os
import numpy as np
import hydra
import torch
import time
from omegaconf import DictConfig
from tqdm import tqdm, trange
import imageio

from dataset import load_blender, get_render_pose
from model import NeRF, get_positional_encoder
from process import run_model_batchify, get_rays, preprocess_rays
from utils import img2mse, mse2psnr, to8b

from configs.config import CONFIG_DIR, LOG_DIR, device


def test(idx, fn_posenc, fn_posenc_d, model, test_imgs, test_poses, hwk, cfg):
    model.eval()
    checkpoint = torch.load(os.path.join(
        LOG_DIR, cfg.training.name, cfg.training.name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_test_dir = os.path.join(
        LOG_DIR, cfg.training.name, cfg.training.name+'_{}'.format(idx), 'test_result')
    os.makedirs(save_test_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    losses = []
    psnrs = []
    result_best = {'i': 0, 'loss': 0, 'psnr': 0}
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(test_poses)):
            rays_o, rays_d = get_rays(
                img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            rays = preprocess_rays(rays_o, rays_d, cfg)  # [3]
            pred_rgb, pred_disp, pred_acc, predextras = run_model_batchify(rays=rays,
                                                                           fn_posenc=fn_posenc,
                                                                           fn_posenc_d=fn_posenc_d,
                                                                           model=model,
                                                                           cfg=cfg)
            # SAVE test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            disp = torch.reshape(pred_disp, [img_h, img_w])
            rgb_np = rgb.cpu().numpy()
            disp_np = disp.cpu().numpy()

            rgb8 = to8b(rgb_np)
            savefilename = os.path.join(save_test_dir, '{:03d}.png'.format(i))
            imageio.imwrite(savefilename, rgb8)

            # GET loss & psnr
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

    print('BEST Result for Testing) idx : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr']))

    f = open(os.path.join(save_test_dir, "_result.txt"), 'w')
    for i in range(len(losses)):
        line = 'idx:{}\tloss:{}\tpsnr:{}\n'.format(i, losses[i], psnrs[i])
        f.write(line)
    f.close()


def render(idx, fn_posenc, fn_posenc_d, model, hwk, cfg):
    '''
    default ) n_angle : 40 / single_angle = -1
    if single_angle is not -1 , it would result single rendering image.
    '''
    n_angle = 40
    single_angle = -1
    render_poses = get_render_pose(n_angle=n_angle, single_angle=single_angle)

    model.eval()
    checkpoint = torch.load(os.path.join(
        LOG_DIR, cfg.training.name, cfg.training.name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_render_dir = os.path.join(
        LOG_DIR, cfg.training.name, cfg.training.name+'_{}'.format(idx), 'render_result')
    os.makedirs(save_render_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    rgbs = []
    disps = []
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(render_poses)):
            print('RENDERING... idx: {}'.format(i))
            rays_o, rays_d = get_rays(
                img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            rays = preprocess_rays(rays_o, rays_d, cfg)  # [3]
            pred_rgb, pred_disp, pred_acc, predextras = run_model_batchify(rays=rays,
                                                                           fn_posenc=fn_posenc,
                                                                           fn_posenc_d=fn_posenc_d,
                                                                           model=model,
                                                                           cfg=cfg)
            # save test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            disp = torch.reshape(pred_disp, [img_h, img_w])
            acc = torch.reshape(pred_acc, [img_h, img_w])
            depth = torch.reshape(predextras['depth_map'], [img_h, img_w])
            rgb_np = rgb.cpu().numpy()
            disp_np = disp.cpu().numpy()
            rgbs.append(rgb_np)
            disps.append(disp_np)
            if not single_angle == -1:
                savefilename = os.path.join(
                    save_render_dir, '{:03d}.png'.format(i))
                imageio.imwrite(savefilename, to8b(rgb_np))

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

    if single_angle == -1:
        imageio.mimwrite(os.path.join(save_render_dir, "rgb.mp4"),
                         to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_render_dir, "disp.mp4"),
                         to8b(disps / np.max(disps)), fps=30, quality=8)


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    images, poses, hwk, i_split = load_blender(
        cfg.data.root, cfg.data.name, cfg.data.half_res, cfg.data.white_bkgd)
    i_train, i_val, i_test = i_split
    img_h, img_w, img_k = hwk
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    # output_ch = 5 if cfg.model.n_importance > 0 else 4
    skips = [4]
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)
    test(idx=200000,
         fn_posenc=fn_posenc,
         fn_posenc_d=fn_posenc_d,
         model=model,
         test_imgs=torch.Tensor(images[i_test]).to(device),
         test_poses=torch.Tensor(poses[i_test]).to(device),
         hwk=hwk,
         cfg=cfg)

    # render(idx=200000,
    #        fn_posenc=fn_posenc,
    #        fn_posenc_d=fn_posenc_d,
    #        model=model,
    #        hwk=hwk,
    #        cfg=cfg)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device)
    main()
