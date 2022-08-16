import os
import numpy as np
import hydra
import torch
import time
from omegaconf import DictConfig
from tqdm import tqdm, trange
import imageio
import visdom

from dataset import load_blender, load_custom, get_render_pose
from model import NeRF, get_positional_encoder
from process import run_model_batchify, get_rays, preprocess_rays
from utils import getSSIM, getLPIPS, img2mse, mse2psnr, to8b, saveNumpyImage

from configs.config import CONFIG_DIR, LOG_DIR, DATA_NAME


def test(idx, fn_posenc, fn_posenc_d, model, test_imgs, test_poses, hwk, cfg, vis=None):
    print('>>> Start Testing for idx'.format(idx))
    model.eval()
    checkpoint = torch.load(os.path.join(
        LOG_DIR, cfg.testing.name, cfg.testing.name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_test_dir = os.path.join(
        LOG_DIR, cfg.testing.name, cfg.testing.name+'_{}'.format(idx), 'test_result')
    os.makedirs(save_test_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    losses = []
    perform_PSNR = []
    perform_SSIM = []
    perform_LPIPS = []
    result_best = {'i': 0, 'loss': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(test_poses)):
            rays_o, rays_d = get_rays(
                img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            rays = preprocess_rays(rays_o, rays_d, cfg)  # [3]
            ret = run_model_batchify(rays=rays,
                                     fn_posenc=fn_posenc,
                                     fn_posenc_d=fn_posenc_d,
                                     model=model,
                                     cfg=cfg)
            # SAVE test image
            rgb = torch.reshape(ret['rgb_map'], [img_h, img_w, 3])
            disp = torch.reshape(ret['disp_map'], [img_h, img_w])
            rgb_np = rgb.cpu().numpy()
            disp_np = disp.cpu().numpy()

            rgb8 = to8b(rgb_np)
            savefilename = os.path.join(save_test_dir, '{:03d}.png'.format(i))
            imageio.imwrite(savefilename, rgb8)

            # GET loss & psnr
            target_img_flat = torch.reshape(test_imgs[i], [-1, 3])
            # >> GET PSNR
            img_loss = img2mse(ret['rgb_map'], target_img_flat)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # GET SSIM & LPIPS
            loss_ssim = getSSIM(pred=rgb, gt=test_imgs[i])
            loss_lpips = getLPIPS(pred=rgb, gt=test_imgs[i])

            losses.append(img_loss)
            perform_PSNR.append(psnr)
            perform_SSIM.append(loss_ssim)
            perform_LPIPS.append(loss_lpips)
            print('idx:{} | Loss:{} | PSNR:{} | SSIM:{} | LPIPS:{}'.format(
                i, img_loss, psnr, loss_ssim, loss_lpips))

            # save best result
            if result_best['psnr'] < psnr:
                result_best['i'] = i
                result_best['loss'] = loss
                result_best['psnr'] = psnr
                result_best['ssim'] = psnr
                result_best['lpips'] = psnr

    print('BEST Result for Testing) idx : {} , LOSS : {} , PSNR : {}'.format(
        result_best['i'], result_best['loss'], result_best['psnr'], result_best['ssim'], result_best['lpips']))

    f = open(os.path.join(save_test_dir, "_result.txt"), 'w')
    result_sum = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    result_best = {'psnr': 0, 'ssim': 0, 'lpips': 1}
    for i in range(len(losses)):
        line = 'idx:{}\tloss:{}\tpsnr:{}\tssim:{}\tlpips:{}\n'.format(
            i, losses[i], perform_PSNR[i], perform_SSIM[i], perform_LPIPS[i])
        result_sum['psnr'] = result_sum['psnr'] + perform_PSNR[i]
        result_sum['ssim'] = result_sum['ssim'] + perform_SSIM[i]
        result_sum['lpips'] = result_sum['lpips'] + perform_LPIPS[i]
        result_best['psnr'] = perform_PSNR[i] if result_best['psnr'] < perform_PSNR[i] else result_best['psnr']
        result_best['ssim'] = perform_SSIM[i] if result_best['ssim'] < perform_SSIM[i] else result_best['ssim']
        result_best['lpips'] = perform_LPIPS[i] if result_best['lpips'] > perform_LPIPS[i] else result_best['lpips']
        f.write(line)
    f.write('\nBest Value ) PSNR : {}\tSSIM : {}\tLPIPS : {}\n'.format(
        result_best['psnr'], result_best['ssim'], result_best['lpips']))
    f.write('Mean Value ) PSNR : {}\tSSIM : {}\tLPIPS : {}'.format(
        result_sum['psnr']/len(losses), result_sum['ssim']/len(losses), result_sum['lpips']/len(losses)))
    f.close()


def render(idx, fn_posenc, fn_posenc_d, model, hwk, cfg, device, n_angle=40, single_angle=-1):
    '''
    default ) n_angle : 40 / single_angle = -1
    if single_angle is not -1 , it would result single rendering image.
    '''
    print('>>> Start Rendering for idx'.format(idx))

    render_poses = get_render_pose(
        n_angle=n_angle, single_angle=single_angle, phi=cfg.testing.phi).to(device)

    model.eval()
    checkpoint = torch.load(os.path.join(
        LOG_DIR, cfg.testing.name, cfg.testing.name+'_{}.pth.tar'.format(idx)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_render_dir = os.path.join(
        LOG_DIR, cfg.testing.name, cfg.testing.name+'_{}'.format(idx), 'render_result')
    os.makedirs(save_render_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    rgbs = []
    disps = []
    depths = []
    accs = []
    with torch.no_grad():
        for i, test_pose in enumerate(tqdm(render_poses)):
            print('RENDERING... idx: {}'.format(i))
            rays_o, rays_d = get_rays(
                img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            rays = preprocess_rays(rays_o, rays_d, cfg)  # [3]
            ret = run_model_batchify(rays=rays,
                                     fn_posenc=fn_posenc,
                                     fn_posenc_d=fn_posenc_d,
                                     model=model,
                                     cfg=cfg)
            # save test image
            rgb = torch.reshape(ret['rgb_map'], [img_h, img_w, 3])
            disp = torch.reshape(ret['disp_map'], [img_h, img_w])
            acc = torch.reshape(ret['acc_map'], [img_h, img_w])
            depth = torch.reshape(ret['depth_map'], [img_h, img_w])

            rgb_np = rgb.cpu().numpy()
            disp_np = disp.cpu().numpy()
            depth_np = depth.cpu().numpy()
            acc_np = acc.cpu().numpy()
            rgbs.append(rgb_np)
            disps.append(disp_np)
            depths.append(depth_np)
            accs.append(acc_np)
            if not single_angle == -1:
                imageio.imwrite(os.path.join(save_render_dir, '{}_{}_rgb.png'.format(
                    cfg.testing.single_angle, str(cfg.testing.phi))), to8b(rgb_np))
                # imageio.imwrite(os.path.join(save_render_dir, '{}_{}_depth.png'.format(cfg.testing.single_angle,str(cfg.testing.phi))), to8b(depth_np))
                # imageio.imwrite(os.path.join(save_render_dir, '{}_{}_disp.png'.format(cfg.testing.single_angle,str(cfg.testing.phi))), to8b(disp_np / np.max(disp_np)))
                # imageio.imwrite(os.path.join(save_render_dir, '{}_{}_acc.png'.format(cfg.testing.single_angle,str(cfg.testing.phi))), to8b(acc_np))

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

    if single_angle == -1:
        # imageio.mimwrite(os.path.join(save_render_dir, "rgb.mp4"),
        #                  to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_render_dir, "rgb.gif"),
                         to8b(rgbs), duration=0.04)
        # imageio.mimwrite(os.path.join(save_render_dir, "disp.mp4"),
        #                  to8b(disps / np.max(disps)), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_render_dir, "disp.gif"),
                         to8b(disps / np.max(disps)), duration=0.04)
        # imageio.mimwrite(os.path.join(save_render_dir, "depth.mp4"),
        #                  to8b(depths), fps=30, quality=8)


@hydra.main(config_path=CONFIG_DIR, config_name=DATA_NAME)
def main(cfg: DictConfig):
    # == visdom ==
    if cfg.visualization.visdom:
        vis = visdom.Visdom(port=cfg.visualization.visdom_port)
    else:
        vis = None

    if cfg.data.type == 'blender':
        images, poses, hwk, i_split = load_blender(
            data_root=cfg.data.root,
            data_name=cfg.data.name,
            testskip=cfg.testing.testskip,
            bkg_white=cfg.data.white_bkgd,
            reduce_res=cfg.data.reduce_res)
        i_train, i_val, i_test = i_split
        img_h, img_w, img_k = hwk
    elif cfg.data.type == 'custom':
        images, poses, hwk, i_split = load_custom(
            data_root=cfg.data.root,
            data_name=cfg.data.name,
            testskip=cfg.testing.testskip,
            bkg_white=cfg.data.white_bkgd,
            reduce_res=cfg.data.reduce_res)
        i_train = i_split[0]
        i_val = []
        i_test = []
        img_h, img_w, img_k = hwk

    device = torch.device('cuda:{}'.format(
        cfg.device.gpu_ids[cfg.device.rank]))

    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    skips = [4]
    model = NeRF(D=cfg.model.netDepth, W=cfg.model.netWidth,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)


    if cfg.testing.mode_test:
        test(idx=cfg.testing.test_iter,
             fn_posenc=fn_posenc,
             fn_posenc_d=fn_posenc_d,
             model=model,
             test_imgs=torch.Tensor(images[i_test]).to(device),
             test_poses=torch.Tensor(poses[i_test]).to(device),
             hwk=hwk,
             cfg=cfg,
             vis=vis)
    if cfg.testing.mode_render:
        render(idx=cfg.testing.test_iter,
               fn_posenc=fn_posenc,
               fn_posenc_d=fn_posenc_d,
               model=model,
               hwk=hwk,
               cfg=cfg,
               device=device,
               n_angle=cfg.testing.n_angle,
               single_angle=cfg.testing.single_angle
               )


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(device)
    main()
