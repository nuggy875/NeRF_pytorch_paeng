# NeRF in Pytorch

Pytorch Re-Implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields)

Paper : https://arxiv.org/abs/2003.08934


# Preparation


## Environment

```
conda env create -f environment.yml
conda activate nerf
```

---

## Dataset

Download data for two example datasets: `lego` and `fern`

```
bash download_data.sh
```

Download more dataset from link below
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

synthetic datasets : [chair, drums, ficus, hotdog, lego, materials, mic, ship]
llff datasets : [fern, flower, fortress, horns, leaves, orchids, room, trex]


### Data Directory

## Blender Dataset
```
nerf_synthetic
    |-- chair
        |-- train
            |-- r_0.png
            |-- r_1.png
                    ...
            |-- r_99.png
        |-- test
        |-- val
        |-- transforms_train.json
        |-- transforms_test.json
        |-- transforms_val.json
    |-- drums
    ...
```

## LLFF Dataset
```
nerf_llff_data
    |-- fern
        |-- images
            |-- IMG_0000.JPG
            |-- IMG_0001.JPG
                    ...
        |-- sparse
        |-- database.db
        |-- poses_bounds.npy
            ...
    |-- flower
    ...
```

---

# Experiments


## Training




**set 'data_root' path in config file to dataset root**

```
python main.py --config configs/blender/lego.txt
```
```
python main.py --config configs/llff/fern.txt
```


---

# Results

## • Official Paper Results (Blender)

| data      | model       | Batch rays | resolution    | PSNR      | SSIM      | LPIPS     |
| --------- | ----------- | ---------- | ------------- | --------- | --------- | --------- |
| chair     | 200,000     | 4096       | 800 x 800     | 33.00     | 0.967     | 0.046     |
| drums     | 200,000     | 4096       | 800 x 800     | 25.01     | 0.925     | 0.091     |
| ficus     | 200,000     | 4096       | 800 x 800     | 30.13     | 0.964     | 0.044     |
| hotdog    | 200,000     | 4096       | 800 x 800     | 36.18     | 0.974     | 0.121     |
| lego      | 200,000     | 4096       | 800 x 800     | 32.54     | 0.961     | 0.050     |
| materials | 200,000     | 4096       | 800 x 800     | 29.62     | 0.949     | 0.063     |
| mic       | 200,000     | 4096       | 800 x 800     | 32.91     | 0.980     | 0.028     |
| ship      | 200,000     | 4096       | 800 x 800     | 28.65     | 0.856     | 0.206     |
| **mean**  | **200,000** | **4096**   | **800 x 800** | **31.01** | **0.947** | **0.081** |

## • This Repo Results (Blender)

| data      | iter        | Batch rays | resolution    | PSNR      | SSIM      | LPIPS     |
| --------- | ----------  | ---------- | ------------- | --------- | --------- | --------- |
| chair     | 200,000     | 4096       | 800 x 800     | 33.92     | 0.966     | 0.048     |
| drums     | 200,000     | 4096       | 800 x 800     | 24.96     | 0.922     | 0.102     |
| ficus     | 200,000     | 4096       | 800 x 800     | 29.82     | 0.960     | 0.063     |
| hotdog    | 200,000     | 4096       | 800 x 800     | 36.20     | 0.974     | 0.049     |
| lego      | 200,000     | 4096       | 800 x 800     | 32.02     | 0.958     | 0.060     |
| materials | 200,000     | 4096       | 800 x 800     | 29.41     | 0.943     | 0.076     |
| mic       | 200,000     | 4096       | 800 x 800     | 33.21     | 0.981     | 0.028     |
| ship      | 200,000     | 4096       | 800 x 800     | 28.47     | 0.854     | 0.186     |
| **mean**  | **200,000** | **4096**   | **800 x 800** | **31.00** | **0.945** | **0.077** |

---

**TEST RESULT** of 200 images from Test Dataset

## • This Repo Results (LLFF)

| data      | model           | Batch rays | resolution    | PSNR      | SSIM      | LPIPS     |
| --------- | --------------- | ---------- | ------------- | --------- | --------- | --------- |
| fern      | Coarse+Fine     | 4096       | 800 x 800     | 23.98     | 0.773     | 0.224     |
| flower    | Coarse+Fine     | 4096       | 800 x 800     | 28.73     | 0.896     | 0.110     |
| fortress  | Coarse+Fine     | 4096       | 800 x 800     | 32.02     | 0.925     | 0.081     |
| horns     | Coarse+Fine     | 4096       | 800 x 800     | 30.11     | 0.921     | 0.125     |
| leaves    | Coarse+Fine     | 4096       | 800 x 800     | 22.51     | 0.815     | 0.187     |
| orchids   | Coarse+Fine     | 4096       | 800 x 800     | 20.74     | 0.734     | 0.210     |
| room      | Coarse+Fine     | 4096       | 800 x 800     | 32.78     | 0.963     | 0.094     |
| trex      | Coarse+Fine     | 4096       | 800 x 800     | 26.38     | 0.924     | 0.149     |
| **mean**  | **Coarse+Fine** | **4096**   | **800 x 800** | **27.16** | **0.869** | **0.148** |


SSIM code from https://github.com/dingkeyan93/IQA-optimization

LPIPS from https://pypi.org/project/lpips/

## Train Environment

```
ubuntu 20.04
GeForce RTX 3090
cuda : v11.1
cuDNN : v8.0.5
```


## • Render Results (Blender)

<table>
      <thead>
      <tr>
            <th width="500px">Render RGB</th>
            <th width="500px">Render DISP</th>
      </tr>
      </thead>
      <tbody>
            <tr width="500px">
                  <td><img src="./figures/blender/chair_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/chair_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/drums_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/drums_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/ficus_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/ficus_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/hotdog_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/hotdog_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/lego_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/lego_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/materials_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/materials_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/mic_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/mic_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/blender/ship_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/blender/ship_disp.gif" width="100%" height="100%"></td>
            </tr>
      </tbody>
</table>

## • Render Results (LLFF)

<table>
      <thead>
      <tr>
            <th width="500px">Render RGB</th>
            <th width="500px">Render DISP</th>
      </tr>
      </thead>
      <tbody>
            <tr width="500px">
                  <td><img src="./figures/llff/fern_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/fern_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/flower_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/flower_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/fortress_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/fortress_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/horns_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/horns_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/leaves_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/leaves_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/orchids_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/orchids_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/room_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/room_disp.gif" width="100%" height="100%"></td>
            </tr>
            <tr width="500px">
                  <td><img src="./figures/llff/trex_rgb.gif" width="100%" height="100%"></td>
                  <td><img src="./figures/llff/trex_disp.gif" width="100%" height="100%"></td>
            </tr>
      </tbody>
</table>
