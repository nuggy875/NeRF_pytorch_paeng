# NeRF Pytorch

Pytorch Implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields)

## Environment

```
conda env create -f environment.yml
conda activate nerf_paeng
```
---
## Dataset

Download data for two example datasets: `lego` and `fern`

```
bash download_data.sh
```

Download more dataset from link below
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

---
## Training

configs/.yaml 파일에서 chunk 숫자 조정하여 batchify 가능.

( cuda memory issue )


Testing : 1 GPU RTX3090
800x800 / chunk_ray : 4096 / chunk_pts : 65536

### Check Analysis Note Link below
https://pointed-tarragon-1bf.notion.site/NeRF-7a61cff78fa0415dbc1e8c304b944404
