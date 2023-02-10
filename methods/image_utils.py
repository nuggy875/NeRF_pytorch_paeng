import os
import cv2
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import LOG_DIR, get_args_parser
from dataset import load_blender, load_llff, load_custom

'''
## extract_image_from_video 함수

input video 로 부터 frame 별로 image를 불러서 저장합니다.
batch를 높힐수록 frame을 띄엄띄엄 불러옵니다.

specific_img 값을 0이 아닌 값으로 두면 해당 id의 이미지만 불러옵니다.
'''


def saveNumpyImage(img, filename: str):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save(LOG_DIR+'/_save_image/{}.png'.format(filename))


def extract_image_from_video(data_root, batch=1, specific_img=0):
    file_path = os.path.join(data_root, 'video.MOV')
    save_path = os.path.join(data_root, 'images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = cv2.VideoCapture(file_path)
    
    currentframe = 1
    if specific_img:
        print(f'Create only one specific image , Id:{specific_img}')
        save_path = os.path.join(LOG_DIR, '_video_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    else:
        print (f'Creating Image from video... : {file_path}')
        
    while(True):
        # reading from frame
        ret, frame = cam.read()
    
        if ret:
            # one specific image for testing
            if specific_img:
                if currentframe == specific_img:
                    name = os.path.join(save_path, str(currentframe)+'.jpg')
                    name_lap = os.path.join(save_path, str(currentframe)+'_lap.jpg')
                    print (f'Creating... {name}')
                    frame = cv2.flip(frame, 0)
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame_lap =  cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
                    frame_lap_var =  cv2.Laplacian(gray, cv2.CV_8U, ksize=5).var()
            
                    # writing the extracted images
                    cv2.imwrite(name, frame)
                    cv2.imwrite(name_lap, frame_lap)
            else:
                if currentframe % batch == 0:
                    # if video is still left continue creating images
                    name = os.path.join(save_path, format(currentframe, '05')+'.jpg')
                    # print (f'Creating... {name}')
                    frame = cv2.flip(frame, 0)
                    frame = cv2.flip(frame, 1)
            
                    # writing the extracted images
                    cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break
    dirListing = os.listdir(save_path)
    print(f'Num of Image Extracted : {len(dirListing)}\n')
    cam.release()
    

    cv2.destroyAllWindows()


if __name__ == "__main__":
    '''
    # extract image from video
    data_root = '/home/brozserver3/brozdisk/data/nerf/custom/minionsmusic'
    # extract_image_from_video(data_root=data_root, batch=30)
    extract_image_from_video(data_root=data_root, batch=1, specific_img=40)
    
    '''
    '''
    # Save Image from loaded Dataset
    opts = get_args_parser()

    print(f'\n>> Loading Dataset : {opts.data_type}')
    if opts.data_type == "blender":
        images, gt_camera_param, hw, i_split = load_blender(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            bkg_white=opts.bkg_white
        )
        render_poses = None
    elif opts.data_type == 'llff':
        images, gt_camera_param, hw, i_split, render_poses = load_llff(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            colmap_relaunch=opts.colmap_relaunch
        )
    elif opts.data_type == 'custom':
        images, gt_camera_param, hw, i_split, nf = load_custom(
            data_root=opts.data_root,
            downsample=opts.downsample,
            testskip=opts.testskip,
            video_batch=opts.video_batch,
            colmap_relaunch=opts.colmap_relaunch
        )
        render_poses = None
        opts.near, opts.far = nf
    i_train, i_val, i_test = i_split
    img_h, img_w = hw
    (gt_intrinsic, gt_extrinsic) = gt_camera_param
    print(f"\n\n>>> Image shape : {images.shape}")
    '''

    

    # saveNumpyImage(images[4], 'totoro')
    