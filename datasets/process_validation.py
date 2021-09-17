import os, cv2
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.data import degradations as degradations
import math
import numpy as np
import torch
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
## Loading Textures
path_textures = "/home/egs1@laccan.net/dip/blind-face-restoration/gfpgan/data/textures"

textures = []
for filename in os.listdir(path_textures):
    img = cv2.imread(os.path.join(path_textures,filename))
    if img is not None:
        textures.append(img)

num_textures = len(textures)

#Loading images
path_imgs = "/home/egs1@laccan.net/dip/blind-face-restoration/datasets/ffhq/validation"
paths = []
for filename in os.listdir(path_imgs):
    img = cv2.imread(os.path.join(path_imgs,filename))
    if img is not None:
        paths.append(os.path.join(path_imgs,filename))


####################################### Process
# def get_component_coordinates(index, status):
#     components_list = torch.load(opt.get('component_path'))
#     components_bbox = self.components_list[f'{index:08d}']
#     if status[0]:  # hflip
#         # exchange right and left eye
#         tmp = components_bbox['left_eye']
#         components_bbox['left_eye'] = components_bbox['right_eye']
#         components_bbox['right_eye'] = tmp
#         # modify the width coordinate
#         components_bbox['left_eye'][0] = 512 - components_bbox['left_eye'][0]
#         components_bbox['right_eye'][0] = 512 - components_bbox['right_eye'][0]
#         components_bbox['mouth'][0] = 512 - components_bbox['mouth'][0]

#     # get coordinates
#     locations = []
#     for part in ['left_eye', 'right_eye', 'mouth']:
#         mean = components_bbox[part][0:2]
#         half_len = components_bbox[part][2]
#         if 'eye' in part:
#             half_len *= self.eye_enlarge_ratio
#         loc = np.hstack((mean - half_len + 1, mean + half_len))
#         loc = torch.from_numpy(loc).float()
#         locations.append(loc)
#     return locations
def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img
def color_jitter_pt(img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

file_client = FileClient('disk')

for index in range(len(paths)):

    # load gt image
    gt_path = paths[index]
    print("GT PATH:", gt_path)
    img_bytes = file_client.get(gt_path)
    img_gt = imfrombytes(img_bytes, float32=True)

    # random horizontal flip
    img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
    h, w, _ = img_gt.shape


    # locations = self.get_component_coordinates(index, status)
    # loc_left_eye, loc_right_eye, loc_mouth = locations

    # ------------------------ generate lq image ------------------------ #
    # blur
    kernel = degradations.random_mixed_kernels(
        ['iso', 'aniso'],
        [0.5, 0.5],
        41,
        [0.1, 10],
         [0.1, 10], [-math.pi, math.pi],
        noise_range=None)
    img_lq = cv2.filter2D(img_gt, -1, kernel)
    downsample_range =  [0.8, 8]
    # downsample
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    # noise
    #if self.noise_range is not None:
    img_lq = degradations.random_add_gaussian_noise(img_lq, [0, 20])
    # jpeg compression
    #if self.jpeg_range is not None:
    img_lq = degradations.random_add_jpg_compression(img_lq, [60, 100])

    # resize to original size
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # ======================== Textures ===============================

    texture = textures[ np.random.randint(num_textures) ]

    if( np.random.rand() >= 0.25):
        texture = cv2.rotate(texture, np.random.randint(3))

    x_start = np.random.randint(texture.shape[0] - img_lq.shape[0])
    x_end = x_start + img_lq.shape[0]
    y_start = np.random.randint(texture.shape[1] - img_lq.shape[1])
    y_end = y_start + img_lq.shape[1]

    texture = texture[x_start:x_end, y_start:y_end, :]

    #Vertical flip
    if (np.random.rand() < 0.5):
        texture = cv2.flip(texture, 0)

    #Horizontal flip
    if (np.random.rand() < 0.5):
        texture = cv2.flip(texture, 1)

    #alpha ~ U(-0.5, -0.5)
    alpha = np.random.rand() - 0.5 
    
    # img_lq = img_lq + alpha * texture
    img_lq = cv2.addWeighted(img_lq, 1, texture, alpha, 0, dtype=cv2.CV_32F)

    # ======================== Textures ===============================

    # random color jitter (only fihape(or lq)
    if (np.random.uniform() < 0.3):
        img_lq = color_jitter(img_lq, 20)

    # random to gray (only for lq)
    if np.random.uniform() < 0.01:
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
        img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        #if self.opt.get('gt_gray'):
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

    # BGR to RGB, HWC to CHW, numpy to tensor
    img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

    # random color jitter (pytorch version) (only for lq)
    if (np.random.uniform() < 0.3):
        brightness =  (0.5, 1.5)
        contrast = (0.5, 1.5)
        saturation = (0, 1.5)
        hue = (-0.1, 0.1)
        img_lq = color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

    # round and clip
    img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
    
    # normalize
    normalize(img_gt,[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    normalize(img_lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

    # if self.crop_components:
    #     return_dict = {
    #         'lq': img_lq,
    #         'gt': img_gt,
    #         'gt_path': gt_path,
    #         'loc_left_eye': loc_left_eye,
    #         'loc_right_eye': loc_right_eye,
    #         'loc_mouth': loc_mouth
    #     }
    #     return return_dict
    # else:

    img_lq = tensor2img(img_lq)
    img_gt = tensor2img(img_gt)

    path_validation_input = gt_path.replace('validation','validation_input')
    path_validation_output = gt_path.replace('validation','validation_output')
    cv2.imwrite(path_validation_output,img_lq)
    cv2.imwrite(path_validation_input,img_gt)
    #img_lq = tensor2img(img_lq)
    #img_gt = tensor2img(img_gt)
    print({'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path})
