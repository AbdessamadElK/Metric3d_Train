import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp
import glob
from PIL import Image

#os.system('nvidia-smi')

import cupy

from pathlib import Path

import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.custom_data import load_from_annos, load_data

from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud, ply_to_obj
from mono.utils.transform import gray_to_colormap
from mono.utils.visualization import vis_surface_normal


import matplotlib.pyplot as plt

from tqdm import tqdm


cfg_large = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
model_large = get_configured_monodepth_model(cfg_large, )
model_large, _,  _, _ = load_ckpt('./weight/metric_depth_vit_large_800k.pth', model_large, strict_match=False)
model_large.eval()


device = "cuda"
model_large.to(device)


def predict_depth_normal(img, fx=1000.0, fy=1000.0, state_cache={}):

    model = model_large
    cfg = cfg_large

    
    if img is None:
        return None, None, None, None, state_cache, "Please upload an image and wait for the upload to complete."


    cv_image = np.array(img) 
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                    model = model,
                    input = rgb_input,
                    cam_model = cam_models_stacks,
                    pad_info = pad,
                    scale_info = label_scale_factor,
                    gt_depth = None,
                    normalize_scale = cfg.data_basic.depth_range[1],
                    ori_shape=[img.shape[0], img.shape[1]],
                )

        pred_normal = output['normal_out_list'][0][:, :3, :, :] 
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth<0] = 0
    pred_color = gray_to_colormap(pred_depth)

    pred_normal = torch.nn.functional.interpolate(pred_normal, [img.shape[0], img.shape[1]], mode='bilinear').squeeze()
    pred_normal = pred_normal.permute(1,2,0)
    pred_color_normal = vis_surface_normal(pred_normal)
    pred_normal = pred_normal.cpu().numpy()


    ##formatted = (output * 255 / np.max(output)).astype('uint8')
    img_depth = Image.fromarray(pred_color)
    img_normal = Image.fromarray(pred_color_normal)
    return img_depth, pred_depth, img_normal, pred_normal


TMA_DATASET = "/mnt/c/users/abdessamad/TMA/datasets/dsec_full/trainval"
root = Path(TMA_DATASET)

for seq in root.iterdir():
    img_paths = (seq / 'images').iterdir()

    depth_vis_dir = seq / 'depth'
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    normal_vis_dir = seq / 'normal'
    normal_vis_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = seq / 'raw_depth'
    depth_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = seq / 'raw_normal'
    normal_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(img_paths), desc=seq.name):
        basename = p.name
        img = Image.open(p)
        img_depth, pred_depth, img_normal, pred_normal = predict_depth_normal(img)

        img_depth.save(depth_vis_dir / basename)
        img_normal.save(normal_vis_dir / basename)

        np.save(depth_dir / f"{p.stem}.npy", pred_depth)
        np.save(normal_dir / f"{p.stem}.npy", pred_normal)





# paths = sorted(glob.glob('./low/train/low/*.png'))

# for p in paths:
#     basename = os.path.basename(p)
#     img = Image.open(p)
#     print()
#     img_depth, pred_depth, img_normal, pred_normal, features = predict_depth_normal(img)
#     features = features.cpu().numpy()

#     #for feat in features[0]:
#     feat_img = np.sum(features[0], axis=0)
#     feat_img = feat_img -np.min(feat_img)
#     feat_img = 255*feat_img/np.max(feat_img)
#     feat_img = cv2.applyColorMap(np.array(feat_img, dtype=np.uint8), cv2.COLORMAP_JET)
#     # cv2.imshow('feat', feat_img)
#     # cv2.waitKey()

#     img_depth.save('./low/train/depth/{}'.format(basename))
#     img_normal.save('./low/train/surface/{}'.format(basename))
#     cv2.imwrite('./low/train/feat_img/{}'.format(basename), feat_img)

#     np.save('./low/train/features/{}'.format(basename[:-3]), features)
#     print(basename + ' done')

    