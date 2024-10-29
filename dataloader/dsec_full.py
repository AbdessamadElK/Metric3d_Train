import numpy as np
import torch
import torch.utils.data as data

import random
import os
from pathlib import Path
import glob

from augment import Augmentor

import imageio.v2 as imageio
import cv2

class DSECfull(data.Dataset):
    def __init__(self, phase, crop:bool = True, flip:bool=True, spatial_aug:bool=True):
        assert phase in ["train", "trainval", "test", "prog"]

        self.init_seed = False
        self.phase = phase
        self.files = []
        self.flows = []

        self.augment = crop or flip or spatial_aug
        crop_size = [288, 384] if crop else None

        ### Please change the root to satisfy your data saving setting.
        root = 'datasets/dsec_sample'
        if phase == 'train' or phase == 'trainval':
            self.root = os.path.join(root, 'trainval')
            self.augmentor = Augmentor(crop_size, do_flip=flip, spatial_aug = spatial_aug)
        else:
            self.root = os.path.join(root, 'test')

        if phase == 'prog':
            self.augment = False
            self.root = 'datasets/dsec_prog/trainval'

        # Event files
        self.files = glob.glob(os.path.join(self.root, '*', '*.npz'))
        self.files.sort()

        # Optical flow files
        self.flows = glob.glob(os.path.join(self.root, '*', 'flow_*.npy'))
        self.flows.sort()

        # Image files
        self.images = glob.glob(os.path.join(self.root, '*', 'images', '*.png'))
        self.images.sort()

        # Depth files
        self.depths = glob.glob(os.path.join(self.root, '*', 'raw_depth', '*.npy'))
        self.depths.sort()

        # Normal files
        self.normals = glob.glob(os.path.join(self.root, '*', 'raw_normal', '*.npy'))
        self.normals.sort()
        
        

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        #events
        events_file = np.load(self.files[index])
        voxel1 = events_file['events_prev'].transpose(1, 2, 0)
        voxel2 = events_file['events_curr'].transpose(1, 2, 0)

        #image
        img = imageio.imread(self.images[index])

        #depth
        depth = np.load(self.depths[index])

        #normal
        normal = np.load(self.normals[index])


        print(img.shape)
        print(depth.shape)
        print(normal.shape)

        #flow
        if self.phase != "test":
            
            flow_16bit = np.load(self.flows[index])
            flow_map, valid2D = flow_16bit_to_float(flow_16bit)

            if self.augment:
                voxel1, voxel2, flow_map, valid2D, img, depth, normal = self.augmentor(voxel1, voxel2, flow_map, valid2D, img, depth, normal)
            
            flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()
            valid2D = torch.from_numpy(valid2D).float()

        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        depth = torch.from_numpy(depth).float()

        normal = torch.from_numpy(normal).permute(2, 0, 1).float()

        if self.phase == "test":
            # Include submission coordinates (seuence name, file index)
            file_path = Path(self.files[index])
            sequence_name = file_path.parent.name
            file_index = int(file_path.stem)
            submission_coords = (sequence_name, file_index)
            return voxel1, voxel2, depth, submission_coords
        
        return voxel1, voxel2, img, depth, normal

    
    def __len__(self):
        return len(self.files)
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def make_data_loader(phase, batch_size, num_workers, crop = True, flip = True, spatial_aug = True, sec_input = 'depth'):
    dset = DSECfull(phase, crop, flip, spatial_aug, sec_input=sec_input)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader

if __name__ == '__main__':

    dset = DSECfull('trainval')
    v1, v2, img, depth, normal = dset[0]

    print("Voxel1 :", v1.shape)
    print("Voxel2 :", v2.shape)
    print("Image :", img.shape)
    print("Depth :", depth.shape)
    print("Normal :", normal.shape)

    print("\n\n\n")

    print("Min :", np.min(depth.numpy()))
    print("Max :", np.max(depth.numpy()))
    print("Mean :", np.mean(depth.numpy()))
