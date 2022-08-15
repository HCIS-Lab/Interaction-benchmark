"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, is_train, data_aug_conf, grid_conf):
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.samples = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()


        print(self)

    def prepro(self):
        samples = []
        if self.is_train:
            samples += self.add_scenarios('data/5_27_5_c_f_f_0_0', 'ClearNoon_', 57, 155, 5)
            samples += self.add_scenarios('data/3_17_2_c_f_f_0_0', 'ClearNoon_', 56, 117, 2)
            samples += self.add_scenarios('data/3_28_2_c_f_f_0_0', 'ClearNoon_', 57, 142, 2)
            samples += self.add_scenarios('data/3_9_0_c_f_f_0_0', 'ClearNoon_', 56, 162, 0)
            samples += self.add_scenarios('data/3_14_2_c_f_f_0_0', 'ClearNoon_', 56, 168, 2)
            samples += self.add_scenarios('data/3_2_5_c_f_f_0_0', 'ClearNoon_', 54, 137, 5)
            samples += self.add_scenarios('data/5_3_3_c_f_f_0_0', 'ClearNoon_', 57, 204, 3)
            samples += self.add_scenarios('data/5_20_3_c_f_f_0_0', 'ClearNoon_', 57, 199, 3)
            samples += self.add_scenarios('data/5_13_3_c_f_f_0_0', 'ClearNoon_', 57, 171, 3)
            samples += self.add_scenarios('data/3_30_4_c_f_f_0_0', 'ClearNoon_', 57, 180, 4)
            samples += self.add_scenarios('data/3_16_0_c_f_f_0_0', 'ClearNoon_', 57, 137, 0)
            samples += self.add_scenarios('data/5_23_1_c_f_f_0_0', 'ClearNoon_', 57, 223, 1)
            samples += self.add_scenarios('data/3_6_5_c_f_f_0_0', 'ClearNoon_', 54, 113, 5)
            samples += self.add_scenarios('data/3_10_1_c_f_f_0_0', 'ClearNoon_', 56, 129, 1)
            samples += self.add_scenarios('data/3_18_4_c_f_f_0_0', 'ClearNoon_', 56, 175, 4)
            samples += self.add_scenarios('data/5_6_5_c_f_f_0_0', 'ClearNoon_', 57, 125, 5)
            samples += self.add_scenarios('data/5_8_4_c_f_f_0_0', 'ClearNoon_', 57, 177, 4)
            samples += self.add_scenarios('data/5_26_0_c_f_f_0_0', 'ClearNoon_', 57, 182, 0)
            samples += self.add_scenarios('data/5_15_4_c_f_f_0_0', 'ClearNoon_', 57, 165, 4)
            samples += self.add_scenarios('data/3_15_2_c_f_f_0_0', 'ClearNoon_', 56, 138, 2)
            samples += self.add_scenarios('data/3_7_4_c_f_f_0_0', 'ClearNoon_', 56, 211, 4)
            samples += self.add_scenarios('data/3_21_5_c_f_f_0_0', 'ClearNoon_', 57, 126, 5)
            samples += self.add_scenarios('data/5_25_1_c_f_f_0_0', 'ClearNoon_', 57, 152, 1)
            samples += self.add_scenarios('data/3_26_1_c_f_f_0_0', 'ClearNoon_', 57, 153, 1)
            samples += self.add_scenarios('data/3_25_4_c_f_f_0_0', 'ClearNoon_', 57, 225, 4)
            samples += self.add_scenarios('data/3_27_1_c_f_f_0_0', 'ClearNoon_', 57, 108, 1)
            samples += self.add_scenarios('data/3_4_2_c_f_f_0_0', 'ClearNoon_', 54, 142, 2)
            samples += self.add_scenarios('data/5_18_1_c_f_f_0_0', 'ClearNoon_', 57, 141, 1)
            samples += self.add_scenarios('data/5_1_2_c_f_f_0_0', 'ClearNoon_', 57, 120, 2)
            samples += self.add_scenarios('data/3_24_2_c_f_f_0_0', 'ClearNoon_', 57, 135, 2)
            samples += self.add_scenarios('data/3_29_3_c_f_f_0_0', 'ClearNoon_', 57, 218, 3)
            samples += self.add_scenarios('data/5_9_3_c_f_f_0_0', 'ClearNoon_', 57, 190, 3)
            samples += self.add_scenarios('data/5_19_0_c_f_f_0_0', 'ClearNoon_', 57, 141, 0)
            samples += self.add_scenarios('data/3_3_4_c_f_f_0_0', 'ClearNoon_', 54, 155, 4)
            samples += self.add_scenarios('data/5_21_1_c_f_f_0_0', 'ClearNoon_', 57, 150, 1)
            samples += self.add_scenarios('data/3_20_4_c_f_f_0_0', 'ClearNoon_', 57, 177, 4)
            samples += self.add_scenarios('data/5_4_2_c_f_f_0_0', 'ClearNoon_', 57, 168, 2)
            samples += self.add_scenarios('data/3_13_4_c_f_f_0_0', 'ClearNoon_', 56, 180, 4)
            samples += self.add_scenarios('data/5_17_4_c_f_f_0_0', 'ClearNoon_', 57, 213, 4)
            samples += self.add_scenarios('data/5_11_3_c_f_f_0_0', 'ClearNoon_', 57, 162, 3)
            samples += self.add_scenarios('data/5_28_0_c_f_f_0_0', 'ClearNoon_', 57, 178, 0)
            samples += self.add_scenarios('data/5_14_5_c_f_f_0_0', 'ClearNoon_', 57, 124, 5)
            samples += self.add_scenarios('data/5_5_2_c_f_f_0_0', 'ClearNoon_', 57, 133, 2)
            samples += self.add_scenarios('data/5_16_2_c_f_f_0_0', 'ClearNoon_', 57, 138, 2)
            samples += self.add_scenarios('data/5_30_3_c_f_f_0_0', 'ClearNoon_', 57, 229, 3)
            samples += self.add_scenarios('data/3_11_0_c_f_f_0_0', 'ClearNoon_', 56, 132, 0)
            samples += self.add_scenarios('data/3_31_3_c_f_f_0_0', 'ClearNoon_', 57, 201, 3)
        else:
            samples += self.add_scenarios('data/3_22_5_c_f_f_0_0', 'ClearNoon_', 57, 117, 5)
            samples += self.add_scenarios('data/5_24_3_c_f_f_0_0', 'ClearNoon_', 57, 210, 3)
            samples += self.add_scenarios('data/5_7_5_c_f_f_0_0', 'ClearNoon_', 57, 152, 5)
            samples += self.add_scenarios('data/3_5_1_c_f_f_0_0', 'ClearNoon_', 54, 148, 1)
            samples += self.add_scenarios('data/3_23_2_c_f_f_0_0', 'ClearNoon_', 57, 170, 2)
            samples += self.add_scenarios('data/5_12_5_c_f_f_0_0', 'ClearNoon_', 57, 133, 5)
        return samples

    def add_scenarios(self, path, scene, frame_begin, frame_end, cmd):
        return [{'path': path, 'scene': scene, 'frame': i, 'control_frame': i - frame_begin, 'cmd': cmd}
                    for i in range(frame_begin, frame_end + 1)]

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if False:
        # if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, index, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        img_path = os.path.join(self.samples[index]['path'], self.samples[index]['scene'])
        tf_path = os.path.join(self.samples[index]['path'], 'transformation')

        for cam in cams:
            # read image
            imgname = os.path.join(img_path, cam, "{:08d}".format(self.samples[index]['frame']) + '.png')

            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # read transformation
            tf_name = os.path.join(tf_path, cam + '.txt')
            with open(tf_name, 'r') as fp:
                # intrinsic
                line = fp.readline().rstrip()

                m00, m01, m02, m10, m11, m12, m20, m21, m22 = line.split(" ")
                intrin = torch.Tensor([[float(m00), float(m01), float(m02)], 
                                       [float(m10), float(m11), float(m12)], 
                                       [float(m20), float(m21), float(m22)]])
                # trans
                line = fp.readline().rstrip()
                x, y, z = line.split(" ")
                tran = torch.Tensor([float(x), float(y), float(z)])
                # rot
                line = fp.readline().rstrip()
                w, x, y, z = line.split(" ")
                rot = torch.Tensor(Quaternion([float(w), float(x), float(y), float(z)]).rotation_matrix)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img.convert('RGB')))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)


        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_binimg(self, index):
        bin_path = os.path.join(self.samples[index]['path'], 'GT/')
        bins = []
        segs = ['cross_walk', 'other_cars', 'white_broken_lane', 
                'yelow_solid_lane', 'drivable_lane', 'shoulder', 
                'white_solid_lane', 'yellow_broken_lane']

        for seg in segs:
            bin_name = os.path.join(bin_path, seg, str(self.samples[index]['frame']) + '.npy')
            bin = np.load(bin_name)
            bins.append(bin)

        return torch.Tensor(bins)

    def get_driving_parameters(self, index):

        # Carla specific
        control_file = os.path.join(self.samples[index]['path'], 'GT/control.npy')
        control_frame = self.samples[index]['control_frame']
        control = np.load(control_file)[control_frame][1:4] # Extract brake, steer, throttle

        cmd = self.samples[index]['cmd']
        return torch.Tensor([cmd]), torch.Tensor(control)

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):

        return 0

class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):

        cams = ['left', 'front', 'right']
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(index, cams)
        binimg = self.get_binimg(index)
        cmd, control = self.get_driving_parameters(index)
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, cmd, control


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
