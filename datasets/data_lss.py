import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm
from tool import img_transform, normalize_img, gen_dx_bx
from torchvision import transforms

class NuscData(Dataset):
    def __init__(self, data_root, grid_conf, data_aug_conf, is_train):
        self.data_root = data_root
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.type_list = ['interactive']
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.is_train = is_train
        self.samples = self.prepro()
        

    def __len__(self):
        return len(self.samples)

    def prepro(self):
        
        if self.is_train:
            samples = []
            basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0])) if '6_' in s[:2] or '7_' in s[:2]]
            
            for s in tqdm(basic_scenarios, file=sys.stdout): 
                variants_path = os.path.join(s, 'variant_scenario')
                variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
                for v in variants:
                    if os.path.isdir(os.path.join(v, 'rgb')):
                        img_path = os.path.join(v, 'rgb')
                    for cam in os.listdir(img_path):
                        img_subpath = os.path.join(img_path, cam)
                        names = []
                        for img_name in os.listdir(img_subpath):
                            names.append(int(img_name.split('.png')[0]))
                        names = sorted(names)
                        start_frame = min(names)
                        end_frame = max(names)
                        break

                    samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1)]

            return samples
        else:
            samples = []
            basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0])) if '10_' in s[:3]]
            for s in tqdm(basic_scenarios, file=sys.stdout): 
                variants_path = os.path.join(s, 'variant_scenario')
                variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
                for v in variants:
                    if os.path.isdir(os.path.join(v, 'rgb')):
                        img_path = os.path.join(v, 'rgb')
                    for cam in os.listdir(img_path):
                        img_subpath = os.path.join(img_path, cam)
                        names = []
                        for img_name in os.listdir(img_subpath):
                            names.append(int(img_name.split('.png')[0]))
                        names = sorted(names)
                        start_frame = min(names)
                        end_frame = max(names)
                        break

                    samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1)]

            return samples
        # samples = []
    
        # path = os.path.join(self.data_root, '1_s-6_0_0_0_f_0_0/variant_scenario/CloudyNoon_mid_')
        # img_path = os.path.join(path, 'rgb')
        # for cam in os.listdir(img_path):
        #     img_subpath = os.path.join(img_path,cam)
        #     names = []
        #     for img_name in os.listdir(img_subpath):
        #         names.append(int(img_name.split('.png')[0]))
        #     names = sorted(names)
        #     start_frame = min(names)
        #     end_frame = max(names)


        # samples += [{'path': path, 'frame': i} for i in range(start_frame, end_frame+1)]

        # return samples

    def sample_augmentation(self):
        H, W = 720, 1280
        fH, fW = 128, 352
        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean((0.0, 0.22)))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate  
        

    def get_image_data(self, index, cams):  
        
        images = []
        intrins = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        
        img_path = os.path.join(self.samples[index]['path'], 'rgb')

        for i, cam in enumerate(cams): 
            if os.path.isfile(os.path.join(img_path, cam, '{:08d}'.format(self.samples[index]['frame']) + '.png')):
                img_name = os.path.join(img_path, cam, '{:08d}'.format(self.samples[index]['frame']) + '.png') 
                img = Image.open(img_name) 
            else:
                img = Image.fromarray(np.zeros((720, 1280)))
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            
            intrin = torch.Tensor([[float(5), float(0), float(640)], 
                                    [float(0), float(5), float(360)], 
                                    [float(0), float(0), float(1)]])

            tran = torch.Tensor([float(2.71671180725), float(0), float(0)])
            if i == 0:
                rot = torch.Tensor(Quaternion([float(0.6744282), float(-0.6744282), float(0.2124774), float(-0.2124774)]).rotation_matrix)
            elif i == 1:
                rot = torch.Tensor(Quaternion([float(0.5), float(-0.5), float(0.5), float(-0.5)]).rotation_matrix)
            elif i == 2:
                rot = torch.Tensor(Quaternion([float(-0.2124774), float(0.2124774), float(-0.6744282), float(0.6744282)]).rotation_matrix)
              
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate)
            
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            images.append(normalize_img(img.convert('RGB')))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        return (
            torch.stack(images), torch.stack(rots), torch.stack(trans),
            torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)
        )

    def get_binimg(self, index):
        if os.path.isdir(os.path.join(self.samples[index]['path'], 'instance_segmentation/lbc_ins')):
            
            bin_path = os.path.join(self.samples[index]['path'], 'instance_segmentation/lbc_ins')
            if os.path.isfile(os.path.join(bin_path, "{:08d}".format(self.samples[index]['frame']) + '.png')):
                bin_name = os.path.join(bin_path, "{:08d}".format(self.samples[index]['frame']) + '.png')
                bin_img = Image.open(bin_name).convert('RGB').resize((200, 200))
                bin_img = transforms.ToTensor()(bin_img) * 255
                new_bin_img = torch.zeros((10, 200, 200))
                for i in range(200):
                    for j in range(200):
                        if bin_img[0][i, j] == 1: #buildings
                            new_bin_img[0][i, j] = 1
                        elif bin_img[0][i, j] == 2: #fences
                            new_bin_img[1][i, j] = 1
                        elif bin_img[0][i, j] == 4: #Pedestrains 
                            new_bin_img[2][i, j] = 1
                        elif bin_img[0][i, j] == 5: #Poles
                            new_bin_img[3][i, j] = 1
                        elif bin_img[0][i, j] == 6: #Roadlines
                            new_bin_img[4][i, j] = 1
                        elif bin_img[0][i, j] == 7: #Roads
                            new_bin_img[5][i, j] = 1
                        elif bin_img[0][i, j] == 8: #Sidewalks
                            new_bin_img[6][i, j] = 1
                        elif bin_img[0][i, j] == 10: #Vehicles
                            new_bin_img[7][i, j] = 1
                        elif bin_img[0][i, j] == 11: #Walls
                            new_bin_img[8][i, j] = 1
                        else: #Others
                            new_bin_img[9][i, j] = 1

                return new_bin_img
            else:
                return torch.zeros((10, 200, 200))
        else:
            return torch.zeros((10, 200, 200))


class SegmentationData(NuscData):
    def __init__(self, data_root, grid_conf, data_aug_conf, is_train):
        super(SegmentationData, self).__init__(data_root, grid_conf, data_aug_conf, is_train)

    def __getitem__(self, index):
        cams = ['left', 'front', 'right']
        images, rots, trans, intrins, post_rots, post_trans = self.get_image_data(index, cams)
        binimg = self.get_binimg(index)
        return images, rots, trans, intrins, post_rots, post_trans, binimg


def compile_data(data_root, grid_conf, data_aug_conf, batch_size):
    train_data = SegmentationData(data_root=data_root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, is_train=True)
    val_data = SegmentationData(data_root=data_root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, is_train=False)
    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=10
                            )
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=10
                            )
    return train_loader, val_loader