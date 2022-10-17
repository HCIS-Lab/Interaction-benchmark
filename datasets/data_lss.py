import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pyquaternion import Quaternion
from tqdm import tqdm
from tool import img_transform, normalize_img, gen_dx_bx
import common
import random

class NuscData(Dataset):
    def __init__(self, data_root, grid_conf, data_aug_conf, is_train, only_town_10, is_visual=False):
        self.data_root = data_root
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.type_list = ['interactive']
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.is_train = is_train
        self.is_visual = is_visual
        self.only_town_10 = only_town_10
        self.samples = self.prepro()
            

    def __len__(self):
        return len(self.samples)

    def prepro(self):
        if self.is_visual:
            samples = []
            root_path = os.path.join(self.data_root, self.type_list[0])
            basic_scenarios = [
                root_path + '/1_t1-1_1_m_f_r_1_rl',
                root_path + '/2_t1-1_1_m_r_l_1_0',
                root_path + '/2_t2-4_1_t_u_r_1_rl',
                root_path + '/2_t3-7_1_p_c_f_1_rl',
                root_path + '/3_i-5_1_b_f_u_1_0',
                root_path + '/3_r-1_0_p_j_f_1_j_1',
                root_path + '/3_t3-8_1_c_l_l_1_0_1',
                root_path + '/5_i-1_0_m_l_f_1_0',
                root_path + '/5_i-7_1_c_l_f_1_0',
                root_path + '/5_t2-2_0_m_r_f_1_0',
                root_path + '/6_i-1_1_m_f_r_1_rl',
                root_path + '/6_i-2_1_c_sr_f_1_0',
                root_path + '/6_i-4_1_p_c_f_1_0',
                root_path + '/6_s-6_0_t_sl_sr_1_0',
                root_path + '/7_i-3_0_b_r_l_1_0',
                root_path + '/7_t2-5_0_m_r_f_1_0',
                root_path + '/10_s-1_0_c_sl_f_1_0',
                root_path + '/10_s-4_0_c_f_sr_1_0_1',
                root_path + '/10_s-6_0_p_j_f_1_j',
                root_path + '/10_t3-1_1_c_u_l_1_0'
            ]
            
            for s in tqdm(basic_scenarios): 
                if not os.path.isdir('./results/' + s[47:]):
                    os.mkdir('./results/' + s[47:])
                if not os.path.isdir('./results/' + s[47:] + '/variant_scenario'):
                    os.mkdir('./results/' + s[47:] + '/variant_scenario')
                    
                variants_path = os.path.join(s, 'variant_scenario')
                variants = [(os.path.join(variants_path, v), v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
                for v, v2 in variants:
                    if not os.path.isdir('./results/' + s[47:] + '/variant_scenario/' + v2):
                        os.mkdir('./results/' + s[47:] + '/variant_scenario/' + v2)
                        
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
                     
        elif self.is_train:
            if self.only_town_10:
                samples = []
                basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0])) if '10_' in s[:3]]
                basic_scenarios = random.sample(basic_scenarios, 25)    
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

                        samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1, 5)]

                return samples
            else:    
                samples = []
                lbc_variants = []
                basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0]))]
                for s in basic_scenarios: 
                    variants_path = os.path.join(s, 'variant_scenario')
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
                    for v in variants:
                        if(os.path.isdir(os.path.join(v, 'instance_segmentation/lbc_ins'))):
                            lbc_variants.append(v)
                            
                random.shuffle(lbc_variants)
                for v in tqdm(lbc_variants[:420]):
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

                    samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1, 5)]

                return samples
        else:
            if self.only_town_10:
                samples = []
                basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0])) if '10_' in s[:3]]
                basic_scenarios = random.sample(basic_scenarios, 4)    
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

                        samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1, 5)]

                return samples
            else:
                samples = []
                lbc_variants = []
                basic_scenarios = [os.path.join(self.data_root, self.type_list[0], s) for s in os.listdir(os.path.join(self.data_root, self.type_list[0]))]
                for s in basic_scenarios: 
                    variants_path = os.path.join(s, 'variant_scenario')
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
                    for v in variants:
                        if(os.path.isdir(os.path.join(v, 'instance_segmentation/lbc_ins'))):
                            lbc_variants.append(v)
                            
                random.shuffle(lbc_variants)           
                for v in tqdm(lbc_variants[420:]):
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

                    samples += [{'path': v, 'frame': i} for i in range(start_frame, end_frame+1, 5)]

                return samples

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
            self.samples[index]['path'][48:], self.samples[index]['frame'],
            torch.stack(images), torch.stack(rots), torch.stack(trans),
            torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)
        )

    def preprocess_semantic(self, topdown):
        topdown = common.CONVERTER[topdown]
        topdown = torch.LongTensor(topdown)
        topdown = F.one_hot(topdown, 10).permute(2, 0, 1).float()
        return topdown


    def get_binimg(self, index):
        if os.path.isdir(os.path.join(self.samples[index]['path'], 'instance_segmentation/lbc_ins')):
            
            bin_path = os.path.join(self.samples[index]['path'], 'instance_segmentation/lbc_ins')
            if os.path.isfile(os.path.join(bin_path, "{:08d}".format(self.samples[index]['frame']) + '.png')):
                bin_name = os.path.join(bin_path, "{:08d}".format(self.samples[index]['frame']) + '.png')
                with Image.open(bin_name) as topdown_image:
                    
                    topdown_image = topdown_image.resize((256, 256)).crop((0, 0, 256, 128))
                    topdown = np.array(topdown_image)
                    topdown = topdown[:, :, 0]
                    topdown = self.preprocess_semantic(topdown)
                    return topdown
            else:
                return torch.zeros((10, 128, 256))
        else:
            return torch.zeros((10, 128, 256))


class SegmentationData(NuscData):
    def __init__(self, data_root, grid_conf, data_aug_conf, is_train, only_town_10, is_visual):
        super(SegmentationData, self).__init__(data_root, grid_conf, data_aug_conf, is_train, only_town_10, is_visual)

    def __getitem__(self, index):
        cams = ['left', 'front', 'right']
        
        path, frame, images, rots, trans, intrins, post_rots, post_trans = self.get_image_data(index, cams)
        if not self.is_visual:
            binimg = self.get_binimg(index)
            return images, rots, trans, intrins, post_rots, post_trans, binimg
        else:
            return path, frame, images, rots, trans, intrins, post_rots, post_trans

def compile_data(data_root, grid_conf, data_aug_conf, batch_size, only_town_10):
    train_data = SegmentationData(data_root=data_root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, is_train=True, only_town_10=only_town_10, is_visual=False)
    val_data = SegmentationData(data_root=data_root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, is_train=False, only_town_10=only_town_10, is_visual=False)
    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=10,
                            drop_last=True
                            )
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=10
                            )
    return train_loader, val_loader

def compile_visual_data(data_root, grid_conf, data_aug_conf, batch_size):
    visual_data = SegmentationData(data_root=data_root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, is_train=False, is_visual=True, only_town_10=False)
    visual_loader = DataLoader(visual_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=10
                        )
    return visual_loader