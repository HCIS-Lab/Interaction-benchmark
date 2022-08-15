"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision.models.resnet import resnet18
from pyquaternion import Quaternion
from tool import gen_dx_bx, cumsum_trick, QuickCumsum, get_rot
import numpy as np
from PIL import Image
device = torch.device('cuda:1')

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1    # batchnorm
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # x1: torch.Size([4, 64, 100, 100])
        x = self.layer2(x1) 
        x = self.layer3(x) # x: torch.Size([4, 256, 25, 25])

        x = self.up1(x, x1) # x up1: torch.Size([4, 256, 100, 100])
        x = self.up2(x) # x up2: torch.Size([4, 4, 200, 200]) (if outC = 4)
        return x

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # x1: torch.Size([4, 64, 100, 100])
        x = self.layer2(x1) 
        x = self.layer3(x) # x: torch.Size([4, 256, 25, 25])

        x = self.up1(x, x1) # x up1: torch.Size([4, 256, 100, 100])
        # x = self.up2(x) # x up2: torch.Size([4, 4, 200, 200]) (if outC = 4)
        return x


class LSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, scale=4, num_cam=3):
        super(LSS, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.num_cam = num_cam
        self.scale = scale

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                              self.grid_conf['ybound'],
                              self.grid_conf['zbound'],
                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape   # torch.Size([41, 8, 22, 3])
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        self.normalize_img = torchvision.transforms.Compose((
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ))
    def get_image_data(self, imgs):
        
        h, w = 720//self.scale, 1280//self.scale

        batch_size = len(imgs[0])
        b_images = []
        b_intrins = []
        b_rots = []
        b_trans = []
        b_post_rots = []
        b_post_trans = []

        for i in range(batch_size):
            images = []
            intrins = []
            rots = []
            trans = []
            post_rots = []
            post_trans = []
            for j in range(self.num_cam):
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                
                intrin = torch.Tensor([[float(5), float(0), float(w)], 
                                        [float(0), float(5), float(h)], 
                                        [float(0), float(0), float(1)]])

                tran = torch.Tensor([float(2.71671180725), float(0), float(0)])
                if j == 0:
                    rot = torch.Tensor(Quaternion([float(0.5), float(-0.5), float(0.5), float(-0.5)]).rotation_matrix)
                elif j == 1:
                    rot = torch.Tensor(Quaternion([float(0.67), float(-0.67), float(0.21), float(-0.21)]).rotation_matrix)
                elif j == 2:
                    rot = torch.Tensor(Quaternion([float(0.21), float(-0.21), float(0.67), float(-0.67)]).rotation_matrix)

                resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
                new_img, post_tran2, post_rot2 = self.img_transform(imgs[j][i], post_tran, post_rot, resize, resize_dims, crop, flip, rotate)
                
                post_rot = torch.eye(3)
                post_tran = torch.zeros(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                images.append(self.normalize_img(new_img.convert('RGB')).permute(1, 2, 0))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)
                
            b_images.append(torch.stack(images))
            b_intrins.append(torch.stack(intrins))
            b_rots.append(torch.stack(rots))
            b_trans.append(torch.stack(trans))
            b_post_rots.append(torch.stack(post_rots))
            b_post_trans.append(torch.stack(post_trans))
        return (
            torch.stack(b_images).to(device),
            torch.stack(b_rots).to(device),
            torch.stack(b_trans).to(device),
            torch.stack(b_intrins).to(device),
            torch.stack(b_post_rots).to(device),
            torch.stack(b_post_trans).to(device) 
        )

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


    def img_transform(self, img, post_rot, post_tran,
                    resize, resize_dims, crop,
                    flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate/180*np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1).to(device)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1,3)
        a = torch.inverse(post_rots).view(B,N,1,1,1,3,3)
        points = a.matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins)).to(device)

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, imH, imW,C  = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W
        # flatten x
        x = x.reshape(Nprime, C)
        
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # torch.Size([4, 5, 41, 8, 22, 3])
        print(geom.shape)
        x = self.get_cam_feats(x)   # torch.Size([4, 5, 41, 8, 22, 64])
        x = self.voxel_pooling(geom, x) # torch.Size([4, 64, 200, 200])

        return x

    def forward(self, imgs):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(imgs)
        # BEV segmentation
        imgs = self.get_voxels(imgs, rots, trans, intrins, post_rots, post_trans)
        segs = self.bevencode(imgs)
        # Driving parameters
        return segs

    def features(self, imgs):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(imgs)
        # BEV segmentation
        imgs = self.get_voxels(imgs, rots, trans, intrins, post_rots, post_trans)
        f = self.bevencode.features(imgs)
        # Driving parameters
        return f
