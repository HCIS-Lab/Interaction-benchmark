from PIL import Image
import numpy as np
import torch 
from model_lss import compile_model
from data_lss import compile_data
import os

device = torch.device('cuda:1')

COLOR = np.uint8([
        (0, 0, 0),
        (66, 62, 64),
        (116, 191, 101),
        (255, 255, 255),
        (136, 138, 133),
        (0, 0, 142),
        (220, 20, 60),
        (0,0,1)
        ])


def save_feature(features, path_list):
    for i in range(features.shape[0]):
        f = features[i,:,:,:]
        f = f.cpu().detach().numpy()
        np.save(path_list[i], f)

root='/data/carla_dataset/data_collection/'
save_root='/data/carla_feature/data_collection'

type_list = ['interactive']

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0]
}

data_aug_conf = {
    'resize_lim': (0.193, 0.225),
    'final_dim': (128, 352),
    'rotate_lim': (-5.4, 5.4),
    'H': 720, 'W': 1280,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'cams': ['left', 'front', 'right'],
    'Ncams': 3
}
    

train_loader = compile_data(data_root=root, grid_conf=grid_conf, data_aug_conf=data_aug_conf)
print(f'len: {len(train_loader)}')
model = compile_model(grid_conf, data_aug_conf, outC=6)
pretrained_dict = torch.load('../models/model139000.pt')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)
model.to(device)

model.eval()

if not os.path.isdir('./results'):
    os.mkdir('./results')

for file in os.listdir('./results'):
    os.remove('./results/' + file)

with torch.no_grad():
    total_intersect = 0.0
    total_union = 0.0
    for batchi, data in enumerate(train_loader):
        imgs, rots, trans, intrins, post_rots, post_trans = data
        pred_segs = model(
                        imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device)
                        )

        segs = pred_segs.clone().sigmoid().cuda()
        mi, _ = torch.max(segs, dim=1)
        mi = mi.view((1, 1, 200, 200))
        drivable_area = torch.zeros_like(mi)
        drivable_area[mi > 0.5] = 1
        non_drive = torch.ones((1, 1, 200, 200)).cuda() - drivable_area
        new_class = torch.ones((1, 1, 200, 200)).cuda()
        new_class[:, :, :100, :] = 0

        pred_cat = torch.cat((non_drive, torch.unsqueeze(segs[:,3,:,:],1), torch.unsqueeze(segs[:,1,:,:],1),
                            torch.unsqueeze(segs[:,4,:,:],1), torch.unsqueeze(segs[:,5,:,:],1),
                            torch.unsqueeze(segs[:,0,:,:],1),
                            torch.unsqueeze(segs[:,2,:,:],1), new_class), 1) # shape(_, 8, 200, 200)
        
        new_pred_cat = np.zeros((1, 8, 200, 200))
        pred_cat_c = np.asarray(pred_cat.clone().cpu())[:, :, 0:100, 0:200]
        new_pred_cat[:, :, 0:100, :] = pred_cat_c
        segs = torch.from_numpy(new_pred_cat).cuda()
        
        result = Image.fromarray(COLOR[segs[0].argmax(0).detach().cpu().numpy()])
        
        result.save('results/eval_{}.png'.format(str(batchi).zfill(4)))


