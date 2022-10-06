from PIL import Image
import numpy as np
import torch 
from model_lss import compile_model
from data_lss import compile_data
import os

device = torch.device('cuda:1')

COLOR = np.uint8([
    (0, 0, 0),
    (70, 70, 70),
    (100, 40, 40),
    (220, 20, 60),
    (153, 153, 153),
    (157, 234, 50),
    (128, 64, 128),
    (244, 35, 232),
    (0, 0, 142),
    (102, 102, 156)
])


def save_feature(features, path_list):
    for i in range(features.shape[0]):
        f = features[i,:,:,:]
        f = f.cpu().detach().numpy()
        np.save(path_list[i], f)

root='/data/carla_dataset/data_collection/'
# save_root='/data/carla_feature/data_collection'

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
    

train_loader, val_loader = compile_data(data_root=root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, batch_size=1)
print(f'len: {len(train_loader)}')
model = compile_model(grid_conf, data_aug_conf, outC=10)
model.load_state_dict(torch.load('model.pt'))

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
        imgs, rots, trans, intrins, post_rots, post_trans, binimgs = data
        pred_segs = model(
                        imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device)
                        )
        pred_segs = pred_segs[:, :, :128, :]
        segs = pred_segs.clone().sigmoid()
        seg_class = torch.argmax(segs, 1)
        print(seg_class.shape)
        result = Image.fromarray(COLOR[seg_class[0].detach().cpu().numpy()])
        result.save('results/eval_{}.png'.format(str(batchi).zfill(4)))
        

