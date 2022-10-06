import os
import numpy as np
import torch 
from model_lss import LiftSplatShoot
from data_lss import compile_data
device = torch.device('cuda:1')
###LSS
# from pyquaternion import Quaternion

def save_feature(feature, path_list):
    f = feature.cpu().detach().numpy()
    print(path_list)
    np.save(path_list, f)

scale=4
root='/data/carla_dataset/data_collection'
save_root='/data/carla_dataset/data_collection'

        
scale = float(scale)

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
def main():
    model = LiftSplatShoot(grid_conf=grid_conf, data_aug_conf=data_aug_conf, outC=10).to(device)
    model.load_state_dict(torch.load('model_feature.pt'))
    print('Searching data...')
    val_loader = compile_data(data_root=root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, batch_size=16)
    model.eval()
    
    with torch.no_grad():
        print('Saving training features...')
        # for i, data in enumerate(train_loader):
        #     path, frame, imgs, rots, trans, intrins, post_rots, post_trans, _ = data
        #     x1, x = model.features(
        #         imgs.to(device),
        #         rots.to(device),
        #         trans.to(device),
        #         intrins.to(device),
        #         post_rots.to(device),
        #         post_trans.to(device)
        #     )
        #     for i in range(len(path)):
        #         f = str(frame[i].item()).zfill(8)
        #         s_path = os.path.join(path[i], 'features')
        #         if not os.path.isdir(s_path):
        #             os.mkdir(s_path)
                
        #         s_path = os.path.join(s_path, 'rgb')
        #         if not os.path.isdir(s_path):
        #             os.mkdir(s_path)
                    
        #         save_path = os.path.join(s_path, 'x1')
        #         if not os.path.isdir(save_path):
        #             os.mkdir(save_path)
        #         save_feature(x1[i], save_path + '/' + f + '.npy')

        #         save_path = os.path.join(s_path, 'x')
        #         if not os.path.isdir(save_path):
        #             os.mkdir(save_path)
        #         save_feature(x[i], save_path  + '/' + f + '.npy')
        print('Saving validating features...')
        for i, data in enumerate(val_loader):
            path, frame, imgs, rots, trans, intrins, post_rots, post_trans, _ = data
            
            x1, x = model.features(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device)
            )
            for i in range(len(path)):
                f = str(frame[i].item()).zfill(8)
                s_path = os.path.join(path[i], 'features')
                if not os.path.isdir(s_path):
                    os.mkdir(s_path)
                
                s_path = os.path.join(s_path, 'rgb')
                if not os.path.isdir(s_path):
                    os.mkdir(s_path)

                save_path = os.path.join(s_path, 'x1')
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                save_feature(x1[i], save_path + '/' + f + '.npy')

                save_path = os.path.join(s_path, 'x')
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                save_feature(x[i], save_path  + '/' + f + '.npy')

if __name__ == '__main__':
    main()