import os
from tqdm import tqdm
import torch
import torch.optim as optim
from argparse import ArgumentParser 
from model_lss import compile_model
from data_lss import compile_data
from tool import SimpleLoss, get_batch_iou, get_pedestrian_ratio
from torch.utils.tensorboard import SummaryWriter

root='/data/carla_dataset/data_collection/'
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
    
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight_decay')
    parser.add_argument('--pos_weight', type=float, default=2.13, help='pos_weight')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max grad norm')
    args = parser.parse_args()

    device = torch.device('cuda:1')
    print(device)
    writer = SummaryWriter()
    print('Searching data...')
    train_loader, val_loader = compile_data(data_root=root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, batch_size=args.batch_size)
    model = compile_model(grid_conf, data_aug_conf, outC=10).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    criterion = SimpleLoss(pos_weight=args.pos_weight).to(device)
    print('Start training...')
    best_iou, best_individual_iou = 0, torch.zeros(10)

    if os.path.isfile('iou_record.txt'):
        os.remove('iou_record.txt')
    # if len(os.listdir('runs')) != 0:
    #     for tfboard_dir in os.listdir('runs'):
    #         for tfboard_file in os.listdir(os.path.join('runs', tfboard_dir)):
    #             os.remove(os.path.join('runs', tfboard_dir, tfboard_file))
    #         os.rmdir(os.path.join('runs', tfboard_dir))
        
    for epoch in tqdm(range(1, args.epochs+1)):
        # train
        model.train()
        total_train_loss, total_val_loss = 0, 0
        total_intersect, total_union = 0, 0
        individual_intersects, individual_unions, individual_iou = torch.zeros(10).to(device), torch.zeros(10).to(device), torch.zeros(10).to(device)
        
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = data
            pred_segs = model(imgs.to(device),
                            rots.to(device),
                            trans.to(device),
                            intrins.to(device),
                            post_rots.to(device),
                            post_trans.to(device))
            pred_segs = pred_segs[:, :, :128, :]
            binimgs = binimgs.to(device)
            train_loss = criterion(pred_segs, binimgs)
            total_train_loss += train_loss.item()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        # validation 
        model.eval()                    
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs = data
                pred_segs = model(imgs.to(device),
                                rots.to(device),
                                trans.to(device),
                                intrins.to(device),
                                post_rots.to(device), 
                                post_trans.to(device))
                pred_segs = pred_segs[:, :, :128, :]
                binimgs = binimgs.to(device)
                val_loss = criterion(pred_segs, binimgs)
                intersects, unions, intersect, union = get_batch_iou(pred_segs, binimgs)
                #  ratio_predict = get_pedestrian_ratio(pred_segs, binimgs)
                individual_intersects = torch.add(individual_intersects, intersects)
                individual_unions = torch.add(individual_unions, unions)
                total_intersect += intersect
                total_union += union
                total_val_loss += val_loss.item()

            individual_iou = torch.div(individual_intersects, individual_unions)
            total_iou = total_intersect / total_union
            
            total_train_loss = total_train_loss / len(train_loader)
            total_val_loss = total_val_loss / len(val_loader)

            if total_iou > best_iou:
                best_iou = total_iou
                best_individual_iou = individual_iou.detach().cpu()
                with open('iou_record.txt', 'a') as f:
                    f.write(f'Best mIOU: {best_iou:04f}\n')
                    f.write(f'Best IOU: ')
                    for i in range(10):
                        f.write(f'{best_individual_iou[i]} ')
                    f.write('\n')
                    # f.write(f'The ratio: {ratio_predict:04f}\n')
                torch.save(model.state_dict(), f'model.pt')

            writer.add_scalar('train/loss', total_train_loss, epoch)
            writer.add_scalar('val/loss', total_val_loss, epoch)
            writer.add_scalar('val/iou', total_iou, epoch)
            writer.add_scalars('comp/loss', {'train': total_train_loss, 
                                            'validation': total_val_loss}, epoch)

if __name__ == '__main__':
    main()
