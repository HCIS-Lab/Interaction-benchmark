from tqdm import tqdm
import torch
import torch.optim as optim
from argparse import ArgumentParser 
from model_lss import compile_model
from data_lss import compile_data
from tool import SimpleLoss, get_batch_iou
from torch.utils.tensorboard import SummaryWriter

root='/data/scenario_retrieval/carla-1/PythonAPI/examples/data_collection/'
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
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight_decay')
    parser.add_argument('--pos_weight', type=float, default=2.13, help='pos_weight')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max grad norm')
    args = parser.parse_args()

    device = torch.device('cuda:1')
    writer = SummaryWriter()
    counter = 0
    print('Searching data...')
    train_loader, val_loader = compile_data(data_root=root, grid_conf=grid_conf, data_aug_conf=data_aug_conf, batch_size=args.batch_size)
    model = compile_model(grid_conf, data_aug_conf, outC=10).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    criterion = SimpleLoss(pos_weight=args.pos_weight).to(device)
    print('Start training...')
    
    for epoch in tqdm(range(args.epochs)):
        # train
        model.train()
        total_train_loss, total_val_loss = 0, 0
        total_intersect, total_union = 0, 0
        
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = data
            if torch.equal(binimgs, torch.zeros((args.batch_size, 10, 200, 200))):
                continue
            else:
                pred_segs = model(imgs.to(device),
                                rots.to(device),
                                trans.to(device),
                                intrins.to(device),
                                post_rots.to(device),
                                post_trans.to(device))
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
                if torch.equal(binimgs, torch.zeros((args.batch_size, 10, 200, 200)).type(torch.long)):
                    continue
                else:
                    pred_segs = model(imgs.to(device),
                                    rots.to(device),
                                    trans.to(device),
                                    intrins.to(device),
                                    post_rots.to(device), 
                                    post_trans.to(device))
                    binimgs = binimgs.to(device)
                    val_loss = criterion(pred_segs, binimgs)
                    intersect, union, _ = get_batch_iou(pred_segs, binimgs)
                    total_intersect += intersect
                    total_union += union
                    total_val_loss += val_loss.item()
                    
            total_iou = total_intersect / total_union

        # print(f'Total train loss: {total_train_loss:.4f}')
        # print(f'Total val loss: {total_val_loss:.4f}')
        # print(f'Total batch iou: {total_iou:.4f}')
            writer.add_scalar('train/loss', total_train_loss, epoch)
            writer.add_scalar('val/loss', total_val_loss, epoch)
            writer.add_scalar('val/iou', total_iou, epoch)

        if epoch % 20 == 0:
            torch.save(model, f'model_{epoch}.pt')

if __name__ == '__main__':
    main()
