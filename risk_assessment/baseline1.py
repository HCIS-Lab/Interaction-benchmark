import baseline1_model
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import argparse

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
VIDEO_PATH = '../../Anticipating-Accidents/dataset/videos/testing/positive/'
def get_parser():
    parser = argparse.ArgumentParser(description="Baseline 2")
    parser.add_argument(
        '-m',
        '--mode',
        default="training",
        help="training, testing or demo",
        type=str
    )
    parser.add_argument(
        '--model',
        # default="training",
        help="which model to use",
        type=str
    )
    parser.add_argument(
        '--dataset',
        # default="training",
        help="which dataset is used",
        type=str
    )
    return parser

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    data = []
    labels = []
    file_name = []
    for sample in batch:
        a,b,c = sample
        data.append(a)
        labels.append(b)
        file_name.append(c)
    data = np.array(data)
    labels = np.array(labels)
    return torch.from_numpy(data).to('cuda'), torch.from_numpy(labels).to('cuda'),file_name

def custom_learning_rate(optimizer, epoch, lr):
    if(epoch>4):
        optimizer.param_groups[0]["lr"] = lr/4.0


def training(batch_size,n_epoch,learning_rate, dataset):
    net = baseline1_model.Baseline_Jinkyu().to('cuda')
    print(net)
    criterion = baseline1_model.custom_loss(80)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    ###
    SA_dataset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA_cube')
    dataset_size = len(SA_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    random_seed = 20
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    trainloader = torch.utils.data.DataLoader(SA_dataset, batch_size=batch_size, 
                                            sampler=train_sampler)#,collate_fn=detection_collate)
    testloader = torch.utils.data.DataLoader(SA_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)#,collate_fn=detection_collate)
#    trainset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA_Maskformer' ,True)
#    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
#    testset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA_Maskformer' ,False)
#    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # scheduler = LambdaLR(optimizer, lr_lambda = lambda1)
    total_batch = len(trainloader)
    val_total_batch = len(testloader)
    best_vloss = 100000.0
    loss_list = []
    val_loss_list = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        # custom_learning_rate(optimizer,epoch,learning_rate)
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        net.train()
        start_t = time.time()
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels,_ = data
            optimizer.zero_grad()
            # zero the parameter gradients
            # forward + backward + optimize
            pred, _ = net(inputs)
            labels = labels.long()
            loss = criterion(pred, labels)
            pred = pred.argmax(dim=2)[80,:]
            correct += (pred==labels).sum().float().cpu().numpy()
            loss.backward()
            # print statistics
            running_loss += loss.item()
            # running_loss = 0.0
            optimizer.step()
            print("\tBatch : %d/%d" % (i+1,total_batch),end='\r')
        loss_list.append(running_loss/total_batch)
        print('\tloss: %f'%(running_loss/total_batch))
        print('\taccuracy: %f%%' % (correct/float(total_batch*batch_size)*100))
        print("Validating...")
        net.eval()
        running_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs,labels,file_name = data
                pred, _ = net(inputs)
                labels = labels.long()
                loss = criterion(pred, labels)
                pred = pred.argmax(dim=2)[80,:]
                correct += (pred==labels).sum().float().cpu().numpy()
                # print statistics
                running_loss += loss.item()
                # running_loss = 0.0
            val_loss_list.append(running_loss/val_total_batch)
        print('\tloss: %f'%(running_loss/val_total_batch))
        print('\taccuracy: %f%%' % (correct/float(val_total_batch*batch_size)*100))
        print('\tTime taken: ',time.time()-start_t,' seconds')
        # print("LEARNING RATE: {}".format(my_lr_scheduler.get_last_lr()))
        # my_lr_scheduler.step()
        if running_loss < best_vloss:
            best_vloss = running_loss
            model_path = 'baseline1_' + dataset +'/model_{}'.format(epoch+1)
            print("Saving model..")
            torch.save(net.state_dict(), model_path)
    print("Training finish.")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(range(1,n_epoch+1),loss_list, label = 'train')
    plt.scatter(range(1,n_epoch+1),val_loss_list, color = 'red', label = 'val')
    plt.show()
    return

def demo(batch_size, model_path, dataset):
    net = baseline1_model.Baseline_Jinkyu().to('cuda')
    net.load_state_dict(torch.load(model_path))
    net.to('cuda')
    net.eval()
    testset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA_' + dataset,False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    with torch.no_grad():
        for data in testloader:
            inputs,labels,file_names = data
            preds, probs, _ = net(inputs) # txbx2, txbx(20x10)
            preds = nn.functional.softmax(preds,dim=2)
            # print(preds)
            for i,label in enumerate(labels):
                if label:
                    pred = preds[:,i].cpu().numpy()
                    weight = probs[:,i].view(-1,10,20).cpu().numpy()
                    plt.figure(figsize=(14,5))
                    plt.plot(pred[:90,1],linewidth=3.0)
                    plt.ylim(0, 1)
                    plt.ylabel('Probability')
                    plt.xlabel('Frame')
                    plt.show()
                    file_name = file_names[i]
                    # bboxes = det[i]
                    # new_weight = weight*255
                    # counter = 0 
                    print(file_name)
                    # print(weight)
                    # for j in range(100):
                    #     imC = cv2.applyColorMap(weight[j], cv2.COLORMAP_JET)
                    #     cv2.imshow(imC)
                    #     c = cv2.waitKey(50)
                    # cv2.destroyAllWindows()
                    cap = cv2.VideoCapture(VIDEO_PATH+str(file_name)) 
                    ret, frame = cap.read()
                    fig = plt.figure()
                    sub_plot = fig.add_subplot(111)
                    # sub_plot.set_ylim(0,9)
                    frame_list = []
                    while ret:
                        frame = cv2.resize(frame,(640,320),interpolation=cv2.INTER_AREA)
                        frame_list.append(frame) # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ret, frame = cap.read()
                    for j in range(100):
                        # ret, frame = cap.read()
                        # frame = cv2.resize(frame,(640,320),interpolation=cv2.INTER_AREA)
                        sub_plot.set_title('{}th frame heatmap.'.format(j+1))
                        im1 = sub_plot.imshow(weight[j], cmap='viridis',interpolation = 'bilinear')
                        if j==0:
                            plt.colorbar(im1)
                        fig.canvas.draw()
                        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
                        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        # img is rgb, convert to opencv's default bgr
                        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        cv2.imshow("plot",img)
                        # display camera feed
                        cv2.imshow("cam",frame_list[j])

                        k = cv2.waitKey(33) & 0xFF
                        if k == 27:
                            break
                    cv2.destroyAllWindows()
                    cap.release()

def precision(batch_size,model_path):
    net = baseline1_model.Baseline_Jinkyu('camera').to('cuda')
    net.load_state_dict(torch.load(model_path))
    net.to('cuda')
    net.eval()
    testset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA_Maskformer3',False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs,labels,_ = data
            preds, _, _ = net(inputs) # txbx2, txbx(20x10)
            preds = nn.functional.softmax(preds,dim=2)
            # print(preds)
            for i,label in enumerate(labels):
                if label:
                    pred = preds[:,i] # 100x2
                    pred = pred.argmax(dim=1)[90].cpu().numpy()
                    if pred == 1:
                        correct+=1
                    total += 1
    print("Precision: ",correct/total)
                        

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()
    learning_rate = 0.0005
    batch_size = 10
    n_epoch = 30
    model_path = args.model
    dataset = args.dataset
    if args.mode == "training":
        training(batch_size, n_epoch, learning_rate, dataset)
    elif args.mode == "demo":
        demo(batch_size, model_path, dataset)
    # elif args.mode == "testing":
    #     evaluation(batch_size, model_path,dataset)
    # else:
    #     train_on_checkpoint(batch_size, n_epoch, learning_rate, dataset, model_path)
