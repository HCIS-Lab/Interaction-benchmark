import baseline1_model
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

def training(batch_size,n_epoch,learning_rate):
    net = baseline1_model.Baseline_Jinkyu('camera').to('cuda')
    criterion = baseline1_model.custom_loss(100)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA',True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    testset = baseline1_model.SADataset('/mnt/sdb/Dataset/SA',False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    total_batch = len(trainloader)*batch_size
    best_vloss = 100000.0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels,_ = data
            # zero the parameter gradients
            # forward + backward + optimize
            pred, prob= net(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            # print statistics
            running_loss += loss.item()
            # running_loss = 0.0
            print("Batch : %d" % (i+1),end='\r')
        print('loss: %f'%(running_loss/total_batch))
        optimizer.step()
        optimizer.zero_grad()
        if running_loss < best_vloss:
            best_vloss = running_loss
            model_path = 'model_{}'.format(epoch+1)
            print("Saving model..")
            torch.save(net.state_dict(), model_path)
    return

if __name__ == '__main__':
    learning_rate = 0.0001
    batch_size = 10
    n_epoch = 40
    training(batch_size,n_epoch,learning_rate)