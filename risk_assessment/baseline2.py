import baseline2_model
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import cv2
import time

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
    bbox = []
    for sample in batch:
        a,b,c,d = sample
        data.append(a)
        labels.append(b)
        file_name.append(c)
        bbox.append(d)
    data = np.array(data)
    labels = np.array(labels)
    return torch.from_numpy(data).to('cuda'), torch.from_numpy(labels).to('cuda'),file_name,bbox

def training(batch_size,n_epoch,learning_rate):
    net = baseline2_model.Baseline_SA(False).to('cuda')
    net.train()
    criterion = baseline2_model.custom_loss(100)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA',True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    # testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA',False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        #   shuffle=True, num_workers=0,collate_fn=detection_collate)
    total_batch = len(trainloader)
    best_vloss = 100000.0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels,_,_ = data
            # zero the parameter gradients
            # optimizer.zero_grad()
            # forward + backward + optimize
            pred, prob, pred_loss= net(inputs)
            loss = criterion(pred_loss, labels)
            loss.backward()
            # print statistics
            running_loss += loss.item()
            # print(loss.item(),running_loss)
            # running_loss = 0.0
            print("Batch : %d/%d" % (i+1,total_batch),end='\r')
        optimizer.step()
        optimizer.zero_grad()
        print('loss: %f'%(running_loss/total_batch))
        if running_loss < best_vloss:
            best_vloss = running_loss
            model_path = 'baseline2/model_{}'.format(epoch+1)
            print("Saving model..")
            torch.save(net.state_dict(), model_path)
    return
def demo(batch_size,model_path):
    net = baseline2_model.Baseline_SA(False)
    net.load_state_dict(torch.load(model_path))
    net.to('cuda')
    net.eval()
    print(net.parameters)
    testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA',False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)
    with torch.no_grad():
        for data in testloader:
            inputs,labels,file_names,det = data
            preds, probs, _= net(inputs)
            for i,label in enumerate(labels):
                if label:
                    pred = preds[i].cpu().numpy()
                    prob = probs[i]
                    plt.figure(figsize=(14,5))
                    plt.plot(pred[:90,1],linewidth=3.0)
                    plt.ylim(0, 1)
                    plt.ylabel('Probability')
                    plt.xlabel('Frame')
                    plt.show()
                    continue
                    file_name = file_names[i]
                    bboxes = det[i]
                    # new_weight = weight[:,:,i]*255
                    pass
                    counter = 0 
                    print(file_name)
                    cap = cv2.VideoCapture(video_path+'00'+file_name+'.mp4') 
                    ret, frame = cap.read()
                    while(ret):
                        attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                        now_weight = new_weight[counter,:]
                        new_bboxes = bboxes[counter,:,:]
                        index = np.argsort(now_weight)
                        # print(now_weight)
                        for num_box in index:
                            if now_weight[num_box]/255.0>0.4:
                                cv2.rectangle(frame,(new_bboxes[num_box,1],new_bboxes[num_box,0]),(new_bboxes[num_box,3],new_bboxes[num_box,2]),(0,255,0),3)
                            else:
                                cv2.rectangle(frame,(new_bboxes[num_box,1],new_bboxes[num_box,0]),(new_bboxes[num_box,3],new_bboxes[num_box,2]),(255,0,0),2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(new_bboxes[num_box,1],new_bboxes[num_box,0]), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                            attention_frame[int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2]),int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3])] = now_weight[num_box]
                            
                            cv2.putText(frame,str(num_box),(int(new_bboxes[num_box,3]),int(new_bboxes[num_box,2])), font, 1,(0,0,255),1,cv2.LINE_AA)

                        attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                        dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                        cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
                        cv2.imshow('result',dst)
                        time.sleep(0.05)
                        c = cv2.waitKey(50)
                        ret, frame = cap.read()
                        if c == ord('q') and c == 27 and ret:
                            break
                        counter += 1
                    cap.release()
                
                cv2.destroyAllWindows()


if __name__ == '__main__':
    learning_rate = 0.0001
    batch_size = 10
    n_epoch = 40
    model_path = 'baseline2/model_0'
    training(batch_size,n_epoch,learning_rate)
    # demo(batch_size,model_path)