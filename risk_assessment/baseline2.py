import baseline2_model
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time
import zipfile
import os

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
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
        '--data',
        default="",
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
    bbox = []
    for sample in batch:
        a, b, c, d = sample
        data.append(a)
        labels.append(b)
        file_name.append(c)
        bbox.append(d)
    data = np.array(data)
    labels = np.array(labels)
    return torch.from_numpy(data).to('cuda'), torch.from_numpy(labels).to('cuda'), file_name, bbox

def training(batch_size,n_epoch,learning_rate,dataset):
    net = baseline2_model.Baseline_SA(False,features_size=256*7*7).to('cuda')  # detectron: 1024 or 256*7*7, maskformer: 256*7*7                                                           # maskformer2: 256x7x7 
    print(net)
    criterion = baseline2_model.custom_loss(80)                                                      
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA',dataset,True)    #SA_Detectron ,SA_Maskformer
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,collate_fn=detection_collate)
    testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA',dataset,False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,collate_fn=detection_collate)


    total_batch = len(trainloader)
    val_total_batch = len(testloader)
    best_vloss = 100000.0
    loss_list = []
    val_loss_list = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        net.train() 
        start_t = time.time()
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels,_,_ = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pred, _, pred_loss= net(inputs)
            pred = pred.argmax(dim=2)[:,80]
            labels = labels.long()
            correct += (pred==labels).sum().float()
            loss = criterion(pred_loss, labels)
            loss.backward()
            # print statistics
            running_loss += loss.item()
            # print(i, loss.item(), running_loss)
            print("\tBatch : %d/%d" % (i+1,total_batch),end='\r')
            optimizer.step()
        loss_list.append(running_loss/total_batch)
        print('\tloss: %f'%(running_loss/total_batch))
        print('\taccuracy: %f%%' % (correct/float(total_batch*batch_size)*100))
        if running_loss < best_vloss:
            best_vloss = running_loss
            model_path = 'baseline2_'+ dataset +'/model_{}'.format(epoch+1) # Detectron: 1024 Detectron2: 256x7x7
            # model_path = 'baseline2_test/model_{}'.format(epoch+1)
            print("Saving model..")
            torch.save(net.state_dict(), model_path)
        print("Validating...")
        net.eval()
        running_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs,labels,_,_ = data
                pred, _, pred_loss= net(inputs)
                labels = labels.long()
                loss = criterion(pred_loss, labels)
                pred = pred.argmax(dim=2)[:,80]
                correct += (pred==labels).sum().float()
                # print statistics
                running_loss += loss.item()
                # running_loss = 0.0
            val_loss_list.append(running_loss/val_total_batch)
        print('\tloss: %f'%(running_loss/val_total_batch))
        print('\taccuracy: %f%%' % (correct/float(val_total_batch*batch_size)*100))
        print('\tTime taken: ',time.time()-start_t,' seconds')
    print("Training finish.")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(range(1,n_epoch+1),loss_list)
    plt.plot(range(1,n_epoch+1),val_loss_list, color = 'red')
    plt.legend(["training","validation"])
    plt.show()
    return

def testing(batch_size, model_path):
    net = baseline2_model.Baseline_SA(False, features_size=256*7*7) # features_size: 1024 (detectron)
    net.load_state_dict(torch.load(model_path))                  #                2048*2*2 (maskformer res5, roi_align kernel:2x2)
    net.to('cuda')
    net.eval()

    testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA', False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,collate_fn=detection_collate)
    total_batch = len(testloader)
    correct = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs,labels,_,_ = data
    
            pred, prob, pred_loss= net(inputs)
            pred = pred.argmax(dim=2)[:,80]
            labels = labels.long()
            correct += (pred==labels).sum().float()

            print("\tBatch : %d/%d" % (i+1,total_batch),end='\r')
        print('\taccuracy: %f' %(correct/float(total_batch*batch_size)))
def evaluation(batch_size,model_path,dataset):
    features_size = 256*7*7
    criterion = baseline2_model.custom_loss(80)  
    net = baseline2_model.Baseline_SA(False, features_size=features_size)
    net.load_state_dict(torch.load(model_path))
    net.to('cuda')
    net.eval()
    if dataset == 'CARLA':
        testset = baseline2_model.SADataset('/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/dataset/collision', dataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    else:
        testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA', False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,collate_fn=detection_collate)
    correct = 0.0
    correct_2 =0.0
    total = 0.0
    total_2 = len(testloader)
    running_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs,labels,_,_ = data
            preds, _, pred_loss = net(inputs) # txbx2, txbx(20x10)
            pred_2 = preds.argmax(dim=2)[:,80]
            labels = labels.long()
            loss = criterion(pred_loss, labels)
            running_loss += loss.item()
            correct_2 += (pred_2==labels).sum().float().cpu().numpy()
            # print(preds)
            for i,label in enumerate(labels):
                if label:
                    pred = preds[i] # 100x2
                    pred = pred.argmax(dim=1)[80].cpu().numpy()
                    if pred == 1:
                        correct+=1
                    total += 1
        
    print('loss: %f'%(running_loss/total_2))
    print("Accuracy: ",correct_2/float(total_2*batch_size))
    print("Precision: ",correct/total)

def load_CARLA_scenario(path):
    img_archive = zipfile.ZipFile(os.path.join(path,'rgb','front.zip'), 'r')
    zip_file_list = img_archive.namelist()
    img_file_list = sorted(zip_file_list)[1:] # the first element is a folder
    # Read bbox
    bbox_archive = zipfile.ZipFile(os.path.join(path,'bbox','front.zip'), 'r')
    zip_file_list = bbox_archive.namelist()
    bbox_file_list = sorted(zip_file_list)[2:]
    # Align frame number
    index = -1
    while img_file_list[index][-7:-4]!=bbox_file_list[-1][-8:-5]:
        index -= 1
    if index!=-1:
        img_file_list = img_file_list[:index+1]
    img_file_list = img_file_list[-100:]
    return img_archive, img_file_list

def demo(batch_size,model_path,dataset):
    # if dataset == "Detectron" :
    #     features_size = 1024
    # elif dataset == "Maskformer":
    features_size = 256*7*7

    net = baseline2_model.Baseline_SA(False,features_size = features_size) 
    net.load_state_dict(torch.load(model_path))
    net.to('cuda')
    net.eval()
    if dataset == 'CARLA':
        testset = baseline2_model.SADataset('/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/dataset/collision', dataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    else:
        testset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA', False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,collate_fn=detection_collate)
    with torch.no_grad():
        for data in testloader:
            inputs,labels,file_names,det = data
            preds, probs, _= net(inputs)
            for i,label in enumerate(labels):
                if label:
                    pred = preds[i].cpu().numpy()
                    weight = probs[i].cpu().numpy()
                    plt.figure(figsize=(14,5))
                    plt.plot(pred[:90,1],linewidth=3.0)
                    plt.ylim(0, 1)
                    plt.ylabel('Probability')
                    plt.xlabel('Frame')
                    plt.show()
                    file_name = file_names[i]
                    bboxes = det[i]
                    new_weight = weight* 255
                    counter = 0 
                    print(file_name.split('/')[-3],file_name.split('/')[-1])
                    if dataset == 'CARLA':
                        img_archive, img_file_list = load_CARLA_scenario(file_name)
                        img_file = img_file_list[counter]
                        imgdata = img_archive.read(img_file)
                        frame = cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        cap = cv2.VideoCapture(VIDEO_PATH+str(file_name)) 
                        ret, frame = cap.read()
                    while(True):
                        frame = cv2.resize(frame,(640,320),interpolation=cv2.INTER_LINEAR)
                        attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                        now_weight = new_weight[counter,:]
                        new_bboxes = bboxes[counter,:]
                        for num_box in range(20):
                            if now_weight[num_box]/255.0>0.4:
                                cv2.rectangle(frame,(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])),(int(new_bboxes[num_box,2]),int(new_bboxes[num_box,3])),(0,255,0),1)
                            else:
                                cv2.rectangle(frame,(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])),(int(new_bboxes[num_box,2]),int(new_bboxes[num_box,3])),(255,0,0),1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])), font, 0.4,(0,0,255),1,cv2.LINE_AA)
                            attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]
                            # cv2.putText(frame,str(num_box),(int(new_bboxes[num_box,3]),int(new_bboxes[num_box,2])), font, 1,(0,0,255),1,cv2.LINE_AA)
                        attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                        dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                        cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
                        cv2.imshow('result',dst)
                        time.sleep(0.05)
                        c = cv2.waitKey(50)
                        counter += 1
                        if counter == 100:
                            break
                        if dataset == 'CARLA':
                            img_file = img_file_list[counter]
                            imgdata = img_archive.read(img_file)
                            frame = cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
                        else:
                            ret, frame = cap.read()
                        if c == ord('q') and c == 27 and ret:
                            break
                    if dataset != 'CARLA':
                        cap.release()
                
                cv2.destroyAllWindows()

def train_on_checkpoint(batch_size,n_epoch,learning_rate,dataset,model_path):
    if dataset == "Detectron" or dataset == "Maskformer":
        features_size = 1024
    elif dataset == "Maskformer2":
        features_size = 256*7*7
    net = baseline2_model.Baseline_SA(False,features_size=features_size).to('cuda')  
    net.load_state_dict(torch.load(model_path))
    net.train()                                                             
    criterion = baseline2_model.custom_loss(80)                                                      
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainset = baseline2_model.SADataset('/mnt/sdb/Dataset/SA_' + dataset,True)    #SA_Detectron ,SA_Maskformer
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,collate_fn=detection_collate)

    total_batch = len(trainloader)
    best_vloss = 100000.0
    loss_list = []
    for epoch in range(40,n_epoch):  # loop over the dataset multiple times
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        start_t = time.time()
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels,_,_ = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pred, _, pred_loss= net(inputs)
            pred = pred.argmax(dim=2)[:,80]
            labels = labels.long()
            correct += (pred==labels).sum().float()
            loss = criterion(pred_loss, labels)
            loss.backward()
            # print statistics
            running_loss += loss.item()
            # print(i, loss.item(), running_loss)
            print("\tBatch : %d/%d" % (i+1,total_batch),end='\r')
            optimizer.step()
        # optimizer.zero_grad()
        loss_list.append(running_loss/total_batch)
        print('\tloss: %f'%(running_loss/total_batch))
        print('\taccuracy: %f%%'%(correct/float(total_batch*batch_size)*100))
        print('\tTime taken: ',time.time()-start_t,' seconds')
        if running_loss < best_vloss:
            best_vloss = running_loss
            model_saving_path = 'baseline2_'+ dataset +'/model_{}'.format(epoch+1)
            print("Saving model..")
            torch.save(net.state_dict(), model_saving_path)
    print("Training finish.")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(range(41,n_epoch+1),loss_list)
    plt.show()
    return


if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()
    learning_rate = 0.0005
    batch_size = 10
    n_epoch = 30
    model_path = args.model
    dataset = args.data
    #model_path = 'baseline2/model_40'
    if args.mode == "training":
        training(batch_size, n_epoch, learning_rate, dataset)
    elif args.mode == "demo":
        demo(batch_size, model_path, dataset)
    elif args.mode == "testing":
        evaluation(batch_size, model_path,dataset)
    else:
        train_on_checkpoint(batch_size, n_epoch, learning_rate, dataset, model_path)
