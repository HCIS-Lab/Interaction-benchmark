import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy


def val_epoch(model, data_loader, criterion, device, is_print=False):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        # for (data, targets) in data_loader:
        #     data, targets = data.to(device), targets.to(device)
        #     outputs = model(data)  

        #     loss = criterion(outputs, targets)
        #     acc = calculate_accuracy(outputs, targets)

        #     losses.update(loss.item(), data.size(0))
        #     accuracies.update(acc, data.size(0))

        for batch_idx, (data_front, data_right, data_left, targets) in enumerate(data_loader):
            data_front, data_right, data_left, targets = data_front.to(device), data_right.to(device),data_left.to(device), targets.to(device)
            outputs = model(data_front, data_right, data_left, targets)
            batch_size = data_front.shape[0]
            num_frame = data_front.shape[1]

            outputs = outputs[:, 1:, :]
            targets = targets[:, 1:]

            # [b, l_f, c] -> [c, b*l_f] == [16, 16, 36] -> [6*16, 16*6]
            # print(outputs.shape)
            outputs = outputs.view(batch_size*num_frame, -1)
            # print(outputs.shape)
            # [b, l] -> [b*l] == [16*6]
            # print(targets.shape)
            targets = targets.view(-1)
            # print(targets.shape)
            loss = criterion(outputs, targets)
            if is_print:
                print(batch_idx)
                
            acc = calculate_accuracy(outputs, targets, is_print)
            losses.update(loss.item(), data_front.size(0))
            accuracies.update(acc, data_front.size(0))
    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg
