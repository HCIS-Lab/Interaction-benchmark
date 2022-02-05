import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
 
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
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
        # acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data_front.size(0))
        # accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data_front), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\t'.format(
        len(data_loader.dataset), losses.avg))

    return losses.avg