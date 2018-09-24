# -*- coding: utf-8 -*-
'''
    Created on wed Sept 22 16:46 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 22 17:48 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from src.dataset.RSseti import RSseti
from src.model.resnet34 import ResNet34

def train(model, root_dir, anno_file, max_epoch, batch_size, lr, momentum,
                                weight_decay, check_point, weight_file_name):
    '''
        Args:
             model          : (nn.Module) untrained darknet
             root_dir       : (string) directory of root
             anno_file      : (string) directory to list file
             max_epoch      : (int) max epoches
             batch_size     : (int) batch size
             lr             : (float) learn rate
             momentum       : (float) momentum
             weight_decay   : (float) weight decay
             check_point    : (int) interval between weights saving
             weight_file_name
                            : (string) name of the weight file
        Returns:
             Output training info and save weight
    '''
    data_loader = DataLoader(RSseti(root_dir, anno_file, 64), batch_size=batch_size,
                             shuffle=True)

    cuda = torch.cuda.is_available()

    # MSE loss function
    mseloss = nn.MSELoss()

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)

    # Prepare a txt file to restore loss info
    loss_recorder = open('loss_recorder.txt', 'a+')

    # Perform training process
    for epoch in range(max_epoch):
        for idx, (canvas, target) in enumerate(data_loader):
            if cuda:
                canvas = canvas.cuda()
                target = target.cuda()

            # Back propagation
            optimizer.zero_grad()

            output = model(canvas)
            loss = mseloss(output, target)
            loss.backward()

            optimizer.step()

            # Calc recall
            nG = float(len(target))
            output = torch.max(output, 1)[1]
            target = torch.max(target, 1)[1]

            nCorrect = float(sum(output == target))
            recall = nCorrect/nG

            # Output train info
            print('[Epoch %d/%d, Batch %d/%d] [Losses: %f recall: %.5f]' %
                  (epoch + 1, max_epoch, idx + 1, len(data_loader), loss.item(), recall))
            loss_recorder.write("loss: %f, recall: %f\n" % (loss.item(), recall))

        if epoch % check_point == 0:
            torch.save(model.state_dict(), weight_file_name)

    torch.save(model.state_dict(), weight_file_name)
    loss_recorder.close()

if __name__ == '__main__':
    # Initialize ResNet34
    model = ResNet34(2)
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()
    train(model, "D:/KailinXu", "train.txt", 100, 64, 0.005, 0.9, 0.005, 1,
          "resnet34.pkl")
