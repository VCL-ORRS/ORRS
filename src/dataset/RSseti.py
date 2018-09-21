# -*- coding: utf-8 -*-
'''
    Created on wed Sept 22 16:46 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 22 16:46 2018

South East University Automation College, 211189 Nanjing China
'''

import glob
import copy
import random
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import sys

class RSseti(Dataset):
    def __init__(self, root_dir, anno_file, img_size):
        '''
        Args:
             root_dir          : (string) directory of root file
             anno_file         : (string) directory to annotation file
             img_size          : (int) dimensions of input image
        '''
        anno_dir = os.path.join(root_dir, anno_file)

        # Open annotation file
        with open(anno_dir) as file:
            list = file.split('\n')

        self.list = []
        for i, content in enumerate(list):
            dic = {'filename': None, 'class': None}
            content = content.split()

            # Read filename
            dic['filename'] = content[0]

            # Get class
            dic['class'] = int(content[1])

            self.list.append(dic)

        self.img_size = img_size

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        '''
            Args:
                 idx            : (int) index of required object
            Returns:
                 Image tensor and label
        '''
        path = os.path.join(self.root_dir, self.list[idx]['filename'])
        img = cv2.imread(path)

        # Read the dimensions of image and calc the target image dimensions
        img_h, img_w = img.shape[0], img.shape[1]
        new_h = int(img_h*min(self.img_size/img_h, self.img_size/img_w))
        new_w = int(img_w*min(self.img_size/img_h, self.img_size/img_w))

        # Resize and pad
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.img_size, self.img_size, 3), 128)

        # Set the border color to [168, 38, 61]
        canvas[..., 0] = 168
        canvas[..., 1] = 38
        canvas[..., 2] = 61
        canvas[(self.img_size - new_h)//2:(self.img_size - new_h)//2 + new_h,
        (self.img_size - new_w)//2:(self.img_size - new_w)//2 + new_w, :] = img

        # set BGR to RGB, HWC to CHW and normalization
        canvas = torch.FloatTensor(canvas)
        canvas = canvas[:, :, ::-1].transpose(2, 0, 1).div(255.)

        # Get label
        label_ = self.list[idx]['class']
        label = torch.FloatTensor([0., 0.])
        label[label_] = 1

        return canvas, label

