# -*- coding: utf-8 -*-
'''
    Created on Sept 21 21:44 2018

    Author	        : Shaoshu Yang
    Email           : 13558615057@163.com
	Last edit date  : Sept 21 21:47

South East University Automation College, 211189 Nanjing China
'''

import os
import cv2
import numpy as np
import math

class processor():
    def __init__(self, filepath, destpath):
        '''
            Args:
                 filepath      : (string) directory of file that stores images
                 destpath      : (string) directory of the destination
        '''
        self.dir = filepath
        self.destdir = destpath

        # Get file names under filepath
        self.list = os.listdir(filepath)

    def rotate(self):
        '''
            Returns:
                 Save target images in self.destdir
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(path)
                img_h, img_w = img.shape[1], img.shape[2]

                # Generate rotation degree using Gaussian distribution
                deg_rot = np.random.normal(0, 30, size=1)
                new_h = int(img_w*abs(math.sin(deg_rot)) + img_h*abs(math.cos(deg_rot)))
                new_w = int(img_h*abs(math.sin(deg_rot)) + img_w*abs(math.cos(deg_rot)))

                # Generate rotation matrix, perform rotation process
                Mrot = cv2.getRotationMatrix2D((img_h/2, img_w/2), deg_rot, 1.0)
                img = cv2.warpAffine(img, Mrot, (new_h, new_w), borderValue=(168, 38, 61))

                # Save target images
                cv2.imwrite(self.destdir + "/%s"%path, img)

    def rotate_test(self, waitkey):
        '''
            Args:
                 waitkey       : (int) parameter of cv2.waitKey()
            Returns:
                 Print out target images
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(path)
                img_h, img_w = img.shape[1], img.shape[2]

                # Generate rotation degree using Gaussian distribution
                deg_rot = np.random.normal(0, 30, size=1)
                new_h = int(img_w * abs(math.sin(deg_rot)) + img_h * abs(math.cos(deg_rot)))
                new_w = int(img_h * abs(math.sin(deg_rot)) + img_w * abs(math.cos(deg_rot)))

                # Generate rotation matrix, perform rotation process
                Mrot = cv2.getRotationMatrix2D((img_h / 2, img_w / 2), deg_rot, 1.0)
                img = cv2.warpAffine(img, Mrot, (new_h, new_w), borderValue=(168, 38, 61))

                # Display target image
                cv2.imshow("target", img)
                cv2.waitKey(waitkey)


    def crop(self, left_top, right_bottom):
        '''
            Args:
                 left_top          : (list) left-top coordinate
                 right_bottom      : (list) right-bottom coordinate
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                # Perform cropping process
                img = img[:, left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]

                cv2.imwrite(self.destdir + "/%s"%path, img)

    def crop_test(self, left_top, right_bottom, waitkey):
        '''
            Args:
                 left_top          : (list) left-top coordinate
                 right_bottom      : (list) right-bottom coordinate
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                # Perform cropping process
                img = img[:, left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]

                # Display target image
                cv2.imshow("target", img)
                cv2.waitKey(waitkey)

    def flip(self, flipcode):
        '''
            Args:
                 flipcode        : (int) defines how to flip
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                # Perform flipping process
                img = cv2.flip(img, flipCode=flipcode)
                cv2.imwrite(self.destdir + "/%s" % path, img)

    def flip_test(self, flipcode, waitkey):
        '''
            Args:
                 flipcode        : (int) defines how to flip
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                # Perform flipping process
                img = cv2.flip(img, flipCode=flipcode)
                
                # Display target image
                cv2.imshow("target", img)
                cv2.waitKey(waitkey)

    def blur(self, rate):
        '''
            Args:
                 rate           : (float) defines the blurry
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                img_h, img_w = img.shape[1], img.shape[2]
                new_h, new_w = int(rate*img_h), int(rate*img_w)

                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(self.destdir + "/%s" % path, img)

    def blur_test(self, rate, waitkey):
        '''
            Args:
                 rate           : (float) defines the blurry
                 waitkey       : (int) parameter of cv2.waitKey()
        '''
        for i in range(0, len(self.list)):
            path = os.path.join(self.dir, list[i])

            # Read image if the path points to a file
            if os.path.isfile(path):
                img = cv2.imread(img)

                img_h, img_w = img.shape[1], img.shape[2]
                new_h, new_w = int(rate * img_h), int(rate * img_w)

                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

                # Display target image
                cv2.imshow("target", img)
                cv2.waitKey(waitkey)

