import os
import torch
import numpy as np
import cv2

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *
import random

class KITTIRoadLoader(data.Dataset):
    """KITTI Road Dataset Loader

    http://www.cvlibs.net/datasets/kitti/eval_road.php

    Data is derived from KITTI
    """
    mean_rgb = [103.939, 116.779, 123.68] # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(1280, 384), augmentations=None, version='pascal', phase='train'):
        """__init__

        :param root:
        :param split:
        :param is_transform: (not used)
        :param img_size: (not used)
        :param augmentations  (not used)
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2
        self.img_size = img_size 
        self.mean = np.array(self.mean_rgb)
        self.files = {}

        if phase == 'train':
            self.images_base = os.path.join(self.root, 'training', 'image_2')
            self.lidar_base = os.path.join(self.root, 'training', 'ADI')
            self.annotations_base = os.path.join(self.root, 'training', 'gt_image_2')
            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
        else:
            self.images_base = os.path.join(self.root, 'testing', 'image_2')
            self.lidar_base = os.path.join(self.root, 'testing', 'ADI')
            self.annotations_base = os.path.join(self.root, 'testing', 'gt_image_2')
            self.split = 'test'

            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.im_files = sorted(self.im_files)

        self.data_size = len(self.im_files)
        self.phase = phase

        print("Found %d %s images" % (self.data_size, self.split))

    def __len__(self):
        """__len__"""
        return self.data_size

    def im_paths(self):
        return self.im_files

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.im_files[index].rstrip()
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lidar = cv2.imread(os.path.join(self.lidar_base, im_name_splits[0] + '_' + im_name_splits[1] + '.png'), cv2.IMREAD_UNCHANGED)
        lidar = np.array(lidar, dtype=np.uint8)

        if self.phase == 'train':
            lbl_path = os.path.join(self.annotations_base,
                                    im_name_splits[0] + '_road_' + im_name_splits[1] + '.png')

            lbl_tmp = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            lbl_tmp = np.array(lbl_tmp, dtype=np.uint8)
            
            lbl = 255 + np.zeros( (img.shape[0], img.shape[1]), np.uint8)
            lbl[lbl_tmp[:,:,0] > 0] = 1
            lbl[(lbl_tmp[:,:,2] > 0) & (lbl_tmp[:,:,0] == 0)] = 0
            
            img, lidar, lbl = self.transform(img, lidar, lbl)

            return img, lidar, lbl
        else:
            tr_img = img.copy()
            tr_lidar = lidar.copy()
            tr_img, tr_lidar = self.transform(tr_img, tr_lidar)
    
            return img, tr_img, lidar, tr_lidar


    def transform(self, img, lidar, lbl=None):
        """transform

        :param img:
        :param lbl:
        """
        img = img.astype(np.float64)
        img -= self.mean

        lidar = lidar.astype(np.float64) / 128.
        lidar = lidar - np.mean(lidar[lidar>0]) 

        img = cv2.resize(img, self.img_size)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        lidar = cv2.resize(lidar, self.img_size)
        lidar = lidar[np.newaxis, :, :] 
        lidar = torch.from_numpy(lidar).float()

        if lbl is not None:
            lbl = cv2.resize(lbl, (int(self.img_size[1]), int(self.img_size[0])), interpolation=cv2.INTER_NEAREST)
            lbl = torch.from_numpy(lbl).long()
            return img, lidar, lbl
        else:
            return img, lidar

