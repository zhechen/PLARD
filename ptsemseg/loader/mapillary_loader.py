import os
import torch
import numpy as np
#import scipy.misc as m
import cv2

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *
import random
import pdb

SCALES = [(640,832), (704,928),(800,1056),(896,1216)]

def augmentation(img, lbl):
    """transform

    :param img:
    :param lbl:
    """
    base_scale_id = 2
    base_w = SCALES[base_scale_id][1]
    base_h = SCALES[base_scale_id][0]
    #base_w = 1056; base_h = 800
    #base_w = 928; base_h = 704

    if random.randint(0, 1) == 0:
        img = img[:,::-1,:]
        lbl = lbl[:,::-1]

    scale_id = random.randint(0,2)
    scale = SCALES[scale_id]
    img = cv2.resize(img, (scale[1], scale[0]))
    lbl = cv2.resize(lbl, (scale[1], scale[0]), interpolation=cv2.INTER_NEAREST)

    if scale_id > base_scale_id:
        crop_x = random.randint(0,scale[1] - base_w - 1)
        crop_y = random.randint(0,scale[0] - base_h - 1) 
        img = img[crop_y:crop_y + base_h, crop_x:crop_x+base_w,:]
        lbl = lbl[crop_y:crop_y + base_h, crop_x:crop_x+base_w]

    elif scale_id < base_scale_id:
        img_tmp = np.zeros( (base_h, base_w, 3), img.dtype)
        img_tmp[:scale[0], :scale[1],:] = img
        lbl_tmp = np.zeros( (base_h, base_w), lbl.dtype) + 255
        lbl_tmp[:scale[0], :scale[1]] = lbl

        img = img_tmp
        lbl = lbl_tmp

    #base_w = SCALES[base_scale_id][1]
    #base_h = SCALES[base_scale_id][0]

    #if random.randint(0, 1) == 0:
    #    img = img[:,::-1,:]
    #    lbl = lbl[:,::-1]

    #scale_id = random.randint(0,2)
    #scale = SCALES[scale_id]
    #img = cv2.resize(img, (scale[1], scale[0]))
    #lbl = cv2.resize(lbl, (scale[1], scale[0]), interpolation=cv2.INTER_NEAREST)

    ##if scale_id > base_scale_id:
    #crop_x = 0; crop_y = 0;
    #if scale[1] > base_w:
    #    crop_x = random.randint(0,scale[1] - base_w - 1)
    #if scale[0] > base_h:
    #    crop_y = random.randint(0,scale[0] - base_h - 1) 
    #im_h = min(scale[0], base_h); im_w = min(scale[1], base_w);
    #img = img[crop_y:crop_y + im_h, crop_x:crop_x+im_w,:]
    #lbl = lbl[crop_y:crop_y + im_h, crop_x:crop_x+im_w]

    ##elif scale_id < base_scale_id:
    #img_tmp = np.zeros( (base_h, base_w, 3), img.dtype)
    #lbl_tmp = np.zeros( (base_h, base_w), lbl.dtype) + 255
    #img_tmp[:im_h, :im_w, :] = img
    #lbl_tmp[:im_h, :im_w] = lbl
    #img = img_tmp
    #lbl = lbl_tmp

    return img, lbl

class MapillaryLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    mean_rgb = [103.939, 116.779, 123.68] # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, split="train", is_transform=False, use_multi_scale=False,
                 img_size=(600, 800), augmentations=None, img_norm=True, version='pascal', phase='train', fpn=False, norm=False):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 15 #9 #15
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb)
        self.files = {}

        #if phase == 'train':
        self.images_base = os.path.join(self.root, 'images')
        self.annotations_base = os.path.join(self.root, 'labelsRelabelling')
        self.im_files = recursive_glob(rootdir=self.images_base, suffix='.jpg')

        self.data_size = len(self.im_files)
        self.phase = phase
        self.fpn = fpn
        self.norm = norm
        #import pdb.set_trace()

    
        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        #self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        #self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
        #                    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
        #                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
        #                    'motorcycle', 'bicycle']

        #self.ignore_index = 250
        #self.class_map = dict(zip(self.valid_classes, range(19))) 

        #if not self.files[split]:
        #    raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (self.data_size, split))

    def __len__(self):
        """__len__"""
        return self.data_size

    def im_paths(self):
        return self.im_files

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        #import pdb
        #pdb.set_trace()
        img_path = self.im_files[index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-1].split('.')[0] + '.png')

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.phase == 'train':
            lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            lbl = np.array(lbl, dtype=np.uint8)
            #lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
            
            #if self.augmentations is not None:
            #    img, lbl = self.augmentations(img, lbl)
            
            img, lbl = augmentation(img, lbl)

            #if self.is_transform:
            img, lbl = self.transform(img, lbl)

            if type(lbl) is list:
                for i in range(len(lbl)):
                    lbl[i][lbl[i] >= self.n_classes] = 255
            else:
                lbl[lbl >= self.n_classes] = 255

            #print('(%d,%d)-(%d,%d)'%(img.shape[1], img.shape[2], lbl.shape[0], lbl.shape[1]))
            return img, lbl
        else:
            img = cv2.resize(img, (1056,800))
            tr_img, _ = self.transform(img.copy(), lbl=None)
            return img, tr_img


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        #img = cv2.resize(img, (self.img_size[1], self.img_size[0])) # uint8 with RGB mode
        img = img.astype(np.float64)
        if self.norm:
            img = img[:,:,::-1] / 255.
            img[:,:,0] = (img[:,:,0] - 0.485) / 0.229
            img[:,:,1] = (img[:,:,1] - 0.456) / 0.224
            img[:,:,2] = (img[:,:,2] - 0.406) / 0.225
        else:
            img -= self.mean

        #if self.img_norm:
        #    # Resize scales images from 0 to 255, thus we need
        #    # to divide by 255.0
        #    img = img.astype(float) / 255.0

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        #classes = np.unique(lbl)
        if lbl is not None:
            lbl_size = lbl.shape
            if self.fpn:
                lbl_4x = cv2.resize(lbl, (lbl_size[1]/4, lbl_size[0]/4), interpolation=cv2.INTER_NEAREST)
                lbl_8x = cv2.resize(lbl, (lbl_size[1]/8, lbl_size[0]/8), interpolation=cv2.INTER_NEAREST)
                """
                lbl_16x = cv2.resize(lbl, (lbl_size[1]/16, lbl_size[0]/16), interpolation=cv2.INTER_NEAREST)
                lbl_32x = cv2.resize(lbl, (lbl_size[1]/32, lbl_size[0]/32), interpolation=cv2.INTER_NEAREST)
                """
                lbl_16x = cv2.resize(lbl, (lbl_size[1]/8, lbl_size[0]/8), interpolation=cv2.INTER_NEAREST)
                lbl_32x = cv2.resize(lbl, (lbl_size[1]/8, lbl_size[0]/8), interpolation=cv2.INTER_NEAREST)
                #"""

                lbl_4x = torch.from_numpy(lbl_4x).long()
                lbl_8x = torch.from_numpy(lbl_8x).long()
                lbl_16x = torch.from_numpy(lbl_16x).long()
                lbl_32x = torch.from_numpy(lbl_32x).long()

            #else:
            #    lbl = cv2.resize(lbl, (lbl_size[1]/8, lbl_size[0]/8), interpolation=cv2.INTER_NEAREST)
            lbl = torch.from_numpy(lbl).long()

        if self.fpn:
            #return img, [lbl_32x, lbl_16x, lbl_8x, lbl_4x] # lbl_4x,
            return img, [lbl_32x, lbl_16x] # lbl_4x,
        else:
            return img, lbl

    #def decode_segmap(self, temp):
    #    r = temp.copy()
    #    g = temp.copy()
    #    b = temp.copy()
    #    for l in range(0, self.n_classes):
    #        r[temp == l] = self.label_colours[l][0]
    #        g[temp == l] = self.label_colours[l][1]
    #        b[temp == l] = self.label_colours[l][2]

    #    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    #    rgb[:, :, 0] = r / 255.0
    #    rgb[:, :, 1] = g / 255.0
    #    rgb[:, :, 2] = b / 255.0
    #    return rgb

    #def encode_segmap(self, mask):
    #    #Put all void classes to zero
    #    for _voidc in self.void_classes:
    #        mask[mask==_voidc] = self.ignore_index
    #    for _validc in self.valid_classes:
    #        mask[mask==_validc] = self.class_map[_validc]
    #    return mask

#if __name__ == '__main__':
#    import torchvision
#    import matplotlib.pyplot as plt
#
#    augmentations = Compose([Scale(2048),
#                             RandomRotate(10),
#                             RandomHorizontallyFlip()])
#
#    local_path = '/home/meetshah1995/datasets/cityscapes/'
#    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
#    bs = 4
#    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
#    for i, data in enumerate(trainloader):
#        imgs, labels = data
#        imgs = imgs.numpy()[:, ::-1, :, :]
#        imgs = np.transpose(imgs, [0,2,3,1])
#        f, axarr = plt.subplots(bs,2)
#        for j in range(bs):      
#            axarr[j][0].imshow(imgs[j])
#            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
#        plt.show()
#        a = raw_input()
#        if a == 'ex':
#            break
#        else:
#            plt.close()
