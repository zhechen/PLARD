import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import collections

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict
import cv2

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    print("Building " + model_name)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, phase='test')
    im_paths = loader.im_paths()
    n_classes = loader.n_classes 
    testloader = data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False)
 
    # Setup Model
    model = get_model(model_name, n_classes) 
    state = torch.load(args.model_path)['model_state']

    model.load_state_dict(state)
    model.eval()   
    model.cuda()

    # Run test for KITTI Road dataset
    for i, (image, tr_image, lidar, tr_lidar) in enumerate(testloader):
        im_name_splits = im_paths[i].split('/')[-1].split('.')[0].split('_')
        task = im_name_splits[0]

        print('processing %d-th image'%i)
        t0 = time.time()
        orig_h, orig_w = image.shape[1:3]
        with torch.no_grad():
            tr_image = Variable(tr_image.cuda())
            tr_lidar = Variable(tr_lidar.cuda())
            outputs = model([tr_image, tr_lidar])
            outputs = outputs.cpu().numpy().transpose((2,3,1,0)).squeeze()
            outputs = cv2.resize(outputs, (orig_w, orig_h))
            outputs = outputs[:,:,1]

        print('Time({:d}'.format(i) + ') {0:.3f}'.format(time.time() - t0))
        output_fg = outputs * 255.
        output_fg[output_fg > 255] = 255
        output_fg = output_fg.astype(np.uint8)

        cv2.imwrite('./outputs/results/' + im_name_splits[0] + '_road_' + im_name_splits[1] + '.png', output_fg)
        print('write to ./outputs/results/' + im_name_splits[0] + '_road_' + im_name_splits[1] + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='kitti_road', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    args = parser.parse_args()
    test(args)

