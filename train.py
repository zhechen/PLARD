import sys, os
import torch
import visdom
import argparse
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import collections

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

def adjust_learning_rate(optimizer, epoch, lr, decay, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (decay ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, logger):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_cols, args.img_rows))

    n_classes = t_loader.n_classes
    nw = args.batch_size if args.batch_size > 1 else 0
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=nw, shuffle=True)
        
    # Setup Model
    model = get_model(args.arch, n_classes)

    if args.pretrained is not None:                                         
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict_without_classification(checkpoint['model_state'])
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) 
    model.cuda()
    
    mom = 0.99
    wd = 5e-4 
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=mom, weight_decay=wd) #0.99 5e-4

    print('Params: l_rate %f, l_rate_decay: %.2f, l_rate_step: %d, batch_size: %d, mom: %.2f, wd: %f'%(
        args.l_rate, args.l_rate_decay, args.l_rate_step, args.batch_size, mom, wd)) 
    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        logger.info('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
            logger.info("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 
            logger.info("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        adjust_learning_rate(optimizer, epoch, args.l_rate, args.l_rate_decay, args.l_rate_step)
        model.train()
        #if args.pretrained is not None:
        model.module.freeze_bn()

        avg_loss = 0.
        for i, (images, lidars, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            if type(labels) == list:
                var_labels = []
                for ii in range(len(labels)):
                    var_labels.append(Variable(labels[ii].cuda()))
            else:
                var_labels = Variable(labels.cuda())
            lidars = Variable(lidars.cuda())

            optimizer.zero_grad()

            loss = model([images, lidars, labels])

            optimizer.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            avg_loss += loss.detach().cpu().numpy().mean() #.data.item()
            #avg_loss += loss.data.item()
            if (i+1) % 10 == 0:
                avg_loss = avg_loss / 10.
                print("Epoch [%d/%d] [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), avg_loss))
                logger.info("Epoch [%d/%d] [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), avg_loss))
                avg_loss = 0.

        if epoch > 0:
            if (args.n_epoch <= 10 and epoch % 2 == 1) or epoch % 20 == 0:
                logger.info('saving models to ' + "{}_{}_{}.pkl".format(args.arch, args.dataset,epoch))
                print('saving models to ' + "{}_{}_{}.pkl".format(args.arch, args.dataset,epoch))
                state = {'epoch': epoch+1,
                         'model_state': model.module.state_dict(),
                         'optimizer_state' : optimizer.state_dict(),}
                torch.save(state, "./output-model/{}_{}_{}.pkl".format(args.arch, args.dataset,epoch))

    logger.info('saving models to ' + "{}_{}_{}.pkl".format(args.arch, args.dataset, args.n_epoch))
    print('saving models to ' + "{}_{}_{}.pkl".format(args.arch, args.dataset,epoch))
    state = {'epoch': epoch+1,
             'model_state': model.module.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}
    torch.save(state, "./output-model/{}_{}_{}.pkl".format(args.arch, args.dataset, args.n_epoch))
 
def setup_logging(name, filename=None):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    if filename is None:
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=filename)
    logger = logging.getLogger(name)
    return logger       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='pspnet', 
                        help='Architecture to use [\'plard, fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='mapillary', 
                        help='Dataset to use [\'kitti_road, pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=384, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=1280, 
                        help='Width of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=5, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-5, 
                        help='Learning Rate')
    parser.add_argument('--l_rate_decay', nargs='?', type=float, default=0.1, 
                        help='Learning Rate Decay')
    parser.add_argument('--l_rate_step', nargs='?', type=int, default=1, 
                        help='Learning Rate Step')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrained', nargs='?', type=str, default=None,    
                        help='pretriain')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()

    logger = setup_logging(__name__, filename='./'+args.arch+'.out')
    train(args, logger)
