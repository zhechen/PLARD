import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

from ptsemseg import caffe_pb2
from ptsemseg.models.utils import *
from ptsemseg.loss import *

# Bottleneck for lidar net
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, final_relu=True, use_gn=False):
        super(Bottleneck, self).__init__()
        bn_func = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if use_gn:
            if planes // 32 < 2:
                self.bn1 = nn.GroupNorm(4, planes) 
            else:
                self.bn1 = nn.GroupNorm(16, planes) 
        else:
            self.bn1 = bn_func(planes) 

        self.conv21 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        if use_gn:
            self.bn21 = nn.GroupNorm(planes//4, planes) 
        else:
            self.bn21 = bn_func(planes) 

        self.conv22 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=4*dilation, bias=False, dilation=4*dilation)
        if use_gn:
            self.bn22 = nn.GroupNorm(planes//4, planes) 
        else:
            self.bn22 = bn_func(planes) 

        self.conv23 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=8*dilation, bias=False, dilation=8*dilation)
        if use_gn:
            self.bn23 = nn.GroupNorm(planes//4, planes) 
        else:
            self.bn23 = bn_func(planes) 

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if use_gn:
            self.bn3 = nn.GroupNorm((self.expansion * planes)//4, self.expansion * planes) 
        else:
            self.bn3 = bn_func(planes * self.expansion) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.final_relu = final_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # hybrid convolution
        out_1 = self.conv21(out)
        out_1_bn = self.bn21(out_1)
        out_1_bn = self.relu(out_1_bn)

        out_2 = self.conv22(out_1_bn)
        out_2_bn = self.bn22(out_2)
        out_2_bn = self.relu(out_2_bn)

        out_3 = self.conv23(out_2_bn)
        
        out = out_1 + out_2 + out_3

        out = self.bn23(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.final_relu:
            out = self.relu(out)

        return out

class plard(nn.Module):
    """
    Progressive LiDAR Adaptation for Road Detection (PLARD)

    This implmentation is based on the pspnet (URL: https://arxiv.org/abs/1612.01105)
    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow
    """

    def __init__(self, 
                 n_classes=2, 
                 block_config=[3, 4, 23, 3], 
                 input_size=(1280,384), 
                 version=None):

        super(plard, self).__init__()
        self.block_config = block_config 
        self.n_classes = n_classes

        
        # Visual Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=64,
                                                 padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1) 
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1) 
        
        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2) 
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4) 
        
        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [5,10,20,40]) 
       
        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False, use_gn=True)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False, use_gn=True)
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)
        self.relu_no_inplace = nn.ReLU(inplace=False)

        #### Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d
        ####

        # LiDAR Net
        block = Bottleneck
        layers = [3, 4, 23, 3]
        self.inplanes = 16
        self.lidar_conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.lidar_bn1 = nn.GroupNorm(4, 16)
        self.lidar_layer1 = self._make_layer(block, 16, layers[0], use_gn=True) 
        self.lidar_layer2 = self._make_layer(block, 32, layers[1], stride=2, use_gn=True) 
        self.lidar_layer3 = self._make_layer(block, 64, layers[2], stride=1, dilation=2, use_gn=True) 
        self.lidar_layer4 = self._make_layer(block, 128, layers[3], stride=1, dilation=4, use_gn=True) 

        # FA 1
        self.layer1_tr = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer1_tr = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer1_feat = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer1_tr_alpha = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer1_tr_beta = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        # Fuse 1
        self.lidar_layer1_fuse = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)

        # FA 2
        self.layer2_tr = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer2_tr = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer2_feat = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer2_tr_alpha = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer2_tr_beta = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        # Fuse 2
        self.lidar_layer2_fuse = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True)

        # FA 3
        self.layer3_tr = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer3_tr = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer3_feat = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer3_tr_alpha = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer3_tr_beta = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        # Fuse 3
        self.lidar_layer3_fuse = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=True)

        # FA 4
        self.layer4_tr = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer4_tr = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer4_feat = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer4_tr_alpha = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lidar_layer4_tr_beta = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        # Fuse 4
        self.lidar_layer4_fuse = nn.Conv2d(256, 2048, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_lidar_cls = nn.Conv2d(512, self.n_classes, 1, 1, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, final_relu=True, use_gn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_gn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(4, planes * block.expansion), 
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation, use_gn=use_gn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, dilation=dilation, use_gn=use_gn))
        layers.append(block(self.inplanes, planes, dilation=dilation, use_gn=use_gn, final_relu=final_relu))

        return nn.Sequential(*layers)

    def forward(self, inputs, no_iter=-1):
        if len(inputs) == 3: # training input
            x = inputs[0]
            lidar = inputs[1]
            target = inputs[2]
        elif len(inputs) == 2: # testing input
            x = inputs[0]
            lidar = inputs[1]
        
            x = x[None,...] if len(x.shape) < 4 else x
            lidar = lidar[None,...] if len(lidar.shape) < 4 else lidar

        inp_shape = x.shape[2:]

        with torch.no_grad():
            # H, W -> H/2, W/2
            x = self.convbnrelu1_1(x)
            x = self.convbnrelu1_2(x)
            x = self.convbnrelu1_3(x)
            # H/2, W/2 -> H/4, W/4
            x = F.max_pool2d(x, 3, 2, 1)

        lidar_x = self.lidar_conv1(lidar)
        lidar_x = self.lidar_bn1(lidar_x)
        lidar_x = F.max_pool2d(lidar_x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x_4x = self.res_block2(x)
        x_4x = self.dropout(x_4x)
        lidar_x_4x = self.lidar_layer1(lidar_x) 

        x_4x_tr = self.layer1_tr(x_4x.detach())
        x_4x_tr_relu = self.relu(x_4x_tr)
        lidar_x_4x_tr = self.lidar_layer1_tr(lidar_x_4x)
        lidar_x_4x_tr_relu = self.relu(lidar_x_4x_tr)

        cond_4x = torch.cat( (lidar_x_4x_tr_relu, x_4x_tr_relu), 1)
        alpha_4x = self.lidar_layer1_tr_alpha(cond_4x) + 1.
        beta_4x = self.lidar_layer1_tr_beta(cond_4x)

        lidar_feat_4x = self.lidar_layer1_feat(lidar_x_4x) 
        lidar_feat_4x = alpha_4x * lidar_feat_4x + beta_4x
        lidar_feat_4x = self.relu(lidar_feat_4x)
        lidar_feat_fuse = self.lidar_layer1_fuse(lidar_feat_4x)

        x_4x = x_4x + 0.1 * lidar_feat_fuse 
        x_4x = self.relu(x_4x)

        x_8x = self.res_block3(x_4x)
        x_8x = self.dropout(x_8x)
        lidar_x_8x = self.lidar_layer2(lidar_x_4x) 

        x_8x_tr = self.layer2_tr(x_8x.detach())
        x_8x_tr_relu = self.relu(x_8x_tr)
        lidar_x_8x_tr = self.lidar_layer2_tr(lidar_x_8x)
        lidar_x_8x_tr_relu = self.relu(lidar_x_8x_tr)

        cond_8x = torch.cat( (lidar_x_8x_tr_relu, x_8x_tr_relu), 1)
        alpha_8x = self.lidar_layer2_tr_alpha(cond_8x) + 1.
        beta_8x = self.lidar_layer2_tr_beta(cond_8x)

        lidar_feat_8x = self.lidar_layer2_feat(lidar_x_8x) 
        lidar_feat_8x = alpha_8x * lidar_feat_8x + beta_8x
        lidar_feat_8x = self.relu(lidar_feat_8x)
        lidar_feat_fuse = self.lidar_layer2_fuse(lidar_feat_8x)

        x_8x = x_8x + 0.1 * lidar_feat_fuse 
        x_8x = self.relu(x_8x)

        x_8x_res4 = self.res_block4(x_8x)
        x_8x_res4 = self.dropout(x_8x_res4)
        lidar_x_8x_res4 = self.lidar_layer3(lidar_x_8x) 

        x_8x_res4_tr = self.layer3_tr(x_8x_res4.detach())
        x_8x_res4_tr_relu = self.relu(x_8x_res4_tr)
        lidar_x_8x_res4_tr = self.lidar_layer3_tr(lidar_x_8x_res4)
        lidar_x_8x_res4_tr_relu = self.relu(lidar_x_8x_res4_tr)

        cond_8x_res4 = torch.cat( (lidar_x_8x_res4_tr_relu, x_8x_res4_tr_relu), 1)
        alpha_8x_res4 = self.lidar_layer3_tr_alpha(cond_8x_res4) + 1.
        beta_8x_res4 = self.lidar_layer3_tr_beta(cond_8x_res4)

        lidar_feat_8x_res4 = self.lidar_layer3_feat(lidar_x_8x_res4) 
        lidar_feat_8x_res4 = alpha_8x_res4 * lidar_feat_8x_res4 + beta_8x_res4
        lidar_feat_8x_res4 = self.relu(lidar_feat_8x_res4)
        lidar_feat_fuse = self.lidar_layer3_fuse(lidar_feat_8x_res4)

        x_8x_res4 = x_8x_res4 + 0.1 * lidar_feat_fuse 
        x_8x_res4 = self.relu(x_8x_res4)
            
        x_aux = self.convbnrelu4_aux(x_8x_res4)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)
        x_8x_res5 = self.res_block5(x_8x_res4)
        x_8x_res5 = self.dropout(x_8x_res5)
        lidar_x_8x_res5 = self.lidar_layer4(lidar_x_8x_res4)

        lidar_x_drop = self.dropout(lidar_x_8x_res5)
        lidar_x_cls = self.aux_lidar_cls(lidar_x_drop)

        x_8x_res5_tr = self.layer4_tr(x_8x_res5.detach())
        x_8x_res5_tr_relu = self.relu(x_8x_res5_tr)
        lidar_x_8x_res5_tr = self.lidar_layer4_tr(lidar_x_8x_res5)
        lidar_x_8x_res5_tr_relu = self.relu(lidar_x_8x_res5_tr)

        cond_8x_res5 = torch.cat( (lidar_x_8x_res5_tr_relu, x_8x_res5_tr_relu), 1)
        alpha_8x_res5 = self.lidar_layer4_tr_alpha(cond_8x_res5) + 1.
        beta_8x_res5 = self.lidar_layer4_tr_beta(cond_8x_res5)

        lidar_feat_8x_res5 = self.lidar_layer4_feat(lidar_x_8x_res5) 
        lidar_feat_8x_res5 = alpha_8x_res5 * lidar_feat_8x_res5 + beta_8x_res5
        lidar_feat_8x_res5 = self.relu(lidar_feat_8x_res5)
        lidar_feat_fuse = self.lidar_layer4_fuse(lidar_feat_8x_res5)

        x_8x_res5 = x_8x_res5 + 0.1 * lidar_feat_fuse 
        x_8x_res5 = self.relu(x_8x_res5)

        x = self.pyramid_pooling(x_8x_res5)
        
        fx = self.cbr_final(x, True)

        x = self.classification(fx)
        x = F.interpolate(x, size=inp_shape, mode='bilinear',align_corners=True)
        if self.training:
            target_x = target

            x_aux = F.interpolate(x_aux, size=inp_shape, mode='bilinear',align_corners=True)
            lidar_x_cls = F.interpolate(lidar_x_cls, size=inp_shape, mode='bilinear',align_corners=True)

            vis = True if no_iter%10 == 0 else False
            loss = self.loss([x, lidar_x_cls, x_aux], [target_x, target_x, target_x],
                        scale_weight=[1.0, 0.4, 0.16], vis=vis)
            loss.backward()
            return loss
        else: # eval mode
            return F.softmax(x, dim=1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

