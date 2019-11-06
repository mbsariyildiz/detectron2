# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import math

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.layers import FrozenBatchNorm2d

__all__ = [ 'VGG16', 'build_vgg16_backbone']


class VGG16(Backbone):

    def __init__(self, features, sobel):
        super().__init__()
        self.features = features
        self.d_ft = 512
        self._out_features = ["features"]
        self._out_feature_strides = {"features": 16}
        self._out_feature_channels = {"features": 512}

        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        return { "features": x }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512] # removed the last max-pooling layer , 'M'
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, input_shape):
    """
    Creates a VGG16 instance from the config.

    Returns:
        VGG16: a :class:`VGG16` instance.
    """
    sobel = False # cfg.MODEL.SOBEL
    bn = True
    dim = 2 + int(not sobel)
    model = VGG16(make_layers(dim, bn), sobel)

    freeze = cfg.MODEL.BACKBONE.FREEZE_AT > 0
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    return model

