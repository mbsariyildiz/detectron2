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

__all__ = [ 'AlexNet', 'build_alexnet_backbone']


# (number of filters, kernel size, stride, pad)
ARCH_CFG = {
    'caron': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1)],  # We removed the last max-pool from config, see the forward function.
    'pytorch': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1)]  # We removed the last max-pool from config, see the forward function.
}

class AlexNet(Backbone):

    def __init__(self, features, sobel):
        super(AlexNet, self).__init__()
        self.features = features
        self._out_features = ["features"]
        self._out_feature_strides = {"features": 32}
        self._out_feature_channels = {"features": 256}

        self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        image_size = x.size(2)
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        # we apply adaptive max pooling to fix the stride of AlexNet to 32.
        out_size = image_size // self._out_feature_strides["features"]
        x = nn.functional.adaptive_max_pool2d(x, (out_size, out_size))
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


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


@BACKBONE_REGISTRY.register()
def build_alexnet_backbone(cfg, input_shape):
    """
    Creates a VGG16 instance from the config.

    Returns:
        VGG16: a :class:`VGG16` instance.
    """
    arch_cfg = cfg.MODEL.BACKBONE.ARCH_CFG
    sobel = cfg.MODEL.BACKBONE.SOBEL
    bn = cfg.MODEL.BACKBONE.BN
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(ARCH_CFG[arch_cfg], dim, bn), sobel)

    freeze_bn = cfg.MODEL.BACKBONE.FREEZE_BN
    if freeze_bn and bn:
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)

    freeze = cfg.MODEL.BACKBONE.FREEZE_AT > 0
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


