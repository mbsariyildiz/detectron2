# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.layers import FrozenBatchNorm2d

__all__ = [ 'AlexNet', 'build_alexnet_backbone']


# For both config items, we removed the last max-pool from config.
# When we do that, the stride of the network becomes around 16:
# image_size:224 activation_size:(13, 13) stride:17.2
# image_size:480 activation_size:(29, 29) stride:16.5
# image_size:512 activation_size:(31, 31) stride:16.5
# image_size:544 activation_size:(33, 33) stride:16.4
# image_size:576 activation_size:(35, 35) stride:16.4
# image_size:608 activation_size:(37, 37) stride:16.4
# image_size:640 activation_size:(39, 39) stride:16.4
# image_size:672 activation_size:(41, 41) stride:16.3
# image_size:704 activation_size:(43, 43) stride:16.3
# image_size:736 activation_size:(45, 45) stride:16.3
# image_size:768 activation_size:(47, 47) stride:16.3
# image_size:800 activation_size:(49, 49) stride:16.3
# To overcome this problem, we do either of the followings so that the stride
# becomes 16 sharp:
# i) resize the activation map with bicubic interpolation .
# ii) pad the activation plane
# (number of filters, kernel size, stride, pad)
ARCH_CFG = {
    'caron': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1)],
    'pytorch': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1)]
}

class AlexNet(Backbone):

    def __init__(self, features, sobel):
        super(AlexNet, self).__init__()
        self.features = features
        self._out_features = ["features"]
        self._out_feature_strides = {"features": 16}
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
        ih, iw = x.shape[2:]
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)

        # to have a fixed stride, compute the size of the new output map
        oh = ih // self._out_feature_strides["features"]
        ow = iw // self._out_feature_strides["features"]
        bs, nc, xh, xw = x.size()
        assert oh >= xh
        assert ow >= xw

        # fix the stride by padding the output plane
        x_new = torch.zeros(bs, nc, oh, ow, dtype=x.dtype, device=x.device)
        x_new[:, :, :xh, :xw] = x

        # fix the stride by interpolating the output plane
        # x_new = nn.functional.interpolate(x, (oh, ow), mode='bicubic', align_corners=False)

        return { "features": x_new }

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

