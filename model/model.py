import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.upsample1 = Upsample(512, 256, 256 + 512, 512)
        self.upsample2 = Upsample(512, 256, 256 + 256, 512)
        self.cls_head = nn.Sequential(nn.Conv2d(512, 512, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, num_classes, 1, 1))

        self.reg_head = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 1, 1, 1),
                                      nn.ReLU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        x1 = self.features[: 19](x)
        x2 = self.features[19: 28](x1)
        x3 = self.features[28:](x2)

        feat = self.upsample1(x3, x2)
        feat = self.upsample2(feat, x1)

        cls_score = self.cls_head(feat)
        pred_den = self.reg_head(feat)
        cls_score_max = cls_score.max(dim=1, keepdim=True)[0]
        cls_score = cls_score - cls_score_max
        return pred_den, cls_score

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU(),
                                     nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])

    def forward(self, low, high):
        low = self.up(low)
        low = self.conv1(low)

        x = torch.cat([high, low], dim=1)

        x = self.conv2(x)
        return x

def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), num_classes)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model