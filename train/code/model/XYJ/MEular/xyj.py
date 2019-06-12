from model import common
import torch
import torch.nn as nn
import torch.nn.functional as f

def make_model(args, parent=False):
    return XYJ(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## PRE-activation Residual Unit (PRE)
class PRE(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(PRE, self).__init__()
        modules_body = []
        for i in range(2):
            if bn: modules_body.append(nn.GroupNorm(16,64))
            modules_body.append(act)
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Modified Eular (ME)
class MEGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(MEGroup, self).__init__()
        res1_body = []
        res1_body = [
            PRE(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)) \
            for _ in range(n_resblocks)]
        res1_body.append(conv(n_feat, n_feat, kernel_size))
        self.res1 = nn.Sequential(*res1_body)
        
        res2_body = []
        res2_body = [
            PRE(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)) \
            for _ in range(n_resblocks)]
        res2_body.append(conv(n_feat, n_feat, kernel_size))
        self.res2 = nn.Sequential(*res2_body)
        self.scale = 0.5
        
    def forward(self, x):
        res1 = self.res1(x)
        x = x + res1
        x = x + (res1 + self.res2(x)).mul(self.scale)
        return x


## Residual Channel Attention Network (XYJ)
class XYJ(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(XYJ, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            MEGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))