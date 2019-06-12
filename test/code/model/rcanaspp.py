from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return RCAN(args)

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

## Residual Channel Attention Block (RCAB)
class RSKN(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,bias=True):
        super(RSKN, self).__init__()
        self.n_feat = n_feat
        modules_1 = [conv(n_feat, n_feat, 1, bias=bias),nn.ReLU(inplace=True)]
        modules_2 = [conv(n_feat, n_feat, 3, bias=bias), nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*modules_1)
        self.conv2 = nn.Sequential(*modules_2)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat*2, 1, padding=0, bias=True)
        )
        self.softmax = nn.Softmax(dim=2)
        self.tail = conv(n_feat,n_feat,kernel_size)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        sum = conv1 + conv2
        se = self.softmax(torch.reshape(self.se(sum),(-1,self.n_feat,2,1,1)))

        print(se[0,0,:,0,0])
        c_1 = se[:, :, 0, :, :]
        c_2 = se[:, :, 0, :, :]
        conv1 = conv1 * c_1
        conv2 = conv2 * c_2

        sum = conv1 + conv2
        res = x + self.tail(sum)
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class UPASPP(nn.Module):
    def __init__(self, conv, n_feat, scale, reduction,):
        super(UPASPP, self).__init__()
        modules_tail_1 = [common.Upsampler(conv, scale, n_feat, 1, act=False)]
        modules_tail_2 = [common.Upsampler(conv, scale, n_feat, 3, act=False)]
        modules_tail_3 = [common.Upsampler(conv, scale, n_feat, 3, dilation=2,act=False)]

        self.conv = conv(n_feat * 3, 3, 3)

        self.tail_1 = nn.Sequential(*modules_tail_1)
        self.tail_2 = nn.Sequential(*modules_tail_2)
        self.tail_3 = nn.Sequential(*modules_tail_3)
        '''
        self.n_feat = n_feat
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat*2, 1, padding=0, bias=True),
        )
        self.softmax = nn.Softmax(dim=2)
        self.conv = conv(n_feat,3,3)
        '''

    def forward(self, x):
        x_1 = self.tail_1(x)
        x_2 = self.tail_2(x)
        x_3 = self.tail_3(x)

        x = self.conv(torch.cat([x_1,x_2,x_3],1))
        return x



## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
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
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.up = UPASPP(conv, n_feats, scale, reduction)



    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.up(res)
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