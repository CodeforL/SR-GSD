from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return SRGSD(args)

## Residual Group (RG)
class Conv_Bn_Relu(nn.Module):
    def __init__(self, conv, in_feat,out_feat, kernel_size,stride=1):
        super(Conv_Bn_Relu, self).__init__()
        modules_body = []
        modules_body.append(conv(in_feat, out_feat, kernel_size,stride=stride))
        modules_body.append(nn.BatchNorm2d(out_feat))
        modules_body.append(nn.ReLU(True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


## Residual Channel Attention Network (RCAN)
class BPN(nn.Module):
    def __init__(self, args,kernel_size, conv=common.default_conv):
        super(BPN, self).__init__()

        n_feats = args.n_feats

        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [Conv_Bn_Relu(conv, n_feats, n_feats, kernel_size, stride=1), \
                        Conv_Bn_Relu(conv, n_feats, n_feats * 2, kernel_size, stride=2), \
                        Conv_Bn_Relu(conv, n_feats * 2, n_feats * 2, kernel_size, stride=1), \
                        Conv_Bn_Relu(conv, n_feats * 2, n_feats * 2, kernel_size, stride=1), \
                        Conv_Bn_Relu(conv, n_feats * 2, n_feats * 4, kernel_size, stride=2), \
                        Conv_Bn_Relu(conv, n_feats * 4, n_feats * 4, kernel_size, stride=1), \
                        Conv_Bn_Relu(conv, n_feats * 4, n_feats * 4, kernel_size, stride=1), \
                        Conv_Bn_Relu(conv, n_feats * 4, n_feats * 8, kernel_size, stride=2), \
                        Conv_Bn_Relu(conv, n_feats * 8, n_feats * 8, kernel_size, stride=1)]

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        modules_kernel = [conv(n_feats * 8, 225, kernel_size=1),
                          nn.Softmax(dim=1)]
        modules_noise =  [conv(n_feats * 8, 1, kernel_size=1), nn.Sigmoid()]

        self.bpn_head = nn.Sequential(*modules_head)
        self.bpn_body = nn.Sequential(*modules_body)
        self.bpn_kernel = nn.Sequential(*modules_kernel)
        self.bpn_noise = nn.Sequential(*modules_noise)

    def forward(self, x):
        x = self.bpn_head(x)
        x = self.bpn_body(x)
        x = self.avg_pooling(x)
        kernel = self.bpn_kernel(x)
        kernel = kernel * 255 - 127.5
        noise = self.bpn_noise(x)
        noise = noise * 70

        return kernel,noise

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
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
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

## Blind Super Resolution Network (SRGSD)
class SRGSD(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRGSD, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        scale = args.scale[0]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.bpn = nn.Sequential(*[BPN(args,kernel_size)])
        #fc_model = [conv(225, 225, kernel_size=1),nn.ReLU(True),conv(225, 15, kernel_size=1)]

        #kernel_reduction = [conv(225, 15, kernel_size=1,bias=False)]
        #self.kernel_reduct = nn.Sequential(*kernel_reduction)

        self.kernel_reduct = conv(225, 15, kernel_size=1,bias=False)
        self.noise_reduct = conv(1, 1, kernel_size=1, bias=False)



        modules_head = [conv(args.n_colors + 15 + 1, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.srn_head = nn.Sequential(*modules_head)
        self.srn_body = nn.Sequential(*modules_body)
        self.srn_tail = nn.Sequential(*modules_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        h,w = x.shape[2:]
        kernel, noise = self.bpn(x)

        kernel_hw = self.kernel_reduct((kernel + 127.5)/255).repeat(1,1,h,w)
        noise_hw = self.noise_reduct(noise).repeat(1,1,h,w)

        x = torch.cat([x,kernel_hw,noise_hw],1)
        x = self.srn_head(x)

        res = self.srn_body(x)
        res = res + x

        x = self.srn_tail(res)
        x = self.add_mean(x)

        degrade = torch.cat([kernel,noise],1).squeeze()
        return x, degrade


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