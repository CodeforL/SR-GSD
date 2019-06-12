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

class QNNS(nn.Module): #QNNS
    def __init__(
        self, conv, n_feat,
        bias=True, bn=False, act=nn.ReLU(True)):
        
        super(QNNS,self).__init__()
        self.conv1_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        
        self.conv1_1 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3,stride=1,padding=1),
        )
        self.conv2_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feat, n_feat , 5, padding=2, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv3_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.max3_1 = nn.Sequential(nn.MaxPool2d(3,stride=1,padding=1),)
        
        self.conv4_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv4_1 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv5_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv6_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        
        self.conv = nn.Sequential(
                nn.Conv2d(n_feat*6, n_feat , 1, padding=0, bias=True),
                #nn.ReLU(inplace=True),
        )
        self.ca = CALayer(n_feat)
        
    def forward(self,x):
        temp1 = self.conv1_0(x)
        index1 = self.conv1_1(temp1)
        temp2 = self.conv3_0(x)
        index2 = self.conv2_0(temp1) + temp2
        index3 = self.max3_1(temp2)
        temp3 = self.conv4_0(x)
        index4 = self.conv4_1(temp2+temp3)
        index5 = self.conv5_0(temp3)
        index6 = self.conv6_0(x)
        
        res = x + self.ca(self.conv(torch.cat([index1,index2,index3,index4,index5,index6],1)))
        
        return res
        

class QNNB(nn.Module):
    def __init__(
        self, conv, n_feat,
        bias=True, bn=False, act=nn.ReLU(True)):
        
        super(QNNB,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        
        self.conv2_0 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv2_1 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 5, padding=2, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
                nn.Conv2d(n_feat*2, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 5, padding=2, bias=True),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(3,stride=1,padding=1)
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(n_feat, n_feat , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
                nn.Conv2d(n_feat*5, n_feat , 1, padding=0, bias=True),
                #nn.ReLU(inplace=True),
        )
        
    def forward(self,x):
        index1 = self.conv1(x)
        index2_0 = self.conv2_0(x)
        index2_1 = self.conv2_1(index2_0)
        index2 = index2_1 + index2_0
        index3 = self.conv2_2(torch.cat([index2_0,index2_1],1))
        index4 = self.conv3(x)
        index5 = self.conv4(x)
        
        return self.conv(torch.cat([index1,index2,index3,index4,index5],1))
        

class AGGA(nn.Module): #只聚合
    def __init__(
        self, conv, n_feat,in_channel,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(AGGA,self).__init__()
        
        modules_body = []
        modules_body.append(conv(in_channel, n_feat, kernel_size))
        if bn:
            modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(act)
        
        self.body = nn.Sequential(*modules_body)    
        
    def forward(self,x):
        return self.body(x)
        
class AGGB(nn.Module): #残差聚合
    def __init__(
        self, conv, n_feat,in_channel,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(AGGB,self).__init__()
        
        self.n_feat = n_feat
        modules_body = []
        modules_body.append(conv(in_channel, n_feat, 3))
        if bn:
            modules_body.append(nn.BatchNorm2d(n_feat))
        
        self.body = nn.Sequential(*modules_body)    
        self.act = act
        
    def forward(self,x):
        return self.act(x[:,:self.n_feat,:,:] + self.body(x))
        
        
class Bone(nn.Module):
    def __init__(
        self, conv, n_feat,aggregation=False,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(Bone,self).__init__()
        self.block1 = None
        self.block2 = None
        self.aggregation = aggregation
        self.agg = None
        
        self.block1 = QNNS(conv, n_feat, bias, bn, act)
        self.block2 = QNNS(conv, n_feat, bias, bn, act)
        
        if aggregation:
            self.agg = AGGB(conv,n_feat,n_feat*3, bias, bn, act)
        else:
            self.agg = AGGB(conv,n_feat,n_feat*2, bias, bn, act)
        
        
    def forward(self,x):
        stage1 = self.block1(x)
        stage2 = self.block2(stage1)
        if self.aggregation:
            return self.agg(torch.cat([stage2,stage1,x],1))
        else:
            return self.agg(torch.cat([stage2,stage1],1))
        
class HDA(nn.Module):
    def __init__(
        self, conv, layer, n_feat,iterative=False,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(HDA,self).__init__()
        
        self.layer = layer
        self.iterative = iterative
        modules_body = []
        for i in range(layer-1):
            if i == 0:
                modules_body.append(Bone(conv, n_feat,False,bias, bn, act))
            else:
                modules_body.append(Bone(conv, n_feat,True ,bias, bn, act))
        self.body = nn.Sequential(*modules_body)
        self.qnnb1 = QNNS(conv, n_feat, bias, bn, act)
        self.qnnb2 = QNNS(conv, n_feat, bias, bn, act)
        
        self.agg = None
        if iterative:
            if layer == 1:
                self.agg = AGGB(conv,n_feat,n_feat*3,bias, bn, act)
            else:
                self.agg = AGGB(conv,n_feat,n_feat*4,bias, bn, act)
        else:
            if layer == 1:
                self.agg = AGGB(conv,n_feat,n_feat*2,bias, bn, act)
            else:
                self.agg = AGGB(conv,n_feat,n_feat*3,bias, bn, act)

    def forward(self,x):
        if self.iterative:
            if self.layer == 1:
                qnnb1 = self.qnnb1(x)
                qnnb2 = self.qnnb2(qnnb1)
                out = torch.cat([qnnb2,qnnb1,x],1)
            else:
                body = self.body(x)
                qnnb1 = self.qnnb1(body)
                qnnb2 = self.qnnb2(qnnb1)
                out = torch.cat([qnnb2,qnnb1,body,x],1)
        else:
            if self.layer == 1:
                qnnb1 = self.qnnb1(x)
                qnnb2 = self.qnnb2(qnnb1)
                out = torch.cat([qnnb2,qnnb1],1)
            else:
                body = self.body(x)
                qnnb1 = self.qnnb1(body)
                qnnb2 = self.qnnb2(qnnb1)
                out = torch.cat([qnnb2,qnnb1,body],1)
        
        return self.agg(out)
            

## Residual Channel Attention Network (XYJ)
class XYJ(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(XYJ, self).__init__()
        
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        layers = [2,3,4,1]
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = []
        for i,layer in enumerate(layers):
            if i == 0:
                modules_body.append(HDA(conv,layer,n_feats,iterative=False,bias=True,bn=False, act=nn.ReLU(True)))
            else:
                modules_body.append(HDA(conv,layer,n_feats,iterative=True,bias=True,bn=False, act=nn.ReLU(True)))
        
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