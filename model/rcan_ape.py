## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torch
import torch.nn as nn

from utility import print_params

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

        self.test_only = args.test_only
        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        # define regressor
        m_regressor = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feats,1),
            nn.Tanh()
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.regressor = nn.Sequential(*m_regressor)

    def forward(self, x):
        if not self.test_only: # training mode / eval mode
            x = self.sub_mean(x)
            x = self.head(x)
            res = x

            outputs = []
            ics = []
            for i, layer in enumerate(self.body[:-1]):
                res = layer(res)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.add_mean(self.tail(x + self.body[-1](res)))
                    ic = self.regressor(res)
                    outputs.append(output)
                    ics.append(ic)
                    # output.append(self.add_mean(self.tail(x + res)))

            # x = self.tail(res)
            # x = self.add_mean(x)

            return outputs, ics
        else: # test mode
            x = self.sub_mean(x)
            x = self.head(x)
            res = x * 1.0 # to assign, avoid just reference

            exit_index = torch.ones(x.shape[0],device=x.device) * (-1.)
            pass_index = torch.arange(0,x.shape[0],device=x.device)
            for i, layer in enumerate(self.body[:-1]):
                if len(pass_index) > 0:
                    res[pass_index,...] = layer(res[pass_index,...])
                if i % self.exit_interval == (self.exit_interval-1):
                    ic = self.regressor(res)
                    pass_index = torch.where(ic<self.exit_threshold)[0]

                    remain_id = torch.where(exit_index < 0.0)[0]
                    exit_id = torch.where(ic>=self.exit_threshold)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = (i-(self.exit_interval-1))//self.exit_interval

            output = self.add_mean(self.tail(x + self.body[-1](res)))
            return output, exit_index, ic

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
