import math
from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.test_only = args.test_only

        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

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
                    decision = self.regressor(res)
                    pass_index = torch.where(decision<self.exit_threshold)[0]

                    remain_id = torch.where(exit_index < 0.0)[0]
                    exit_id = torch.where(decision>=self.exit_threshold)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = (i-(self.exit_interval-1))//self.exit_interval

            output = self.add_mean(self.tail(x + self.body[-1](res)))
            return output, exit_index, decision


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))