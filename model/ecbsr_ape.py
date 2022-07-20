import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ecb import ECB
# from ecb import ECB
from model import common

def make_model(args, parent=False):
    return ECBSR(args)

class ECBSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ECBSR, self).__init__()

        scale = args.scale[0]
        self.scale = scale
        self.n_stage  = args.m_ecbsr
        self.n_feats  = args.c_ecbsr
        self.with_idt = args.idt_ecbsr 
        self.n_colors = args.n_colors

        self.test_only = args.test_only
        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold


        self.dm = args.dm_ecbsr
        self.act_type = args.act
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        # define the head
        m_head = [ECB(self.n_colors, out_planes=self.n_feats, depth_multiplier=self.dm, act_type=self.act_type, with_idt = self.with_idt)]

        # define the body
        m_body = []
        for i in range(self.n_stage):
            m_body.append(ECB(self.n_feats, out_planes=self.n_feats, depth_multiplier=self.dm, act_type=self.act_type, with_idt = self.with_idt))
        m_body.append(ECB(self.n_feats, out_planes=self.scale * self.scale * self.n_colors, depth_multiplier=self.dm, act_type='linear', with_idt = self.with_idt))

        # define the tail
        m_tail = [nn.PixelShuffle(self.scale)]

        # define regressor
        m_regressor = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.n_feats,1),
            nn.Tanh()
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.regressor = nn.Sequential(*m_regressor)

    def forward(self, x):
        if not self.test_only: # training mode / eval mode
            unshuffle_x = F.pixel_unshuffle(self.upsample(x), self.scale)
            x = self.head(x)
            res = x
            outputs = []
            ics = []

            for i, layer in enumerate(self.body[:-1]):
                res = layer(res)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.tail(unshuffle_x + self.body[-1](res))
                    ic = self.regressor(res)
                    outputs.append(output)
                    ics.append(ic)
            return outputs, ics
        else: # test mode
            unshuffle_x = F.pixel_unshuffle(self.upsample(x), self.scale)
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

            output = self.tail(unshuffle_x + self.body[-1](res))
            return output, exit_index, ic


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
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

if __name__ == "__main__":
    # RGB
    input = torch.rand((1,3,20,20))
    # model = ECBSR(4,32,4,'prelu',2,3)
    # output = model(input)
    # Y
    # input = torch.ones((1,1,20,20))
    # model = ECBSR(4,32,4,'prelu',2,1)
    # output = model(input)