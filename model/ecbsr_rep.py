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
        kernel_size = 3
        self.n_stage  = args.m_ecbsr
        self.n_feats  = args.c_ecbsr
        self.with_idt = args.idt_ecbsr 
        self.n_colors = args.n_colors

        self.dm = args.dm_ecbsr
        self.act_type = args.act
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        # define the head
        m_head = [conv(self.n_colors, self.n_feats, kernel_size)]

        # define the body
        m_body = []
        for i in range(self.n_stage):
            m_body.append(conv(self.n_feats, self.n_feats, kernel_size))
        m_body.append(conv(self.n_feats, self.scale * self.scale * self.n_colors, kernel_size))

        # define the tail
        m_tail = [nn.PixelShuffle(self.scale)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # head = self.head(x)
        # body = self.body(head)
        # x = F.pixel_unshuffle(self.upsample(x), self.scale)
        # out = self.tail(body) + x
        # out  = self.upsampler(out)
        unshuffle_x = F.pixel_unshuffle(self.upsample(x), self.scale)
        x = self.head(x)
        res = self.body(x)
        res += unshuffle_x
        x  = self.tail(res)
        
        return x

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