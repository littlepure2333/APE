from turtle import forward
import torch
from torch import nn as nn
from torch.nn import functional as F


def make_model(args, parent=False):
    return RRDBNet(args)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, args, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()

        self.scale = args.scale[0]
        assert self.scale==4, "only support scale 4"

        num_block = args.n_resblocks
        self.test_only = args.test_only
        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold

        # define head module
        modules_head = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)]

        # define body module
        modules_body = [RRDB(num_feat) for _ in range(num_block)]
        modules_body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        # define regressor
        m_regressor = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_feat,1),
            nn.Tanh()
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = Tail(num_out_ch, num_feat)
        self.regressor = nn.Sequential(*m_regressor)


    def forward(self, x):
        # feat = self.head(x)
        # body_feat = self.body(feat)
        # feat = feat + body_feat

        # out = self.tail(feat)

        # return out

        if not self.test_only: # training mode / eval mode
            feat = self.head(x)
            body_feat = feat

            outputs = []
            ics = []
            for i, layer in enumerate(self.body[:-1]):
                body_feat = layer(body_feat)
                if (i % self.exit_interval == (self.exit_interval-1)) or (i==22):
                    output = self.tail(feat + self.body[-1](body_feat))
                    ic = self.regressor(body_feat)
                    outputs.append(output)
                    ics.append(ic)
                    # output.append(self.add_mean(self.tail(x + res)))

            # x = self.tail(res)
            # x = self.add_mean(x)

            return outputs, ics
        else: # test mode
            feat = self.head(x)
            body_feat = feat * 1.0 # to assign, avoid just reference

            exit_index = torch.ones(x.shape[0],device=x.device) * (-1.)
            pass_index = torch.arange(0,x.shape[0],device=x.device)
            for i, layer in enumerate(self.body[:-1]):
                if len(pass_index) > 0:
                    body_feat[pass_index,...] = layer(body_feat[pass_index,...])
                if (i % self.exit_interval == (self.exit_interval-1)) or (i==22):
                    ic = self.regressor(body_feat)
                    # print("ic", ic)
                    pass_index = torch.where(ic<self.exit_threshold)[0]

                    remain_id = torch.where(exit_index < 0.0)[0]
                    exit_id = torch.where(ic>=self.exit_threshold)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = (i-(self.exit_interval-1))//self.exit_interval

            output = self.tail(feat + self.body[-1](body_feat))
            return output, exit_index, ic
    

class Tail(nn.Module):
    def __init__(self, num_out_ch, num_feat=64):
        super(Tail, self).__init__()

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))

        return x
