import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import numpy as np
import cv2

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def valid_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, bias=bias)

def adaConv(input, kernel, bias=None):
    input = F.pad(input,(1,1,1,1)) # same padding
    B,C_in,H,W = input.shape
    B,C_out,C_in,h,w = kernel.shape
    H_out = H - h + 1
    W_out = W - w + 1

    inp_unf = torch.nn.functional.unfold(input, (h,w))
    out_unf = inp_unf.transpose(1,2) # (B, H_out*W_out, C_in*h*w)
    w_tran = kernel.view(kernel.size(0),kernel.size(1),-1).transpose(1,2) # (B, C_in*h*w, C_out)
    out_unf = out_unf.matmul(w_tran).transpose(1,2) # (B, C_out, H_out*W_out)
    out = out_unf.view(B,C_out,H_out,W_out)
    b = bias.reshape(B,C_out,1,1).repeat(1,1,H_out,W_out)
    out = out + b

    return out

# class AdaConv(nn.Module):
#     def __init__(self, kernel_size):
#         super(AdaConv, self).__init__()
#         self.weight = None
#         self.bias = None
#         self.padding = (kernel_size//2)
    
#     def update_param(self, weight, bias):
#         self.weight = weight
#         self.bias = bias
    
#     def forward(self, x):
#         out = F.conv2d(x, weight=self.weight, bias=self.bias, 
#                        stride=1, padding=self.padding)
#         return out

class Space2Batch(nn.Module):
    def __init__(self, kernel_size):
        super(Space2Batch, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = self.kernel_size - (H % self.kernel_size)
        pad_w = self.kernel_size - (W % self.kernel_size)

        x = F.pad(x, (0, pad_w, 0, pad_h)) # (B,C,H,W)
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        x = x.permute(0,2,1)
        x = x.contiguous().view(x.shape[0]*x.shape[1], C, self.kernel_size, self.kernel_size)
        # (B*N, C, h, w), N is the number of patches

        return x, B, C, H, W, pad_h, pad_w

class Batch2Space(nn.Module):
    def __init__(self, kernel_size):
        super(Batch2Space, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x, B, C, H, W):
        # (B*N, C, h, w), N is the number of patches
        x = x.contiguous().view(B, int(x.shape[0]/B), -1)
        x = x.permute(0,2,1)
        x = F.fold(x, output_size=(H,W), kernel_size=self.kernel_size, stride=self.stride)
        # (B, C, H, W)

        return x


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResBlockSA(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockSA, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x, d_map):
        res = self.body(x).mul(self.res_scale)
        res = res * d_map
        res += x

        return res

class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv = default_conv,
         bias=True, bn=True, act=nn.LeakyReLU(0.2, True), res_scale=1):

        super(Bottleneck, self).__init__()
        m_res = []
        # conv1
        m_res.append(conv(in_channels, out_channels//4, 1, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels//4))
        m_res.append(act)
        # conv2
        m_res.append(conv(out_channels//4, out_channels//4, 3, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels//4))
        m_res.append(act)
        # conv3
        m_res.append(conv(out_channels//4, out_channels, 1, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels))

        self.res = nn.Sequential(*m_res)
        self.res_scale = res_scale

        m_shortcut = [conv(in_channels, out_channels, 1, bias=bias)]
        if bn:
            m_shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*m_shortcut)

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        res += self.shortcut(x)

        return res

class MyResBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MyResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act:
            m.append(act)
        m.append(conv(out_channels, out_channels, kernel_size, bias=bias))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.identity = conv(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        res = self.body(x)
        x = self.identity(x)
        x += res

        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges