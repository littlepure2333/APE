import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data
from augments import cutblur

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=False):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.cutblur = args.cutblur

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def __getitem__(self, index: int):
        lr, hr, filename = super(Benchmark, self).__getitem__(index)
        # if self.assistant:
        #     return lr, hr, filename, index
        # else:
        
        if self.cutblur is None:
            return lr, hr, filename
        elif self.cutblur > 0:
            hr, lr = cutblur(hr, lr, alpha = self.cutblur, train=self.train)
        elif self.cutblur == 0:
            hr, lr = cutblur(hr, lr, alpha = self.cutblur, train=False)

        return lr, hr, filename