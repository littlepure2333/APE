import os
from data import srdata
import glob

class TEST8K(srdata.SRData):
    def __init__(self, args, name='TEST8K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.train = train
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(TEST8K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
    
    def __getitem__(self, index: int):
        lr, hr, filename = super(TEST8K, self).__getitem__(index)
        return lr, hr, filename

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

