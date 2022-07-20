import os
from data import srdata
from augments import cutblur

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        # self.assistant = args.assistant
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.trian = train
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.cutblur = args.cutblur
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
    
    def __getitem__(self, index: int):
        lr, hr, filename = super(DIV2K, self).__getitem__(index)
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

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

