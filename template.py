machine  = {"A3":"/data/shizun/dataset/", 
            "B3":"/data/shizun/"}

import datetime
import socket
hostname = socket.gethostname() # 获取当前主机名
dir_data = machine[hostname]
today = datetime.datetime.now().strftime('%Y%m%d')

def set_template(args):
    if args.template == 'EDSR':
        # model
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "4"
        args.dir_data = dir_data
        args.ext = "sep"
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.data_range = '1-800/801-810'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10

        # experiemnt
        args.reset = True
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks)

        # resume
        # args.reset = False
        # args.load = "xxx"
        # args.resume = -1

        # test
        # args.data_test = 'TEST8K'
        # args.data_test = 'DIV2K'
        # if args.data_test == 'DIV2K':
        #     args.data_range = '801-810'
        # elif args.data_test == 'TEST8K':
        #     args.data_range = '1-100'
        # args.test_only = True
        # args.ssim = True
        # args.save_gt = True
        # args.save_results = True
        # args.pre_train = "xxx/model/model_best.pt"
        # args.save = "{}_{}_x{}_{}".format(today, args.model, args.scale, args.data_test)

    elif args.template == 'EDSR_APE':
        # model
        args.model = 'EDSR_APE'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "4"
        args.dir_data = dir_data
        args.ext = "sep"
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.data_range = '1-800/801-810'
        args.patch_size = 192

        # device
        args.device = "0"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.APE = True
        args.exit_interval = 4

        # experiemnt
        args.reset = True
        args.pre_train = "xxx/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)
        
        # resume
        # args.reset = False
        # args.load = "xxx"
        # args.resume = -1

    elif args.template == 'EDSR_test':
        # model
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'
        args.data_test = 'DIV2K'
        if args.data_test == 'DIV2K':
            args.data_range = '801-810'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "0"
        args.n_GPUs = 1

        # pipeline
        args.APE = True
        args.test_only = True
        args.n_parallel = 500
        args.save_results = True
        args.save_gt = True
        args.ssim = True

        # experiment
        args.reset = True
        args.pre_train = "xxx/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_ps{}_st{}_n{}".format(today, args.model, args.scale, args.data_test, args.patch_size, args.step, args.n_resgroups)

    elif args.template == 'EDSR_APE_test':
        # model
        args.model = 'EDSR_APE'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"
        args.dir_data = dir_data
        args.data_test = 'DIV2K'
        # args.data_test = 'TEST8K'
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "0"
        args.n_GPUs = 1

        # pipeline
        args.APE = True
        args.test_only = True
        args.exit_interval = 4
        args.exit_threshold = 1
        args.n_parallel = 500
        args.save_results = True
        args.save_gt = True
        args.ssim = True
        # args.add_mask = True

        # experiment
        args.reset = True
        args.pre_train = "xxx_APE/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_ps{}_st{}_n{}_i{}_th{}".format(today, args.model, args.scale, args.data_test, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.exit_threshold)

    elif args.template == 'RCAN':
        # model
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

    elif args.template == 'VDSR':
        # model
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64

    elif args.template == 'ECBSR':
        # model
        args.model = 'ECBSR'
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'

    elif args.template == 'RRDB':
        # model
        args.model = 'RRDB'
        args.n_resblocks = 23

    elif args.template == 'SWINIR':
        # model
        args.model = 'SWINIR'
        args.n_resblocks = 6
