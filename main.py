from numpy import index_exp
import torch

import utility
import data
import model
import loss
from option import args

if args.APE:
    from trainer_ape import Trainer
else:
    from trainer import Trainer
import os
# if torch.cuda.is_available():
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            if args.switchable:
                loader = []
                for part in args.data_part_list:
                    args.file_suffix = part
                    loader.append(data.Data(args))
            else:
                loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            checkpoint.done()

if __name__ == '__main__':
    main()
