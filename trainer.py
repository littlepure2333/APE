import os
import math
from decimal import Decimal
import numpy as np

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from model.patchnet import PatchNet
import lpips
from pytorch_msssim import ssim
from data.utils_image import calculate_ssim
from torch.nn import functional as F

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ssim = args.ssim
        # self.lpips_alex = args.lpips_alex
        # self.lpips_vgg = args.lpips_vgg
        # self.loss_fn_alex = lpips.LPIPS(net='alex')
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        if self.args.patchnet:
            self.patchnet = PatchNet(args).to(self.device)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            if self.args.patchnet:
                trainability = self.patchnet(sr)
                trainability = torch.squeeze(trainability)
                loss = trainability * loss
                loss = torch.sum(loss)
                self.loss.log[-1, 0] += loss.item()
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        torch.cuda.empty_cache()

    def warm_up(self):
        self.ckp.write_log("warming up...\n")
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for i, (lr, hr, filename) in enumerate(d):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    if i > 2:
                        break
        # torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.ckp.write_log("warm up ended\n")

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        # self.warm_up()

        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        timer_pre = utility.timer()
        timer_model = utility.timer()
        timer_post = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                ssim_total = 0
                # lpips_vgg_total = 0
                # lpips_alex_total = 0

                pbar = tqdm(d)
                for lr, hr, filename in pbar:
                    timer_pre.tic()
                    lr, hr = self.prepare(lr, hr)
                    timer_pre.hold()
                    timer_model.tic()
                    if self.args.model.find("SWINIR") >= 0:
                        window_size = 8
                        scale = self.args.scale[0]
                        mod_pad_h, mod_pad_w = 0, 0
                        _, _, h, w = lr.size()
                        if h % window_size != 0:
                            mod_pad_h = window_size - h % window_size
                        if w % window_size != 0:
                            mod_pad_w = window_size - w % window_size
                        lr = F.pad(lr, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                        sr = self.model(lr, idx_scale)
                        _, _, h, w = sr.size()
                        sr = sr[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
                    else:
                        sr = self.model(lr, idx_scale)
                    torch.cuda.synchronize()
                    timer_model.hold()
                    timer_post.tic()
                    sr = utility.quantize(sr, self.args.rgb_range)
                    b,c,h,w = sr.size()

                    save_list = [sr]
                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    # pbar.set_description("{} shape:{}\tPSNR:{:.4f}".format(filename, sr.shape, item_psnr))
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr.cpu()

                    if self.ssim:
                        ssim_item = ssim(sr, hr, data_range=255, size_average=False).item()
                        ssim_total += ssim_item
                        self.ckp.write_log("{}\tPSNR:{:.3f}\tSSIM:{:.4f}".format(filename, item_psnr, ssim_item))
                    else:
                        self.ckp.write_log("{}\tPSNR:{:.3f}".format(filename, item_psnr))

                    # if self.lpips_vgg:
                    #     #print('sr',torch.reshape(torch.Tensor(sr),(1,c,h,w)).size())
                    #     #print('hr',torch.reshape(torch.Tensor(hr),(1,c,h,w)).size())
                    #     lpips_vgg = self.loss_fn_vgg(torch.reshape(torch.Tensor(sr),(b,c,h,w)),torch.reshape(torch.Tensor(hr),(b,c,h,w)))
                    #     lpips_vgg_total += lpips_vgg
                    # if self.lpips_alex:
                    #     lpips_alex = self.loss_fn_alex(torch.reshape(torch.Tensor(sr),(b,c,h,w)),torch.reshape(torch.Tensor(hr),(b,c,h,w)))
                    #     lpips_alex_total += lpips_alex
                    #     #print('alex',lpips_alex_total[0][0][0][0]/len(d),type(lpips_alex_total))
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    # torch.cuda.empty_cache()
                    timer_post.hold()

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                best = self.ckp.log.max(0)
                if self.ssim:
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f}\tSSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            ssim_total/len(d),
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                # if self.lpips_vgg:
                #      self.ckp.write_log(
                #         '[{} x{}]\tLPIPS-vgg: {:.4f}'.format(
                #             d.dataset.name,
                #             scale,
                #             lpips_vgg_total[0][0][0][0]/len(d)
                #         )
                #     )
                # if self.lpips_alex:
                #      self.ckp.write_log(
                #         '[{} x{}]\tLPIPS-alex: {:.4f}'.format(
                #             d.dataset.name,
                #             scale,
                #             lpips_alex_total[0][0][0][0]/len(d)
                #         )
                #     )

        # self.ckp.write_log('Model forward: {:.2f}s\n'.format(timer_model.release(reset=False)))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            '[whole]\t {:.4f}s\n'.format(timer_test.toc()), refresh=True
        )
        self.ckp.write_log(
            '[Total]\t pre:{:.4f}s\tmodel:{:.4f}s\tpost:{:.4f}s\n'.format(
                timer_pre.release(reset=False),
                timer_model.release(reset=False),
                timer_post.release(reset=False)
            ),refresh=True
        )
        self.ckp.write_log(
            '[Average]\t pre:{:.4f}s\tmodel:{:.4f}s\tpost:{:.4f}s\n'.format(
                timer_pre.release(avg=True),
                timer_model.release(avg=True),
                timer_post.release(avg=True)
            ),refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

