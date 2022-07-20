import os
import math
from decimal import Decimal
import numpy as np
import time
import utility
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.patchnet import PatchNet
# import lpips
from torch.nn import functional as F
from pytorch_msssim import ssim

from data.utils_image import calculate_ssim

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ssim = args.ssim

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.de_loss = nn.MSELoss()

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
            sr, ic = self.model(lr, 0)
            loss = 0

            pre_psnr = 0
            for sr_i, ic_i in zip(sr, ic):
                now_psnr = utility.calc_psnr(sr_i, hr, self.scale[0], self.args.rgb_range)
                ic_i_gt = 1 - torch.tanh(torch.tensor(now_psnr - pre_psnr))
                pre_psnr = now_psnr
                loss = loss + self.loss(sr_i, hr) + self.de_loss(ic_i.T.squeeze(), ic_i_gt)
            
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

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        if self.args.model.find("RCAN") >= 0:
            exit_len = int(self.args.n_resgroups/self.args.exit_interval)
        elif self.args.model.find("EDSR") >= 0:
            exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        elif self.args.model.find("VDSR") >= 0:
            exit_len = int((self.args.n_resblocks-2)/self.args.exit_interval)
        elif self.args.model.find("ECBSR") >= 0:
            exit_len = int((self.args.m_ecbsr)/self.args.exit_interval)
        elif self.args.model.find("FSRCNN") >= 0:
            exit_len = int(4/self.args.exit_interval)
        elif self.args.model.find("RRDB") >= 0:
            exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        elif self.args.model.find("SWINIR") >= 0:
            exit_len = int(6/self.args.exit_interval)
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), exit_len)
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                ssim_total = 0
                save_dict = {}

                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
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
                        sr, ic = self.model(lr, idx_scale)
                        for i in range(len(sr)):
                            _, _, h, w = sr[i].size()
                            sr[i] = sr[i][:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
                    else:
                        sr, ic = self.model(lr, idx_scale)
                    # sr, ics = self.model(lr, idx_scale)
                    for i, sr_i in enumerate(sr):
                        sr_i = utility.quantize(sr_i, self.args.rgb_range)
                        save_dict['SR-{}'.format(i)] = sr_i
                        item_psnr = utility.calc_psnr(sr_i, hr, scale, self.args.rgb_range, dataset=d)
                        self.ckp.log[-1, idx_data, i] += item_psnr.cpu()
                    
                    if self.ssim:
                        sr_i_np = sr_i.squeeze().cpu().permute(1,2,0).numpy()
                        hr_np = hr.squeeze().cpu().permute(1,2,0).numpy()
                        ssim = calculate_ssim(sr_i_np, hr_np)
                        ssim_total += ssim

                    if self.args.save_gt:
                        save_dict['LR'] = lr
                        save_dict['HR'] = hr

                    if self.args.save_results:
                        self.ckp.save_results_dynamic(d, filename[0], save_dict, scale)
                    torch.cuda.empty_cache()

                self.ckp.log[-1, idx_data, :] /= len(d)

                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, -1],
                        best[0][idx_data, -1],
                        best[1][idx_data, -1] + 1
                    )
                )
                if self.ssim:
                    self.ckp.write_log(
                        '[{} x{}]\tSSIM: {:.4f}'.format(
                            d.dataset.name,
                            scale,
                            ssim_total/len(d)
                        )
                    )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def test_dynamic(self):
        torch.set_grad_enabled(False)

        if self.args.model.find("RCAN") >= 0:
            exit_len = int(self.args.n_resgroups/self.args.exit_interval)
        elif self.args.model.find("EDSR") >= 0:
            exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        elif self.args.model.find("VDSR") >= 0:
            exit_len = int((self.args.n_resblocks-2)/self.args.exit_interval)
        elif self.args.model.find("ECBSR") >= 0:
            exit_len = int((self.args.m_ecbsr)/self.args.exit_interval)
        elif self.args.model.find("FSRCNN") >= 0:
            exit_len = int(4/self.args.exit_interval)
        elif self.args.model.find("RRDB") >= 0:
            exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        elif self.args.model.find("SWINIR") >= 0:
            exit_len = int(6/self.args.exit_interval)
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
                save_dict = {}
                exit_list = torch.zeros((1,exit_len))
                AVG_exit = 0
                pass_time_list = []
                total_exit_index_list = torch.Tensor()

                for lr, hr, filename in d:
                    pass_start = time.time()
                    timer_pre.tic()
                    lr, hr = self.prepare(lr, hr) # (B,C,H,W)
                    lr_list, num_h, num_w, new_h, new_w = utility.crop_parallel(lr, self.patch_size//scale, self.step//scale)
                    timer_pre.hold()
                    sr_list = torch.Tensor()
                    exit_index_list = torch.Tensor()
                    avg_exit = 0

                    pbar = tqdm(range(len(lr_list)//self.args.n_parallel + 1), ncols=120)
                    for lr_patch_index in pbar:
                        timer_model.tic()
                        sr_patches, exit_index, ic = self.model(lr_list[lr_patch_index*self.args.n_parallel:(lr_patch_index+1)*self.args.n_parallel], idx_scale)
                        torch.cuda.synchronize()
                        timer_model.hold()
                        sr_list = torch.cat([sr_list, sr_patches.cpu()])
                        for index in exit_index:
                            exit_list[-1, int(index)] += 1
                        exit_index_list = torch.cat([exit_index_list,exit_index.cpu()])

                    total_exit_index_list = torch.cat([total_exit_index_list,exit_index_list])
                    timer_post.tic()
                    sr = utility.combine(sr_list, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step)
                    pass_end = time.time()
                    pass_time = pass_end-pass_start
                    pass_time_list.append(pass_time)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_dict['SR'] = sr
                    timer_post.hold()

                    if self.args.add_mask:
                        sr_mask = utility.add_mask(sr, scale, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step, exit_index_list)
                        save_dict['SR_MASK'] = sr_mask
                    hr = hr[:, :, 0:new_h*scale, 0:new_w*scale].cpu()
                    
                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr.cpu()

                    avg_exit = utility.calc_avg_exit(exit_list[-1])
                    avg_flops, avg_flops_percent = utility.calc_flops(exit_list[-1], self.args.model, scale, self.args.exit_interval)
                    # self.ckp.write_log("{}\tPSNR:{:.3f}\taverage exit:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.2f}%) pass time:{:.3f}s".format(filename, item_psnr, avg_exit, exit_len-1, avg_flops, avg_flops_percent, pass_time))

                    if self.ssim:
                        ssim_item = ssim(sr, hr, data_range=255, size_average=False).item()
                        ssim_total += ssim_item
                        self.ckp.write_log("{}\tPSNR:{:.3f}\tSSIM:{:.4f}\taverage exit:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.2f}%) pass time:{:.3f}s".format(filename, item_psnr, ssim_item, avg_exit, exit_len-1, avg_flops, avg_flops_percent, pass_time))
                    else:
                        self.ckp.write_log("{}\tPSNR:{:.3f}\taverage exit:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.2f}%) pass time:{:.3f}s".format(filename, item_psnr, avg_exit, exit_len-1, avg_flops, avg_flops_percent, pass_time))

                    if self.args.save_gt:
                        save_dict['LR'] = lr
                        save_dict['HR'] = hr

                    if self.args.save_results:
                        self.ckp.save_results_dynamic(d, filename[0], save_dict, scale)
                    # torch.cuda.empty_cache()
                    exit_list = torch.cat([exit_list,torch.zeros((1,exit_len))])

                self.ckp.log[-1, idx_data, :] /= len(d)

                AVG_exit = utility.calc_avg_exit(exit_list)
                AVG_flops, AVG_flops_percent = utility.calc_flops(exit_list, self.args.model, scale, self.args.exit_interval)
                AVG_pass_time = np.array(pass_time_list)[1:].mean()

                if self.ssim:
                    self.ckp.write_log(
                        '[{} x{}] PSNR: {:.3f}\tSSIM: {:.4f}\tThreshold: {}\tAverage exits:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.3f}%) avg pass time: {:.2f}s'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, -1],
                            ssim_total/len(d),
                            self.args.exit_threshold,
                            AVG_exit,
                            exit_len-1,
                            AVG_flops,
                            AVG_flops_percent,
                            AVG_pass_time
                        )
                    )
                else:
                    self.ckp.write_log(
                        '[{} x{}] PSNR: {:.3f}\tThreshold: {}\tAverage exits:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.3f}%) avg pass time: {:.2f}s'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, -1],
                            self.args.exit_threshold,
                            AVG_exit,
                            exit_len-1,
                            AVG_flops,
                            AVG_flops_percent,
                            AVG_pass_time
                        )
                    )

        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.save_exit_list(total_exit_index_list)

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

    def test_static(self):
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
                save_dict = {}

                for lr, hr, filename in d:
                    timer_pre.tic()
                    lr, hr = self.prepare(lr, hr) # (B,C,H,W)
                    lr_list, num_h, num_w, new_h, new_w = utility.crop_parallel(lr, self.patch_size//scale, self.step//scale)
                    timer_pre.hold()
                    sr_list = torch.Tensor()

                    pbar = tqdm(range(len(lr_list)//self.args.n_parallel + 1), ncols=120)
                    for lr_patch_index in pbar:
                        timer_model.tic()
                        sr_patches = self.model(lr_list[lr_patch_index*self.args.n_parallel:(lr_patch_index+1)*self.args.n_parallel], idx_scale)
                        torch.cuda.synchronize()
                        timer_model.hold()
                        sr_list = torch.cat([sr_list, sr_patches.cpu()])

                    timer_post.tic()
                    # sr = utility.combine(sr_list, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step)
                    sr = utility.seamless_combine(sr_list, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_dict['SR'] = sr
                    timer_post.hold()
                    hr = hr[:, :, 0:new_h*scale, 0:new_w*scale].cpu()

                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr.cpu()

                    if self.ssim:
                        sr_np = sr.squeeze().cpu().permute(1,2,0).numpy()
                        hr_np = hr.squeeze().cpu().permute(1,2,0).numpy()
                        ssim = calculate_ssim(sr_np, hr_np)
                        ssim_total += ssim
                        self.ckp.write_log("{}\tPSNR:{:.3f}\tSSIM:{:.4f}".format(filename, item_psnr, ssim))
                    else:
                        self.ckp.write_log("{}\tPSNR:{:3f}".format(filename, item_psnr))

                    if self.args.save_gt:
                        save_dict['LR'] = lr
                        save_dict['HR'] = hr

                    if self.args.save_results:
                        self.ckp.save_results_dynamic(d, filename[0], save_dict, scale)
                    torch.cuda.empty_cache()

                self.ckp.log[-1, idx_data, :] /= len(d)
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

        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
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


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            if self.args.model.find("APE") >= 0:
                self.test_dynamic()
            else:
                self.test_static()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

