import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import pickle
import cv2

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.times = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        ''' accumulate (toc-tic) and hold times'''
        self.acc += self.toc()
        self.times += 1

    def release(self, avg=False, reset=True):
        ''' return all accumulated (toc-tic) in sum/avg mode, then reset'''
        ret = self.acc / self.count() if avg else self.acc
        if reset: self.reset()

        return ret

    def count(self):
        return self.times

    def reset(self):
        self.acc = 0
        self.times = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def save_exit_list(self, exit_list):
        with open(self.get_path('exit_list.pt'), 'wb') as _f:
            pickle.dump(exit_list, _f)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=True, print_time=True):
        if print_time:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            log = '[' + current_time + '] ' + log
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    def save_results_dynamic(self, dataset, filename, save_dict, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(self.args.data_test[0]),
                '{}_x{}_'.format(filename, scale)
            )

            # postfix = ('SR', 'LR', 'HR')
            # for v, p in zip(save_list, postfix):
            for key, value in save_dict.items():
                normalized = value[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, key), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    '''
    automatically recognize dims
    [C,H,W] -> 1
    [B,C,H,W] -> [B]
    '''
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean((-1,-2,-3)).squeeze()
    # mse = diff.pow(2).mean()
    # if mse <= 0:
    #     print(mse)
    #     print(sr)
    #     print(hr)
    #     raise ValueError
    
    # psnr = -10 * math.log10(mse)
    psnr = -10 * torch.log10(mse)

    return psnr

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    if len(milestones) == 1:
        kwargs_scheduler = {'step_size': milestones[0], 'gamma': args.gamma}
        scheduler_class = lrs.StepLR
    else:
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def crop(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list[0].device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img

def seamless_combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list[0].device)
    border = [1,1,1,1]
    for i in range(num_h):
        if i == 0:  # top side
            border[1] = 0
            border[3] = 1
        elif i < num_h-1: # middle
            border[1] = 1
            border[3] = 1
        else: # bottom side
            border[1] = 1
            border[3] = 0
        for j in range(num_w):
            if j == 0:  # left side
                border[0] = 0
                border[2] = 1
            elif j < num_w-1: # middle
                border[0] = 1
                border[2] = 1
            else: # right side
                border[0] = 1
                border[2] = 0
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += fade_border(sr_list[index], patch_size-step, border)
            index+=1

    return sr_img

def fade_border(img, border_size, border=[1,1,1,1]):
    ''' 
    gradually fade the border while maintain the center,
    "border" indicates fading at [left, top, right, bottom]
    '''
    if border_size > 0: # overlap
        if border[0] != 0: # left border
            img[:, :, :border_size] *= torch.linspace(0, 1, border_size).unsqueeze(0).unsqueeze(0)
        if border[1] != 0: # top border
            img[:, :border_size, :] *= torch.linspace(0, 1, border_size).unsqueeze(0).transpose(1,0).unsqueeze(0)
        if border[2] != 0: # right border
            img[:, :, -border_size:] *= torch.linspace(1, 0, border_size).unsqueeze(0).unsqueeze(0)
        if border[3] != 0: # bottom border
            img[:, -border_size:,:] *= torch.linspace(1, 0, border_size).unsqueeze(0).transpose(1,0).unsqueeze(0)
        return img
    else: # non-overlap
        return img


def crop_parallel(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine_parallel(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list.device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img


def add_mask(sr_img, scale, num_h, num_w, h, w, patch_size, step, exit_index, show_number=True):
    # white and 7-rainbow
    # color_list = [(255,255,255),(255,0,0),(255,165,0),(255,255,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]
    color_list = [(255,255,255),(255,225,0),(255,165,0),(240,0,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]

    idx = 0
    sr_img = sr_img.squeeze().permute(1,2,0).numpy() # (H,W,C)
    mask = np.zeros((sr_img.shape), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            bbox = [j * step + 2*scale, 
                     i * step + 2*scale,
                     j * step + patch_size - (2*scale+1),
                     i * step + patch_size - (2*scale+1)]  # xl,yl,xr,yr

            color = color_list[int(exit_index[idx])]
            cv2.rectangle(mask, (bbox[0]+1, bbox[1]+1), (bbox[2]-1, bbox[3]-1), color=color, thickness=-1)
            cv2.putText(mask, '{}'.format(int(exit_index[idx]+1)), 
                        (bbox[0]+4*scale, bbox[3]-4*scale), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            idx += 1

    # add_mask
    alpha = 0.7
    beta = 1 - alpha
    gamma = 0
    sr_mask = cv2.addWeighted(sr_img, alpha, mask, beta, gamma)
    sr_mask = torch.from_numpy(sr_mask).permute(2,0,1).unsqueeze(0)

    return sr_mask



def calc_avg_exit(exit_list):
    if exit_list.ndim == 2:
        exit_list = exit_list.sum(0)
    num = exit_list.sum()
    index = torch.arange(0,len(exit_list),1).float()
    avg = (index*exit_list).sum() / num

    return avg

def calc_flops(exit_list, model_name, scale, exit_interval):
    
    if exit_list.ndim == 2:
        exit_list = exit_list.sum(0)
    
    if model_name.find("EDSR") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([9.60,12.32,15.04,17.76,20.47,23.19,25.91,28.63,31.35,34.07,36.79,39.51,42.23,44.95,47.67,50.38,53.10,55.82,58.54,61.26,63.98,66.70,69.42,72.14,74.86,77.57,80.29,83.01,85.73,88.45,91.17,93.89])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([16.48,19.19,21.91,24.63,27.35,30.07,32.79,35.51,38.23,40.95,43.67,46.39,49.10,51.82,54.54,57.26,59.98,62.70,65.42,68.14,70.86,73.58,76.30,79.01,81.73,84.45,87.17,89.89,92.61,95.33,98.05,100.77])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([31.54,34.26,36.98,39.70,42.42,45.14,47.86,50.58,53.29,56.01,58.73,61.45,64.17,66.89,69.61,72.33,75.05,77.77,80.49,83.20,85.92,88.64,91.36,94.08,96.80,99.52,102.24,104.96,107.68,110.40,113.11,115.83])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("RCAN") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([3.94,7.43,10.92,14.41,17.90,21.39,24.88,28.38,31.87,35.36])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([4.38,7.87,11.36,14.86,18.35,21.84,25.33,28.82,32.31,35.80])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([5.35,8.84,12.33,15.82,19.31,22.80,26.29,29.79,33.28,36.77])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("VDSR") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([0.37,0.72,1.06,1.40,1.74,2.08,2.42,2.76,3.10,3.44,3.78,4.12,4.47,4.81,5.15,5.49,5.83,6.17])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([0.84,1.61,2.38,3.14,3.91,4.68,5.44,6.21,6.98,7.75,8.51,9.28,10.05,10.81,11.58,12.35,13.12,13.88])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([1.50,2.86,4.22,5.59,6.95,8.32,9.68,11.04,12.41,13.77,15.13,16.50,17.86,19.22,20.59,21.95,23.32,24.68])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("ECBSR") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([0.11,0.19,0.28,0.36,0.45,0.53,0.62,0.70,0.79,0.87,0.96,1.04,1.13,1.21,1.30,1.38])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([0.13,0.21,0.30,0.38,0.47,0.55,0.64,0.72,0.81,0.89,0.98,1.06,1.15,1.23,1.32,1.40])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([0.15,0.24,0.32,0.41,0.49,0.58,0.66,0.75,0.83,0.92,1.00,1.09,1.17,1.26,1.34,1.43])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("RRDB") >= 0:
        if scale == 4:
            flops_list = torch.Tensor([4.88,6.54,8.20,9.85,11.51,13.17,14.83,16.49,18.14,19.80,21.46,23.12,24.77,26.43,28.09,29.75,31.41,33.06,34.72,36.38])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("SWINIR") >= 0:
        if scale == 4:
            flops_list = torch.Tensor([6.52, 11.09, 15.67, 20.25, 24.83, 29.41])
            flops_list = flops_list[exit_interval-1::exit_interval]

    num = exit_list.sum()
    flops = (flops_list*exit_list).sum() / num
    percent = flops / flops_list[-1] * 100.0

    return flops, percent


if __name__ == "__main__":
    import time
    tic = time.time()
    
    img1 = torch.ones(32,3,96,96)*101
    img2 = torch.ones(32,3,96,96)*100
    # print(img1)
    # print(img2)

    psnr = calc_psnr(img1, img2, 2, 255)
    print(psnr)

    toc = time.time()
    print(toc-tic)