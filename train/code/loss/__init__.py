import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        ##default  1*L1
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
           
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.ker_loss = []
        self.ker_loss.append({
            'type': "Degrade_L1",
            'weight': float(10),
            'function': nn.L1Loss()}
        )

        self.sr_log = torch.Tensor()
        self.ker_log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, ker, ker_y):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.sr_log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.sr_log[-1, i] += self.loss[i - 1]['function'].loss

        for i, l in enumerate(self.ker_loss):
            if l['function'] is not None:
                loss = l['function'](ker, ker_y)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.ker_log[-1, i] += effective_loss.item()

        loss_sum = sum(losses)
        '''
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        '''
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.sr_log = torch.cat((self.sr_log, torch.zeros(1, len(self.loss))))
        self.ker_log = torch.cat((self.ker_log, torch.zeros(1, len(self.ker_loss))))

    def end_log(self, n_batches):
        self.sr_log[-1].div_(n_batches)
        self.ker_log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.sr_log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        for l, c in zip(self.ker_loss, self.ker_log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = 'SR {} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.sr_log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/SR_loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

        for i, l in enumerate(self.ker_loss):
            label = 'Ker {} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.ker_log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/Ker_loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.sr_log, os.path.join(apath, 'loss_sr_log.pt'))
        torch.save(self.ker_log, os.path.join(apath, 'loss_ker_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.sr_log = torch.load(os.path.join(apath, 'loss_sr_log.pt'))
        self.ker_log = torch.load(os.path.join(apath, 'loss_ker_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.sr_log)): l.scheduler.step()

