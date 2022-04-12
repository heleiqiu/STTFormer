#!/usr/bin/env python
import argparse
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm


def init_seed(seed):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Tuples Transformer')
    parser.add_argument('--work_dir', default='./work_dir/ntu60/temp', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/ntu60_xsub_joint.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--run_mode', default='train', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--save_epoch', type=int, default=80, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for model testing')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--cuda_visible_device', default='2,3', help='')
    parser.add_argument('--device', type=int, default=[0,1], nargs='+', help='the indexes of GPUs for training or testing')

    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)

    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Processor():
    """ Processor for Skeleton-based Action Recgnition """

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.load_model()
        self.load_data() 

        if arg.run_mode == 'train':
            result_visual = os.path.join(arg.work_dir, 'runs')
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(result_visual):
                    print('log_dir: ', result_visual, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(result_visual)
                        print('Dir removed: ', result_visual)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', result_visual)
                self.train_writer = SummaryWriter(os.path.join(result_visual, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(result_visual, 'val'), 'val')
                
                self.load_optimizer()
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(result_visual, 'test'), 'test')


        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.run_mode == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log('Data load finished')

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        # print(Model)
        self.model = Model(**self.arg.model_args)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        self.print_log('Model load finished: ' + self.arg.model)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.print_log('Optimizer load finished: ' + self.arg.optimizer)

    def adjust_learning_rate(self, epoch):
        self.print_log('adjust learning rate, using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * ( self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        self.adjust_learning_rate(epoch)
 
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

        for batch, (data, label, sample) in enumerate(tqdm(self.data_loader['train'], desc="Training", ncols=100)):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
            losses.update(loss.item())

            self.train_writer.add_scalar('top1', top1.avg, self.global_step)
            self.train_writer.add_scalar('loss', losses.avg, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

        self.print_log('training: epoch: {}, loss: {:.4f}, top1: {:.2f}%, lr: {:.6f}'.format(
            epoch + 1, losses.avg, top1.avg, self.lr))

        if  epoch + 1 == self.arg.num_epoch:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.work_dir + '/' + self.arg.work_dir.split('/')[-1] + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        for ln in loader_name:
            score_frag = []
            label_list = []
            pred_list = []
            for batch, (data, label, sampie) in enumerate(tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    
                    score_frag.append(output.data.cpu().numpy())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))
                losses.update(loss.item())

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(sampie[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))

            if self.arg.run_mode == 'train':
                self.val_writer.add_scalar('loss', top1.avg, self.global_step)
                self.val_writer.add_scalar('acc', losses.avg, self.global_step)

            self.best_acc = top1.avg if top1.avg > self.best_acc else self.best_acc
        
            self.print_log('evaluating: loss: {:.4f}, top1: {:.2f}%, best_acc: {:.2f}%'.format(losses.avg, top1.avg, self.best_acc))

            if save_score:
                with open('{}/score.pkl'.format(self.arg.work_dir), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):

        if self.arg.run_mode == 'train':

            for argument, value in sorted(vars(self.arg).items()):
                self.print_log('{}: {}'.format(argument, value))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.print_log('###***************start training***************###')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)

                if ((epoch + 1) % self.arg.eval_interval == 0):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')

        elif self.arg.run_mode == 'test':
            if not self.arg.test_feeder_args['debug']:
                weights_path = self.arg.work_dir + '.pt'
                wf = self.arg.work_dir + '/wrong.txt'
                rf = self.arg.work_dir + '/right.txt'
            else:
                wf = rf = None

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.cuda_visible_device
    init_seed(1)
    processor = Processor(arg)
    processor.start()
