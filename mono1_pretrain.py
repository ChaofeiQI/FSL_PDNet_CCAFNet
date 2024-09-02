# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time, os, gc, argparse
from colorama import init, Fore
init()  # Init Colorama
try: from apex import amp  #使用Apex加速运算
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass
from data.datamgr import SimpleDataManager, SetDataManager
from utils import *
from methods.mono1_template import Mono_BaselineTrain

def train(params, base_loader, val_loader, model, stop_epoch):
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    ################################
    # 3 初始化优化器
    ################################
    optimizer = torch.optim.SGD(model.parameters(), lr=params.pre_lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    if params.amp_opt_level != "O0" and amp is not None:  model, optimizer = amp.initialize(model, optimizer, opt_level=params.amp_opt_level)
    
    ################################
    # 4 初始化lr_scheduler
    ################################
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)

    ################################
    # 5 训练+测试
    ################################
    path_file = os.path.join(params.checkpoint_dir, 'train_log.txt')
    log_file = open(path_file, 'w')
    log_file.write(f'params: {params}\n')
    log_file.close()
    start_epoch=0
    if params.check_resume=='True':
        print('***********************************')
        print('加载断点中：')
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(params.checkpoint_dir,'checkpoint_epoch_'+ str(params.check_epoch) +'.pt'))
        print(f'Model loaded from epoch {start_epoch}')
        print('***********************************')
    else: print('从头开始训练！')

    for epoch in range(start_epoch, stop_epoch):
        log_file = open(path_file, 'a')
        start = time.time()
        # 1）训练阶段
        model.train()
        trainObj, top1 = model.train_loop(model, epoch, base_loader, optimizer)
        print(Fore.RED +"train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))    
        log_file.write(f'epoch: {epoch}, train loss: {trainObj}, train acc: {top1}\n')
        
        # 2）验证阶段
        model.eval()
        if params.val in ['meta']:
            if params.val == 'meta': valObj, acc = model.meta_test_loop(val_loader)
            print(Fore.RED +"val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
            log_file.write(f'epoch: {epoch}, val loss: {valObj}, val acc: {acc}\n'
            trlog['val_loss'].append(valObj)
            trlog['val_acc'].append(acc)
            if acc > trlog['max_acc']:
                print(Fore.BLUE +"best model! save...")
                trlog['max_acc'] = acc
                trlog['max_acc_epoch'] = epoch
                max_acc, max_epoch = acc, epoch
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                log_file.write(f'          best model! save...\n'
            log_file.write(f'          current best acc: {max_acc}, best acc epoch: {max_epoch}\n')
            print(Fore.RED +"model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        # 学习率更新
        if params.optimizer == 'SGD':
            lr_scheduler.step()
        elif params.optimizer == 'Adam':
            optimizer.step()
            lr_scheduler.step()
            
        print(Fore.RED +'Current Pre_lr:',  display_learning_rate(optimizer))
        print(Fore.RED +"This epoch use %.2f minutes" % ((time.time() - start) / 60))
        log_file.write(f'          Current Pre_lr: { display_learning_rate(optimizer)}. Epoch Time-consuming: {(time.time() - start) / 60}\n')
        torch.cuda.empty_cache()
        gc.collect()
        log_file.close()

    return model



if __name__ == '__main__':
    option_dataset=['miniimagenet_large','cub_cropped','cifar_fs','aircraft_fs']
    option_models=['PDNet16','PDNet19','CCAFNet']
    option_methods=['protonet']
    
    ################################
    # 0 超参数
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--pre_lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=200, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--dataset', default='miniimagenet_middle', choices=option_dataset)
    parser.add_argument('--val_dataset', default='miniimagenet_middle', choices=option_dataset)
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--val_data_path', type=str, help='validation dataset path')
    parser.add_argument('--model', default='ResNet12', choices=option_models
    parser.add_argument('--method', default='protonet', choices=option_methods)
    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episodes in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of  classes to classify in meta validation')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')
    parser.add_argument('--extra_dir', default='', help='recording additional information')
    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')    
    parser.add_argument('--check_epoch', default=0, type=int, help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument("--check_resume", type=str, default='False', choices=['False', 'True'])  # Input method
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')

    ################################
    # DistributedDataParallel
    ################################
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    params = parser.parse_args()
    num_gpu = set_gpu(params)
    set_seed(params.seed)

    ################################
    # 1 加载数据集
    ################################
    if params.dataset not in ['cross_domain_cub', 'cross_domain_aircraft', 'cross_domain_car']:
        params.val_dataset = params.dataset
        params.val_data_path = params.data_path
    if params.val == 'last': val_file = None
    elif params.val == 'meta': val_file = 'val'
    if  params.dataset == 'miniimagenet_large':
        base_file = 'train'
        params.num_classes = 64
        params.batch_size=params.batch_size*2
    elif params.dataset == 'cub_cropped':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 130
    elif params.dataset == 'cifar_fs':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
    elif params.dataset == 'aircraft_fs':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 50
    else: ValueError('dataset error')
    base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, dataset=params.dataset)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)
    if params.val == 'meta':
        test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(params.val_data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, dataset=params.val_dataset, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else: val_loader = None

    ################################
    # 2 初始化网络模型
    ################################
    model = Mono_BaselineTrain(params, model_dict[params.model], params.num_classes)  # 依据model和method，来初始化BaselineTrain。
    model = model.cuda()

    # model save path
    params.checkpoint_dir = './checkpoints/%s/Mono_%s_%s' % (params.dataset, params.model, params.method) # 依据dataset,model和method，来初始化BaselineTrain。
    params.checkpoint_dir += '/prephase_%s_%s_%s_%s_%s' % (params.optimizer, params.batch_size, float(params.pre_lr), params.val_n_episode, params.epoch)
    params.checkpoint_dir += params.extra_dir
    
    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    print(params)
    print(model)

    # 训练+测试主函数：
    model = train(params, base_loader, val_loader, model, params.epoch)
