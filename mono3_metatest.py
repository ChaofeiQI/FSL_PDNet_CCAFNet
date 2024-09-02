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
import time, os, argparse, tqdm
from colorama import init, Fore
init()  # Init Colorama
color_code = Fore.BLUE    
try: from apex import amp #使用Apex加速运算
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass
from data.datamgr import SetDataManager
from sklearn.linear_model import LogisticRegression
from utils import *
from methods.mono2_spatial_metric_dis import Mono_Spatial_Metric_Dis

if __name__ == '__main__':
    option_dataset=['miniimagenet_large','cub_cropped','cifar_fs','aircraft_fs']
    option_models=['PDNet16','PDNet19','CCAFNet']
    option_methods=['protonet']

    ################################
    # 0 超参数
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
    parser.add_argument('--dataset', default='miniimagenet_middle', choices=option_dataset)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--pre_batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--pre_num_episode', default=1000, type=int, help='number of episodes in meta validation')
    parser.add_argument('--pre_optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument('--pre_lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--pre_epoch', default=200, type=int, help='stopping epoch')
    parser.add_argument('--meta_lr', type=float, default=0.0005, help='initial learning rate of the metatrain')
    parser.add_argument('--meta_epoch', default=100, type=int, help='Stopping epoch')
    parser.add_argument('--meta_train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--meta_val_n_episode', default=300, type=int, help='number of episodes in meta val')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument('--model', default='ResNet12', choices=option_models)
    parser.add_argument('--method', default='stl_deepbdc', choices=option_methods)
    parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')
    parser.add_argument('--test_n_episode', default=1000, type=int, help='number of episodes in test')
    parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
    parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')

    ################################
    # DistributedDataParallel
    ################################
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    params = parser.parse_args()
    novel_file = 'test'

    ################################
    # 1 加载数据集
    ################################
    novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, dataset=params.dataset,  **novel_few_shot_params)
    novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

    ################################
    # 2 初始化网络模型
    ################################
    model = Mono_Spatial_Metric_Dis(params, model_dict[params.model], **novel_few_shot_params)
    num_gpu = set_gpu(params)
    model = model.cuda()
    model.eval()
    print(params.model_path)
    model_file = os.path.join(params.model_path)
    model = load_model(model, model_file)
    print(params)
    params.checkpoint_dir = './checkpoints/%s/Mono_%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '/metaphase_%s_%s_%s_%s_%s_%dway_%dshot' % (params.pre_optimizer, params.pre_batch_size, float(params.pre_lr), params.pre_num_episode, params.pre_epoch, params.test_n_way, params.n_shot)
    params.checkpoint_dir += '/metatest_%s_%s_%s_train_%s_val_%s_test_%s' %(params.optimizer, float(params.meta_lr), params.meta_epoch, params.meta_train_n_episode, params.meta_val_n_episode, params.test_n_episode)
    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)

    ################################
    # 3 网络模型推理
    ################################
    path_file = os.path.join(params.checkpoint_dir, 'test_log.txt')
    log_file = open(path_file, 'w')
    log_file.write(f'params: {params}\n')
    log_file.close()

    iter_num = params.test_n_episode
    acc_all_task, std_all_task, latency_task = [], [], []
    for _ in range(params.test_task_nums):
        log_file = open(path_file, 'a')
        acc_all, latency_all = [],[]
        test_start_time = time.time()
        tqdm_gen = tqdm.tqdm(novel_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))
        for _, (x, _) in enumerate(tqdm_gen):
            with torch.no_grad():
                model.n_query = params.n_query
                s_time = time.time()
                scores = model.set_test_forward(x, False) 
                f_time = time.time()
                period = f_time - s_time     
                scores = torch.softmax(scores, dim=1)
            if params.method in ['meta_deepbdc', 'protonet', 'BinoHeD', 'BinoBCD', 'BinoCaD', 'BinoSML']:
                pred = scores.data.cpu().numpy().argmax(axis=1)
            else: pred = scores
            y = np.repeat(range(params.test_n_way), params.n_query)
            acc = np.mean(pred == y) * 100
            acc_all.append(acc)
            latency_all.append(period*1000)
            tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(Fore.RED +'%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
        log_file.write(f'iter_num: {iter_num}, test acc: {acc_mean}+-{1.96 * acc_std / np.sqrt(iter_num)}, \
                time-consuming: {(time.time()-test_start_time)/60}\n')
        acc_all_task.append(acc_all)
        std_all_task.append(1.96 * acc_std / np.sqrt(iter_num))
        latency_task.append(latency_all)
        log_file.close()
    log_file = open(path_file, 'a')
    acc_all_task_mean, acc_all_task_std, latency_task_mean = np.mean(acc_all_task), np.mean(std_all_task), np.mean(latency_task)
    print(Fore.RED +'%d test mean acc = %4.2f%% std = %4.2f%% latency=%4d' % (params.test_task_nums, acc_all_task_mean, acc_all_task_std, latency_task_mean))
    log_file.write(f'All tasks:{params.test_task_nums} , test mean acc:{acc_all_task_mean}, std:{acc_all_task_std}, latency:{latency_task_mean}')
    log_file.close()