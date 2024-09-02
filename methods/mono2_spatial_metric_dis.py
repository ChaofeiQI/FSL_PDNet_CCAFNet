import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from .mono1_template import Mono_MetaTemplate

class Mono_Spatial_Metric_Dis(Mono_MetaTemplate):
    def __init__(self, params, model_func_l, n_way, n_support):
        super(Mono_Spatial_Metric_Dis, self).__init__(params, model_func_l, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss() 

    ############################
    # 主阶段1.训练阶段（实例化）
    ############################
    def set_forward(self, x, is_feature=False): 
        z_support, z_query = self.parse_feature(x, is_feature)                        
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1) 
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)            
        metric_method = self.params.method 
        if metric_method == 'protonet': logit = self.euclidean_dist(z_query, z_proto)
        return logit

    def set_forward_loss(self, x):                                              
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)) 
        y_query = Variable(y_query.cuda())                                     
        y_label = np.repeat(range(self.n_way), self.n_query)                  
        logit = self.set_forward(x)                                           
        topk_labels = torch.argmax(logit, dim=1)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(logit, y_query), logit


    ############################
    # 主阶段2.验证阶段（实例化）
    ############################
    def set_val_forword(self, x, is_feature=False): 
        z_support, z_query = self.parse_feature(x, is_feature)                        
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1) 
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)            

        metric_method = self.params.method
        if metric_method == 'protonet': logit = self.euclidean_dist(z_query, z_proto)

    def set_val_loss(self, x):                                                   
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))  
        y_query = Variable(y_query.cuda())                                      
        y_label = np.repeat(range(self.n_way), self.n_query)                     
        logit = self.set_val_forword(x)                                         
        topk_labels =  torch.argmax(logit, dim=1)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(logit, y_query), logit


    ###########################
    # 主阶段3.测试阶段（实例化）
    ###########################
    def set_test_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)                        
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1) 
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)            

        metric_method = self.params.method 
        if metric_method == 'protonet': logit = self.euclidean_dist(z_query, z_proto)
        return logit