import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import tqdm, gc
from colorama import init, Fore
init()  # Init Colorama
color_code = Fore.BLUE  

class Mono_BaselineTrain(nn.Module):
    def __init__(self, params, model_func, num_class):
        super(Mono_BaselineTrain, self).__init__()
        self.params = params
        self.num_class = num_class
        self.feature = model_func()  
        if params.method in ['protonet']:
            self.feat_dim = self.feature.feat_dim[0]  
            self.avgpool = nn.AdaptiveAvgPool2d(1)    
        if params.method in ['protonet']:
            self.classifier = nn.Linear(self.feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        self.loss_fn = nn.CrossEntropyLoss()

    def feature_forward(self, x):                             
        feat = self.feature.forward(x)                        
        if self.params.method in ['protonet']:    
            feat = self.avgpool(feat).view(feat.size(0), -1)  
        return feat

    ########################################################################################################
    # 主阶段1：预训练——训练
    ########################################################################################################
    def forward(self, x):
        x = Variable(x.cuda())                  
        feat = self.feature_forward(x)         
        logit = self.classifier.forward(feat)  
        return logit
    
    def forward_loss(self, x, y):
        logit = self.forward(x)                 
        y = Variable(y.cuda())                  
        return self.loss_fn(logit, y), logit    

    def train_loop(self, model, epoch, train_loader, optimizer):
        print_freq = 1
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = len(train_loader) * self.params.batch_size
        for i, (x, y) in enumerate(train_loader):               
            y = Variable(y.cuda())
            loss, logit = self.forward_loss(x, y)                
            pred = torch.argmax(logit, dim=1)                    
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            optimizer.zero_grad()
            if self.params.amp_opt_level != "O0":
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else: loss.backward()
            else: loss.backward()
            optimizer.step() 
            avg_loss = avg_loss + loss.item()
            gc.collect()
            if i % print_freq == 0: print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, self.params.epoch, i, len(train_loader), avg_loss / float(i + 1)))
        return avg_loss / iter_num, float(total_correct) / total * 100
    

    ########################################################################################################
    # 主阶段2：预训练——元验证
    ########################################################################################################
    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        score = -torch.pow(x - y, 2).sum(2)
        return score

    def feature_val_forward(self, x):  
        feat = self.feature.forward(x) 
        if self.params.method in ['protonet']: feat = self.avgpool(feat).view(feat.size(0), -1) 
        return feat

    def forward_meta_val(self, x):          
        x = Variable(x.cuda())             
        x = x.contiguous().view(self.params.val_n_way * (self.params.n_shot + self.params.n_query), *x.size()[2:]) 
        feat = self.feature_val_forward(x)   
        feat = feat.view(self.params.val_n_way, self.params.n_shot + self.params.n_query, -1)   
        support = feat[:, :self.params.n_shot] 
        query = feat[:, self.params.n_shot:]   
        proto = support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        query = query.contiguous().view(self.params.val_n_way * self.params.n_query, -1)        
        if self.params.method in ['protonet']: logit = self.euclidean_dist(query, proto)
        return logit

    def forward_meta_val_loss(self, x):             
        y_query = torch.from_numpy(np.repeat(range(self.params.val_n_way), self.params.n_query))
        y_query = Variable(y_query.cuda())                                      
        y_label = np.repeat(range(self.params.val_n_way), self.params.n_query)  
        logit = self.forward_meta_val(x)              
        topk_labels = torch.argmax(logit, dim=1)  
        topk_ind = topk_labels.cpu().numpy()      
        top1_correct = np.sum(topk_ind == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(logit, y_query), logit
    
    def meta_test_loop(self, test_loader):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):  
                correct_this, count_this, loss, _ = self.forward_meta_val_loss(x)
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean, acc_std = np.mean(acc_all), np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return avg_loss/iter_num, acc_mean
 

##############################
# 元训练类：Mono_MetaTemplate
##############################
class Mono_MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(Mono_MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query  
        self.change_way = change_way
        self.params = params
        self.feature = model_func()
        if params.method in ['protonet']:  self.avgpool = nn.AdaptiveAvgPool2d(1)

    ############################################################################################################
    # 基础函数-特征提取
    ############################################################################################################
    def feature_forward(self, x):                         
        feat = self.feature.forward(x)                    
        feat = self.avgpool(feat).view(feat.size(0), -1) 
        return feat
    def parse_feature(self, x, is_feature):            
        x = Variable(x.cuda())
        if is_feature: feat = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            feat = self.feature_forward(x)                                   
            feat = feat.view(self.n_way, self.n_support + self.n_query, -1) 
        feat_support = feat[:, :self.n_support]           
        feat_query = feat[:, self.n_support:]       
        return feat_support, feat_query
    def euclidean_dist(self, x, y): 
        n = x.size(0)  # x(batch)
        m = y.size(0)  # y(batch)
        d = x.size(1)  # x(channel)
        assert d == y.size(1) # y(channel)
        x = x.unsqueeze(1).expand(n, m, d) 
        y = y.unsqueeze(0).expand(n, m, d)  
        score = -torch.pow(x-y, 2).sum(2)
        return score
    
    ############################################################################################################
    # 主阶段1：元训练
    ############################################################################################################
    @abstractmethod
    def set_forward(self, x, is_feature): pass
    
    @abstractmethod
    def set_forward_loss(self, x): pass

    def train_loop(self, model, epoch, train_loader, optimizer):
        print_freq = 1
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):                
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way = x.size(0)
            correct_this, count_this, loss,  _ = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()  
            acc_all.append(correct_this / count_this * 100)
            optimizer.zero_grad()
            if self.params.amp_opt_level != "O0":
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward()
                else: loss.backward()
            else: loss.backward()
            optimizer.step() 
            if i % print_freq == 0: print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, self.params.epoch, i, len(train_loader), avg_loss / float(i + 1)))
            gc.collect()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        return avg_loss / iter_num, acc_mean

    ############################################################################################################
    # 主阶段2：元验证
    ############################################################################################################
    @abstractmethod
    def set_val(self, x, is_feature): pass
    
    @abstractmethod
    def set_val_loss(self, x): pass

    def test_loop(self, test_loader, record=None):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            tqdm_gen = tqdm.tqdm(test_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))
            for i, (x, _) in enumerate(tqdm_gen, 1):
                self.n_query = x.size(1) - self.n_support
                if self.change_way: self.n_way = x.size(0)
                correct_this, count_this, loss, _ = self.set_val_loss(x)
                avg_loss = avg_loss + loss.item()
                acc_all.append(correct_this / count_this * 100)
                gc.collect()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        torch.cuda.empty_cache()
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return avg_loss / iter_num, acc_mean

    ############################################################################################################
    # 主阶段3：元测试
    ############################################################################################################
    @abstractmethod
    def set_test_forward(self, x, is_feature): pass

################################################################################################################
# end
################################################################################################################