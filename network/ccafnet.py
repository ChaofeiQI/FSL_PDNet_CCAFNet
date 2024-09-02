# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Our Modified Version based on https://github.com/zyrant/CCAFNet/tree/main

import torch
import torch.nn as nn
import math, time
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from colorama import init, Fore
init()  # Init Colorama

###################
# Basic Functions
###################
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class Spatical_Fuse_attention3_GHOST(nn.Module):
    def __init__(self, in_channels,):
        super(Spatical_Fuse_attention3_GHOST, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.active = Hsigmoid()

    def forward(self, x, y):
        input_y = self.conv(y)
        input_y = self.active(input_y)
        # return input_y
        return x + x * input_y

class Channel_Fuse_attention2(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Channel_Fuse_attention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )
    def forward(self, x, y):
        b, c, _, _ = x.size()
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class Gatefusion3(nn.Module):
    def __init__(self, channel):
        super(Gatefusion3, self).__init__()
        self.channel = channel
        self.gate = nn.Sigmoid()

    def forward(self, x, y, fusion_up):
        first_fusion = torch.cat((x, y), dim=1)
        gate_fusion = self.gate(first_fusion)
        gate_fusion = torch.split(gate_fusion, self.channel, dim=1)
        fusion_x = gate_fusion[0] * x + x
        fusion_y = gate_fusion[1] * y + y
        fusion = fusion_x + fusion_y
        fusion = torch.abs((fusion - fusion_up)) * fusion + fusion
        return fusion

class Gatefusion3_fusionup(nn.Module):
    def __init__(self, channel):
        super(Gatefusion3_fusionup, self).__init__()
        self.channel = channel
        self.gate = nn.Sigmoid()

    def forward(self, x, y):
        first_fusion = torch.cat((x, y), dim=1)
        gate_fusion = self.gate(first_fusion)
        gate_fusion = torch.split(gate_fusion, self.channel, dim=1)
        fusion_x = gate_fusion[0] * x + x
        fusion_y = gate_fusion[1] * y + y
        fusion = fusion_x + fusion_y
        return fusion

###################
# Pretrained Models
###################
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

###################################################
# Layer1_r, Layer2_r, Layer3_r, Layer4_r, Layer5_r
###################################################
class vgg_rgb(nn.Module):
    # def __init__(self, pretrained=False):
    def __init__(self, pretrained=True):
        super(vgg_rgb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),    # first block 84*84*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second block 42*42*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third block 21*21*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth block 10*10*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth block 5*5*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict: model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        R1 = self.features[:6](rgb)
        R2 = self.features[6:13](R1)
        R3 = self.features[13:23](R2)
        R4 = self.features[23:33](R3)
        R5 = self.features[33:43](R4)
        
        # print('R1:',R1.shape)  # R1: torch.Size([1, 64, 84, 84])
        # print('R2:',R2.shape)  # R2: torch.Size([1, 128, 42, 42])
        # print('R3:',R3.shape)  # R3: torch.Size([1, 256, 21, 21])
        # print('R4:',R4.shape)  # R4: torch.Size([1, 512, 10, 10])
        # print('R5:',R5.shape)  # R5: torch.Size([1, 512, 5, 5])
        return R1, R2, R3, R4, R5


###################################################
# Layer1_d, Layer2_d, Layer3_d, Layer4_d, Layer5_d
###################################################
class vgg_depth(nn.Module):
    # def __init__(self, pretrained=False):
    def __init__(self, pretrained=True):
        super(vgg_depth, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),     # first block 84*84*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),   # second block 42*42*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third block 21*21*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth block 10*10*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 5*5*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict: model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, thermal):
        D1_d = self.features[:6](thermal)
        D2_d = self.features[6:13](D1_d)
        D3_d = self.features[13:23](D2_d)
        D4_d = self.features[23:33](D3_d)
        D5_d = self.features[33:43](D4_d)

        # print('D1_d:',D1_d.shape)  # D1_d: torch.Size([1, 64, 84, 84])
        # print('D2_d:',D2_d.shape)  # D2_d: torch.Size([1, 128, 42, 42])
        # print('D3_d:',D3_d.shape)  # D3_d: torch.Size([1, 256, 21, 21])
        # print('D4_d:',D4_d.shape)  # D4_d: torch.Size([1, 512, 10, 10])
        # print('D5_d:',D5_d.shape)  # D5_d: torch.Size([1, 512, 5, 5])
        return D1_d, D2_d, D3_d, D4_d, D5_d


#########################
# CCAFNet for FSL
#########################
class CCAFNet_(nn.Module):
    def __init__(self, ):
        super(CCAFNet_, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = vgg_rgb()
        self.depth_pretrained = vgg_depth()

        # rgb Fuse_model
        self.SAG1 = Spatical_Fuse_attention3_GHOST(64)
        self.SAG2 = Spatical_Fuse_attention3_GHOST(128)
        self.SAG3 = Spatical_Fuse_attention3_GHOST(256)

        # depth Fuse_model
        self.CAG4 = Channel_Fuse_attention2(512)
        self.CAG5 = Channel_Fuse_attention2(512)

        self.gatefusion = Gatefusion3_fusionup(512)
        self.conv = nn.Conv2d(512, 512, 1)

        self.feat_dim = [512]
        
    # def forward(self, rgb, depth):
    #     R1, R2, R3, R4, R5 = self.rgb_pretrained(rgb)
    #     D1_d, D2_d, D3_d, D4_d, D5_d = self.depth_pretrained(depth)
    #     SAG1_R = self.SAG1(R1, D1_d)
    #     SAG2_R = self.SAG2(R2, D2_d)
    #     SAG3_R = self.SAG3(R3, D3_d)
    #     CAG5_D = self.CAG5(D5_d, R5)
    #     CAG4_D = self.CAG4(D4_d, R4)
    #     F5 = self.gatefusion(R5, CAG5_D)
    #     out = self.conv(F5)
    #     return out
    def fft_input(self, X, truncate_ratio=0.85):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft
    
    def forward(self, rgb, fpb_sign=True):
        if fpb_sign==True: 
            rgb_novel = self.fft_input(rgb)
            # print('True') 
        else: 
            rgb_novel = rgb
            # print('False')
        R1, R2, R3, R4, R5 = self.rgb_pretrained(rgb)
        D1_d, D2_d, D3_d, D4_d, D5_d = self.depth_pretrained(rgb_novel)
        SAG1_R = self.SAG1(R1, D1_d)
        SAG2_R = self.SAG2(R2, D2_d)
        SAG3_R = self.SAG3(R3, D3_d)
        CAG5_D = self.CAG5(D5_d, R5)
        CAG4_D = self.CAG4(D4_d, R4)
        F5 = self.gatefusion(R5, CAG5_D)
        out = self.conv(F5)
        return out

def CCAFNet(**kwargs):
    """Constructs a CCAFNet model."""
    model = CCAFNet_(**kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'CCAFNet参数:')
    print(Fore.RED+'*********'* 10)
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")   # Total parameters: 2578494
    return model


if __name__=='__main__':
    # 实例化模型(CCAFNet)
    s_time = time.time()
    model = CCAFNet().cuda()
    print(model)
    f_time = time.time()
    print('Model loading time consuming:', f_time-s_time, 's', '\n') 
    # 输入+ 推理 +输出
    for i in range(10): 
        input = torch.randn(64, 3, 84, 84)
        result = model(input.cuda())
        print('Result{}: {}'.format(i, result.shape))

'''
Model loading time consuming: 2.8824892044067383 s 
Result0: torch.Size([64, 512, 5, 5])
Result1: torch.Size([64, 512, 5, 5])
Result2: torch.Size([64, 512, 5, 5])
Result3: torch.Size([64, 512, 5, 5])
Result4: torch.Size([64, 512, 5, 5])
Result5: torch.Size([64, 512, 5, 5])
Result6: torch.Size([64, 512, 5, 5])
Result7: torch.Size([64, 512, 5, 5])
Result8: torch.Size([64, 512, 5, 5])
Result9: torch.Size([64, 512, 5, 5])
'''