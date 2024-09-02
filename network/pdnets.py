# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Our Modified Version based on https://github.com/cai199626/PDNet

import torch, time
import torch.nn as nn
from colorama import init, Fore
init()  # Init Colorama

# Master Network Conv-Blocks with Pooling layer
def MN_conv2_block_1_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # conv-1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

# For PDNet-vgg16 Blocks-3-4
def conv4_block_3_4(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # conv-1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-4
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
# For PDNet-vgg16 Blocks-5
def conv3_block_5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # conv-1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),  # conv-3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

# For PDNet-vgg19 Blocks-3-4
def conv5_block_3_4(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # conv-1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-4
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-5
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
# For PDNet-vgg19 Blocks-5
def conv4_block_5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # conv-1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),# conv-2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), # conv-4
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

# Subsidiary Network with Pooling layer
def Subsidiary_Network(in_channels=3, hid_dim_1=64, hid_dim_2=128, hid_dim_3=256, out_channels=512):
    return nn.Sequential(
        nn.Conv2d(in_channels, hid_dim_1, 3, padding=1),
        nn.BatchNorm2d(hid_dim_1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(hid_dim_1, hid_dim_2, 3, padding=1),
        nn.BatchNorm2d(hid_dim_2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(hid_dim_2, hid_dim_3, 3, padding=1),
        nn.BatchNorm2d(hid_dim_3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(hid_dim_3, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

# Channel Fusion Layer (Master & Subsidiary)
def Depth_enhanced_Fusion(in_channels=1024, out_channels=512):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


#########################
# PDNet16 for FSL
#########################
class PDNet16_(nn.Module):
    def __init__(self, x_dim=3, hid_dim_1=64, hid_dim_2=128, hid_dim_3=256, z_dim=512, **kwargs):
        super().__init__()
        self.feat_dim = [z_dim]
        self.Master_Net = nn.Sequential(
            MN_conv2_block_1_2(x_dim, hid_dim_1),
            MN_conv2_block_1_2(hid_dim_1, hid_dim_2),
            conv4_block_3_4(hid_dim_2, hid_dim_3),
            conv4_block_3_4(hid_dim_3, z_dim),
            conv3_block_5(z_dim, z_dim)
        )
        self.Subsidiary_Net = Subsidiary_Network(x_dim, hid_dim_1, hid_dim_2, hid_dim_3, z_dim)
        self.feature_fusion = Depth_enhanced_Fusion(z_dim*2, z_dim)

    # def forward(self, rgb, depp):
    #     f_rgb = self.Master_Net(rgb)
    #     f_depp= self.Subsidiary_Net(depp)
    #     x = torch.cat((f_rgb, f_depp), dim=1)  # 按通道维度拼接
    #     x = self.feature_fusion(x) 
    #     return x
    def forward(self, rgb):
        f_rgb = self.Master_Net(rgb)
        f_depp= self.Subsidiary_Net(rgb)
        x = torch.cat((f_rgb, f_depp), dim=1)  # 按通道维度拼接
        x = self.feature_fusion(x) 
        return x

def PDNet16(x_dim_=3, hid_dim_1_=64, hid_dim_2_=128, hid_dim_3_=256, z_dim_=512, **kwargs):
    """Constructs a PDNet16 model."""
    model = PDNet16_(x_dim=x_dim_, hid_dim_1=hid_dim_1_, hid_dim_2=hid_dim_2_, hid_dim_3=hid_dim_3_, z_dim=z_dim_, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'PDNet16参数:')
    print(Fore.RED+'*********'* 10)
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")   # Total parameters: 2578494
    
    return model


#########################
# PDNet19 for FSL
#########################
class PDNet19_(nn.Module):
    def __init__(self, x_dim=3, hid_dim_1=64, hid_dim_2=128, hid_dim_3=256, z_dim=512, **kwargs):
        super().__init__()
        self.feat_dim = [z_dim]
        self.Master_Net = nn.Sequential(
            MN_conv2_block_1_2(x_dim, hid_dim_1),
            MN_conv2_block_1_2(hid_dim_1, hid_dim_2),
            conv5_block_3_4(hid_dim_2, hid_dim_3),
            conv5_block_3_4(hid_dim_3, z_dim),
            conv4_block_5(z_dim, z_dim)
        )
        self.Subsidiary_Net = Subsidiary_Network(x_dim, hid_dim_1, hid_dim_2, hid_dim_3, z_dim)
        self.feature_fusion = Depth_enhanced_Fusion(z_dim*2, z_dim)

    # def forward(self, rgb, depp):
    #     f_rgb = self.Master_Net(rgb)
    #     f_depp= self.Subsidiary_Net(depp)
    #     x = torch.cat((f_rgb, f_depp), dim=1)
    #     x = self.feature_fusion(x) 
    #     return x
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
            
        f_rgb = self.Master_Net(rgb)
        f_depp= self.Subsidiary_Net(rgb_novel)
        x = torch.cat((f_rgb, f_depp), dim=1)  
        x = self.feature_fusion(x) 
        return x

def PDNet19(x_dim_=3, hid_dim_1_=64, hid_dim_2_=128, hid_dim_3_=256, z_dim_=512, **kwargs):
    """Constructs a PDNet19 model."""
    model = PDNet19_(x_dim=x_dim_, hid_dim_1=hid_dim_1_, hid_dim_2=hid_dim_2_, hid_dim_3=hid_dim_3_, 
                    z_dim=z_dim_, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'PDNet19参数:')
    print(Fore.RED+'*********'* 10)
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")   # Total parameters: 2578494
    
    return model




if __name__ == '__main__':
    # 实例化模型Ⅰ(PDNet16)
    s_time = time.time()
    model = PDNet16().cuda()
    print(model)
    f_time = time.time()
    print('Model loading time consuming:', f_time-s_time, 's', '\n') 

    # 输入+ 推理 +输出
    for i in range(10): 
        Input_tensor_rgb = torch.rand(64, 3, 84, 84) # torch.Size([64, 3, 84, 84])
        result = model(Input_tensor_rgb.cuda())
        print('PDNet-16 Result{}: {}'.format(i, result.shape))
    '''
    Model loading time consuming: 1.4727532863616943 s 
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

    # 实例化模型Ⅱ(PDNet19)
    s_time = time.time()
    model = PDNet19().cuda()
    print(model)
    f_time = time.time()
    print('Model loading time consuming:', f_time-s_time, 's', '\n') 
    # 输入+ 推理 +输出
    for i in range(10): 
        Input_tensor_rgb = torch.rand(64, 3, 84, 84) # torch.Size([64, 3, 84, 84])
        result = model(Input_tensor_rgb.cuda())
        print('PDNet-19 Result{}: {}'.format(i, result.shape))

    '''
    Model loading time consuming: 0.1854114532470703 s 
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