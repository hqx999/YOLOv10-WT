import torch
import torch.nn as nn
from custom_lib.modules.conv import autopad
from .se import SEAttention 
from .block import Conv
# --------------------- Neck 核心模块 ---------------------
class CIBBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CIBBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class SPPF_SE(nn.Module):
    """带SE注意力的SPPF模块"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.se = SEAttention(c2)  # 添加SE

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.se(self.cv2(torch.cat((x, y1, y2, y3), 1)))

class PAN_SE(nn.Module):
    """带SE注意力的PANet结构"""
    def __init__(self, c1, c2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.cib1 = CIBBlock(c1, c2)
        self.cib2 = CIBBlock(c2, c2)
        self.se = SEAttention(c2)

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([x, y], dim=1)
        x = self.cib2(self.cib1(x))
        return self.se(x)