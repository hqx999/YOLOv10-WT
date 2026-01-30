# optimized_model.py
import torch
import torch.nn as nn
from custom_lib.modules.efficientvit_backbone import EfficientViTBackbone
from custom_lib.modules.repgfpn import RepGFPN, PartialSelfAttention
from custom_lib.modules.se import SEAttention
from custom_lib.modules.assf import SpatialAttention, ChannelAttention
from custom_lib.modules.neck import CIBBlock
from custom_lib.modules.detection_head import DetectionHead
from custom_lib.modules.ghostconv import GhostConv

class OptimizedModel(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone：输出 c5 的形状为 [B, 512, H, W]
        self.backbone = EfficientViTBackbone()  # No ch argument now
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)

        # 优化模块
        self.assf = nn.Sequential(
            SpatialAttention(),
            ChannelAttention(256)
        )
        self.se_attention = SEAttention(256)
        self.repgfpn = RepGFPN(256, 256)
        self.cib = CIBBlock(256, 256)
        self.psa = PartialSelfAttention(256)

        # 检测头
        self.detect = DetectionHead(in_channels=256, num_anchors=num_anchors, num_classes=num_classes)

        # 设定一个默认的 stride 属性
        self.stride = 32  # 这个值可以根据需求调整，通常 YOLO 模型的步幅是 32




    def forward(self, batch, targets=None):
        # ... [保持原有前向传播逻辑] ...
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        if targets is not None:
            loss, loss_items = self.compute_loss(x, targets)
            # 将 loss_items 转换为张量（关键修正！）
            loss_items = torch.cat([loss_items[key] for key in ['box_loss', 'cls_loss', 'dfl_loss']])
            return loss, loss_items

        return (x,)  # 必须返回元组

    def compute_loss(self, predictions, targets):
        # 这里实现你的损失计算逻辑
        box_loss = ...  # 计算边界框损失
        cls_loss = ...  # 计算分类损失
        dfl_loss = ...  # 计算分布焦点损失
        loss_total = box_loss + cls_loss + dfl_loss
        return loss_total, {'box_loss': box_loss, 'cls_loss': cls_loss, 'dfl_loss': dfl_loss}





