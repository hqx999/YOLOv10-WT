import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 直接复用 Ultralytics 自带模块（不重写）
from ultralytics.nn.modules import Conv, C2f, SPPF, GhostConv

__all__ = [
    "PConv", "WTConv", "ECA", "GatedSpatialConv",
    "C2f_WT_Light", "Bottleneck_WT_Light", "v10Detect",
    "GhostConv", "Conv", "C2f", "SPPF", "C2f_UltraLight"
]
# ---------------- PConv（FasterNet 思路，签名对齐 YOLO：c1, c2, k=3, s=1） ----------------
class PConv(nn.Module):
    """更高效的PConv"""
    def __init__(self, c1, c2, k=3, s=1, r=0.125):  # 更小的r
        super().__init__()
        self.r = r
        self.c_part = max(1, int(c1 * r))
        p = k // 2
        
        self.dw = nn.Conv2d(self.c_part, self.c_part, k, s, p, 
                           groups=self.c_part, bias=False)
        self.rest = nn.Identity() if s == 1 else nn.AvgPool2d(2, 2)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        
    def forward(self, x):
        x1 = x[:, :self.c_part]
        x2 = x[:, self.c_part:]
        x1 = self.dw(x1)
        x2 = self.rest(x2)
        y = torch.cat([x1, x2], 1)
        return F.silu(self.bn(self.pw(y)))
        
        
class C2f_UltraLight(nn.Module):
    """增强版超轻量C2f"""
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = c2 = max(4, c2)
        c_ = max(2, c2 // 2)
        
        self.cv1 = EfficientPConv(c1, 2 * c_, 1, 1)
        
        # 交替使用不同卷积类型
        blocks = []
        for i in range(n):
            if i % 3 == 0:  # 每3个块用1个WTConv
                blocks.append(WTConv(c_, c_, 3, 1))
            else:
                blocks.append(EfficientPConv(c_, c_, 3, 1))
        
        self.m = nn.Sequential(*blocks)
        self.cv2 = EfficientPConv((2 + n) * c_, c2, 1, 1)
        
        # 只在较大通道使用注意力
        if c2 >= 32:
            self.attn = UltraECA(c2)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        y = y2
        ys = [y1, y2]
        for b in self.m:
            y = b(y)
            ys.append(y)
        out = self.cv2(torch.cat(ys, 1))
        return self.attn(out)
        

# ---------------- WTConv（DWT -> 3x3 conv -> IDWT），签名对齐：c1, c2, k=3, s=1 ----------------
class HaarDWT(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")
            H, W = x.shape[-2:]
        x = x.view(B, C, H//2, 2, W//2, 2).permute(0,1,3,5,2,4).contiguous()
        return x.view(B, C*4, H//2, W//2)

class HaarIDWT(nn.Module):
    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x = x.view(B, C, 2, 2, H, W).permute(0,1,4,2,5,3).contiguous()
        return x.view(B, C, H*2, W*2)

# 优化版WTConv - 添加残差连接
class WTConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        assert s == 1
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(c1*4, c2*4, k, padding=p, bias=False),
            nn.BatchNorm2d(c2*4),
            nn.SiLU(inplace=True),
        )
        self.use_residual = (c1 == c2)
        
    def forward(self, x):
        H, W = x.shape[-2:]
        identity = x
        y = self.dwt(x)
        y = self.conv(y)
        y = self.idwt(y)
        y = y[..., :H, :W]
        
        if self.use_residual:
            y = y + identity
        return y
        
class WtFusionBlock(nn.Module):
    """
    双分支融合块：
    - 分支1：普通 3x3 Conv（空间域）
    - 分支2：WTConv（小波域）
    - 用通道门控 α 在两者之间自适应加权：
        y = α * y_wt + (1 - α) * y_spatial
    """
    def __init__(self, c, k=3):
        super().__init__()
        # 空间域分支
        self.conv_spatial = Conv(c, c, k=k, s=1)
        # 小波域分支（保留你原来的 3x3 WTConv）
        self.conv_wt = WTConv(c, c, k=k, s=1)

        # 产生门控 α 的轻量通道注意力
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, c // 4)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_spatial = self.conv_spatial(x)
        y_wt = self.conv_wt(x)

        alpha = self.fc(self.gap(x))  # [B,C,1,1]
        # 高频重要时 α→1，更多用 WT 分支；否则偏向普通 Conv
        y = alpha * y_wt + (1.0 - alpha) * y_spatial
        return y

# ---------------- 轻量 CBAM（防止 cr=0） ----------------
class ECA(nn.Module):

    """Efficient Channel Attention (CVPR'20) - 无通道降维，极轻量。"""
    def __init__(self, channels: int, k_size: int | None = None):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 自适应核长（论文公式），保证为奇数
        if k_size is None:
            t = int(abs((math.log2(channels) / 2) + 1))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)  # 至少 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        y = self.gap(x)                    # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y)                   # B,1,C
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * y.expand_as(x)
# ---------------- 门控空间卷积 ----------------
class GatedSpatialConv(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.gate = nn.Sequential(nn.Conv2d(c1, 1, 1, bias=True), nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x * self.gate(x))


# ---------------- C2f_WT_Light（签名与 C2f 一致：c1, c2, n=*, shortcut=False） ----------------
# 优化版C2f_WT_Light - 只添加残差，保持轻量
class C2f_WT_Light(nn.Module):
    """
    增强版 WT-C2f：
    - 外部接口与原来完全一致：C2f_WT_Light(c1, c2, n=3, shortcut=False)
    - 仍然采用 C2f 结构：cv1(1x1) -> n 个 block -> concat -> cv2(1x1)
    - 中间的“WT 位置”使用 WtFusionBlock：
        * 一条普通 3x3 Conv 分支（空间域）
        * 一条 WTConv 分支（小波域）
        * 通道门控 α 做自适应融合
    """
    def __init__(self, c1, c2, n=3, shortcut=False):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, 2 * c_, k=1, s=1)

        blocks = []
        for i in range(n):
            if i == n // 2:
                # 原来这里是单独一个 WTConv，现在升级为“WT+Conv 双分支融合”
                blocks.append(WtFusionBlock(c_, k=3))
            else:
                blocks.append(Conv(c_, c_, k=3, s=1))
        self.m = nn.Sequential(*blocks)

        # 拼接 y1, y2 以及 n 个中间特征，总通道 (2 + n) * c_
        self.cv2 = Conv((2 + n) * c_, c2, k=1, s=1)

        # 轻量注意力沿用原逻辑
        if c2 >= 16:
            self.attn = ECA(c2)
        else:
            self.attn = nn.Identity()

        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        identity = x if self.add else None

        y1, y2 = self.cv1(x).chunk(2, 1)
        ys = [y1, y2]
        y = y2
        for b in self.m:
            y = b(y)
            ys.append(y)

        out = self.cv2(torch.cat(ys, 1))
        out = self.attn(out)

        if identity is not None:
            out = out + identity

        return out

# ---------------- Bottleneck_WT_Light（占位/轻量瓶颈） ----------------
class Bottleneck_WT_Light(nn.Module):
    """最小改动：保持与 Bottleneck 签名一致，内部两层 conv；c1==c2 时支持残差。"""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

# ---------------- v10Detect：用原生 Detect 做壳（延迟导入，最小化循环依赖风险） ----------------
# 放在文件顶部其它 import 之后
from ultralytics.nn.modules.head import Detect as UDetect
import inspect

class v10Detect(UDetect):
    """
    继承官方 Detect，复用官方的 DFL/解码/NMS/后处理，保证与 val/predict 完全一致。
    你的 YAML 里传的是: [nc, [ch_list]]，这里照收，不再自定义 forward。
    """
    def __init__(self, nc=1, ch=(), **kwargs):
        # 兼容不同版本的 Detect.__init__ 签名（8.3.220 没有 reg_max）
        sig = inspect.signature(UDetect.__init__)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        super().__init__(nc=nc, ch=ch, **allowed)

    def fuse(self):
        # 评估/导出阶段有时会调用 fuse，保留即可
        try:
            return super().fuse()
        except Exception:
            return self

# 确保 __all__ 至少包含以下名字（按你实际有的来补）：
__all__ = [
    "PConv", "WTConv", "ECA", "GatedSpatialConv",
    "C2f_WT_Light", "Bottleneck_WT_Light", "v10Detect",
    # 如果你要在 YAML 里用 GhostConv/Conv/C2f/SPPF（原生的），可以不写进 __all__
]

def register_to_ultralytics(verbose=True):
    """
    把本文件中定义的自定义模块，注册进 ultralytics.nn.tasks 的全局命名空间，
    让 parse_model 能用到它们（避免 tasks.py 主动 import 造成循环依赖）。
    """
    import importlib

    # 1) 先导入 Ultralytics 的 tasks 模块（此时不要再 import 回本文件）
    ytasks = importlib.import_module("ultralytics.nn.tasks")

    # 2) 你要暴露给 YAML 的自定义模块名（按你实际有的来填）
    export_names = [
        "PConv", "WTConv", "ECA", "GatedSpatialConv",
        "C2f_WT_Light", "Bottleneck_WT_Light", "v10Detect"
    ]

    # 3) 逐个挂载到 tasks 命名空间
    for name in export_names:
        if name in globals():
            setattr(ytasks, name, globals()[name])
            if verbose:
                print(f"✔ 强制注册: {name}")
        else:
            if verbose:
                print(f"✗ 未找到要注册的类: {name}")

    # 4)（可选）也把部分原生模块名转发进去，方便 YAML 里直接用
    # from ultralytics.nn.modules import Conv, C2f, SPPF, GhostConv
    # setattr(ytasks, "Conv", Conv)
    # setattr(ytasks, "C2f", C2f)
    # setattr(ytasks, "SPPF", SPPF)
    # setattr(ytasks, "GhostConv", GhostConv)

    if verbose:
        print("Custom modules registered successfully!")