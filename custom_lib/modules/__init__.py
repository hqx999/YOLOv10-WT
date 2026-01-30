# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.

This module provides access to various neural network components used in Ultralytics models, including convolution blocks,
attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron.
    >>> from custom_lib.modules import *
    >>> import torch
    >>> import os
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
"""


# 先导入基础模块（不依赖其他模块的）
from .block import Conv, C2f, DWConv

# 然后导入其他模块
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    GhostConv,
)

from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)

# 最后导入可能产生循环依赖的模块
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)

from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ImagePoolingAttn",
    "MaxSigmoidAttnBlock",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PartialSelfAttention",
    "TorchVision",
    "Index",
    "A2C2f",
)

# 在文件末尾添加自定义模块（使用延迟导入避免循环）
try:
    from .efficientvit_backbone import EfficientViTBackbone
    from .efficientvit import EfficientViT_M0
    from .assf import SpatialAttention
    from .repgfpn import RepGFPN, PartialSelfAttention
    from .se import SEAttention
    
    # 扩展 __all__ 包含这些模块
    __all__ += ('EfficientViTBackbone', 'EfficientViT_M0', 'SpatialAttention', 
                'RepGFPN', 'PartialSelfAttention', 'SEAttention')
except ImportError as e:
    print(f"警告: 某些模块导入失败: {e}")

# 自定义数据集模块
try:
    from custom_lib.custom_dataset import CustomDataset
    __all__ += ('CustomDataset',)
except ImportError:
    pass

# 延迟导入自定义模块以避免循环导入
def import_custom_modules():
    """延迟导入自定义模块"""
    try:
        from .custom_modules_ds import (
            ECA, GatedSpatialConv, WTConv, 
            PConv, C2f_WT_Light
        )
        return {
            'ECA': ECA,
            'GatedSpatialConv': GatedSpatialConv, 
            'WTConv': WTConv,
            'C2f_WT_Light': C2f_WT_Light
        }
    except ImportError as e:
        print(f"警告: 自定义模块导入失败: {e}")
        return {}

# 在全局命名空间中添加自定义模块
_custom_modules = import_custom_modules()
for name, module in _custom_modules.items():
    globals()[name] = module
    if name not in __all__:
        __all__ += (name,)

# 确保 __all__ 包含所有必要的模块
__all__ = tuple(sorted(set(__all__)))

from .custom_modules_ds import *

__all__ = [
     'WTConv', 'ECA', 'GatedSpatialConv', 
    'C2f_WT_Light', 'SPPF', 'v10Detect'
]