# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
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
    Proto26,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    ConvNeXtLayerNorm,
    DropPath,
    ConvNeXtStem,
    ConvNeXtDownsample,
    ConvNeXtBlock,
    GRN,
    Timm,
    MaxViTBlock,
    MaxMBConv,
    # DinoV3Backbone
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    ChannelSplit,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
    DeformableConv2d
)
from .head import (
    OBB,
    OBB26,
    OBBShoreAux,
    RotatedFCOS,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    Segment26,
    SegmentShoreAux,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    RTDETROBBDecoder,
    RTDETRSegmentDecoder,
    # CascadeRCNNHead
)
from .rhino import RHINOOBBDecoder
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

from .neck import (
    BaseNeck,
    FPN,
    PAFPN,
    BiFPN,
    PANet,
    AugFPN,
    LibraFPN,
    RepFPN,
    RecursiveFPN,
    ScaleEqualizingFPN,
)
from .legnet import LWEGNet
from .hrvmamba import HRAdd, HRBottleneck, HRConv, HRFusion
from .rcnn import CascadeMaskRCNNHead, MaskRCNNHead, OrientedRCNNHead, RotatedFasterRCNNHead
from .rcnn_backbone import ResNetBackbone, UnravelNetBackbone
from .mask2former import Mask2FormerHead
from .pointrend import (
    Mask2FormerPointRendAdapter,
    PointRendConfig,
    PointRendInstances,
    PointRendPointHead,
    PointRendRefiner,
    PointRendTrainConfig,
    RCNNPointRendAdapter,
    RTDETRPrototypePointRendAdapter,
    YOLOPrototypePointRendAdapter,
    configure_pointrend_from_yaml,
    configure_pointrend_training,
    get_pointrend_adapter,
    has_pointrend,
    prepare_pointrend_weight_transfer,
    register_pointrend_adapter,
)

from .rpn import AnchorGenerator, RPNHead

try:
    from .mamba_yolo import DVSSBlock, EdgeStem, EdgeVSSBlock, SimpleStem, VisionClueMerge, VSSBlock, XSSBlock
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:
    _MAMBA_IMPORT_ERROR = exc

    def _missing_mamba_module(name):
        class _MissingMambaModule:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{name} requires optional Mamba-YOLO dependencies. "
                    "Install 'einops' and build the local selective_scan package first."
                ) from _MAMBA_IMPORT_ERROR

        _MissingMambaModule.__name__ = name
        return _MissingMambaModule

    EdgeStem = _missing_mamba_module("EdgeStem")
    EdgeVSSBlock = _missing_mamba_module("EdgeVSSBlock")
    DVSSBlock = _missing_mamba_module("DVSSBlock")
    SimpleStem = _missing_mamba_module("SimpleStem")
    VisionClueMerge = _missing_mamba_module("VisionClueMerge")
    VSSBlock = _missing_mamba_module("VSSBlock")
    XSSBlock = _missing_mamba_module("XSSBlock")

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
    "ChannelSplit",
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
    "Proto26",
    "Detect",
    "Segment",
    "Segment26",
    "SegmentShoreAux",
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
    "OBB26",
    "OBBShoreAux",
    "RotatedFCOS",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ResNetBackbone",
    "UnravelNetBackbone",
    "MaskRCNNHead",
    "CascadeMaskRCNNHead",
    "RotatedFasterRCNNHead",
    "OrientedRCNNHead",
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
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "ConvNeXtLayerNorm",
    "DropPath",
    "ConvNeXtStem",
    "ConvNeXtDownsample",
    "ConvNeXtBlock",
    "GRN",
    "ConvNeXtV2Block",
    "Timm",
    "DinoV3Backbone",
    "FPN",
    "PAFPN",
    "PANet",
    "BiFPN",
    "AugFPN",
    "LibraFPN",
    "RecursiveFPN",
    "RepFPN",
    "ScaleEqualizingFPN",
    "LWEGNet",
    "HRAdd",
    "HRBottleneck",
    "HRConv",
    "HRFusion",
    "RHINOOBBDecoder",
    "RTDETROBBDecoder",
    "RTDETRSegmentDecoder",
    "Mask2FormerHead",
    "PointRendConfig",
    "PointRendInstances",
    "PointRendPointHead",
    "PointRendRefiner",
    "PointRendTrainConfig",
    "YOLOPrototypePointRendAdapter",
    "RTDETRPrototypePointRendAdapter",
    "Mask2FormerPointRendAdapter",
    "RCNNPointRendAdapter",
    "configure_pointrend_from_yaml",
    "configure_pointrend_training",
    "get_pointrend_adapter",
    "has_pointrend",
    "prepare_pointrend_weight_transfer",
    "register_pointrend_adapter",
    "EdgeStem",
    "EdgeVSSBlock",
    "DVSSBlock",
    "SimpleStem",
    "VisionClueMerge",
    "VSSBlock",
    "XSSBlock",
    # "CascadeRCNNHead"
)
