# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
neck.py — Modular FPN family for Ultralytics

Scope covered with minimal duplication and clean extension points:

E1: FPN Variants
    - FPN (standard top-down with lateral)
    - PANet (bottom-up augmentation)
    - PAFPN (YOLOv4-style: concat-heavy, top-down + bottom-up)
    - BiFPN (weighted fusion, fast normalized fusion; iterations configurable)
    - AugFPN (ratio-invariant adaptive pooling + residual augmentation)  [approximation]
    - LibraFPN (balanced feature pyramid via global balanced fusion)
    - RepFPN (reparam convs in lateral & smoothing)
    - RecursiveFPN (stacked FPN refinement passes)
    - ScaleEqualizingFPN [approximation — scale alignment via cross-level pooling average]

E2: Attention Modules in Neck (optional plugin blocks)
    - SE, ECA, CBAM (full / channel-only / spatial-only), CoordinateAttention, SimAM (param-free)
      Use via `attn_cfg` with keys: {'per_level': <name or None>, 'after_fuse': <name or None>}.

E3: Convolutional Enhancements (pluggable per-conv policy)
    - Depthwise/grouped/dilated/atrous selections (ConvPolicy)
    - DeformableConv2d (toggle via conv_cfg.dcn=True)
    - Simple multi-rate dilation via conv_cfg.dilation

E5: Feature Normalization / Aggregation
    - Optional per-level channel normalization to `out_channels` via 1×1 conv:
        `normalize_channels=True`
    - Fusion modes: 'add' | 'concat' | 'weighted' (BiFPN-style)
      Concat path ends with 1×1 to out_channels.
    - Separate dropout for top-down and bottom-up paths: `drop_td`, `drop_bu`

IMPORTANT:
- No P1/P6 (or extra levels) are created internally. Necks operate **only** on the input list.
- Inputs/outputs are lists of feature maps; lengths and resolutions are respected as-is.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Ultralytics blocks already present in your repo
from .conv import Conv, DWConv, DeformableConv2d, CBAM, RepConv, SE
# from .common import RepConv, SE


# -----------------------------
# Utilities
# -----------------------------

def _make_dropout(p: float | None) -> nn.Module:
    return nn.Dropout2d(p) if p and p > 0.0 else nn.Identity()


def _disable_metadata_cfg(cfg: dict | None) -> dict:
    """Return a shallow config copy with metadata conditioning disabled."""
    cfg = dict(cfg or {})
    metadata_cfg = dict(cfg.get("metadata_cfg", {}))
    metadata_cfg["enabled"] = False
    cfg["metadata_cfg"] = metadata_cfg
    return cfg


class MetadataConditioner(nn.Module):
    """Map a per-image metadata vector to per-level FiLM parameters or channel gates."""

    def __init__(self, out_channels: int, num_levels: int, cfg: dict):
        super().__init__()
        self.mode = cfg.get("mode", "film_affine")
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        dropout = float(cfg.get("dropout", 0.0))
        if self.mode not in {"film_affine", "film_gate"}:
            raise ValueError(f"Unsupported metadata modulation mode '{self.mode}'.")

        self.trunk = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        out_dim = 2 * out_channels if self.mode == "film_affine" else out_channels
        self.level_heads = nn.ModuleList(nn.Linear(self.hidden_dim, out_dim) for _ in range(num_levels))

    def forward(self, xs: List[torch.Tensor], metadata_vec: torch.Tensor) -> List[torch.Tensor]:
        """Apply metadata-driven affine modulation or channel gating to each feature level."""
        if metadata_vec.ndim != 2:
            raise ValueError(f"metadata_vec must have shape [B, M], received {tuple(metadata_vec.shape)}.")
        hidden = self.trunk(metadata_vec)
        outs = []
        for x, head in zip(xs, self.level_heads):
            params = head(hidden).to(device=x.device, dtype=x.dtype)
            if self.mode == "film_affine":
                gamma, beta = params.chunk(2, dim=1)
                x = x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
            else:
                gate = torch.sigmoid(params).unsqueeze(-1).unsqueeze(-1)
                x = x * gate
            outs.append(x)
        return outs


# ================================ CONVOLUTIONAL POLICY ================================

class ConvPolicy(nn.Module):
    """
    Composable convolution policy for lateral/smoothing/fusion operations.
    Supports standard Conv, grouped/depthwise Conv, dilated Conv, and Deformable Conv (E19b).
    
    This abstraction allows swapping convolution types via conv_cfg dict:
        - conv_cfg={'groups': 4} -> Grouped convolution (E21a)
        - conv_cfg={'dilation': 2} -> Dilated/atrous convolution (E20a)
        - conv_cfg={'dcn': True} -> Deformable convolution v2 (E19b)
    
    Args:
        c1, c2: Input/output channels
        k, s: Kernel size and stride
        groups: Number of groups for grouped convolution (c2 for depthwise when c1==c2)
        dilation: Dilation rate for atrous convolution
        dcn: If True, use DeformableConv2d instead of standard Conv
        act: Activation function (True -> SiLU, False -> None, or pass nn.Module)
    
    Example:
        >>> # Standard 3x3 conv
        >>> conv = ConvPolicy(256, 256, k=3, s=1)
        >>> # Deformable conv with dilation
        >>> conv_dcn = ConvPolicy(256, 256, k=3, s=1, dcn=True, dilation=2)
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        groups: int = 1,
        dilation: int = 1,
        dcn: bool = False,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        if dcn:
            # Deformable convolution path: DCN + BN + activation
            act_mod = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
            self.m = nn.Sequential(
                DeformableConv2d(c1, c2, k, s, p=dilation, d=dilation, bias=False),
                nn.BatchNorm2d(c2),
                act_mod,
            )
        else:
            # Standard convolution path (supports groups and dilation)
            self.m = Conv(c1, c2, k=k, s=s, p=dilation, g=groups, d=dilation, act=act)

    def forward(self, x):
        return self.m(x)


def lateral_1x1(c1: int, c2: int) -> nn.Module:
    """
    Create 1x1 convolution for lateral connections in FPN.
    Used to project backbone features to neck's out_channels.
    
    Args:
        c1: Input channels (from backbone)
        c2: Output channels (neck out_channels)
    
    Returns:
        Conv module with 1x1 kernel
    """
    return Conv(c1, c2, k=1, s=1)


def smooth_3x3(c: int, conv_cfg: dict, dcn: bool) -> nn.Module:
    """
    Create 3x3 smoothing convolution applied after feature fusion.
    Respects conv_cfg for grouped/dilated/deformable variants.
    
    Args:
        c: Number of channels (in == out)
        conv_cfg: Dict with optional keys: 'groups', 'dilation', 'dcn'
    
    Returns:
        ConvPolicy module configured with conv_cfg settings
    """
    return ConvPolicy(c, c, k=3, s=1,
                      groups=conv_cfg.get('groups', 1),
                      dilation=conv_cfg.get('dilation', 1),
                      dcn=dcn,
                      act=True)


def make_align_layers(in_channels: Sequence[int], out_channels: int, normalize: bool) -> nn.ModuleList:
    """
    Build channel alignment layers for input features.
    
    If normalize=True: Creates 1x1 convs to project mismatched channels to out_channels
    If normalize=False: Uses nn.Identity (no projection, assumes channels already match)
    
    Args:
        in_channels: List of input channel counts [C_P2, C_P3, C_P4, ...]
        out_channels: Target channel count for neck
        normalize: Whether to normalize/align channels
    
    Returns:
        ModuleList of Conv or nn.Identity layers, one per input level
    
    Example:
        >>> # Align [256, 512, 1024] -> 256
        >>> align = make_align_layers([256, 512, 1024], 256, normalize=True)
        >>> # Results in [nn.Identity(), Conv(512->256), Conv(1024->256)]
    """
    layers = []
    for c in in_channels:
        layers.append(lateral_1x1(c, out_channels) if normalize and c != out_channels else nn.Identity())
    return nn.ModuleList(layers)


# ================================ ATTENTION MODULES (E2) ================================

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (E13b).
    
    Lightweight channel attention via 1D convolution over channel descriptors.
    More efficient than SE by avoiding dimensionality reduction.
    
    Reference: ECA-Net (CVPR 2020)
    
    Args:
        channels: Number of input channels
        k_size: Kernel size for 1D conv (adaptive kernel based on channel dimension)
    
    Example:
        >>> eca = ECABlock(256)
        >>> out = eca(features)  # (B, 256, H, W) -> (B, 256, H, W) with channel attention
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        # 1D conv over channels: (B, C, 1, 1) -> (B, 1, C) -> (B, 1, C)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # Reshape and apply sigmoid: (B, 1, C) -> (B, C, 1, 1)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        # Channel-wise attention
        return x * y


class CoordAttention(nn.Module):
    """
    Coordinate Attention (E15d).
    
    Encodes spatial information along height and width dimensions separately,
    enabling long-range dependencies with precise positional information.
    More effective than standard channel attention for localization tasks.
    
    Reference: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
    
    Args:
        c: Number of input channels
        rd: Reduction ratio for bottleneck (default: 32)
    
    Example:
        >>> coord_attn = CoordAttention(256, rd=32)
        >>> out = coord_attn(features)  # Preserves spatial structure better than SE
    """
    def __init__(self, c: int, rd: int = 32):
        super().__init__()
        m = max(8, c // rd)  # Bottleneck channels (minimum 8)
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(c, m, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(m)
        self.act = nn.SiLU()
        
        # Separate 1x1 convs for height and width attention
        self.conv_h = nn.Conv2d(m, c, kernel_size=1, stride=1, bias=True)
        self.conv_w = nn.Conv2d(m, c, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Aggregate along width: (B, C, H, W) -> (B, C, H, 1)
        x_h = x.mean(dim=3, keepdim=True)
        # Aggregate along height: (B, C, H, W) -> (B, C, 1, W) -> (B, C, W, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        
        # Concatenate: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        
        # Shared transform
        y = self.act(self.bn1(self.conv1(y)))
        
        # Split back into height and width components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W)
        
        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, H, 1)
        a_w = self.sigmoid(self.conv_w(x_w))  # (B, C, 1, W)
        
        # Apply coordinate-wise attention
        return x * a_h * a_w


class SimAM(nn.Module):
    """
    SimAM: Parameter-free attention (E15f).
    
    Computes attention weights based on neuron importance without any learnable parameters.
    Uses simple energy function to measure neuron importance relative to surrounding neurons.
    
    Reference: SimAM: A Simple, Parameter-Free Attention Module (ICML 2021)
    
    Args:
        e_lambda: Regularization parameter for stability
    
    Example:
        >>> simam = SimAM()  # Zero parameters!
        >>> out = simam(features)  # Attention applied with no learned weights
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        
        # Compute squared difference from mean
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)) ** 2
        
        # Compute variance with regularization
        v = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        
        # Energy function (lower energy = more important)
        attn = x_minus_mu_square / (4 * (v + 1e-12)) + 0.5
        
        # Apply sigmoid to get attention weights
        return x * torch.sigmoid(attn)

class CarafePPResample2d(nn.Module):
    """
    CARAFE++-style content-aware resampling (upsample or downsample) in pure PyTorch.
    
    CARAFE++ (Content-Aware ReAssembly of FEatures) is a learnable upsampling/downsampling
    operator that generates position-specific reassembly kernels based on content features.
    Unlike fixed interpolation (bilinear, nearest), it adapts kernels spatially for better
    feature preservation.

    Modes:
      - 'up'   : scale>1   (default scale=2)   -> content-aware upsampling
      - 'down' : scale>1   (default scale=2)   -> content-aware downsampling

    Args:
        channels (int): input/output channels (kept constant throughout)
        mode (str): 'up' for upsampling or 'down' for downsampling
        scale (int): resampling factor (e.g., 2 for 2x up/down)
        kernel (int): reassembly kernel size (e.g., 5 means 5x5 local neighborhood)
        encoder_kernel (int): kernel size for kernel-prediction conv (e.g., 3)
        comp (int): channel compression ratio for efficient kernel prediction
    """
    def __init__(
        self,
        channels: int,
        mode: str = "up",
        scale: int = 2,
        kernel: int = 5,
        encoder_kernel: int = 3,
        comp: int = 4,
    ):
        super().__init__()
        assert mode in ("up", "down")
        assert scale >= 2, "scale must be >=2 for CARAFE++"
        self.mode = mode
        self.scale = int(scale)
        self.kernel = int(kernel)  # Size of local reassembly kernel
        self.pad = self.kernel // 2  # Padding to maintain spatial dimensions

        # Compressed channel dimension for efficient kernel prediction
        mid = max(8, channels // int(comp))

        # ========== Kernel Prediction Network ==========
        # Three-stage pipeline: compress -> encode -> predict
        
        # Stage 1: Channel compression (C -> mid channels)
        # Reduces computational cost of kernel prediction
        self.compress = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Stage 2: Content encoding
        # For upsampling: encode at input resolution
        # For downsampling: encode at output (downsampled) resolution
        stride = 1 if mode == "up" else self.scale
        self.encoder = nn.Conv2d(mid, mid, kernel_size=encoder_kernel, stride=stride,
                                 padding=encoder_kernel // 2, bias=True)

        # Stage 3: Kernel prediction
        if mode == "up":
            # Predict kernels for upsampling:
            # - Output: scale^2 * kernel^2 channels (one kernel per HR pixel)
            # - PixelShuffle rearranges to (kernel^2, H*scale, W*scale)
            # - Each HR location gets its own kernel^2 weights
            out_ch = (self.scale * self.scale) * (self.kernel * self.kernel)
            self.predict = nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
            self.ps = nn.PixelShuffle(self.scale)  # Rearrange to HR grid
        else:
            # Predict kernels for downsampling:
            # - Output: kernel^2 channels (one kernel per LR pixel)
            # - Directly at downsampled resolution
            out_ch = (self.kernel * self.kernel)
            self.predict = nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: content-aware resampling.
        
        Input:  x -> (B, C, H, W)
        Output: y -> (B, C, H*s, W*s) for 'up', or (B, C, H/s, W/s) for 'down'
        
        Process:
        1. Generate position-specific reassembly kernels from content
        2. Extract local neighborhoods from input features
        3. Apply predicted kernels to reassemble features at target resolution
        """
        B, C, H, W = x.shape

        # ========== Step 1: Kernel Prediction ==========
        # Generate content-aware kernels for each output position
        kp = self.act(self.compress(x))  # Compress channels
        kp = self.act(self.encoder(kp))  # Encode content
        # kp shape: (B, mid, H, W) for up, (B, mid, H/s, W/s) for down

        if self.mode == "up":
            # --- Upsampling Path ---
            
            # Predict kernels and rearrange to HR grid
            logits = self.predict(kp)                         # (B, s^2*K^2, H, W)
            kernels = self.ps(logits)                         # (B, K^2, H*s, W*s)
            kernels = torch.softmax(kernels, dim=1)           # Normalize to sum=1
            
            # ========== Step 2: Extract Neighborhoods ==========
            # Extract K×K neighborhoods around each LR pixel
            neigh = F.unfold(x, kernel_size=self.kernel, padding=self.pad, stride=1)
            # neigh: (B, C*K^2, H*W) - flattened neighborhoods
            
            neigh = neigh.view(B, C, self.kernel * self.kernel, H, W)
            # neigh: (B, C, K^2, H, W) - structured neighborhoods
            
            # ========== Step 3: Broadcast to HR Grid ==========
            # Repeat each LR neighborhood across its corresponding s×s HR cells
            neigh = neigh.repeat_interleave(self.scale, dim=3)  # H -> H*s
            neigh = neigh.repeat_interleave(self.scale, dim=4)  # W -> W*s
            # neigh: (B, C, K^2, H*s, W*s)
            
            # ========== Step 4: Reassemble with Predicted Kernels ==========
            # Weighted sum: each HR pixel uses its predicted kernel weights
            out = (neigh * kernels.unsqueeze(1)).sum(dim=2)     # (B, C, H*s, W*s)
            return out

        # --- Downsampling Path ---
        else:
            # Predict kernels at LR resolution
            logits = self.predict(kp)                                # (B, K^2, H/s, W/s)
            kernels = torch.softmax(logits, dim=1)                   # Normalize
            
            # ========== Step 2: Extract Strided Neighborhoods ==========
            # Extract K×K neighborhoods centered on LR grid (stride=scale)
            neigh = F.unfold(x, kernel_size=self.kernel, padding=self.pad, stride=self.scale)
            # neigh: (B, C*K^2, (H/s)*(W/s)) - neighborhoods at LR positions
            
            Hds = kp.shape[-2]  # Height at downsampled resolution
            Wds = kp.shape[-1]  # Width at downsampled resolution
            neigh = neigh.view(B, C, self.kernel * self.kernel, Hds, Wds)
            # neigh: (B, C, K^2, H/s, W/s)
            
            # ========== Step 3: Reassemble with Predicted Kernels ==========
            # Weighted sum: each LR pixel aggregates from its K×K HR neighborhood
            out = (neigh * kernels.unsqueeze(1)).sum(dim=2)          # (B, C, H/s, W/s)
            return out

def build_upsampler(c: int, resample_cfg: Dict) -> nn.Module:
    """
    Factory function to build upsampler modules for top-down feature pyramid paths.
    
    Supports two upsampling strategies:
    1. 'nearest': Simple nearest-neighbor interpolation (fast, no parameters)
    2. 'carafepp': Content-aware learnable upsampling (better quality, more compute)
    
    Args:
        c (int): Number of channels (input = output)
        resample_cfg (Dict): Configuration dictionary with keys:
            - up (str): Upsampler type - 'nearest' or 'carafepp'
            - scale (int): Upsampling factor (default: 2)
            - kernel (int): CARAFE++ reassembly kernel size (default: 5)
            - encoder_kernel (int): CARAFE++ encoder kernel size (default: 3)
            - comp (int): CARAFE++ compression ratio (default: 4)
    
    Returns:
        nn.Module: Upsampler that takes (B, c, H, W) -> (B, c, H*scale, W*scale)
    
    Example:
        # Use nearest-neighbor 2x upsampling
        up = build_upsampler(256, {"up": "nearest", "scale": 2})
        
        # Use CARAFE++ 2x upsampling with custom kernel
        up = build_upsampler(256, {"up": "carafepp", "scale": 2, "kernel": 5})
    """
    typ = (resample_cfg or {}).get("up", "nearest").lower()
    s = (resample_cfg or {}).get("scale", 2)
    
    if typ == "nearest":
        # Fast, parameter-free upsampling via nearest-neighbor interpolation
        return nn.Upsample(scale_factor=s, mode="nearest")
    
    if typ == "carafepp":
        # Content-aware learnable upsampling
        k = resample_cfg.get("kernel", 5)           # Reassembly kernel size
        ek = resample_cfg.get("encoder_kernel", 3)  # Encoder kernel size
        comp = resample_cfg.get("comp", 4)          # Channel compression ratio
        return CarafePPResample2d(c, mode="up", scale=s, kernel=k, encoder_kernel=ek, comp=comp)
    
    raise ValueError(f"Unknown upsampler: {typ}")


def build_downsampler(c: int, resample_cfg: Dict, conv_cfg: Dict, dcn: bool) -> nn.Module:
    """
    Factory function to build downsampler modules for bottom-up feature pyramid paths.
    
    Supports two downsampling strategies:
    1. 'conv': Strided convolution (standard approach, preserves local structure)
    2. 'carafepp': Content-aware learnable downsampling (adaptive aggregation)
    
    Args:
        c (int): Number of channels (input = output)
        resample_cfg (Dict): Configuration dictionary with keys:
            - down (str): Downsampler type - 'conv' or 'carafepp'
            - scale (int): Downsampling factor (default: 2)
            - kernel (int): CARAFE++ reassembly kernel size (default: 5)
            - encoder_kernel (int): CARAFE++ encoder kernel size (default: 3)
            - comp (int): CARAFE++ compression ratio (default: 4)
        conv_cfg (Dict): Configuration for ConvPolicy when using 'conv' mode:
            - groups (int): Convolution groups (default: 1)
            - dilation (int): Dilation rate (default: 1)
            - dcn (bool): Use deformable convolution (default: False)
    
    Returns:
        nn.Module: Downsampler that takes (B, c, H, W) -> (B, c, H/scale, W/scale)
    
    Example:
        # Use strided conv 2x downsampling
        down = build_downsampler(256, {"down": "conv", "scale": 2}, {"groups": 1})
        
        # Use CARAFE++ 2x downsampling
        down = build_downsampler(256, {"down": "carafepp", "scale": 2}, {})
    """
    typ = (resample_cfg or {}).get("down", "conv").lower()
    s = (resample_cfg or {}).get("scale", 2)
    
    if typ == "conv":
        # Standard strided convolution downsampling
        # Uses 3x3 kernel with stride=scale to reduce spatial dimensions
        # Mirrors existing stride-2 downsample pattern in the architecture
        return ConvPolicy(c, c, k=3, s=s,
                          groups=conv_cfg.get("groups", 1),
                          dilation=conv_cfg.get("dilation", 1),
                          dcn=dcn)
    
    if typ == "carafepp":
        # Content-aware learnable downsampling
        # Predicts position-specific kernels to aggregate HR features into LR
        k = resample_cfg.get("kernel", 5)           # Reassembly kernel size
        ek = resample_cfg.get("encoder_kernel", 3)  # Encoder kernel size
        comp = resample_cfg.get("comp", 4)          # Channel compression ratio
        return CarafePPResample2d(c, mode="down", scale=s, kernel=k, encoder_kernel=ek, comp=comp)
    
    raise ValueError(f"Unknown downsampler: {typ}")


def build_attn(name: Optional[str], c: int) -> nn.Module:
    """
    Factory function to build attention modules by name.
    
    Supported attention types (E2 experiments):
        - 'se': Squeeze-and-Excitation (E13a)
        - 'eca': Efficient Channel Attention (E13b)
        - 'cbam': Full CBAM (channel + spatial) (E15a)
        - 'cbam_c': CBAM channel-only (E13c)
        - 'cbam_s': CBAM spatial-only (E14a)
        - 'coord': Coordinate Attention (E15d)
        - 'simam': SimAM parameter-free (E15f)
    
    Args:
        name: Attention type string (case-insensitive) or None for no attention
        c: Number of channels for the attention module
    
    Returns:
        Attention module instance or nn.Identity if name is None
    
    Raises:
        ValueError: If attention name is not recognized
    
    Example:
        >>> attn = build_attn('coord', 256)
        >>> # Use in attn_cfg: {'per_level': 'se', 'after_fuse': 'cbam'}
    """
    if not name:
        return nn.Identity()
    
    name = name.lower()
    
    if name == 'se':
        return SE(c)
    if name == 'eca':
        return ECABlock(c)
    if name == 'cbam':
        return CBAM(c, spatial=True, channel=True)
    if name == 'cbam_c':
        return CBAM(c, spatial=False, channel=True)
    if name == 'cbam_s':
        return CBAM(c, spatial=True, channel=False)
    if name == 'coord':
        return CoordAttention(c)
    if name == 'simam':
        return SimAM()
    
    raise ValueError(f"Unknown attention type: {name}. "
                     f"Supported: se, eca, cbam, cbam_c, cbam_s, coord, simam")


# ================================ FUSION MODULES ================================

class WeightedAdd(nn.Module):
    """
    BiFPN-style fast normalized weighted fusion (E2b variant).
    
    Learns non-negative scalar weights for each input and computes normalized weighted sum.
    Faster than softmax-based weighting used in earlier BiFPN variants.
    
    Formula: output = sum_i (w_i / (sum_j w_j + eps)) * x_i
    where w_i = ReLU(learned_weight_i)
    
    Reference: EfficientDet (CVPR 2020)
    
    Args:
        n_inputs: Number of input tensors to fuse (typically 2 for FPN nodes)
        eps: Small constant for numerical stability in normalization
    
    Example:
        >>> fuser = WeightedAdd(n_inputs=2)
        >>> out = fuser([feature1, feature2])  # Learned weighted combination
    """
    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # Initialize weights to 1.0 (equal weighting initially)
        self.w = nn.Parameter(torch.ones(n_inputs))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # Ensure non-negative weights and match input dtype
        w = torch.relu(self.w).to(xs[0].dtype)
        # Normalize to sum to 1
        w = w / (w.sum() + self.eps)
        
        # Compute weighted sum
        out = w[0] * xs[0]
        for i in range(1, len(xs)):
            out = out + w[i] * xs[i]
        return out


class Fusion(nn.Module):
    """
    Enhanced flexible fusion module supporting multiple fusion strategies (E32 variants).
    
    Fusion Modes:
        1. 'add': Element-wise addition (E32b)
           - Requires aligned channels (all c_in must equal c_out)
           - Most efficient, no extra parameters
           - Assumes equal importance of all inputs
           
        2. 'weighted': BiFPN-style learned static weights (E32d)
           - Fast normalized fusion with learnable scalar weights
           - Each input gets a learned weight, normalized to sum to 1
           - More flexible than 'add', minimal parameter overhead
           
        3. 'concat': Concatenation + 1x1 projection (E32a)
           - Concatenates all inputs along channel dimension
           - Projects concatenated features to c_out via 1x1 conv
           - Preserves all information, highest parameter count
           
        4. 'adapool': Adaptive pooling fusion (similar to AugFPN E5)
           - Concatenates inputs
           - Applies adaptive average pooling to (bins × bins) resolution
           - Upsamples pooled features back to original size
           - Adds as residual to concatenated features
           - Projects to c_out via 1x1 conv
           - Captures multi-scale context information
           
        5. 'fc': Data-dependent MLP gating (NEW - dynamic fusion)
           - Uses global average pooling + MLP to compute input weights
           - Weights are data-dependent (change per sample)
           - Two variants:
             * Scalar gating (fc_channel=False): One weight per input
             * Per-channel gating (fc_channel=True): C weights per input
    
    Args:
        mode (str): Fusion strategy - 'add' | 'weighted' | 'concat' | 'adapool' | 'fc'
        c_in (List[int]): Input channel counts for each input feature
        c_out (int): Output channel count after fusion
        bins (int): Pooling resolution for 'adapool' mode (default: 3)
                    E.g., bins=3 pools to 3×3 grid
        fc_hidden (int | None): Hidden size for MLP in 'fc' mode
                                If None, auto-computed as max(128, total_channels // 2)
        fc_channel (bool): If True, use per-channel gating in 'fc' mode
                           If False, use scalar gating (one weight per input)
    
    Examples:
        >>> # Simple addition (requires aligned channels)
        >>> fuse_add = Fusion('add', [256, 256], 256)
        
        >>> # BiFPN weighted fusion
        >>> fuse_weighted = Fusion('weighted', [256, 256], 256)
        
        >>> # Concatenation fusion
        >>> fuse_concat = Fusion('concat', [128, 256, 512], 256)
        
        >>> # Adaptive pooling fusion (AugFPN-style)
        >>> fuse_adapool = Fusion('adapool', [256, 256], 256, bins=3)
        
        >>> # Data-dependent scalar gating
        >>> fuse_fc_scalar = Fusion('fc', [256, 256, 256], 256, fc_hidden=256, fc_channel=False)
        
        >>> # Data-dependent per-channel gating
        >>> fuse_fc_channel = Fusion('fc', [256, 256, 256], 256, fc_hidden=512, fc_channel=True)
    
    References:
        - 'weighted': EfficientDet (CVPR 2020)
        - 'adapool': AugFPN (arXiv 2020)
        - 'fc': Inspired by attention mechanisms and adaptive fusion
    """
    def __init__(
        self,
        mode: str,
        c_in: List[int],
        c_out: int,
        bins: int = 3,
        fc_hidden: Optional[int] = None,
        fc_channel: bool = False,
    ):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        self.c_in = list(c_in)
        self.c_out = int(c_out)
        self.bins = int(bins)
        self.fc_channel = bool(fc_channel)
        
        # Number of inputs to fuse
        self.num_inputs = len(c_in)

        # =====================================================================
        # MODE: WEIGHTED (BiFPN-style learned static weights)
        # =====================================================================
        if mode == 'weighted':
            # Learnable weights for each input (normalized during forward)
            self.fuser = WeightedAdd(self.num_inputs)
            self.project = nn.Identity()

        # =====================================================================
        # MODE: CONCAT (concatenation + 1x1 projection)
        # =====================================================================
        elif mode == 'concat':
            self.fuser = nn.Identity()
            # Project concatenated features (sum of all c_in) to c_out
            self.project = Conv(sum(c_in), c_out, k=1, s=1)

        # =====================================================================
        # MODE: ADD (element-wise addition)
        # =====================================================================
        elif mode == 'add':
            self.fuser = nn.Identity()
            self.project = nn.Identity()
            # Note: Assumes all c_in are equal to c_out (enforced by caller via normalize_channels)

        # =====================================================================
        # MODE: ADAPOOL (adaptive pooling fusion with residual)
        # =====================================================================
        elif mode == 'adapool':
            # Validate bins parameter
            assert self.bins >= 1, f"bins must be >= 1 for 'adapool', got {self.bins}"
            
            self.fuser = nn.Identity()
            # Project concatenated + pooled features to c_out
            self.project = Conv(sum(c_in), c_out, k=1, s=1)
            
            # No additional learnable parameters needed
            # Pooling and upsampling are done in forward()

        # =====================================================================
        # MODE: FC (data-dependent MLP gating)
        # =====================================================================
        elif mode == 'fc':
            # Validate that all inputs have same channels (required for FC gating)
            T = self.num_inputs  # Number of inputs to fuse
            C = c_in[0]          # Channels per input
            
            if not all(ci == C for ci in c_in):
                raise ValueError(
                    f"'fc' fusion requires all inputs to have the same channels. "
                    f"Got {c_in}. Enable normalize_channels=True in neck config."
                )
            
            # Total feature dimension after GAP and concatenation
            in_dim = T * C
            
            # Auto-compute hidden size if not provided
            if fc_hidden is None:
                fc_hidden = max(128, in_dim // 2)
            
            # ===== Build MLP for gating =====
            if self.fc_channel:
                # Per-channel gating: output is (T * C) logits
                # Reshaped to (T, C) and softmax over T dimension
                # Each channel gets independent weights across inputs
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, fc_hidden, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Linear(fc_hidden, T * C, bias=True),
                )
            else:
                # Scalar gating: output is T logits
                # Softmax over T gives one weight per input
                # All channels of an input share the same weight
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, fc_hidden, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Linear(fc_hidden, T, bias=True),
                )
            
            # Global average pooling to get per-input descriptors
            self.gap = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)
            self.project = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported fusion mode: '{mode}'. "
                f"Supported modes: 'add', 'weighted', 'concat', 'adapool', 'fc'"
            )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through fusion module.
        
        Args:
            xs: List of input tensors to fuse
                Each tensor has shape (B, C_i, H, W) where C_i = self.c_in[i]
                For 'add' and 'fc' modes, all C_i must be equal
        
        Returns:
            Fused tensor of shape (B, c_out, H, W)
        
        Raises:
            RuntimeError: If invalid fusion mode state (should never happen)
            AssertionError: If input shapes are incompatible
        """
        # Validate input count
        assert len(xs) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(xs)}"
        
        # Validate spatial dimensions (all inputs must have same H, W)
        H, W = xs[0].shape[-2:]
        for i, x in enumerate(xs[1:], 1):
            assert x.shape[-2:] == (H, W), \
                f"Input {i} spatial size {x.shape[-2:]} != reference {(H, W)}. " \
                f"All inputs must have same spatial dimensions."
        
        # =====================================================================
        # CONCAT MODE: Concatenate along channel dim + 1x1 projection
        # =====================================================================
        if self.mode == 'concat':
            # Concatenate: [B,C1,H,W], [B,C2,H,W], ... -> [B, sum(Ci), H, W]
            y = torch.cat(xs, dim=1)
            # Project to output channels
            return self.project(y)

        # =====================================================================
        # WEIGHTED MODE: BiFPN-style learned weighted sum
        # =====================================================================
        if self.mode == 'weighted':
            # WeightedAdd computes: sum_i (w_i / sum_j(w_j)) * x_i
            y = self.fuser(xs)
            return self.project(y)

        # =====================================================================
        # ADD MODE: Simple element-wise addition
        # =====================================================================
        if self.mode == 'add':
            # Initialize with first input
            y = xs[0]
            # Add remaining inputs
            for t in xs[1:]:
                y = y + t
            return self.project(y)

        # =====================================================================
        # ADAPOOL MODE: Adaptive pooling with residual connection
        # =====================================================================
        if self.mode == 'adapool':
            # Step 1: Concatenate all inputs
            y = torch.cat(xs, dim=1)  # (B, sum(C_i), H, W)
            
            # Step 2: Adaptive average pooling to (bins × bins) resolution
            # This captures multi-scale context at a fixed resolution
            pooled = F.adaptive_avg_pool2d(y, (self.bins, self.bins))  # (B, sum(C_i), bins, bins)
            
            # Step 3: Upsample pooled features back to original spatial size
            # Uses nearest neighbor to match FPN's upsampling strategy
            pooled_up = F.interpolate(pooled, size=y.shape[-2:], mode='nearest')  # (B, sum(C_i), H, W)
            
            # Step 4: Residual connection with numerical stability
            # Add small epsilon to prevent gradient issues in case of near-zero features
            y = y + pooled_up + 1e-6
            
            # Step 5: Project to output channels
            return self.project(y)

        # =====================================================================
        # FC MODE: Data-dependent MLP gating
        # =====================================================================
        if self.mode == 'fc':
            B, C, H, W = xs[0].shape
            T = len(xs)  # Number of inputs
            
            # Step 1: Global average pooling for each input
            # Reduces spatial dimensions to get per-input descriptors
            gs = [self.gap(t).flatten(1) for t in xs]  # Each: (B, C)
            
            # Step 2: Concatenate descriptors
            g = torch.cat(gs, dim=1)  # (B, T*C)
            
            # Step 3: Check if gradient checkpointing is needed for memory efficiency
            in_dim = T * C
            use_checkpoint = in_dim > 10000  # Threshold for large models
            
            # ===== Per-channel gating variant =====
            if self.fc_channel:
                # Step 3a: MLP produces logits for each input-channel pair
                # Use gradient checkpointing for very large feature dimensions
                if use_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint
                    logits = checkpoint(self.mlp, g, use_reentrant=False)
                else:
                    logits = self.mlp(g)  # (B, T*C)
                
                # Step 4a: Reshape to separate inputs and channels
                logits = logits.view(B, T, C)  # (B, T, C)
                
                # Step 5a: Softmax over inputs (dim=1) for each channel independently
                # This gives T weights per channel that sum to 1
                w = torch.softmax(logits, dim=1)  # (B, T, C)
                
                # Step 6a: Weighted sum with per-channel weights
                # Using Kahan summation for numerical stability with many inputs
                if T > 10:  # Use stable summation for many inputs
                    out = torch.zeros_like(xs[0])
                    compensation = torch.zeros_like(xs[0])  # Kahan summation compensation
                    
                    for i, xi in enumerate(xs):
                        # Extract weights for input i: (B, C, 1, 1)
                        wi = w[:, i, :].unsqueeze(-1).unsqueeze(-1).to(xi.dtype)
                        
                        # Kahan summation algorithm for numerical stability
                        y_term = xi * wi - compensation
                        t = out + y_term
                        compensation = (t - out) - y_term
                        out = t
                else:
                    # Standard summation for few inputs
                    out = sum(xi * w[:, i, :].unsqueeze(-1).unsqueeze(-1).to(xi.dtype) 
                             for i, xi in enumerate(xs))
                
                return self.project(out)
            
            # ===== Scalar gating variant =====
            else:
                # Step 3b: MLP produces scalar logit per input
                # Use gradient checkpointing for very large feature dimensions
                if use_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint
                    logits = checkpoint(self.mlp, g, use_reentrant=False)
                else:
                    logits = self.mlp(g)  # (B, T)
                
                # Step 4b: Softmax over inputs (dim=1)
                # This gives T weights that sum to 1 (one per input)
                w = torch.softmax(logits, dim=1)  # (B, T)
                
                # Step 5b: Weighted sum with scalar weights
                # Using Kahan summation for numerical stability with many inputs
                if T > 10:  # Use stable summation for many inputs
                    out = torch.zeros_like(xs[0])
                    compensation = torch.zeros_like(xs[0])  # Kahan summation compensation
                    
                    for i, xi in enumerate(xs):
                        # Broadcast scalar weight: (B, 1, 1, 1)
                        wi = w[:, i].view(B, 1, 1, 1).to(xi.dtype)
                        
                        # Kahan summation algorithm
                        y_term = xi * wi - compensation
                        t = out + y_term
                        compensation = (t - out) - y_term
                        out = t
                else:
                    # Standard summation for few inputs
                    out = sum(xi * w[:, i].view(B, 1, 1, 1).to(xi.dtype) 
                             for i, xi in enumerate(xs))
                
                return self.project(out)

        # Should never reach here if mode is valid
        raise RuntimeError(f"Invalid fusion mode state: {self.mode}")


# ================================ BASE NECK ================================

class BaseNeck(nn.Module):
    """
    Base class for all FPN-style necks with shared functionality.
    
    SIMPLIFIED API: Takes only 3 arguments:
        1. in_channels: List[int] - Input channels from backbone
        2. out_channels: int - Target output channels
        3. cfg: dict - Configuration with all optional parameters
    
    Configuration keys in cfg:
        - normalize_channels (bool): Align inputs to out_channels via 1x1 convs (default: True)
        - attn_cfg (dict): Attention configuration
            - 'per_level': Attention applied after alignment (default: None)
            - 'after_fuse': Attention applied after fusion (default: None)
        - conv_cfg (dict): Convolution configuration
            - 'groups': Number of groups for grouped conv (default: 1)
            - 'dilation': Dilation rate for atrous conv (default: 1)
            - 'dcn': Use deformable convolution (default: False)
        - fusion (str): Fusion mode - 'add'|'concat'|'weighted' (default: 'add')
        - drop_td (float): Dropout for top-down path (default: 0.0)
        - drop_bu (float): Dropout for bottom-up path (default: 0.0)
    
    Example:
        >>> cfg = {
        ...     'normalize_channels': True,
        ...     'attn_cfg': {'per_level': 'se', 'after_fuse': 'cbam'},
        ...     'conv_cfg': {'dcn': True, 'dilation': 2},
        ...     'fusion': 'add',
        ...     'drop_td': 0.1,
        ...     'drop_bu': 0.1
        ... }
        >>> neck = FPN([256, 512, 1024, 2048], 256, cfg)
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        # Parse configuration with defaults
        cfg = cfg or {}
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.normalize_channels = cfg.get('normalize_channels', True)

        self.fusion_bins = cfg.get("fusion_bins", 3)
        self.fusion_fc_hidden = cfg.get("fusion_fc_hidden", None)
        self.fusion_fc_channel = cfg.get("fusion_fc_channel", False)
        
        # Build channel alignment layers (1x1 convs or Identity)
        self.align = make_align_layers(self.in_channels, self.out_channels, self.normalize_channels)

        # Store configuration dicts
        self.conv_cfg = cfg.get('conv_cfg', {})
        self.attn_cfg = cfg.get('attn_cfg', {})
        self.resample_cfg = cfg.get('resample_cfg', {})
        
        # Build attention modules
        # Per-level attention: applied after channel alignment
        self.attn_per_level = nn.ModuleList([
            build_attn(self.attn_cfg.get('per_level'), self.out_channels)
            for _ in self.in_channels
        ])
        # After-fusion attention: applied after feature fusion in FPN paths
        self.attn_after_fuse = build_attn(self.attn_cfg.get('after_fuse'), self.out_channels)

        self.fusion_mode = cfg.get('fusion', 'add')

        # Path-specific dropout layers
        self.drop_td = _make_dropout(cfg.get('drop_td', 0.0))
        self.drop_bu = _make_dropout(cfg.get('drop_bu', 0.0))
        self.metadata_cfg = cfg.get("metadata_cfg", {})
        self.metadata_enabled = bool(self.metadata_cfg.get("enabled", False))
        self.metadata_conditioner = (
            MetadataConditioner(self.out_channels, len(self.in_channels), self.metadata_cfg) if self.metadata_enabled else None
        )

    def _resize_to(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Resize source tensor to match reference tensor's spatial dimensions.
        Uses nearest neighbor interpolation (standard for FPN).
        
        Args:
            src: Source tensor to resize (B, C, H_src, W_src)
            ref: Reference tensor with target size (B, C, H_ref, W_ref)
        
        Returns:
            Resized source tensor (B, C, H_ref, W_ref)
        """
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode='nearest')

    def _apply_align_and_attn(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply channel alignment and per-level attention to input features.
        
        Pipeline: backbone_features -> 1x1 alignment -> per-level attention
        
        Args:
            xs: Input feature list from backbone
        
        Returns:
            Aligned and attention-enhanced features
        """
        ys = []
        for x, al, attn in zip(xs, self.align, self.attn_per_level):
            y = al(x)       # Channel alignment (1x1 conv or identity)
            y = attn(y)     # Per-level attention (SE/ECA/CBAM/etc or identity)
            ys.append(y)
        return ys

    def _apply_metadata_modulation(
        self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Apply metadata conditioning if enabled and metadata is available."""
        if not self.metadata_enabled or metadata_vec is None:
            return xs
        return self.metadata_conditioner(xs, metadata_vec)
    
    def _select_flag(self, key: str, role: str, i: int, default: bool = False):
        """
        Reads conv_cfg[key] as:
            - bool -> bool
            - dict - {role: bool | list[int | bool], default}
        """

        val = self.conv_cfg.get(key, default)

        if isinstance(val, bool):
            return val
        if isinstance(val, dict):
            role_val = val.get(role, val.get("default", default))
            if isinstance(role_val, bool):
                return role_val
            if isinstance(role_val, (list, tuple)):
                return bool(role_val[i]) if i < len(role_val) else val.get("default", default)
        
        return bool(val)
    
    def _dcn(self, role: str, i: int, default: bool = False):
        return self._select_flag("dcn", role, i, default)


# ================================ FPN VARIANTS ================================

class FPN(BaseNeck):
    """
    Standard Feature Pyramid Network (E1 baseline).
    
    Architecture:
        1. Top-down pathway starting from coarsest level
        2. Lateral connections from backbone at each level
        3. Element-wise fusion (add/concat/weighted)
        4. 3x3 smoothing convolution after fusion
    
    Key properties:
        - Builds high-level semantic features from coarse to fine
        - Each level receives information from coarser level above
        - Lateral connections inject fine-grained spatial details
    
    Args:
        in_channels: Backbone output channels [C_P2, C_P3, C_P4, C_P5]
        out_channels: Output channels for all FPN levels (typically 256)
        cfg: Additional args passed to BaseNeck (attn_cfg, conv_cfg, etc.)
    
    Example:
        >>> # Standard FPN with 256 output channels
        >>> fpn = FPN([256, 512, 1024, 2048], 256)
        >>> outs = fpn([p2, p3, p4, p5])  # All outputs have 256 channels
        
        >>> # FPN with coordinate attention and deformable convs
        >>> fpn_advanced = FPN(
        ...     [256, 512, 1024, 2048], 256,
        ...     attn_cfg={'after_fuse': 'coord'},
        ...     conv_cfg={'dcn': True}
        ... )
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg: dict):
        super().__init__(in_channels, out_channels, cfg)
        L = len(self.in_channels)
        self.num_outs = cfg.get("num_outs", L)
        self.add_extra_convs = cfg.get("add_extra_convs", False)
        self.relu_before_extra_convs = cfg.get("relu_before_extra_convs", False)
        if self.num_outs < L:
            raise ValueError(f"FPN num_outs must be >= number of inputs, got {self.num_outs} < {L}.")
        if self.add_extra_convs not in {False, True, "on_output"}:
            raise ValueError(
                "FPN add_extra_convs must be False, True, or 'on_output'. "
                f"Received {self.add_extra_convs!r}."
            )

        # Lateral 1x1 convs (only needed if not using normalize_channels)
        # When normalize_channels=True, alignment is done in BaseNeck.align
        self.laterals = nn.ModuleList([
            lateral_1x1(self.in_channels[i], self.out_channels) if not self.normalize_channels else nn.Identity()
            for i in range(L)
        ])
        
        # 3x3 smoothing convolutions (applied after fusion to reduce aliasing)
        self.smooth = nn.ModuleList([smooth_3x3(self.out_channels, self.conv_cfg, dcn=self._dcn("smooth_td", i)) for i in range(L)])

        # build the carafe upsampler
        self.upsample = build_upsampler(self.out_channels, self.resample_cfg)
        
        # Fusion modules for combining lateral and top-down features
        # (Not needed for topmost level)
        self._fusions = nn.ModuleList([
            Fusion(
                self.fusion_mode, [self.out_channels, self.out_channels], self.out_channels,
                bins = self.fusion_bins,
                fc_hidden = self.fusion_fc_hidden,
                fc_channel = self.fusion_fc_channel
            )
            for _ in range(L - 1)
        ])
        extra_levels = self.num_outs - L
        self.extra_convs = nn.ModuleList(
            ConvPolicy(
                self.out_channels,
                self.out_channels,
                k=3,
                s=2,
                groups=self.conv_cfg.get("groups", 1),
                dilation=self.conv_cfg.get("dilation", 1),
                dcn=self._dcn("extra", i),
            )
            for i in range(extra_levels)
        )

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            xs: Input features from backbone [P2, P3, P4, P5, ...]
                where P2 is finest (largest spatial size) and P5 is coarsest
        
        Returns:
            FPN output features [P2_out, P3_out, P4_out, P5_out, ...]
            All outputs have out_channels and enhanced multi-scale information
        """
        # Step 1: Channel alignment and per-level attention
        xs = self._apply_align_and_attn(xs)
        xs = self._apply_metadata_modulation(xs, metadata_vec)
        L = len(xs)

        outs = [None] * L
        
        # Step 2: Initialize top-down path from coarsest level
        top = self.laterals[-1](xs[-1])
        outs[-1] = self.smooth[-1](top)

        # Step 3: Top-down fusion from coarse to fine
        for i in range(L - 2, -1, -1):
            # Lateral connection from backbone
            li = self.laterals[i](xs[i])
            
            # Upsample coarser FPN feature to current resolution
            up = self.upsample(outs[i + 1])
            up = self._resize_to(up, li)
            
            # Fuse lateral and upsampled features
            fused = self._fusions[i]([li, up])
            
            # Apply after-fusion attention (if configured)
            fused = self.attn_after_fuse(fused)
            
            # Apply dropout for regularization
            fused = self.drop_td(fused)
            
            # 3x3 smoothing to reduce aliasing from upsampling
            outs[i] = self.smooth[i](fused)

        if not self.extra_convs:
            return outs

        extra_source = outs[-1]
        for extra_conv in self.extra_convs:
            if self.relu_before_extra_convs:
                extra_source = F.relu(extra_source)
            if self.add_extra_convs in {True, "on_output"}:
                extra_source = extra_conv(extra_source)
            else:
                extra_source = F.max_pool2d(extra_source, kernel_size=1, stride=2)
            outs.append(extra_source)

        return outs


class PANet(BaseNeck):
    """
    Path Aggregation Network (E3a: Bottom-up path augmentation).
    
    Architecture:
        1. Standard FPN top-down path (coarse to fine)
        2. Additional bottom-up path (fine to coarse)
        3. Bottom-up path allows low-level features to be propagated upward
    
    Advantages over FPN:
        - Low-level localization features can directly reach high levels
        - Shorter path for information flow from fine to coarse levels
        - Improves detection of small objects
    
    Reference: Path Aggregation Network for Instance Segmentation (CVPR 2018)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        cfg: Additional args (attn_cfg, conv_cfg, fusion, dropout, etc.)
    
    Example:
        >>> panet = PANet([256, 512, 1024, 2048], 256)
        >>> outs = panet([p2, p3, p4, p5])  # Enhanced with bottom-up path
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg: dict):
        super().__init__(in_channels, out_channels, cfg)
        L = len(self.in_channels)
        self.metadata_enabled = False
        self.metadata_conditioner = None

        # Top-down FPN stage (standard FPN)
        self.td_fpn = FPN(in_channels, out_channels, _disable_metadata_cfg(cfg))

        # Bottom-up path: stride-2 convolutions for downsampling
        # self.down = nn.ModuleList([
        #     ConvPolicy(self.out_channels, self.out_channels, k=3, s=2, dcn=self._dcn("down_bu", i))
        #     for i in range(L - 1)
        # ])
        self.down = nn.ModuleList([
            build_downsampler(self.out_channels, self.resample_cfg, self.conv_cfg, dcn = self._dcn("down_bu", i))
            for i in range(L - 1)
        ])
        
        # Bottom-up fusion modules
        self.bu_fuse = nn.ModuleList([
            Fusion(self.fusion_mode, [self.out_channels, self.out_channels], self.out_channels, self.fusion_bins, self.fusion_fc_hidden, self.fusion_fc_channel)
            for _ in range(L - 1)
        ])
        
        # Bottom-up smoothing convolutions
        self.bu_smooth = nn.ModuleList([smooth_3x3(self.out_channels, self.conv_cfg, dcn=self._dcn("smooth_bu", i)) for i in range(L - 1)])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: FPN top-down + bottom-up path augmentation.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            PANet features with bidirectional information flow
        """
        # Step 1: Standard FPN top-down path
        # (alignment and per-level attention handled inside FPN)
        ys = self.td_fpn(xs, metadata_vec=metadata_vec)
        L = len(ys)

        # Step 2: Bottom-up path augmentation (fine to coarse)
        bu = [None] * L
        bu[0] = ys[0]  # Finest level unchanged from FPN
        
        for i in range(L - 1):
            # Downsample finer level
            d = self.down[i](bu[i])
            
            # Fuse downsampled with corresponding FPN level
            fused = self.bu_fuse[i]([d, ys[i + 1]])
            
            # Apply after-fusion attention
            fused = self.attn_after_fuse(fused)
            
            # Apply bottom-up dropout
            fused = self.drop_bu(fused)
            
            # Smooth with 3x3 conv
            bu[i + 1] = self.bu_smooth[i](fused)

        return bu


class PAFPN(BaseNeck):
    """
    YOLOv4-style PAFPN (E10: Concat-heavy bidirectional pyramid).
    
    Architecture:
        1. Top-down path with concatenation fusion
        2. Bottom-up path with concatenation fusion
        3. More parameters than PANet but preserves more information
    
    Key difference from PANet:
        - Uses concatenation instead of addition for all fusion operations
        - Concatenation followed by 1x1 projection + 3x3 smoothing
        - Preserves full feature information at cost of more parameters
    
    Reference: YOLOv4 (arXiv 2020)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Note:
        This implementation manages its own alignment to avoid double-laterals
        since it has a different fusion strategy than standard FPN.
    
    Example:
        >>> pafpn = PAFPN([256, 512, 1024, 2048], 256)
        >>> outs = pafpn([p2, p3, p4, p5])  # YOLOv4-style features
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        L = len(self.in_channels)

        # Top-down path components
        # self.td_upsample = nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest') for _ in range(L - 1)])
        self.td_upsample = nn.ModuleList([
            build_upsampler(self.out_channels, self.resample_cfg)
            for _ in range(L - 1)
        ])
        self.td_fuse = nn.ModuleList([
            Fusion('concat', [self.out_channels, self.out_channels], self.out_channels, self.fusion_bins, self.fusion_fc_hidden, self.fusion_fc_channel) for _ in range(L - 1)
        ])
        self.td_smooth = nn.ModuleList([smooth_3x3(self.out_channels, self.conv_cfg, dcn=self._dcn("smooth_td", i)) for i in range(L - 1)])

        # Bottom-up path components
        # self.bu_down = nn.ModuleList([
        #     ConvPolicy(self.out_channels, self.out_channels, k=3, s=2, dcn=self._dcn("down_bu", i))
        #     for i in range(L - 1)
        # ])
        self.bu_down = nn.ModuleList([
            build_downsampler(self.out_channels, self.resample_cfg, self.conv_cfg, dcn = self._dcn("down_bu", i))
            for i in range(L - 1)
        ])
        self.bu_fuse = nn.ModuleList([
            Fusion('concat', [self.out_channels, self.out_channels], self.out_channels, self.fusion_bins, self.fusion_fc_hidden, self.fusion_fc_channel) for _ in range(L - 1)
        ])
        self.bu_smooth = nn.ModuleList([smooth_3x3(self.out_channels, self.conv_cfg, dcn=self._dcn("smooth_bu", i)) for i in range(L - 1)])

        # PAFPN manages its own alignment to avoid redundancy with BaseNeck
        self.pre_align = make_align_layers(self.in_channels, self.out_channels, self.normalize_channels)
        self.pre_attn = nn.ModuleList([build_attn(self.attn_cfg.get('per_level'), self.out_channels)
                                       for _ in self.in_channels])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: Concat-heavy bidirectional pyramid.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            PAFPN features with full information preservation via concatenation
        """
        # Step 1: Align channels and apply per-level attention
        xs = [attn(al(x)) for x, al, attn in zip(xs, self.pre_align, self.pre_attn)]
        xs = self._apply_metadata_modulation(xs, metadata_vec)
        L = len(xs)

        # Step 2: Top-down path with concatenation
        td = [None] * L
        curr = xs[-1]
        td[-1] = curr
        
        for i in range(L - 2, -1, -1):
            # Upsample coarser level
            up = self.td_upsample[i](curr)
            up = self._resize_to(up, xs[i])
            
            # Concatenate finer level with upsampled coarser level
            fused = self.td_fuse[i]([xs[i], up])
            
            # Apply after-fusion attention
            fused = self.attn_after_fuse(fused)
            
            # Apply top-down dropout
            fused = self.drop_td(fused)
            
            # Smooth with 3x3 conv
            curr = self.td_smooth[i](fused)
            td[i] = curr

        # Step 3: Bottom-up path with concatenation
        bu = [None] * L
        bu[0] = td[0]
        
        for i in range(L - 1):
            # Downsample finer level
            d = self.bu_down[i](bu[i])
            
            # Concatenate downsampled with TD feature
            fused = self.bu_fuse[i]([d, td[i + 1]])
            
            # Apply after-fusion attention
            fused = self.attn_after_fuse(fused)
            
            # Apply bottom-up dropout
            fused = self.drop_bu(fused)
            
            # Smooth with 3x3 conv
            bu[i + 1] = self.bu_smooth[i](fused)

        return bu


class BiFPN(BaseNeck):
    """
    Bidirectional Feature Pyramid Network (E2a-d: BiFPN with iterations).
    
    Architecture:
        1. Weighted feature fusion using fast normalized fusion
        2. Bidirectional cross-scale connections
        3. Stacked iterations for iterative refinement
    
    Key innovations:
        - Learned fusion weights (more efficient than equal weighting)
        - Multiple iterations (E2c: 3, E2d: 5) for feature refinement
        - Cross-scale connections for better information flow
    
    Reference: EfficientDet (CVPR 2020)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        iterations: Number of BiFPN stacks (1=single pass, 3=E2c, 5=E2d)
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Example:
        >>> # Single iteration BiFPN (E2a)
        >>> bifpn = BiFPN([256, 512, 1024, 2048], 256, iterations=1)
        
        >>> # 3-iteration BiFPN (E2c)
        >>> bifpn_3 = BiFPN([256, 512, 1024, 2048], 256, iterations=3)
        
        >>> # 5-iteration BiFPN (E2d)
        >>> bifpn_5 = BiFPN([256, 512, 1024, 2048], 256, iterations=5)
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        cfg["fusion"] = "weighted"
        super().__init__(in_channels, out_channels, cfg)
        self.iterations = max(1, cfg.get("iterations", 1))
        self.metadata_enabled = False
        self.metadata_conditioner = None
        
        # First BiFPN layer processes backbone features
        self.fpn = FPN(in_channels, out_channels, cfg)

        # Additional iterations process uniform-channel features
        L = len(in_channels)
        self.extra_stacks = nn.ModuleList([
            FPN([out_channels] * L, out_channels, _disable_metadata_cfg(cfg))
            for _ in range(self.iterations - 1)
        ])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: Iterative bidirectional feature refinement.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            Refined features after multiple BiFPN iterations
        """
        # First iteration: process backbone features
        y = self.fpn(xs, metadata_vec=metadata_vec)
        
        # Additional iterations: iterative refinement
        for fpn in self.extra_stacks:
            y = fpn(y)
        
        return y


class AugFPN(BaseNeck):
    """
    Augmented FPN (E5 approximation: Ratio-invariant adaptive pooling).
    
    Architecture:
        1. Standard FPN as base
        2. Ratio-invariant adaptive pooling (RAP) to fixed bins
        3. Residual feature augmentation (RFA) added back to each level
    
    Key features:
        - Handles scale variations better through adaptive pooling
        - Adds global context via pooled features
        - Lightweight approximation of full AugFPN
    
    Reference: AugFPN (arXiv 2020)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        pool_bins: Number of bins for adaptive pooling (typically 3)
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Note:
        This is a simplified implementation focusing on RAP+RFA core concepts.
        Full AugFPN includes additional components like semantic enhancement.
    
    Example:
        >>> augfpn = AugFPN([256, 512, 1024, 2048], 256, pool_bins=3)
        >>> outs = augfpn([p2, p3, p4, p5])  # Enhanced with global context
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        self.pool_bins = cfg.get("pool_bins", 3)
        self.metadata_enabled = False
        self.metadata_conditioner = None
        
        # Base FPN
        self.fpn = FPN(in_channels, out_channels, cfg)

        # RAP projection layers
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: FPN + ratio-invariant adaptive pooling + residual augmentation.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            Augmented features with global context
        """
        # Step 1: Standard FPN
        feats = self.fpn(xs, metadata_vec=metadata_vec)
        
        # Step 2: Apply RAP + RFA to each level
        outs = []
        for f in feats:
            H, W = f.shape[-2:]
            
            # Adaptive pooling to fixed bins (ratio-invariant)
            pooled = F.adaptive_avg_pool2d(f, (self.pool_bins, self.pool_bins))
            
            # Project pooled features
            pooled = self.proj(pooled)
            pooled = self.bn(pooled)
            pooled = self.act(pooled)
            
            # Upsample back to original size
            pooled = F.interpolate(pooled, size=(H, W), mode='nearest')
            
            # Residual feature augmentation
            outs.append(f + pooled)
        
        return outs


class LibraFPN(BaseNeck):
    """
    Libra FPN (E9 approximation: Balanced feature pyramid).
    
    Architecture:
        1. Standard FPN as base
        2. Global feature balancing across all levels
        3. Balanced features added back to each level
    
    Key features:
        - Addresses feature imbalance across pyramid levels
        - Global context pooling at reference scale
        - Improves consistency across scales
    
    Reference: Libra R-CNN (CVPR 2019)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Note:
        Full Libra R-CNN includes IoU-balanced sampling and balanced L1 loss,
        which are training components (not architecture). This implements
        the balanced feature pyramid component only.
    
    Example:
        >>> librafpn = LibraFPN([256, 512, 1024, 2048], 256)
        >>> outs = librafpn([p2, p3, p4, p5])  # Balanced features
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        self.metadata_enabled = False
        self.metadata_conditioner = None
        
        # Base FPN
        self.fpn = FPN(in_channels, out_channels, cfg)
        
        # Reprojection layers after adding balanced features
        self.reproj = nn.ModuleList([Conv(out_channels, out_channels, k=3, s=1) for _ in in_channels])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: FPN + global balanced feature injection.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            Balanced features with global context
        """
        # Step 1: Standard FPN
        feats = self.fpn(xs, metadata_vec=metadata_vec)
        
        # Step 2: Compute balanced feature at reference scale (median level)
        ref = feats[len(feats) // 2]
        
        # Resize all features to reference scale
        resized = [self._resize_to(f, ref) for f in feats]
        
        # Average across all levels (balanced feature)
        balanced = torch.stack(resized, dim=0).mean(dim=0)
        
        # Step 3: Add balanced feature back to each level
        outs = []
        for i, f in enumerate(feats):
            # Resize balanced feature to match current level
            b = self._resize_to(balanced, f)
            
            # Add and reproject
            outs.append(self.reproj[i](f + b))
        
        return outs


class RepFPN(BaseNeck):
    """
    Reparameterizable FPN (E11: RepConv for lateral and smoothing).
    
    Architecture:
        1. Standard FPN structure
        2. RepConv (reparameterizable convolution) for lateral connections
        3. RepConv for smoothing operations
    
    Key features:
        - Multi-branch training, single-branch inference
        - Improves representation power during training
        - No additional inference cost after reparameterization
    
    Reference: RepVGG (CVPR 2021)
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Note:
        Call model.fuse() before inference to merge branches for speed.
    
    Example:
        >>> repfpn = RepFPN([256, 512, 1024, 2048], 256)
        >>> # Training
        >>> outs = repfpn([p2, p3, p4, p5])
        >>> # Before inference
        >>> repfpn.eval()
        >>> # Branches are automatically merged in eval mode
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        L = len(in_channels)

        # Channel alignment layers
        self.align = make_align_layers(in_channels, out_channels, self.normalize_channels)

        # RepConv lateral connections (1x1 + RepConv 3x3)
        self.lat = nn.ModuleList([
            nn.Sequential(
                Conv(in_channels[i] if not self.normalize_channels else out_channels, out_channels, k=1, s=1),
                RepConv(out_channels, out_channels, k=3, s=1),
            )
            for i in range(L)
        ])
        
        # RepConv smoothing layers
        self.smooth = nn.ModuleList([RepConv(out_channels, out_channels, k=3, s=1) for _ in range(L)])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: FPN with reparameterizable convolutions.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            FPN features with enhanced representation from RepConv
        """
        # Step 1: Channel alignment
        xs = [al(x) for x, al in zip(xs, self.align)]
        xs = self._apply_metadata_modulation(xs, metadata_vec)
        L = len(xs)
        
        # Step 2: Top-down FPN with RepConv
        outs = [None] * L
        
        # Topmost level
        outs[-1] = self.smooth[-1](self.lat[-1](xs[-1]))
        
        # Top-down fusion
        for i in range(L - 2, -1, -1):
            # Upsample coarser level
            up = self._resize_to(outs[i + 1], xs[i])
            
            # Lateral connection + addition fusion
            fused = self.lat[i](xs[i]) + up
            
            # Apply dropout
            fused = self.drop_td(fused)
            
            # RepConv smoothing
            outs[i] = self.smooth[i](fused)
        
        return outs


class RecursiveFPN(BaseNeck):
    """
    Recursive FPN (E7: Multi-pass refinement).
    
    Architecture:
        1. Standard FPN applied multiple times
        2. Each pass refines features from previous pass
        3. Iterative refinement improves feature quality
    
    Key features:
        - Simple but effective iterative refinement
        - Each pass has same architecture (weight sharing)
        - More passes = more refinement but slower inference
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        passes: Number of FPN passes (default: 2)
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Example:
        >>> # 2-pass recursive FPN
        >>> recfpn = RecursiveFPN([256, 512, 1024, 2048], 256, passes=2)
        
        >>> # 3-pass for more refinement
        >>> recfpn_3 = RecursiveFPN([256, 512, 1024, 2048], 256, passes=3)
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        self.passes = max(1, cfg.get("passes", 2))
        L = len(in_channels)
        self.metadata_enabled = False
        self.metadata_conditioner = None
        
        # Single FPN applied multiple times
        self.fpn_first = FPN(in_channels, out_channels, cfg)
        self.fpn_shared = FPN([out_channels] * L, out_channels, _disable_metadata_cfg(cfg))

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: Apply FPN multiple times for iterative refinement.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            Refined features after multiple FPN passes
        """
        # First pass on backbone features
        y = self.fpn_first(xs, metadata_vec=metadata_vec)
        
        # Additional passes on refined features
        for _ in range(self.passes - 1):
            y = self.fpn_shared(y)
        
        return y


class ScaleEqualizingFPN(BaseNeck):
    """
    Scale-Equalizing FPN (E8 approximation: Cross-level mean feature injection).
    
    Architecture:
        1. Standard FPN as base
        2. Compute mean feature across all levels at reference scale
        3. Inject mean as residual bias to equalize scales
    
    Key features:
        - Addresses scale imbalance in feature pyramid
        - Cross-level information sharing via mean pooling
        - Helps with scale-sensitive tasks
    
    Args:
        in_channels: Backbone output channels
        out_channels: Output channels for all levels
        cfg: Additional args (attn_cfg, conv_cfg, dropout, etc.)
    
    Note:
        This is a lightweight approximation. Full scale-equalizing pyramid
        uses integral loss for scale balance during training.
    
    Example:
        >>> sefpn = ScaleEqualizingFPN([256, 512, 1024, 2048], 256)
        >>> outs = sefpn([p2, p3, p4, p5])  # Scale-equalized features
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, cfg):
        super().__init__(in_channels, out_channels, cfg)
        self.metadata_enabled = False
        self.metadata_conditioner = None
        
        # Base FPN
        self.fpn = FPN(in_channels, out_channels, cfg)
        
        # Post-processing layers after adding mean feature
        self.post = nn.ModuleList([Conv(out_channels, out_channels, k=3, s=1) for _ in in_channels])

    def forward(self, xs: List[torch.Tensor], metadata_vec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass: FPN + cross-level mean feature injection.
        
        Args:
            xs: Input features from backbone
        
        Returns:
            Scale-equalized features with cross-level mean
        """
        # Step 1: Standard FPN
        feats = self.fpn(xs, metadata_vec=metadata_vec)
        
        # Step 2: Compute cross-level mean at reference scale
        ref = feats[len(feats) // 2]  # Median level as reference
        
        # Resize all features to reference scale
        pooled = [self._resize_to(f, ref) for f in feats]
        
        # Mean across all levels
        mean = torch.stack(pooled, dim=0).mean(0)
        
        # Step 3: Add mean to each level as scale-equalizing bias
        outs = []
        for i, f in enumerate(feats):
            # Resize mean to match current level
            m = self._resize_to(mean, f)
            
            # Add and post-process
            outs.append(self.post[i](f + m))
        
        return outs


# ================================ FACTORY ================================

NECKS = {
    'fpn': FPN,
    'panet': PANet,
    'pafpn': PAFPN,
    'bifpn': BiFPN,
    'augfpn': AugFPN,
    'librafpn': LibraFPN,
    'recfpn': RecursiveFPN,
    'repfpn': RepFPN,
    'sefpn': ScaleEqualizingFPN,
}


def build_neck(name: str, in_channels: Sequence[int], out_channels: int, cfg) -> nn.Module:
    """
    Factory function to build neck by name.
    
    Args:
        name: Neck architecture name (case-insensitive)
        in_channels: Backbone output channels
        out_channels: Neck output channels
        cfg: Additional configuration (attn_cfg, conv_cfg, etc.)
    
    Returns:
        Instantiated neck module
    
    Raises:
        KeyError: If neck name is not recognized
    
    Available necks:
        - 'fpn': Standard FPN (E1)
        - 'panet': Path Aggregation Network (E3a)
        - 'pafpn': YOLOv4-style PAFPN (E10)
        - 'bifpn': BiFPN with iterations (E2a-d)
        - 'augfpn': Augmented FPN (E5)
        - 'librafpn': Libra FPN (E9)
        - 'recfpn': Recursive FPN (E7)
        - 'repfpn': Reparameterizable FPN (E11)
        - 'sefpn': Scale-Equalizing FPN (E8)
    
    Example:
        >>> # Build BiFPN with 3 iterations and coordinate attention
        >>> neck = build_neck(
        ...     'bifpn',
        ...     in_channels=[256, 512, 1024, 2048],
        ...     out_channels=256,
        ...     iterations=3,
        ...     attn_cfg={'after_fuse': 'coord'},
        ...     conv_cfg={'dcn': True}
        ... )
    """
    name = name.lower()
    if name not in NECKS:
        raise KeyError(f"Unknown neck: {name}. Available: {list(NECKS.keys())}")
    return NECKS[name](in_channels, out_channels, cfg)
