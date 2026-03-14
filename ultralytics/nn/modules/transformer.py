# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """
    AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """
        Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """
        Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """
    Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment,
    learnable position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """
        Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, w, h].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, w, h].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """
        Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """
    A simple multi-layer perceptron (also called FFN).

    This class implements a configurable MLP with multiple linear layers, activation functions, and optional
    sigmoid output activation.

    Attributes:
        num_layers (int): Number of layers in the MLP.
        layers (nn.ModuleList): List of linear layers.
        sigmoid (bool): Whether to apply sigmoid to the output.
        act (nn.Module): Activation function.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act=nn.ReLU, sigmoid: bool = False
    ):
        """
        Initialize the MLP with specified input, hidden, output dimensions and number of layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            act (nn.Module): Activation function.
            sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the entire MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP.
        """
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension
    while preserving spatial dimensions.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Initialize LayerNorm2d with the given parameters.

        Args:
            num_channels (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Normalized output tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales
    with learnable sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """
        Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, n_levels, 2],
                range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for non-padding
                elements, False for padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


def rotmat(theta):  # theta: (N,)
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s,  c], -1)], -2)  # (N,2,2)

def circular_bias(m, S, M, device):
    # Eq.(10): o_ms = s * (cos(2π m/M), sin(2π m/M)) / max(|cos|,|sin|)
    s_vals = torch.arange(1, S+1, device=device, dtype=torch.float32)
    ang = 2 * math.pi * (m / M)
    c, s = math.cos(ang), math.sin(ang)
    denom = max(abs(c), abs(s))
    vec = torch.stack([torch.full_like(s_vals, c), torch.full_like(s_vals, s)], -1)
    return (s_vals[:, None] * vec) / max(denom, 1e-6)  # (S,2)

def obb_to_gaussian_params(w, h, theta, eps=1e-6):
    """
    Map OBB size/orientation to Gaussian Σ according to Eq. 8:
      Σ^{1/2} = F Λ F^T
    where F(θ) is rotation matrix and Λ = diag(w/2, h/2)
    """
    k = 2.0  # Using factor of 2 as suggested by paper
    sigx = torch.clamp(w / k, min=eps)
    sigy = torch.clamp(h / k, min=eps)
    varx, vary = sigx**2, sigy**2
    
    BQ = w.numel()
    R = rotmat(theta.view(-1))  # (BQ,2,2)
    D = torch.zeros(BQ, 2, 2, device=w.device, dtype=w.dtype)
    D[:, 0, 0], D[:, 1, 1] = varx.view(-1), vary.view(-1)
    Sigma = torch.matmul(torch.matmul(R, D), R.transpose(-2, -1))  # (BQ,2,2)
    
    # Compute inverse and log normalizer
    det = torch.clamp(Sigma[:, 0, 0]*Sigma[:, 1, 1] - Sigma[:, 0, 1]*Sigma[:, 1, 0], min=eps)
    Sigma_inv = torch.linalg.inv(Sigma)
    log_norm = -math.log(2*math.pi) - 0.5 * torch.log(det)
    return Sigma_inv, log_norm  # each (BQ,2,2), (BQ,)

def gaussian_pdf_2d(points_xy, mu_xy, Sigma_inv, log_norm):
    """Compute 2D Gaussian probability density."""
    d = (points_xy - mu_xy)[..., None, :]  # (...,1,2)
    mahal = torch.matmul(torch.matmul(d, Sigma_inv), d.transpose(-1, -2))  # (...,1,1)
    log_pdf = -0.5 * mahal.squeeze(-1).squeeze(-1) + log_norm  # (...)
    return torch.exp(log_pdf)

def multi_scale_oriented_deformable_attn(
    value: torch.Tensor,
    value_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-scale oriented deformable attention sampling.
    
    Args:
        value: (bs, len_v, n_heads, head_dim)
        value_shapes: [(H_0, W_0), (H_1, W_1), ..., (H_L-1, W_L-1)]
        sampling_locations: (bs, len_q, n_heads, n_levels, n_points, 2) in normalized [0,1]
        attention_weights: (bs, len_q, n_heads, n_levels, n_points)
    
    Returns:
        (bs, len_q, n_heads * head_dim)
    """
    bs, len_q, n_heads, n_levels, n_points = attention_weights.shape
    _, len_v, _, head_dim = value.shape
    
    # Split value by levels
    value_list = value.split([H * W for H, W in value_shapes], dim=1)
    
    # Sample from each level
    sampled_values = []
    for level, (H, W) in enumerate(value_shapes):
        # Get value for this level: (bs, H*W, n_heads, head_dim)
        value_l = value_list[level].reshape(bs, H, W, n_heads, head_dim)
        value_l = value_l.permute(0, 3, 4, 1, 2).flatten(0, 1)  # (bs*n_heads, head_dim, H, W)
        
        # Get sampling locations for this level: (bs, len_q, n_heads, n_points, 2)
        sampling_loc_l = sampling_locations[:, :, :, level, :, :]
        sampling_loc_l = sampling_loc_l.flatten(0, 2)  # (bs*len_q*n_heads, n_points, 2)
        
        # Convert to grid_sample format [-1, 1]
        grid = sampling_loc_l * 2.0 - 1.0  # (bs*len_q*n_heads, n_points, 2)
        grid = grid.unsqueeze(2)  # (bs*len_q*n_heads, n_points, 1, 2)
        
        # Sample: (bs*n_heads, head_dim, len_q*n_points, 1)
        sampled = F.grid_sample(
            value_l.repeat(len_q, 1, 1, 1),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampled = sampled.squeeze(-1)  # (bs*n_heads*len_q, head_dim, n_points)
        sampled = sampled.reshape(bs, n_heads, len_q, head_dim, n_points)
        sampled = sampled.permute(0, 2, 1, 3, 4)  # (bs, len_q, n_heads, head_dim, n_points)
        sampled_values.append(sampled)
    
    # Stack all levels: (bs, len_q, n_heads, head_dim, n_levels, n_points)
    sampled_values = torch.stack(sampled_values, dim=-2)
    
    # Apply attention weights: (bs, len_q, n_heads, n_levels, n_points)
    attention_weights = attention_weights.unsqueeze(3)  # (bs, len_q, n_heads, 1, n_levels, n_points)
    output = (sampled_values * attention_weights).sum(dim=(-2, -1))  # (bs, len_q, n_heads, head_dim)
    
    return output.flatten(-2)  # (bs, len_q, n_heads * head_dim)


class MultiScaleOrientedDeformableAttention(nn.Module):
    """
    Multi-scale RO2-DETR Oriented Deformable Attention implementing Eq. 6, 9, 10:
      - Works across multiple feature pyramid levels
      - Gaussian pdf modulates query features in offset calculation (Eq. 9)
      - Offsets use sqrt((w,h)/S) scale factor
      - Circular bias o_ms from Eq. 10
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # Attention weights over (n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # Query-dependent offset modulation for z_q * f(μ,Σ) term in Eq. 9
        # One offset per (head, level, point)
        self.offset_modulation = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        
        # Initialize offset modulation (similar to sampling_offsets in standard deformable DETR)
        nn.init.constant_(self.offset_modulation.weight.data, 0.0)
        # Initialize bias with circular pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.offset_modulation.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize projections
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    @torch.no_grad()
    def _bias_table(self, device):
        """Generate circular bias o_ms for each head (Eq. 10)."""
        # (n_heads, n_points, 2)
        return torch.stack([
            circular_bias(m, self.n_points, self.n_heads, device) 
            for m in range(self.n_heads)
        ], 0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Multi-scale oriented deformable attention forward pass.
        
        Args:
            query: (bs, len_q, C) query features
            refer_bbox: (bs, len_q, n_levels, 5) as (cx, cy, w, h, theta_degrees) in normalized [0,1] coords
            value: (bs, len_v, C) value features from all levels concatenated
            value_shapes: [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask: (bs, len_v) optional mask
            
        Returns:
            (bs, len_q, C) attended features
        """
        bs, len_q, C = query.shape
        len_v = value.shape[1]
        assert C == self.d_model
        assert sum(s[0] * s[1] for s in value_shapes) == len_v
        assert refer_bbox.shape[-1] == 5, "refer_bbox must be (cx, cy, w, h, theta)"
        if refer_bbox.shape[2] == 1:
            refer_bbox = refer_bbox.expand(-1, -1, self.n_levels, -1)
        elif refer_bbox.shape[2] != self.n_levels:
            raise ValueError(
                f"Expected refer_bbox to have 1 or {self.n_levels} feature levels, but got {refer_bbox.shape[2]}."
            )

        # Project values
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.head_dim)

        # Compute attention weights (normalized over all levels and points)
        attention_weights = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Query-dependent offset modulation (z_q component in Eq. 9)
        offset_mod = self.offset_modulation(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Parse OBB parameters for each level
        # refer_bbox: (bs, len_q, n_levels, 5)
        cx = refer_bbox[..., 0]  # (bs, len_q, n_levels)
        cy = refer_bbox[..., 1]
        bw = refer_bbox[..., 2]
        bh = refer_bbox[..., 3]
        theta = (refer_bbox[..., 4] - 0.25) * math.pi  # normalized angle -> radians

        # Compute Gaussian parameters per level (Eq. 7, 8)
        BQL = bs * len_q * self.n_levels
        Sigma_inv, log_norm = obb_to_gaussian_params(
            bw.reshape(-1), bh.reshape(-1), theta.reshape(-1)
        )
        Sigma_inv = Sigma_inv.view(bs, len_q, self.n_levels, 2, 2)
        log_norm = log_norm.view(bs, len_q, self.n_levels)

        # Evaluate Gaussian pdf at reference box centers
        centers = torch.stack([cx, cy], -1)  # (bs, len_q, n_levels, 2)
        pdf_center = gaussian_pdf_2d(
            centers, centers, Sigma_inv, log_norm
        )  # (bs, len_q, n_levels)
        pdf_center = pdf_center[:, :, None, :, None, None]  # (bs, len_q, 1, n_levels, 1, 1)

        # Circular bias o_ms (Eq. 10)
        bias_ms = self._bias_table(query.device)  # (n_heads, n_points, 2)
        bias_ms = bias_ms[None, None, :, None, :, :]  # (1, 1, n_heads, 1, n_points, 2)
        bias_ms = bias_ms.expand(bs, len_q, -1, self.n_levels, -1, -1)

        # Scale factor: sqrt((w,h)/S) as in Eq. 9
        scale = torch.stack([bw, bh], -1) / float(self.n_points) # (bs, len_q, n_levels, 2)
        scale = scale[:, :, None, :, None, :]  # (bs, len_q, 1, n_levels, 1, 2)

        # Compute offsets according to Eq. 9:
        # Δp_mqs = sqrt((w_q, h_q)/S) * (z_q * f(μ_q, Σ_q) + o_ms) * F
        offsets_local = scale * (offset_mod * pdf_center + bias_ms)  # (bs, len_q, n_heads, n_levels, n_points, 2)

        # Rotate offsets by theta (multiply by rotation matrix F)
        # Need to apply per-level rotation
        R = rotmat(theta.reshape(-1)).view(bs, len_q, self.n_levels, 2, 2)  # (bs, len_q, n_levels, 2, 2)
        R = R[:, :, None, :, :, :]  # (bs, len_q, 1, n_levels, 2, 2)
        
        # Rotate: (bs, len_q, n_heads, n_levels, n_points, 2) @ (bs, len_q, 1, n_levels, 2, 2)
        offsets_rotated = torch.einsum("bqhlpd,bqhlij->bqhlpi", offsets_local, R.transpose(-2, -1))
        # offsets_rotated = torch.matmul(
        #     offsets_local.unsqueeze(-2),  # (bs, len_q, n_heads, n_levels, n_points, 1, 2)
        #     R.transpose(-2, -1)  # (bs, len_q, 1, n_levels, 2, 2)
        # ).squeeze(-2)  # (bs, len_q, n_heads, n_levels, n_points, 2)

        # Add to centers to get sampling locations (already in normalized [0,1] coords)
        centers_exp = centers[:, :, None, :, None, :]  # (bs, len_q, 1, n_levels, 1, 2)
        sampling_locations = centers_exp + offsets_rotated  # (bs, len_q, n_heads, n_levels, n_points, 2)
        
        # Clamp to [0, 1] to stay within valid range
        sampling_locations = sampling_locations.clamp(0, 1)

        # Apply multi-scale deformable attention
        output = multi_scale_oriented_deformable_attn(
            value, value_shapes, sampling_locations, attention_weights
        )
        
        return self.output_proj(output)

class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
        use_obb: bool = False,
    ):
        """
        Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
            use_obb (bool): Whether to use oriented bounding boxes.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        if use_obb:
            self.cross_attn = MultiScaleOrientedDeformableAttention(d_model, n_levels, n_heads, n_points)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction
    heads for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """
        Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: list,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            dec_bboxes (torch.Tensor): Decoded bounding boxes.
            dec_cls (torch.Tensor): Decoded classification scores.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
