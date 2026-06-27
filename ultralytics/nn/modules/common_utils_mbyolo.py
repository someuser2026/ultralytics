import importlib
import math
from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .block import DropPath

DropPath.__repr__ = lambda self: f"DropPath({self.drop_prob})"

if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd") and hasattr(torch.amp, "custom_bwd"):
    _amp_custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    _amp_custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    _amp_custom_fwd = torch.cuda.amp.custom_fwd
    _amp_custom_bwd = torch.cuda.amp.custom_bwd


def _import_selective_scan_module(name):
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, exc


selective_scan_cuda_core, _selective_scan_error = _import_selective_scan_module("selective_scan_cuda_core")
if selective_scan_cuda_core is None:
    selective_scan_cuda_core, _fallback_error = _import_selective_scan_module("selective_scan_cuda")
    if selective_scan_cuda_core is None and _fallback_error is not None:
        _selective_scan_error = _fallback_error

selective_scan_cuda_oflex, _ = _import_selective_scan_module("selective_scan_cuda_oflex")
selective_scan_cuda_ndstate, _ = _import_selective_scan_module("selective_scan_cuda_ndstate")
selective_scan_cuda_nrow, _ = _import_selective_scan_module("selective_scan_cuda_nrow")


def _require_selective_scan():
    if selective_scan_cuda_core is None:
        raise ImportError(
            "Mamba-YOLO requires the local selective_scan extension. "
            "Install optional Python deps like 'einops' and build the extension with "
            "'cd selective_scan && pip install .' before using Mamba blocks."
        ) from _selective_scan_error


def _raise_selective_scan_cpu_runtime_error():
    raise RuntimeError(
        "Mamba-YOLO selective scan CPU fallback is available only during model construction stride probing. "
        "Normal CPU execution is not supported; move the model to CUDA for training or inference and keep the "
        "compiled selective_scan extension installed."
    )


def _selective_scan_ref_build_only(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """Pure PyTorch selective scan used only for CPU model-construction fallback."""
    dtype_in = u.dtype
    A = A.to(dtype_in)
    B = B.to(dtype_in)
    C = C.to(dtype_in)
    D = D.to(dtype_in) if D is not None else None
    delta = delta.to(dtype_in)
    delta_bias = delta_bias.to(dtype_in) if delta_bias is not None else None

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B, "... (l two) -> ... l two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C, "... (l two) -> ... l two", two=2))

    state = A.new_zeros((batch, dim, dstate))
    outputs = []
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        delta_b_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    elif B.dim() == 3:
        delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
        delta_b_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])

    for index in range(u.shape[2]):
        state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", state, C)
        elif C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", state, C[:, :, index])
        else:
            y = torch.einsum("bdn,bdn->bd", state, C[:, :, :, index])
        if y.is_complex():
            y = y.real * 2
        outputs.append(y)

    y = torch.stack(outputs, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    return out.to(dtype=dtype_in)


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @_amp_custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        _require_selective_scan()
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @_amp_custom_bwd
    def backward(ctx, dout, *args):
        _require_selective_scan()
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default',
        allow_cpu_fallback_for_build=False,
        no_einsum=False,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        if u.is_cuda:
            _require_selective_scan()
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
        if allow_cpu_fallback_for_build:
            return _selective_scan_ref_build_only(u, delta, A, B, C, D, delta_bias, delta_softplus)
        _raise_selective_scan_cpu_runtime_error()

    xs = CrossScan.apply(x)

    if no_einsum:
        x_dbl = F.conv1d(
            xs.view(B, -1, L),
            x_proj_weight.view(-1, D, 1),
            bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
            groups=K,
        )
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(
            dts.contiguous().view(B, -1, L),
            dt_projs_weight.view(K * D, -1, 1),
            groups=K,
        )
    else:
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)
