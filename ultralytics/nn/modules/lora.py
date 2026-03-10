"""LoRA helpers for timm-backed Ultralytics models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` layer with trainable low-rank adapters."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects an nn.Linear base layer, but received {type(base_layer).__name__}.")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, but received {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, but received {alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must satisfy 0 <= dropout < 1, but received {dropout}.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(self.out_features, self.rank))

        self.reset_parameters()
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def reset_parameters(self):
        """Initialize LoRA weights with a zero-initialized output projection."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def lora_weight(self) -> torch.Tensor:
        """Return the low-rank weight update in matrix form."""
        return torch.matmul(self.lora_B, self.lora_A) * self.scaling

    @property
    def weight(self):
        """Expose the effective weight so modules that read `.weight` still see the LoRA update."""
        return self.base_layer.weight + self.lora_weight()

    @property
    def bias(self):
        """Expose the wrapped linear bias for compatibility with existing code."""
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus the trainable low-rank update."""
        base = self.base_layer(x)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + update * self.scaling


def has_lora_parameters(module: nn.Module) -> bool:
    """Return True when a module tree already contains LoRA parameters."""
    return any(".lora_A" in name or ".lora_B" in name for name, _ in module.named_parameters())


def _matches_target(module_name: str, targets: list[str]) -> bool:
    """Match either an exact leaf name (e.g. ``qkv``) or a dotted suffix (e.g. ``attn.qkv``)."""
    leaf_name = module_name.rsplit(".", 1)[-1]
    for target in targets:
        if "." in target:
            if module_name == target or module_name.endswith(f".{target}"):
                return True
        elif leaf_name == target:
            return True
    return False


def _resolve_submodule(root: nn.Module, module_name: str):
    """Return the parent module and child key for a dotted submodule path."""
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def _replace_submodule(root: nn.Module, module_name: str, module: nn.Module):
    """Replace a submodule addressed by dotted path."""
    parent, child_name = _resolve_submodule(root, module_name)
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora_into_timm(
    timm_layer: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: list[str] | tuple[str, ...] | None = None,
    container_name: str | None = None,
    unit_indices: list[int] | tuple[int, ...] | None = None,
    target_root: nn.Module | None = None,
) -> dict:
    """Inject ``LoRALinear`` adapters into matching linear modules inside a wrapped timm backbone."""
    if not hasattr(timm_layer, "m"):
        raise TypeError("Expected a wrapped timm layer with attribute 'm'.")

    search_root = target_root if target_root is not None else timm_layer.m
    targets = list(target_modules or ("attn.qkv", "attn.proj"))
    selected_units = set(unit_indices) if unit_indices is not None else None
    replacements = []
    skipped_existing = []
    prefix_tokens = None
    if container_name and selected_units is not None:
        prefix_tokens = tuple(f"{container_name}.{idx}." for idx in sorted(selected_units))

    for module_name, module in search_root.named_modules():
        if isinstance(module, LoRALinear):
            skipped_existing.append(module_name)
            continue
        if not isinstance(module, nn.Linear):
            continue
        if prefix_tokens and not module_name.startswith(prefix_tokens):
            continue
        if not _matches_target(module_name, targets):
            continue
        replacements.append((module_name, module))

    for module_name, module in replacements:
        _replace_submodule(search_root, module_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    trainable_params = sum(
        param.numel()
        for name, param in timm_layer.m.named_parameters()
        if (".lora_A" in name or ".lora_B" in name) and param.requires_grad
    )
    return {
        "matched_modules": [name for name, _ in replacements],
        "existing_modules": skipped_existing,
        "trainable_params": trainable_params,
        "num_matched": len(replacements),
    }
