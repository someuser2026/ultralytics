"""Structural and numerical checks for the reference-aligned HRVMamba YAML model."""

from collections import Counter
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
import torch.nn as nn


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "ultralytics"
    / "cfg"
    / "models"
    / "mamba-yolo"
    / "mamba-hrnet-seg-dvss.yaml"
)
MAMBA_TEST_READY = find_spec("cv2") is not None and find_spec("einops") is not None


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import HRVMamba blocks")
def test_hrvmamba_yaml_has_fixed_reference_structure():
    """The explicit YAML must retain the base reference widths, depths, and stochastic-depth schedule."""
    from ultralytics.nn.modules import DVSSBlock, HRFusion
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    config = yaml_model_load(CONFIG)
    assert "scales" not in config
    assert config["depth_multiple"] == 1.0
    assert config["width_multiple"] == 1.0

    model, _, backbone, head = parse_model(deepcopy(config), ch=3, verbose=False)
    blocks = [module for module in model.modules() if isinstance(module, DVSSBlock)]
    fusions = [module for module in model.modules() if isinstance(module, HRFusion)]

    assert len(model) == 109
    assert len(backbone) == 90
    assert len(head) == 19
    assert len(blocks) == 44
    assert Counter(block.hidden_dim for block in blocks) == {80: 14, 160: 14, 320: 12, 640: 4}
    assert Counter(fusion.target_index for fusion in fusions) == {0: 7, 1: 7, 2: 6, 3: 2}
    assert all(isinstance(block.proj_conv, nn.Identity) for block in blocks)
    assert all(block.in_channels == block.hidden_dim for block in blocks)
    assert all(block.norm0.norm.eps == pytest.approx(1e-5) for block in blocks)

    reference_dpr = torch.linspace(0, 0.15, 14).tolist()
    branch_counts = (2, 3, 3, 3, 3, 4, 4)
    expected_drops = []
    for pair_index, num_branches in enumerate(branch_counts):
        pair = reference_dpr[2 * pair_index:2 * pair_index + 2]
        expected_drops.extend(pair * num_branches)
    assert [block.drop_path.drop_prob for block in blocks] == pytest.approx(expected_drops)


def test_hr_fusion_adds_all_aligned_branches_and_backpropagates():
    """Each YAML fusion primitive must align every source and combine them by summation."""
    from ultralytics.nn.modules import HRFusion

    channels = [80, 160, 320, 640]
    spatial_sizes = [16, 8, 4, 2]
    for target_index, (target_channels, target_size) in enumerate(zip(channels, spatial_sizes)):
        fusion = HRFusion(channels, target_index).eval()
        inputs = [
            (torch.ones(1, channel, size, size) * 2).requires_grad_()
            for channel, size in zip(channels, spatial_sizes)
        ]
        output = fusion(inputs)
        assert output.shape == (1, target_channels, target_size, target_size)
        output.sum().backward()
        assert all(source.grad is not None for source in inputs)
        assert all(torch.isfinite(source.grad).all() for source in inputs)
        assert all(torch.count_nonzero(source.grad) > 0 for source in inputs)


def test_dvss_residual_path_is_an_exact_identity_when_branches_are_zero():
    """DVSS must not normalize, project, or activate the shortcut itself."""
    from ultralytics.nn.modules import DVSSBlock

    block = DVSSBlock(8, 8, ssm_ratio=0, mlp_ratio=0)
    for conv in block.esinb_convs:
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
    source = torch.randn(2, 8, 5, 5)
    assert torch.equal(block(source), source)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import HRVMamba blocks")
def test_dvss_all_trainable_parameters_receive_finite_nonzero_gradients():
    """A complete DVSS residual block must propagate useful gradients through every trainable parameter."""
    from ultralytics.nn.modules import DVSSBlock
    from ultralytics.nn.tasks import _enable_mamba_cpu_fallback_for_build

    block = DVSSBlock(8, 8, drop_path=0.0)
    source = torch.randn(2, 8, 4, 4, requires_grad=True)
    with _enable_mamba_cpu_fallback_for_build(block):
        output = block(source)
    output.square().mean().backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} received a non-finite gradient"
        assert torch.count_nonzero(parameter.grad) > 0, f"{name} received an all-zero gradient"


def test_hr_convolution_preserves_reference_syncbn_defaults():
    """Ultralytics initialization must not replace the HRVMamba SyncBN contract."""
    from ultralytics.nn.modules import HRConv

    layer = HRConv(3, 64, 3, 2)
    assert isinstance(layer.norm, nn.SyncBatchNorm)
    assert layer.norm.eps == pytest.approx(1e-5)
    assert layer.norm.momentum == pytest.approx(0.1)
