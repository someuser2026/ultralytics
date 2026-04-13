from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest

LEGNET_MODELS = {
    "legnet-small.yaml": ("Detect", 3),
    "legnet-small-seg.yaml": ("Segment", 3),
    "legnet-small-obb.yaml": ("OBB", 3),
    "legnet-small-fcos.yaml": ("RotatedFCOS", 5),
    "legnet-small-fcos-smallobj.yaml": ("RotatedFCOS", 5),
}
LEGNET_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "legnet"
LEGNET_TEST_READY = find_spec("cv2") is not None and find_spec("torch") is not None


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize(("model_name", "expected"), LEGNET_MODELS.items())
def test_legnet_model_yaml_parses(model_name, expected):
    """Ensure LEGNet model definitions register cleanly and expose three task features."""
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    head_name, num_levels = expected
    model_cfg = yaml_model_load(LEGNET_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name
    assert model[-1].nl == num_levels


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_legnet_backbone_output_shapes():
    """Validate the LEGNet-small backbone returns the expected P2-P5 pyramid."""
    import torch

    from ultralytics.nn.modules import LWEGNet

    backbone = LWEGNet()
    outputs = backbone(torch.randn(1, 3, 640, 640))

    assert backbone.channels == [64, 128, 256, 512]
    assert len(outputs) == 4
    assert outputs[0].shape == (1, 64, 160, 160)
    assert outputs[1].shape == (1, 128, 80, 80)
    assert outputs[2].shape == (1, 256, 40, 40)
    assert outputs[3].shape == (1, 512, 20, 20)


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_legnet_analytic_kernels_match_reference_initialization():
    """LEGNet analytic filters should initialize from the reference Scharr, Gaussian, and LoG kernels."""
    import torch

    from ultralytics.nn.modules.legnet import LWEGNet, _gaussian_kernel, _log_kernel

    backbone = LWEGNet()
    stem = backbone.stem
    stage0_scharr = backbone.stages[0].blocks[0].edge

    scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32)
    scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=torch.float32)
    scharr_x = scharr_x.unsqueeze(0).unsqueeze(0).repeat(stage0_scharr.conv_x.weight.shape[0], 1, 1, 1)
    scharr_y = scharr_y.unsqueeze(0).unsqueeze(0).repeat(stage0_scharr.conv_y.weight.shape[0], 1, 1, 1)
    gaussian = _gaussian_kernel(9, 0.5).repeat(stem.gaussian.gaussian.weight.shape[0], 1, 1, 1)
    log_kernel = _log_kernel(7, 1.0).repeat(stem.log.log.weight.shape[0], 1, 1, 1)

    assert torch.allclose(stage0_scharr.conv_x.weight.detach(), scharr_x)
    assert torch.allclose(stage0_scharr.conv_y.weight.detach(), scharr_y)
    assert torch.allclose(stem.gaussian.gaussian.weight.detach(), gaussian)
    assert torch.allclose(stem.log.log.weight.detach(), log_kernel)

    assert stage0_scharr.conv_x.weight.requires_grad
    assert stage0_scharr.conv_y.weight.requires_grad
    assert stem.gaussian.gaussian.weight.requires_grad
    assert stem.log.log.weight.requires_grad


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_legnet_analytic_kernels_receive_gradients():
    """LEGNet Scharr, Gaussian, and LoG kernels should participate in backpropagation."""
    import torch

    from ultralytics.nn.modules.legnet import LWEGNet

    backbone = LWEGNet()
    outputs = backbone(torch.randn(2, 3, 128, 128))
    loss = sum(output.square().mean() for output in outputs)
    loss.backward()

    tracked_weights = [
        backbone.stages[0].blocks[0].edge.conv_x.weight,
        backbone.stages[0].blocks[0].edge.conv_y.weight,
        backbone.stem.gaussian.gaussian.weight,
        backbone.stem.log.log.weight,
    ]
    for weight in tracked_weights:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
