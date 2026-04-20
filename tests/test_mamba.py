from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

MAMBA_MODELS = (
    "Mamba-YOLO-T.yaml",
    "Mamba-YOLO-B.yaml",
    "Mamba-YOLO-L.yaml",
    "yolo-mamba-seg.yaml",
    "mamba-hrnet-obb.yaml",
    "mamba-hrnet-seg.yaml",
)
MAMBA_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "mamba-yolo"
MAMBA_TEST_READY = find_spec("cv2") is not None and find_spec("einops") is not None
MAMBA_BUILD_CASES = (
    ("Mamba-YOLO-L-obb-demo.yaml", "obb"),
    ("Mamba-YOLO-T.yaml", "detect"),
    ("mamba-hrnet-obb.yaml", "obb"),
    ("mamba-hrnet-seg.yaml", "segment"),
)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
@pytest.mark.parametrize("model_name", MAMBA_MODELS)
def test_mamba_model_yaml_parses(model_name):
    """Ensure Mamba-YOLO model definitions register cleanly with the research-branch parser."""
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    model_cfg = yaml_model_load(MAMBA_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
@pytest.mark.parametrize(("model_name", "task"), MAMBA_BUILD_CASES)
def test_mamba_model_construction_uses_build_only_cpu_fallback(model_name, task):
    """Mamba-YOLO models should finish CPU construction and resolve their head strides."""
    from ultralytics import YOLO

    model = YOLO(MAMBA_ROOT / model_name, task=task)

    expected_stride = torch.tensor([4.0, 8.0, 16.0, 32.0]) if model_name == "mamba-hrnet-seg.yaml" else torch.tensor(
        [8.0, 16.0, 32.0]
    )
    assert torch.equal(model.model.stride.cpu(), expected_stride)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
def test_mamba_cpu_forward_still_requires_build_context():
    """The CPU fallback should remain limited to stride probing during model construction."""
    from ultralytics import YOLO

    model = YOLO(MAMBA_ROOT / "Mamba-YOLO-T.yaml", task="detect")

    with pytest.raises(RuntimeError, match="CPU fallback is available only during model construction stride probing"):
        model.model(torch.zeros(1, 3, 64, 64))
