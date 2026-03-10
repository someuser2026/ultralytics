from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest

MAMBA_MODELS = (
    "Mamba-YOLO-T.yaml",
    "Mamba-YOLO-B.yaml",
    "Mamba-YOLO-L.yaml",
    "yolo-mamba-seg.yaml",
)
MAMBA_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "mamba-yolo"
MAMBA_TEST_READY = find_spec("cv2") is not None and find_spec("einops") is not None


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
