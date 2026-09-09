import re
from pathlib import Path

import yaml

from ultralytics.nn.tasks import yaml_model_load


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ultralytics" / "cfg" / "models"
BENCHMARK_SUBMITTER = ROOT / "jobs" / "train" / "hpc" / "bash_scripts_joint" / "submit_benchmark_models.sh"


def _load(path: Path) -> dict:
    """Load a model YAML for architecture-level comparisons."""
    return yaml.safe_load(path.read_text())


def _benchmark_config_paths() -> tuple[Path, ...]:
    """Extract model YAML paths directly from the shared benchmark job table."""
    paths = re.findall(r"\|(ultralytics/cfg/models/[^|]+\.yaml)\|", BENCHMARK_SUBMITTER.read_text())
    return tuple(ROOT / path for path in paths)


def test_benchmark_yolox_configs_keep_original_architectures() -> None:
    """The one-class x configs must differ from stock YOLO only in configuration metadata."""
    cases = (
        (MODEL_ROOT / "11/yolo11-obb.yaml", MODEL_ROOT / "11/yolo11x-obb-1cls.yaml"),
        (
            MODEL_ROOT / "11/yolo11-seg.yaml",
            MODEL_ROOT / "timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/yolo11x-yolo11x-segment.yaml",
        ),
        (MODEL_ROOT / "12/yolo12-obb.yaml", MODEL_ROOT / "12/yolo12x-obb-1cls.yaml"),
        (
            MODEL_ROOT / "12/yolo12-seg.yaml",
            MODEL_ROOT / "timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml",
        ),
        (MODEL_ROOT / "26/yolo26-obb.yaml", MODEL_ROOT / "26/yolo26x-obb-1cls.yaml"),
        (MODEL_ROOT / "26/yolo26-seg.yaml", MODEL_ROOT / "26/yolo26x-seg-1cls.yaml"),
    )

    for stock_path, benchmark_path in cases:
        stock = _load(stock_path)
        benchmark = _load(benchmark_path)
        assert benchmark["nc"] == 1
        assert benchmark["scales"] == {"x": stock["scales"]["x"]}
        assert benchmark["backbone"] == stock["backbone"]
        assert benchmark["head"] == stock["head"]


def test_benchmark_mamba_yolo_models_share_official_b_architecture() -> None:
    """Mamba-YOLO benchmark variants must share the B backbone and neck, excluding the terminal task head."""
    reference = _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-B.yaml")
    variants = {
        "OBB": _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-B-obb.yaml"),
        "Segment": _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-B-seg.yaml"),
    }

    for terminal, variant in variants.items():
        assert variant["nc"] == 1
        assert variant["scales"] == {"B": reference["scales"]["B"]}
        assert variant["backbone"] == reference["backbone"]
        assert variant["head"][:-1] == reference["head"][:-1]
        assert variant["head"][-1][2] == terminal


def test_benchmark_resnet_models_use_scratch_timm_backbones() -> None:
    """Every benchmark ResNet variant should be scratch-initialized and fully trainable."""
    configs = (
        MODEL_ROOT / "rcnn/rotated_faster_rcnn_r50_fpn_le90_smallobj.yaml",
        MODEL_ROOT / "rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml",
        MODEL_ROOT / "rcnn/mask_rcnn_r50_fpn_smallobj.yaml",
        MODEL_ROOT / "rcnn/pointrend_rcnn_r50_fpn_smallobj.yaml",
        MODEL_ROOT / "fcos/rotated_fcos_r50_fpn_le90.yaml",
    )

    for config_path in configs:
        backbone_layer = _load(config_path)["backbone"][0]
        assert backbone_layer[:3] == [-1, 1, "Timm"]
        args = backbone_layer[3]
        assert args[0] == "resnet50"
        assert args[1] is False  # pretrained
        assert args[8] is False  # freeze_stem
        assert args[9] is False  # freeze


def test_benchmark_transformers_are_scratch_except_dinov3() -> None:
    """Swin/HRNet Mask2Former backbones train from scratch while DINOv3 retains its required pretrained weights."""
    scratch_configs = (
        MODEL_ROOT / "transformer/mask2former-swin-timm-seg.yaml",
        MODEL_ROOT / "transformer/mask2former-hrnet-w32-timm-seg.yaml",
    )
    dino_configs = (
        MODEL_ROOT
        / "timm/obb/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
        "dinov3_7_12_17_22-yolo11x-obb.yaml",
        MODEL_ROOT
        / "timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
        "dinov3_7_12_17_22-yolo11x-segment.yaml",
    )

    for config_path in scratch_configs:
        assert _load(config_path)["backbone"][0][3][1] is False
    for config_path in dino_configs:
        assert _load(config_path)["backbone"][0][3][1] is True


def test_all_benchmark_timm_backbones_are_scratch_except_dinov3() -> None:
    """Inspect every config referenced by the launcher so new benchmark entries cannot silently load weights."""
    saw_dino = False
    for config_path in _benchmark_config_paths():
        config = _load(config_path)
        for layer in config.get("backbone", []) + config.get("head", []):
            if layer[2] != "Timm":
                continue
            is_dino = "dinov3" in str(layer[3][0]).lower()
            saw_dino |= is_dino
            assert layer[3][1] is is_dino
    assert saw_dino


def test_benchmark_submitter_uses_faithful_yolo_and_mamba_configs() -> None:
    """Both dataset launch paths share this submitter, so keep its canonical config references explicit."""
    submitter = BENCHMARK_SUBMITTER.read_text()
    expected = (
        "ultralytics/cfg/models/11/yolo11x-obb-1cls.yaml",
        "ultralytics/cfg/models/12/yolo12x-obb-1cls.yaml",
        "ultralytics/cfg/models/26/yolo26x-obb-1cls.yaml",
        "ultralytics/cfg/models/26/yolo26x-seg-1cls.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/"
        "yolo11x-yolo11x-segment.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B-obb.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B-seg.yaml",
    )
    replaced = (
        "yolo11x-augfpn_512c-obb.yaml",
        "yolo12x-augfpn_512c-obb.yaml",
        "ultralytics/cfg/models/26/yolo26-obb.yaml",
        "ultralytics/cfg/models/26/yolo26-seg.yaml",
        "ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-seg.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
    )

    assert all(path in submitter for path in expected)
    assert all(path not in submitter for path in replaced)


def test_benchmark_dino_and_mamba_hr_task_pairs_share_necks() -> None:
    """Each OBB/segment pair must have the same feature-building neck before its task-specific head."""
    cases = (
        (
            MODEL_ROOT
            / "timm/obb/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
            "dinov3_7_12_17_22-yolo11x-obb.yaml",
            MODEL_ROOT
            / "timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
            "dinov3_7_12_17_22-yolo11x-segment.yaml",
            [25, 28, 31],
        ),
        (
            MODEL_ROOT
            / "timm/obb/final/yolo_neck/transformer/dinov3_7_12_17_22/2cls/"
            "dinov3_7_12_17_22-yolo11x-obb.yaml",
            MODEL_ROOT
            / "timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/2cls/"
            "dinov3_7_12_17_22-yolo11x-segment.yaml",
            [25, 28, 31],
        ),
        (
            MODEL_ROOT / "mamba-yolo/mamba-hrnet-obb.yaml",
            MODEL_ROOT / "mamba-yolo/mamba-hrnet-seg.yaml",
            [70, 73, 76, 79],
        ),
    )

    for obb_path, segment_path, expected_levels in cases:
        obb = _load(obb_path)
        segment = _load(segment_path)
        assert obb["nc"] == segment["nc"]
        if "mamba-hrnet" in obb_path.name:
            assert obb["scale"] == segment["scale"] == "B"
        assert obb["backbone"] == segment["backbone"]
        assert obb["head"][:-1] == segment["head"][:-1]
        assert obb["head"][-1][0] == segment["head"][-1][0] == expected_levels
        assert obb["head"][-1][2] == "OBB"
        assert segment["head"][-1][2] == "Segment"


def test_mamba_hr_explicit_b_scale_survives_yaml_loading() -> None:
    """The model loader must not replace an explicit nonstandard scale with an empty filename inference."""
    for name in ("mamba-hrnet-obb.yaml", "mamba-hrnet-seg.yaml"):
        config = yaml_model_load(MODEL_ROOT / "mamba-yolo" / name)
        assert config["scale"] == "B"
