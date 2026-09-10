import numpy as np

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils.metrics import ap_per_class, smooth


OBB_FITNESS_WEIGHTS = {
    "precision": 0.0,
    "recall": 0.05,
    "mAP50": 0.0,
    "mAP50_95": 0.25,
    "f1": 0.0,
    "f2": 0.70,
}

SEGMENT_FITNESS_WEIGHTS = {
    "precision": 0.0,
    "recall": 0.0,
    "mAP50": 0.0,
    "mAP50_95": 0.0,
    "f1": 0.0,
    "f2": 0.0,
    "mask_precision": 0.0,
    "mask_recall": 0.05,
    "mask_mAP50": 0.0,
    "mask_mAP50_95": 0.20,
    "mask_f1": 0.0,
    "mask_f2": 0.65,
    "dice": 0.10,
    "miou": 0.0,
    "boundary_f1": 0.0,
    "boundary_iou": 0.0,
}


def test_task_validators_use_their_recall_oriented_fitness_profiles() -> None:
    """OBB and segmentation validators must use separate normalized checkpoint-selection profiles."""
    obb_validator = OBBValidator(args=get_cfg(DEFAULT_CFG))
    segment_validator = SegmentationValidator(args=get_cfg(DEFAULT_CFG))

    assert obb_validator.metrics.fitness_weights == OBB_FITNESS_WEIGHTS
    assert segment_validator.metrics.fitness_weights == SEGMENT_FITNESS_WEIGHTS
    assert sum(obb_validator.metrics.fitness_weights.values()) == 1.0
    assert sum(segment_validator.metrics.fitness_weights.values()) == 1.0


def test_ap_per_class_reports_metrics_at_f2_optimal_threshold() -> None:
    """Point metrics used by fitness must come from max F2 when max F1 occurs at a different confidence."""
    is_true_positive = np.array([True] * 5 + [False] * 15 + [True] * 5)
    tp = is_true_positive[:, None]
    conf = np.linspace(1.0, 0.1, len(tp))
    pred_cls = np.zeros(len(tp))
    target_cls = np.zeros(10)

    results = ap_per_class(tp, conf, pred_cls, target_cls)
    selected_f2, f1_curve, f2_curve = results[5], results[10], results[11]
    best_f1_index = smooth(f1_curve.mean(0), 0.1).argmax()
    best_f2_index = smooth(f2_curve.mean(0), 0.1).argmax()

    assert best_f1_index != best_f2_index
    np.testing.assert_allclose(selected_f2, f2_curve[:, best_f2_index])
    assert not np.allclose(selected_f2, f2_curve[:, best_f1_index])
