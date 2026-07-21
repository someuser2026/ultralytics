# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import torch

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.models.yolo.classify import train as classify_train_module
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR


def test_func(*args):  # noqa
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


class _DummyDataset:
    def __init__(self, batch_count):
        self.batch_count = batch_count

    def __len__(self):
        return self.batch_count


class _DummyLoader:
    num_workers = 0

    def __init__(self, batch_count):
        self.batch_count = batch_count
        self.dataset = _DummyDataset(batch_count)

    def __iter__(self):
        for _ in range(self.batch_count):
            yield {"img": torch.zeros(1, 3, 32, 32), "cls": torch.zeros(1)}

    def __len__(self):
        return self.batch_count

    def reset(self):
        pass


class _DummyStopper:
    possible_stop = False

    def __call__(self, *args, **kwargs):
        return False


class _DummyEMA:
    def __init__(self, model):
        self.ema = model
        self.updates = 0

    def update(self, model):
        self.updates += 1

    def update_attr(self, model, include=None):
        pass


class _DummyScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {}


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return self.weight.square(), torch.tensor([1.0])


class _TimeLimitTrainer(BaseTrainer):
    def __init__(self, tmp_path, epochs, time_hours, explicit_epoch_limit, batch_count=2):
        self.args = SimpleNamespace(
            warmup_epochs=0,
            time=time_hours,
            imgsz=32,
            close_mosaic=0,
            nbs=1,
            warmup_bias_lr=0.1,
            warmup_momentum=0.8,
            momentum=0.937,
            compile=False,
            val=False,
            save=False,
            plots=False,
            cos_lr=False,
            lrf=0.01,
            epochs=epochs,
        )
        self.device = torch.device("cpu")
        self.save_dir = tmp_path
        self.csv = tmp_path / "results.csv"
        self.callbacks = defaultdict(list)
        self.batch_size = 1
        self.epochs = epochs
        self.start_epoch = 0
        self.plot_idx = []
        self.world_size = 0
        self._explicit_epoch_limit = explicit_epoch_limit
        self.metrics = {}
        self.best_fitness = 0.0
        self.fitness = 0.0
        self.loss = None
        self.tloss = None
        self.stop = False
        self.freeze_layer_names = []
        self.unfreeze_layer_names = []
        self.batch_count = batch_count
        self.batches_seen = 0
        self.validate_calls = 0
        self.scheduler_epochs = []

    def _setup_train(self):
        self.model = _DummyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self._setup_scheduler()
        self.stopper = _DummyStopper()
        self.train_loader = _DummyLoader(self.batch_count)
        self.scaler = _DummyScaler()
        self.ema = _DummyEMA(self.model)
        self.accumulate = 1
        self.amp = False

    def _setup_scheduler(self):
        super()._setup_scheduler()
        self.scheduler_epochs.append(self.epochs)

    def progress_string(self):
        return ""

    def preprocess_batch(self, batch):
        self.batches_seen += 1
        return batch

    def validate(self):
        self.validate_calls += 1
        return {}, 0.0

    def save_metrics(self, metrics):
        pass

    def save_model(self):
        pass

    def final_eval(self):
        pass

    def _restore_timm_train_modes(self):
        pass


class _TimeSequence:
    def __init__(self, *values):
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self):
        return next(self.values, self.last)


class _DummyTQDM:
    def __init__(self, iterable, total=None):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, description):
        pass


def test_time_and_epochs_stop_at_first_limit(monkeypatch, tmp_path):
    """Explicit time and epoch limits should finish the current epoch, then stop at the first limit hit."""
    trainer = _TimeLimitTrainer(tmp_path, epochs=3, time_hours=1, explicit_epoch_limit=True, batch_count=2)
    monkeypatch.setattr(trainer_module.time, "time", _TimeSequence(0, 0, 100, 200, 300, 300, 4001, 4002, 4003, 4004, 4005))
    monkeypatch.setattr(trainer_module, "TQDM", _DummyTQDM)
    trainer._do_train()

    assert trainer.epoch == 1
    assert trainer.epochs == 3
    assert trainer.batches_seen == 4
    assert trainer.validate_calls == 1
    assert trainer.time_limit_reached
    assert trainer._should_preserve_last_checkpoint()


def test_time_only_training_keeps_current_epoch_estimation(monkeypatch, tmp_path):
    """Time-only training should still stretch beyond the initial epoch cap, but stop after the current epoch."""
    trainer = _TimeLimitTrainer(tmp_path, epochs=2, time_hours=1, explicit_epoch_limit=False, batch_count=2)
    monkeypatch.setattr(trainer_module.time, "time", _TimeSequence(0, 0, 100, 200, 300, 300, 4001, 4002, 4003, 4004, 4005))
    monkeypatch.setattr(trainer_module, "TQDM", _DummyTQDM)
    trainer._do_train()

    assert trainer.epoch == 1
    assert trainer.scheduler_epochs[1] > trainer.scheduler_epochs[0]
    assert trainer.batches_seen == 4
    assert trainer.validate_calls == 1
    assert trainer.time_limit_reached
    assert not trainer._should_preserve_last_checkpoint()


def test_preserve_last_checkpoint_requires_unfinished_explicit_epoch_target():
    """Only a timed stop before an explicit epoch target should preserve last.pt."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(time=1)
    trainer._explicit_epoch_limit = True
    trainer.time_limit_reached = True
    trainer.epochs = 3

    trainer.epoch = 1
    assert trainer._should_preserve_last_checkpoint()

    trainer.epoch = 2
    assert not trainer._should_preserve_last_checkpoint()

    trainer.epoch = 1
    trainer.time_limit_reached = False
    assert not trainer._should_preserve_last_checkpoint()

    trainer.time_limit_reached = True
    trainer._explicit_epoch_limit = False
    assert not trainer._should_preserve_last_checkpoint()


class _DummyFinalValidator:
    def __init__(self):
        self.args = SimpleNamespace(plots=False, compile=True, data=None)
        self.calls = []

    def __call__(self, model):
        self.calls.append(model)
        return {"fitness": 1.0}


def _make_final_eval_trainer(trainer_cls, tmp_path, preserve):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.last.write_bytes(b"last")
    trainer.best.write_bytes(b"best")
    trainer.args = SimpleNamespace(time=1, plots=False, data="data.yaml")
    trainer._explicit_epoch_limit = True
    trainer.time_limit_reached = preserve
    trainer.epoch = 0 if preserve else 1
    trainer.epochs = 2
    trainer.validator = _DummyFinalValidator()
    trainer.callbacks = defaultdict(list)
    trainer.metrics = {}
    trainer.read_results_csv = lambda: {"epoch": [1], "metric": [0.5]}
    return trainer


def test_final_eval_preserves_resumable_last_but_finalizes_best(monkeypatch, tmp_path):
    """An incomplete timed run should preserve last.pt and still finalize best.pt with current results."""
    trainer = _make_final_eval_trainer(BaseTrainer, tmp_path, preserve=True)
    strip_calls = []

    def fake_strip(path, updates=None):
        strip_calls.append((path, updates))
        return {"train_results": {"old": []}}

    monkeypatch.setattr(trainer_module, "strip_optimizer", fake_strip)
    trainer.final_eval()

    assert [call[0] for call in strip_calls] == [trainer.best]
    assert strip_calls[0][1] == {"train_results": {"epoch": [1], "metric": [0.5]}}
    assert trainer.validator.calls == [trainer.best]


def test_final_eval_strips_last_after_epoch_target(monkeypatch, tmp_path):
    """A completed epoch target should retain the existing finalization behavior."""
    trainer = _make_final_eval_trainer(BaseTrainer, tmp_path, preserve=False)
    strip_calls = []

    def fake_strip(path, updates=None):
        strip_calls.append((path, updates))
        return {"train_results": {"epoch": [1, 2]}}

    monkeypatch.setattr(trainer_module, "strip_optimizer", fake_strip)
    trainer.final_eval()

    assert [call[0] for call in strip_calls] == [trainer.last, trainer.best]
    assert strip_calls[1][1] == {"train_results": {"epoch": [1, 2]}}


def test_classification_final_eval_preserves_last(monkeypatch, tmp_path):
    """Classification should follow the same timed-checkpoint preservation rule."""
    trainer = _make_final_eval_trainer(classify.ClassificationTrainer, tmp_path, preserve=True)
    strip_calls = []
    monkeypatch.setattr(classify_train_module, "strip_optimizer", lambda path: strip_calls.append(path))

    trainer.final_eval()

    assert strip_calls == [trainer.best]
    assert trainer.validator.calls == [trainer.best]


def test_export():
    """Test model exporting functionality by adding a callback and verifying its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo11n.yaml").model)
    YOLO(f)(ASSETS)  # exported model inference


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_segment():
    """Test image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo11n-seg.yaml",
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo11n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo11n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"
