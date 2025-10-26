# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for Cascade R-CNN detection and segmentation models."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cv2", reason="OpenCV is required for Cascade model smoke tests")

from ultralytics.nn.tasks import CascadeMaskRCNNModel, CascadeRCNNDetectionModel
from ultralytics.models.cascade_rcnn.train import CascadeMaskRCNNTrainer, CascadeRCNNTrainer


@pytest.fixture(autouse=True)
def _disable_model_ema(monkeypatch):
    """Replace ModelEMA with a lightweight stub to simplify smoke training."""

    from ultralytics.utils import torch_utils
    from ultralytics.engine import trainer as trainer_mod

    class _NullEMA:
        def __init__(self, model) -> None:
            self.ema = model
            self.updates = 0

        def update(self, model) -> None:  # noqa: D401 - no-op
            return

        def update_attr(self, *args, **kwargs) -> None:  # noqa: D401 - no-op
            return

    monkeypatch.setattr(torch_utils, "ModelEMA", _NullEMA)
    monkeypatch.setattr(trainer_mod, "ModelEMA", _NullEMA)


class _DummyCascadeDataset(torch.utils.data.Dataset):
    """Minimal dataset emitting synthetic images and targets for cascade models."""

    def __init__(self, samples: int = 4, img_size: int = 128, mask_size: int | None = None):
        self.samples = samples
        self.img_size = img_size
        self.mask_size = mask_size

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str]]:
        img = torch.randint(0, 256, (3, self.img_size, self.img_size), dtype=torch.uint8)
        cls = torch.tensor([[index % 2]], dtype=torch.long)
        bboxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32)
        sample = {
            "img": img,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros((1, 1), dtype=torch.long),
            "im_file": f"dummy_{index}.jpg",
        }
        if self.mask_size:
            masks = torch.zeros(1, self.mask_size, self.mask_size, dtype=torch.uint8)
            masks[:, self.mask_size // 4 : -self.mask_size // 4, self.mask_size // 4 : -self.mask_size // 4] = 1
            sample["masks"] = masks
        return sample

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        batch: dict[str, torch.Tensor | list] = {}
        samples = [dict(sorted(s.items())) for s in samples]
        keys = samples[0].keys()
        for key in keys:
            values = [s[key] for s in samples]
            if key == "img":
                batch[key] = torch.stack(values, 0)
            elif key in {"cls", "bboxes", "masks"}:
                batch[key] = torch.cat(values, 0)
            elif key == "batch_idx":
                batch[key] = torch.cat([v + i for i, v in enumerate(values)], 0)
            else:
                batch[key] = values
        return batch


class _SimpleInfiniteLoader:
    """Lightweight DataLoader mimic with reset support for tests."""

    def __init__(self, dataset: _DummyCascadeDataset, batch_size: int = 2, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.num_workers = 0
        self._batch_indices: list[list[int]] = []
        self.reset()

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        for indices in self._batch_indices:
            samples = [self.dataset[i] for i in indices]
            yield self.dataset.collate_fn(samples)

    def reset(self) -> None:
        order = list(range(len(self.dataset)))
        if self.shuffle and len(order) > 1:
            random.shuffle(order)
        self._batch_indices = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]


class _DummyValidator:
    """Validator stub returning empty metrics."""

    def __init__(self):
        self.metrics = SimpleNamespace(keys=[])

    def __call__(self, *args, **kwargs):
        return {}


class _DummyCascadeDetectionTrainer(CascadeRCNNTrainer):
    """Trainer wired to dummy dataloaders for quick smoke training."""

    def get_dataset(self):
        return {"train": "dummy", "val": "dummy", "names": {0: "cls0", 1: "cls1"}, "nc": 2, "channels": 3}

    def get_dataloader(self, dataset_path, batch_size=2, rank=0, mode: str = "train"):
        dataset = _DummyCascadeDataset(samples=4, img_size=128)
        return _SimpleInfiniteLoader(dataset, batch_size=batch_size, shuffle=mode == "train")

    def get_validator(self):
        return _DummyValidator()

    def validate(self):
        return {}, 0.0

    def final_eval(self):
        return

    def save_model(self):
        return


class _DummyCascadeSegmentationTrainer(CascadeMaskRCNNTrainer):
    """Segmentation trainer using synthetic batches."""

    def get_dataset(self):
        return {"train": "dummy", "val": "dummy", "names": {0: "cls0", 1: "cls1"}, "nc": 2, "channels": 3}

    def get_dataloader(self, dataset_path, batch_size=2, rank=0, mode: str = "train"):
        dataset = _DummyCascadeDataset(samples=4, img_size=128, mask_size=64)
        return _SimpleInfiniteLoader(dataset, batch_size=batch_size, shuffle=mode == "train")

    def get_validator(self):
        return _DummyValidator()

    def validate(self):
        return {}, 0.0

    def final_eval(self):
        return

    def save_model(self):
        return


def _make_detection_batch(img_size: int = 256, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Generate a minimal detection batch compatible with v8 losses."""

    imgs = torch.rand(2, 3, img_size, img_size, device=device)
    cls = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.3, 0.7, 0.2, 0.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    return {"img": imgs, "cls": cls, "bboxes": bboxes, "batch_idx": batch_idx}


def _make_segmentation_batch(
    img_size: int = 256, mask_size: int = 64, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Generate a minimal segmentation batch with square instance masks."""

    batch = _make_detection_batch(img_size=img_size, device=device)
    masks = torch.zeros(2, mask_size, mask_size, device=device)
    masks[0, 16:48, 16:48] = 1.0
    masks[1, 8:40, 24:56] = 1.0
    batch["masks"] = masks
    return batch


def test_cascade_rcnn_detection_loss_smoke():
    """Ensure the Cascade R-CNN detection model produces finite training losses."""

    model = CascadeRCNNDetectionModel(
        cfg="ultralytics/cfg/models/cascade/cascade-rcnn.yaml", nc=2, ch=3, verbose=False
    )
    model.train()
    batch = _make_detection_batch()

    loss_vec, loss_items = model.loss(batch)
    assert loss_vec.shape == loss_items.shape == (3,)
    assert torch.isfinite(loss_vec).all()
    assert torch.isfinite(loss_items).all()

    model.zero_grad(set_to_none=True)
    loss_vec.sum().backward()


def test_cascade_mask_rcnn_segmentation_loss_smoke():
    """Ensure the Cascade Mask R-CNN segmentation model produces finite training losses."""

    model = CascadeMaskRCNNModel(
        cfg="ultralytics/cfg/models/cascade/cascade-mask-rcnn.yaml", nc=2, ch=3, verbose=False
    )
    model.train()
    batch = _make_segmentation_batch()

    loss_vec, loss_items = model.loss(batch)
    assert loss_vec.shape == loss_items.shape == (4,)
    assert torch.isfinite(loss_vec).all()
    assert torch.isfinite(loss_items).all()

    model.zero_grad(set_to_none=True)
    loss_vec.sum().backward()


def test_cascade_rcnn_dummy_training(tmp_path):
    """Run a short Cascade R-CNN training loop on dummy data."""

    overrides = {
        "model": "ultralytics/cfg/models/cascade/cascade-rcnn.yaml",
        "data": "dummy.yaml",
        "epochs": 1,
        "batch": 2,
        "imgsz": 128,
        "device": "cpu",
        "save": False,
        "plots": False,
        "val": False,
        "workers": 0,
        "project": str(tmp_path),
        "name": "cascade-det-dummy",
        "task": "detect",
        "amp": False,
    }
    trainer = _DummyCascadeDetectionTrainer(overrides=overrides)
    trainer.train()
    assert trainer.epoch >= 0


def test_cascade_mask_rcnn_dummy_training(tmp_path):
    """Run a short Cascade Mask R-CNN training loop on dummy data."""

    overrides = {
        "model": "ultralytics/cfg/models/cascade/cascade-mask-rcnn.yaml",
        "data": "dummy.yaml",
        "epochs": 1,
        "batch": 2,
        "imgsz": 128,
        "device": "cpu",
        "save": False,
        "plots": False,
        "val": False,
        "workers": 0,
        "project": str(tmp_path),
        "name": "cascade-seg-dummy",
        "task": "segment",
        "amp": False,
    }
    trainer = _DummyCascadeSegmentationTrainer(overrides=overrides)
    trainer.train()
    assert trainer.epoch >= 0
