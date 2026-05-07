from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from ultralytics.data.augment import Format, Mosaic, RandomPerspective
from ultralytics.utils.instance import Instances


def _empty_instances() -> Instances:
    return Instances(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 1000, 2), dtype=np.float32),
        bbox_format="xywh",
        normalized=False,
    )


def test_random_perspective_uses_zero_border_fill() -> None:
    transform = RandomPerspective(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, border=(1, 1))
    labels = {
        "img": np.full((2, 2, 3), 255, dtype=np.uint8),
        "cls": np.zeros((0, 1), dtype=np.float32),
        "instances": _empty_instances(),
    }

    result = transform(labels)

    assert result["img"].shape == (4, 4, 3)
    assert np.all(result["img"][0] == 0)
    assert np.all(result["img"][-1] == 0)
    assert np.all(result["img"][:, 0] == 0)
    assert np.all(result["img"][:, -1] == 0)


def test_mosaic_uses_zero_canvas_fill() -> None:
    random.seed(0)
    transform = Mosaic(SimpleNamespace(cache=None), imgsz=4, p=1.0, n=4)

    def label(img: np.ndarray, name: str) -> dict:
        return {
            "img": img,
            "im_file": name,
            "ori_shape": img.shape[:2],
            "resized_shape": img.shape[:2],
            "cls": np.zeros((0, 1), dtype=np.float32),
            "instances": _empty_instances(),
        }

    labels = label(np.full((2, 2, 3), 255, dtype=np.uint8), "a.jpg")
    labels["mix_labels"] = [
        label(np.full((2, 2, 3), 200, dtype=np.uint8), "b.jpg"),
        label(np.full((2, 2, 3), 150, dtype=np.uint8), "c.jpg"),
        label(np.full((2, 2, 3), 100, dtype=np.uint8), "d.jpg"),
    ]

    result = transform._mosaic4(labels)

    assert result["img"].shape == (8, 8, 3)
    assert result["img"].min() == 0


def test_format_img_bgr_flip_keeps_channel_scales_contiguous(monkeypatch) -> None:
    """Regression test for PyTorch rejecting negative-stride channel-scale views."""
    monkeypatch.setattr("ultralytics.data.augment.random.uniform", lambda *_args: 1.0)
    formatter = Format(channel_scale_factors={1: 1.0, 2: 2.0, 3: 4.0}, bgr=0.0)
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    result = formatter._format_img(img)

    expected = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1]).astype(np.float32)
    expected /= np.array([4.0, 2.0, 1.0], dtype=np.float32)[:, None, None]
    np.testing.assert_allclose(result.numpy(), expected)
