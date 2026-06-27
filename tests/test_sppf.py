from io import BytesIO

import torch

from ultralytics.nn.modules import SPPF


def torch_load(buffer: BytesIO):
    """Load test modules across PyTorch versions with different weights_only defaults."""
    buffer.seek(0)
    try:
        return torch.load(buffer, weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer)


def test_sppf_forward_restores_legacy_attrs():
    module = SPPF(c1=16, c2=32).eval()
    x = torch.randn(1, 16, 8, 8)

    with torch.no_grad():
        expected = module(x)

    del module.n
    del module.add

    with torch.no_grad():
        actual = module(x)

    assert module.n == 3
    assert module.add is False
    assert torch.allclose(actual, expected)


def test_sppf_pickle_load_restores_legacy_attrs():
    module = SPPF(c1=16, c2=32).eval()
    del module.n
    del module.add

    buffer = BytesIO()
    torch.save(module, buffer)

    loaded = torch_load(buffer).eval()
    x = torch.randn(1, 16, 8, 8)

    assert loaded.n == 3
    assert loaded.add is False
    with torch.no_grad():
        assert loaded(x).shape == (1, 32, 8, 8)


def test_sppf_keeps_current_configured_behavior():
    module = SPPF(c1=16, c2=16, n=2, shortcut=True).eval()
    x = torch.randn(1, 16, 8, 8)

    assert module.n == 2
    assert module.add is True
    with torch.no_grad():
        assert module(x).shape == x.shape
