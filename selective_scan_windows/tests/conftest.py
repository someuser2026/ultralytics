"""Independent test setup: no repository-wide conftest or cleanup hooks."""

import importlib
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true", help="Fail instead of skipping Windows GPU qualification")


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda"):
        try:
            from selective_scan_windows.diagnostics import require_backend
            require_backend()
        except Exception as exc:
            raise pytest.UsageError(str(exc)) from exc


@pytest.fixture(scope="session")
def backend(request):
    import torch

    if sys.platform != "win32" or not torch.cuda.is_available():
        if request.config.getoption("--require-cuda"):
            pytest.fail("Native Windows CUDA is required")
        pytest.skip("Native Windows CUDA unavailable; this does not establish GPU correctness")
    # On Windows with CUDA, a missing/broken installation is a failure, never a skip.
    from selective_scan_windows.diagnostics import require_backend
    require_backend()
    return importlib.import_module("selective_scan_windows._C")
