"""Independent CUDA core with lazy loading so diagnostics work after failed builds."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["fwd", "bwd"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    import torch  # Load PyTorch's DLL dependencies before our .pyd.

    try:
        core = import_module("selective_scan_windows._C")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The selective-scan-windows CUDA binary could not be loaded. "
            "Build this package on native Windows using the toolchain in its README. "
            "No original selective_scan backend or CPU fallback will be loaded."
        ) from exc
    globals().update(fwd=core.fwd, bwd=core.bwd)
    return globals()[name]
