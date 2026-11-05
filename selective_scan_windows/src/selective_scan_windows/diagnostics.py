"""Report prerequisites and verify the installed compatibility import without fallbacks."""

import importlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import struct
import sys


def collect():
    """Return a JSON-ready report. Never import an unexpected legacy backend."""
    from . import __version__

    report = {"package_version": __version__, "python": sys.version, "platform": platform.platform(),
              "errors": [], "status": "failed"}
    errors = report["errors"]
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        errors.append("Native Windows x64 is required for GPU qualification.")
    try:
        original = metadata.distribution("selective_scan")
    except metadata.PackageNotFoundError:
        pass
    else:
        errors.append(f"Conflicting original selective_scan distribution at {original.locate_file('')}. Use a separate environment.")

    expected_shim = Path(__file__).resolve().parent.parent / "selective_scan_cuda_core.py"
    shim_spec = util.find_spec("selective_scan_cuda_core")
    report["compatibility_module"] = shim_spec.origin if shim_spec else None
    shim_ok = bool(shim_spec and shim_spec.origin and Path(shim_spec.origin).resolve() == expected_shim)
    if not shim_ok:
        errors.append("selective_scan_cuda_core does not resolve to this package's compatibility module.")

    try:
        import torch

        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.6":
            errors.append("The target runtime is torch 2.8.0 with CUDA 12.6.")
        if not torch.cuda.is_available():
            errors.append("PyTorch cannot access a CUDA GPU.")
        else:
            report["gpus"] = [
                {"index": i, "name": torch.cuda.get_device_name(i),
                 "capability": list(torch.cuda.get_device_capability(i)),
                 "total_memory": torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())
            ]
        # Import only our fully qualified native module, regardless of shim origin.
        core = importlib.import_module("selective_scan_windows._C")
        report["binary"] = core.__file__
        if Path(core.__file__).resolve().parent != Path(__file__).resolve().parent:
            errors.append("Native binary is outside this package.")
        if shim_ok and not errors:
            shim = importlib.import_module("selective_scan_cuda_core")
            if shim.fwd is not core.fwd or shim.bwd is not core.bwd:
                errors.append("Compatibility functions are not the Windows extension's functions.")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if not errors:
        report["status"] = "passed"
    return report


def require_backend():
    report = collect()
    if report["errors"]:
        raise RuntimeError("Windows backend verification failed:\n" + "\n".join(report["errors"]))
    return report


def main():
    report = collect()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
