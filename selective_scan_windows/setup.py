"""Build only our vendored core. Never import the original extension or setup.py."""

import os
from importlib import metadata
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "csrc/selective_scan/cus/selective_scan.cpp",
    "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
    "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
]


def cuda_extension():
    """Defer toolchain checks until compilation, so sdists need no CUDA compiler."""
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        raise RuntimeError("selective-scan-windows builds require native Windows x64 Python.")
    try:
        metadata.distribution("selective_scan")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("The original selective_scan is installed. Build in a separate Windows environment.")

    import torch
    from packaging.version import Version
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

    if Version(torch.__version__).base_version != "2.8.0" or torch.version.cuda != "12.6":
        raise RuntimeError("Install torch==2.8.0 from the cu126 index before building (see README).")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc.exe").is_file():
        raise RuntimeError("CUDA Toolkit 12.6 with nvcc.exe is required; check CUDA_PATH.")
    if shutil.which("cl") is None:
        raise RuntimeError("MSVC cl.exe is missing. Use the VS 2022 x64 developer shell with toolset 14.38.")
    if shutil.which("ninja") is None:
        raise RuntimeError("Install ninja in this Python environment before building.")
    nvcc = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc.exe"), "--version"], text=True)
    if "release 12.6," not in nvcc:
        raise RuntimeError("This initial build targets CUDA Toolkit 12.6; nvcc reports:\n" + nvcc)
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and not torch.cuda.is_available():
        raise RuntimeError("No visible CUDA GPU. Set TORCH_CUDA_ARCH_LIST to the target GPU capability.")

    return CUDAExtension(
        "selective_scan_windows._C",
        sources=SOURCES,
        include_dirs=[str(ROOT / "csrc/selective_scan")],
        define_macros=[("NOMINMAX", "1")],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17"],
            "nvcc": [
                "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "--use_fast_math", "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ],
        },
    )


class WindowsBuildExtension(build_ext):
    """Delegate actual compilation to PyTorch, keeping metadata commands portable."""

    def run(self):
        extension = cuda_extension()
        from torch.utils.cpp_extension import BuildExtension

        # Setuptools must know this is a binary distribution even for metadata-only
        # commands. Replace the placeholder with the CUDA-aware extension here.
        self.distribution.ext_modules = [extension]
        command = BuildExtension(self.distribution)
        command.initialize_options()
        for name in ("build_lib", "build_temp", "inplace", "debug", "force", "parallel", "plat_name"):
            setattr(command, name, getattr(self, name))
        command.ensure_finalized()
        command.run()


if __name__ == "__main__":
    from setuptools import Extension

    setup(
        ext_modules=[Extension("selective_scan_windows._C", sources=SOURCES)],
        cmdclass={"build_ext": WindowsBuildExtension},
    )
