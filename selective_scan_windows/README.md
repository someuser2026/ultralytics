# Native Windows selective scan

This is a self-contained copy and Windows adaptation of the **core** CUDA
selective-scan implementation used by this custom Ultralytics fork. It includes
forward and backward kernels. It does not import the original `selective_scan`
distribution, read its source at build time, or change existing model files.

## Validation status

The target toolchain is Windows 11 x64, Python 3.13.5, PyTorch 2.8.0 + cu126,
CUDA Toolkit 12.6 Update 3, and Visual Studio 2022 MSVC 19.38 (toolset 14.38).
The source has portable packaging/import/reference tests; **native Windows
compilation, numerical GPU tests, and actual checkpoint inference must pass on
the PC before this package is considered qualified**. Mac CPU tests cannot
establish Windows or CUDA correctness. Other toolchains are not qualified.

There is no CPU fallback. Only the core backend is included; `oflex`, `ndstate`,
and `nrow` extensions are not required by the active model wrapper and are not
built. Full model training qualification is separate from kernel gradient tests;
any training smoke test must run two full epochs under the repository rules.

## Compatibility interface

The distribution is named `selective-scan-windows`. Its native binary is
`selective_scan_windows._C`. A new `selective_scan_cuda_core.py` compatibility
module exposes that binary's `fwd` and `bwd` functions under the import name
already used by the model. Existing model code, class names, parameters, and
checkpoint formats remain unchanged.

Use a Windows environment **without the original `selective_scan` package**.
Both implementations provide the same compatibility import name; installing
both can cause Python to choose an older `.pyd` ahead of this module. Do not
install this package in an HPC environment. No installation step below removes
or replaces the original distribution automatically.

```python
from selective_scan_windows import fwd, bwd
# Identical function objects exposed for the unchanged Ultralytics wrapper:
from selective_scan_cuda_core import fwd, bwd
```

The low-level positional API is unchanged:

```text
fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows) -> [out, x]
bwd(u, delta, A, B, C, D, delta_bias, dout, x, delta_softplus, nrows)
    -> [du, ddelta, dA, dB, dC, dD, ddelta_bias]
```

`u` and `delta` have shape `[batch, channels, length]`; `A` is
`[channels, state]`; `B` and `C` are `[batch, groups, state, length]`.
Channels must be divisible by groups, state must be at most 256, and the last
input dimension must be contiguous (or have size 1). Inputs use FP32, FP16,
or BF16; `A`, optional `D`, and optional delta bias use FP32. All tensors must
be on the same CUDA device. The core executes one row regardless of the legacy
`nrows` argument, matching the original; the active wrapper passes `1`.
The FP32 workspace is `[batch, channels, ceil(length/2048), 2*state]`.
Optional absent gradients are returned as `None`.

## Windows installation

These commands assume the Windows prerequisites and `ultralytics_contrib`
environment from the PC setup guide are installed. The CUDA toolkit is a
separate installation from PyTorch's CUDA runtime. See the official
[CUDA Windows guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-microsoft-windows/index.html)
and [PyTorch extension guide](https://docs.pytorch.org/docs/2.8/cpp_extension.html).

Use **x64 Native Tools Command Prompt for VS 2022** for all commands in this
section (Command Prompt syntax, not PowerShell). Adjust `SSW_REPO` and the
Visual Studio installation path if necessary. `micromamba run` selects the
existing Python environment without relying on shell activation.

```bat
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.38
set "SSW_REPO=%USERPROFILE%\rip-work\repos\ultralytics"
set "MAX_JOBS=2"
cl
nvcc --version
nvidia-smi
micromamba run -n ultralytics_contrib python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
micromamba run -n ultralytics_contrib python -m pip show selective_scan
```

`cl` should report 19.38; its missing-source-file error is normal for this
version check. `nvcc` must report 12.6; PyTorch must report 2.8.0 and CUDA 12.6.
The final `pip show` should say the original package was not found. If it is
installed, use a separate environment instead of proceeding.

Install prerequisites, then build only this package:

```bat
micromamba run -n ultralytics_contrib python -m pip install setuptools wheel ninja packaging pytest numpy
cd /d "%SSW_REPO%\selective_scan_windows"
micromamba run -n ultralytics_contrib python -m pip install --no-build-isolation --no-deps -v .
micromamba run -n ultralytics_contrib python -m selective_scan_windows.diagnostics
micromamba run -n ultralytics_contrib python -m pytest -c pytest.ini --require-cuda
```

The integration tests also require the custom Ultralytics checkout and its
`mamba` dependencies, already covered by the PC setup guide. If not installed:

```bat
micromamba run -n ultralytics_contrib python -m pip install -e "%SSW_REPO%[mamba]"
```

`--no-build-isolation --no-deps` uses the installed CUDA PyTorch without
letting the extension install replace it. The build deliberately rejects other
PyTorch/CUDA versions. Use an ordinary wheel install, not editable mode, for
initial qualification. Rebuild after changing Python, PyTorch, or the toolkit.

PyTorch determines architectures from visible GPUs. Alternatively, set
`TORCH_CUDA_ARCH_LIST` explicitly to the capability printed above, such as
`set "TORCH_CUDA_ARCH_LIST=8.6"` **only if the actual device reports `(8, 6)`**.
Without a visible GPU, this variable is required to build. Compilation alone
does not replace runtime tests on the target GPU.

Diagnostics report package/binary origin, any original-distribution conflict,
PyTorch/CUDA versions, GPU identity and capability. They exit nonzero if the
runtime or compatibility module is incorrect. They never load a foreign
compatibility module as a fallback. A missing compiler/DLL/build error should
be resolved using the captured message; do not bypass compiler compatibility
checks with `--allow-unsupported-compiler`.

## Isolated source-distribution verification

This verifies that the source archive builds without the original repository.
In the same developer Command Prompt, use a **new** staging directory:

```bat
cd /d "%SSW_REPO%\selective_scan_windows"
micromamba run -n ultralytics_contrib python setup.py sdist
set "SSW_STAGE=%TEMP%\selective-scan-windows-qualification"
mkdir "%SSW_STAGE%"
tar -xf dist\selective_scan_windows-0.1.0.tar.gz -C "%SSW_STAGE%"
cd /d "%SSW_STAGE%\selective_scan_windows-0.1.0"
micromamba run -n ultralytics_contrib python -m pip wheel --no-build-isolation --no-deps -v . -w dist
micromamba run -n ultralytics_contrib python -m pip install --no-deps --force-reinstall --no-index --find-links=dist selective-scan-windows==0.1.0
micromamba run -n ultralytics_contrib python -m selective_scan_windows.diagnostics
micromamba run -n ultralytics_contrib python -m pytest -c pytest.ini --require-cuda
```

No original extension source is needed at any point. Tests use an independent
PyTorch recurrence, with the original test's dtype-specific tolerances. The
integration tests import the installed custom Ultralytics model wrapper, not
the old extension. `--require-cuda` prevents skipped GPU tests from being
mistaken for a qualified build.

## Checkpoint inference verification

Supply an existing local trusted research `.pt` checkpoint and a small set of
representative **448 x 448 tiles**, including challenging sites. Checkpoints
and images are not downloaded. The command requires a new output directory:

```bat
cd /d "%SSW_REPO%\selective_scan_windows"
micromamba run -n ultralytics_contrib python tools\validate_checkpoint.py --checkpoint "%USERPROFILE%\rip-work\models\best.pt" --images "%USERPROFILE%\rip-work\data\tile1.png" "%USERPROFILE%\rip-work\data\tile2.png" --output validation-output\first-run
```

The tool uses device 0, batch 1, `imgsz=448`, FP32, two warm-up passes and five
measured passes. It checks finite predictions and writes `report.json` plus
`predictions.npz`, without plots, saved images, uploads, or data cleanup.
The report includes median/minimum/maximum time per tile and peak allocated
and reserved VRAM. Timing includes preprocessing and postprocessing.

After warming all inputs, live GPU allocation at matching end-of-pass points
must not grow by more than the larger of 16 MiB or 5% of the first measured
pass. Reserved memory is recorded separately because it can reflect caching.
This is a bounded repeatability check, not proof that a long job cannot leak.

Set `--conf`, `--iou`, `--max-det`, and `--retina-masks` to the intended pipeline
settings when needed; defaults are 0.25, 0.7, 300, and disabled. No model/site
quality claim is made from these default thresholds alone.

To compare runs, add `--baseline path\to\report.json`; its neighboring
`predictions.npz` must exist. The checkpoint/image hashes, image order, and
settings must match. Ordered boxes/OBBs/scores use FP32 scan tolerances;
binary masks must match exactly. Count, ordering, or threshold-boundary
differences cause failure and require investigation. A supplied HPC baseline
must follow the same report identity and array format. This tool never imports
the original extension to produce a baseline. Without a baseline it reports
`baseline_status: not_provided`, not prediction parity.

## Portable development checks

On a machine without Windows CUDA, run the package's own tests only:

```text
python -m pytest -c /absolute/path/to/selective_scan_windows/pytest.ini /absolute/path/to/selective_scan_windows/tests
```

The CPU tests check archive contents, isolated imports, missing-binary errors,
reference mathematics, and baseline comparison. GPU and integration cases are
reported as skipped. The package has its own pytest configuration and does not
use the repository test suite's destructive cleanup hooks.

See `THIRD_PARTY_NOTICES.md` for the original revision, hashes, retained notices,
and an adaptation summary. All implementation files are in this new directory;
the original extension and `ultralytics/` files are unchanged.
