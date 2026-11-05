"""Portable checks exercise isolated archives/imports without compiling CUDA."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_python(code, cwd, source=ROOT / "src"):
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd, env=env,
                          text=True, capture_output=True)


def test_no_original_build_or_runtime_dependency():
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            assert not any(n == "selective_scan" or n.startswith("selective_scan.") for n in names)
            assert not any(n in {"selective_scan_cuda", "selective_scan_cuda_oflex", "selective_scan_cuda_ndstate"}
                           for n in names)
    includes = ROOT / "csrc/selective_scan"
    for path in includes.rglob("*"):
        if path.is_file():
            for name in re.findall(r'^#include "([^"]+)"', path.read_text(), re.M):
                candidates = [(path.parent / name).resolve(), (includes / name).resolve()]
                assert any(p.is_file() and p.is_relative_to(includes) for p in candidates), (path, name)
    setup = (ROOT / "setup.py").read_text()
    assert '"selective_scan_windows._C"' in setup
    assert "FORCE_CXX11_ABI" not in setup and "-gencode" not in setup


def test_compatibility_shim_uses_only_own_binary(tmp_path):
    result = run_python('''
import importlib.abc
import sys
import types
class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"selective_scan", "selective_scan_cuda", "selective_scan_cuda_oflex", "selective_scan_cuda_ndstate"}:
            raise AssertionError("Attempted original import: " + fullname)
sys.meta_path.insert(0, BlockOriginal())
core = types.ModuleType("selective_scan_windows._C")
core.fwd = lambda *args: None
core.bwd = lambda *args: None
sys.modules[core.__name__] = core
import selective_scan_windows
import selective_scan_cuda_core
assert selective_scan_cuda_core.fwd is core.fwd
assert selective_scan_cuda_core.bwd is core.bwd
assert selective_scan_windows.fwd is core.fwd
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_binary_has_no_fallback(tmp_path):
    result = run_python('''
import importlib.abc
import sys
class MissingBinary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "selective_scan_windows._C":
            raise ImportError("deliberately absent binary")
        if fullname in {"selective_scan", "selective_scan_cuda"}:
            raise AssertionError("Original package import attempted")
sys.meta_path.insert(0, MissingBinary())
try:
    import selective_scan_cuda_core
except ImportError as exc:
    assert "No original selective_scan backend" in str(exc), str(exc)
else:
    raise AssertionError("Missing binary unexpectedly imported")
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_require_cuda_rejects_unbuilt_source(tmp_path):
    # Source-only import cannot be qualified, even on a machine with a GPU.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-c", str(ROOT / "pytest.ini"),
         str(ROOT / "tests/test_selective_scan.py"), "--require-cuda"],
        cwd=tmp_path, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "Windows backend verification failed" in result.stderr


def test_diagnostics_does_not_import_foreign_compatibility_module(tmp_path):
    result = run_python('''
from importlib.machinery import ModuleSpec
from unittest.mock import patch
import selective_scan_windows.diagnostics as d
real_find = d.util.find_spec
real_import = d.importlib.import_module
def find(name, *args, **kwargs):
    if name == "selective_scan_cuda_core":
        return ModuleSpec(name, None, origin="/foreign/selective_scan_cuda_core.pyd")
    return real_find(name, *args, **kwargs)
def load(name, *args, **kwargs):
    assert name != "selective_scan_cuda_core", "Must not import the foreign backend"
    return real_import(name, *args, **kwargs)
with patch.object(d.util, "find_spec", side_effect=find), patch.object(d.importlib, "import_module", side_effect=load):
    report = d.collect()
assert report["status"] == "failed"
assert any("does not resolve" in message for message in report["errors"])
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sdist_is_self_contained(tmp_path):
    isolated = tmp_path / "source"
    shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"))
    result = subprocess.run([sys.executable, "-B", "setup.py", "sdist", "--dist-dir", str(tmp_path / "dist")],
                            cwd=isolated, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    archive = next((tmp_path / "dist").glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        names = {"/".join(Path(member.name).parts[1:]) for member in tar.getmembers()}
    for source in (ROOT / "csrc").rglob("*"):
        if source.is_file():
            assert str(source.relative_to(ROOT)).replace(os.sep, "/") in names
    for required in ("src/selective_scan_cuda_core.py", "src/selective_scan_windows/__init__.py",
                     "src/selective_scan_windows/diagnostics.py", "LICENSE", "THIRD_PARTY_NOTICES.md",
                     "tests/reference.py", "tools/validate_checkpoint.py"):
        assert required in names
    assert not any(name.endswith((".pyd", ".pyc")) for name in names)


def test_wheel_contains_delegated_binary_and_compatibility_module(tmp_path):
    # Exercise the setuptools/PyTorch build-command delegation without pretending
    # to compile CUDA here. The placeholder binary exists only in this temp wheel.
    isolated = tmp_path / "source"
    shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"))
    result = run_python('''
import importlib.util
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import torch.utils.cpp_extension as cpp
spec = importlib.util.spec_from_file_location("ssw_build", "setup.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class FakeCompile(build_ext):
    def run(self):
        for extension in self.extensions:
            output = Path(self.get_ext_fullpath(extension.name))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"test placeholder, not a native binary")
module.cuda_extension = lambda: Extension("selective_scan_windows._C", sources=module.SOURCES)
cpp.BuildExtension = FakeCompile
setup(ext_modules=[Extension("selective_scan_windows._C", sources=module.SOURCES)],
      cmdclass={"build_ext": module.WindowsBuildExtension},
      script_args=["bdist_wheel", "--dist-dir", "dist"])
''', isolated, source=isolated / "src")
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((isolated / "dist").glob("*.whl"))
    assert not wheel.name.endswith("none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "selective_scan_cuda_core.py" in names
        assert "selective_scan_windows/diagnostics.py" in names
        binaries = [n for n in names if n.startswith("selective_scan_windows/_C.")]
        assert len(binaries) == 1
        assert archive.read(binaries[0]) == b"test placeholder, not a native binary"
        assert any(n.endswith("THIRD_PARTY_NOTICES.md") for n in names)


def load_validator():
    spec = importlib.util.spec_from_file_location("ssw_validation_tool", ROOT / "tools/validate_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_checks_identity_shapes_and_values(tmp_path):
    validator = load_validator()
    identity = {"checkpoint_sha256": "abc", "image_sha256": ["def"], "settings": {"imgsz": 448}}
    arrays = {"image_0_boxes": np.array([[1., 2., 3., 4., .9, 0.]]),
              "image_0_masks": np.array([[[True, False]]])}
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"identity": identity}))
    np.savez(tmp_path / "predictions.npz", **arrays)
    validator.compare_baseline(path, identity, arrays)
    with pytest.raises(ValueError, match="hashes"):
        validator.compare_baseline(path, {}, arrays)
    with pytest.raises(AssertionError):
        validator.compare_baseline(path, identity, {**arrays, "image_0_boxes": arrays["image_0_boxes"] + 1})
    with pytest.raises(AssertionError):
        validator.compare_baseline(path, identity, {**arrays, "image_0_masks": ~arrays["image_0_masks"]})
    with pytest.raises(AssertionError, match="shape"):
        validator.compare_baseline(path, identity, {**arrays, "image_0_boxes": np.array(0.)})


def test_validator_help_does_not_require_gpu():
    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/validate_checkpoint.py"), "--help"],
                            text=True, capture_output=True)
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout and "--baseline" in result.stdout
