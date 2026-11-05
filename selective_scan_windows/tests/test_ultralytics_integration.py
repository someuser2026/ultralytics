import importlib

import pytest
import torch

from .reference import assert_gradients, make_inputs, selective_scan_ref, tolerances


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unchanged_model_wrapper(backend, dtype):
    module = importlib.import_module("ultralytics.nn.modules.common_utils_mbyolo")
    assert module.selective_scan_cuda_core.fwd is backend.fwd
    assert module.selective_scan_cuda_core.bwd is backend.bwd
    values = make_inputs(length=129, dtype=dtype)
    with torch.autocast(device_type="cuda", dtype=dtype if dtype != torch.float32 else torch.float16,
                        enabled=dtype != torch.float32):
        output = module.SelectiveScanCore.apply(*values, True, 1, 1, True)
    reference, _ = selective_scan_ref(*values, True)
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
    dout = torch.ones_like(output)
    expected = list(torch.autograd.grad(reference, values, dout))
    actual = torch.autograd.grad(output, values, dout)
    assert_gradients(actual, values, expected, dtype)


@pytest.mark.cuda
@pytest.mark.integration
def test_wrapper_makes_last_dimension_contiguous(backend):
    module = importlib.import_module("ultralytics.nn.modules.common_utils_mbyolo")
    assert module.selective_scan_cuda_core.fwd is backend.fwd
    values = make_inputs(length=258)
    for i in (0, 1, 3, 4):
        values[i] = values[i][..., ::2].detach().requires_grad_()
    output = module.SelectiveScanCore.apply(*values, True, 1, 1, True)
    reference, _ = selective_scan_ref(*values, True)
    torch.testing.assert_close(output, reference, rtol=6e-4, atol=2e-3)
    expected = list(torch.autograd.grad(reference, values, torch.ones_like(output)))
    actual = torch.autograd.grad(output, values, torch.ones_like(output))
    assert_gradients(actual, values, expected, torch.float32)
