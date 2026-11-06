import itertools

import pytest
import torch

from .reference import assert_gradients, make_inputs, selective_scan_ref, tolerances


def test_reference_known_recurrence_and_derivatives():
    # A=0, B=C=1, dt=1: cumulative input plus the D skip connection.
    u = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    output, state = selective_scan_ref(u, torch.ones_like(u), torch.zeros(1, 1),
                                       torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3), torch.tensor([2.]))
    torch.testing.assert_close(output, torch.tensor([[[3., 7., 12.]]]))
    torch.testing.assert_close(state, torch.tensor([[[6.]]]))
    output.sum().backward()
    torch.testing.assert_close(u.grad, torch.tensor([[[5., 4., 3.]]]))


def test_first_step_does_not_depend_on_transition():
    values = make_inputs(length=1, device="cpu")
    output, _ = selective_scan_ref(*values, True)
    grads = torch.autograd.grad(output.sum(), values)
    assert torch.count_nonzero(grads[2]) == 0
    assert_gradients(grads, values, list(grads), torch.float32)


BOUNDARIES = [1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4096]
# Exercise aligned/tail launch specializations with softplus both on and off.
# Boundary coverage in FP32; mixed precision covers both single and multiple chunks.
CASES = [(length, torch.float32, 1, 4, 8, True, True, softplus, False)
         for length, softplus in itertools.product(BOUNDARIES, [False, True])]
CASES += [(length, dtype, 2, 2, 16, True, True, True, False)
          for length, dtype in itertools.product([129, 2049], [torch.float16, torch.bfloat16])]
CASES += [(65, torch.float32, 2, 1, 1, d, bias, softplus, False)
          for d, bias, softplus in itertools.product([False, True], repeat=3)]
CASES += [(129, torch.float32, 1, 4, 256, True, True, True, False),
          (257, torch.float32, 2, 2, 8, True, True, True, True)]


@pytest.mark.cuda
@pytest.mark.parametrize("length,dtype,batch,groups,state,has_D,has_bias,softplus,strided", CASES)
def test_forward_backward(backend, length, dtype, batch, groups, state, has_D, has_bias, softplus, strided):
    values = make_inputs(length, dtype, batch, groups, state, has_D, has_bias, outer_strided=strided)
    out, workspace = backend.fwd(*values, softplus, 1)
    reference, last_state = selective_scan_ref(*values, softplus)
    assert out.dtype == dtype and out.shape == values[0].shape
    assert workspace.dtype == torch.float32
    assert workspace.shape == (batch, 8, (length + 2047) // 2048, state * 2)
    assert torch.isfinite(out).all() and torch.isfinite(workspace).all()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(workspace[:, :, -1, 1::2], last_state, rtol=rtol, atol=atol)
    dout = torch.linspace(-0.5, 0.5, out.numel(), device="cuda", dtype=dtype).reshape_as(out)
    expected = list(torch.autograd.grad(reference, [v for v in values if v is not None], dout))
    gradients = backend.bwd(*values, dout, workspace, softplus, 1)
    assert len(gradients) == 7
    assert_gradients(gradients, values, expected, dtype)


@pytest.mark.cuda
def test_image_feature_length(backend):
    with torch.no_grad():
        values = make_inputs(length=12544, state=1)
        output, workspace = backend.fwd(*values, True, 1)
        reference, state = selective_scan_ref(*values, True)
        torch.testing.assert_close(output, reference, rtol=6e-4, atol=2e-3)
        torch.testing.assert_close(workspace[:, :, -1, 1::2], state, rtol=6e-4, atol=2e-3)


@pytest.mark.cuda
def test_nondefault_stream(backend):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = make_inputs(length=257)
        out, workspace = backend.fwd(*values, True, 1)
        reference, _ = selective_scan_ref(*values, True)
        dout = torch.ones_like(out)
        expected = list(torch.autograd.grad(reference, values, dout))
        got = backend.bwd(*values, dout, workspace, True, 1)
    stream.synchronize()
    torch.testing.assert_close(out, reference, rtol=6e-4, atol=2e-3)
    assert_gradients(got, values, expected, torch.float32)


@pytest.mark.cuda
@pytest.mark.parametrize("invalid", ["cpu", "dtype", "state", "stride", "groups"])
def test_invalid_inputs(backend, invalid):
    values = make_inputs(length=64, state=257 if invalid == "state" else 8)
    if invalid == "cpu":
        values[0] = values[0].cpu()
    elif invalid == "dtype":
        values[1] = values[1].double()
    elif invalid == "stride":
        values[0] = torch.zeros(1, 8, 128, device="cuda")[:, :, ::2]
    elif invalid == "groups":
        values[3] = torch.zeros(1, 3, 8, 64, device="cuda")
    with pytest.raises(RuntimeError):
        backend.fwd(*values, True, 1)
