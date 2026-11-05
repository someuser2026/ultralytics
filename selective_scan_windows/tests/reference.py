"""Small differentiable real-valued recurrence; no extension imports or production fallback."""

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    dtype = u.dtype
    u, delta, A, B, C = (v.float() for v in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    if B.ndim == 3:
        B = B.unsqueeze(1)
    if C.ndim == 3:
        C = C.unsqueeze(1)
    channels_per_group = u.shape[1] // B.shape[1]
    # Expanding only one time step keeps long-sequence reference memory bounded.
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t, None]
        bt = B[:, :, :, t].repeat_interleave(channels_per_group, dim=1)
        ct = C[:, :, :, t].repeat_interleave(channels_per_group, dim=1)
        state = torch.exp(dt * A) * state + dt * bt * u[:, :, t, None]
        outputs.append((state * ct).sum(-1))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + D.float()[None, :, None] * u
    return output.to(dtype), state


def make_inputs(length=129, dtype=torch.float32, batch=1, groups=4, state=8,
                has_D=True, has_bias=True, device="cuda", outer_strided=False):
    generator = torch.Generator(device=device).manual_seed(4301)
    dim = 8

    def random(shape, kind=dtype):
        return torch.rand(shape, generator=generator, device=device, dtype=kind)

    u = random((batch, dim, length)) - 0.5
    delta = 0.01 + 0.1 * random((batch, dim, length))
    A = -0.1 - random((dim, state), torch.float32)
    B = random((batch, groups, state, length)) - 0.5
    C = random((batch, groups, state, length)) - 0.5
    D = random((dim,), torch.float32) if has_D else None
    bias = random((dim,), torch.float32) * 0.1 if has_bias else None
    values = [u, delta, A, B, C, D, bias]
    if outer_strided:
        for i in (0, 1, 3, 4):
            v = values[i]
            shape = list(v.shape)
            shape[1] *= 2
            storage = torch.empty(shape, dtype=v.dtype, device=device)
            storage[:, ::2] = v
            values[i] = storage[:, ::2]
    return [v.detach().requires_grad_() if v is not None else None for v in values]


def tolerances(dtype):
    # Copied values from the source test, without its extension imports.
    if dtype == torch.float32:
        return 6e-4, 2e-3
    if dtype == torch.float16:
        return 3e-3, 5e-3
    return 3e-2, 5e-2


def assert_gradients(actual, inputs, expected, dtype):
    rtol, atol = tolerances(dtype)
    limits = [(rtol * 2, atol * 2), (rtol * 5, atol * 10), (1e-3, 5e-3),
              (rtol, atol), (rtol, atol), (1e-3, 1e-3), (1e-3, 1e-3)]
    for i, (got, source) in enumerate(zip(actual, inputs)):
        if source is None:
            assert got is None
            continue
        want = expected.pop(0)
        assert got.shape == source.shape and got.dtype == source.dtype
        assert torch.isfinite(got).all()
        # In particular, dA is correctly zero at length 1: the initial state is
        # zero, so the first output cannot depend on the transition matrix.
        if torch.count_nonzero(want) > 0:
            assert torch.count_nonzero(got) > 0
        torch.testing.assert_close(got, want, rtol=limits[i][0], atol=limits[i][1])
