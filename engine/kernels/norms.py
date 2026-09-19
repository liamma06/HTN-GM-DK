"""RMSNorm in Triton for the decode step, matching Qwen3RMSNorm's arithmetic.

Same cast placement as the reference: normalise in fp32, round to the input
dtype, then multiply by the weight. Rows are read through strides, so a slice of
a larger tensor (such as q or k inside the fused qkv output) needs no copy.
"""

import torch
import triton
import triton.language as tl

MAX_BLOCK = 8192


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, stride_b, stride_i, inner, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // inner
    i = row % inner
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + b * stride_b + i * stride_i + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)

    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim of a [rows, n] or [batch, inner, n] tensor.

    The last dim must be contiguous; other strides are arbitrary. Returns a new
    contiguous tensor of the same shape.
    """
    two_d = x.dim() == 2
    if two_d:
        x = x.unsqueeze(1)
    batch, inner, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK or x.stride(2) != 1:
        raise ValueError("unsupported layout for fused rms_norm")
    out = torch.empty((batch, inner, n_cols), dtype=x.dtype, device=x.device)
    _rms_norm_kernel[(batch * inner,)](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        inner,
        n_cols,
        eps,
        BLOCK=block,
        num_warps=max(4, min(16, block // 256)),
    )
    return out.squeeze(1) if two_d else out


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return weight * hidden.to(x.dtype)


def self_check(device, dtype=torch.bfloat16, eps: float = 1e-6) -> bool:
    """True when the kernel agrees with the reference on shapes the engine uses."""
    gen = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)

    hidden = rand(16, 2560)
    wide = rand(4, 6144)
    cases = [
        (hidden, rand(2560)),
        (wide[:, :4096].view(4, 32, 128), rand(128)),
        (wide[:, 4096:5120].view(4, 8, 128), rand(128)),
    ]
    for x, w in cases:
        got = rms_norm(x, w, eps)
        want = _reference(x, w, eps)
        if got.shape != want.shape or not torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2):
            return False
    return True
