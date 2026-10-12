import torch

from .kernels import *


def mul_and_silu_masked(hidden_states: torch.Tensor, masked_m: torch.Tensor):
    assert hidden_states.shape[0] == masked_m.shape[0]
    B, M, twoH = hidden_states.shape
    H = twoH // 2
    out = torch.empty(
        (B, M, H),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    BLOCK_M = 32
    BLOCK_H = 64
    grid = (
        B,
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(H, BLOCK_H),
    )
    mul_silu_masked_kernel[grid](
        hidden_states,
        out,
        masked_m,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H=H,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )

    return out


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None, reverse=False):
    """
    input: [..., 2 * hidden_size]
    out:   [..., hidden_size]
    """
    assert input.is_cuda
    assert input.dim() >= 2

    hidden_size = input.shape[-1] // 2
    assert input.shape[-1] == 2 * hidden_size

    x = input.contiguous()
    N = x.numel() // (2 * hidden_size)

    x2d = x.view(N, 2 * hidden_size)

    if out is None:
        out = torch.empty(
            (*input.shape[:-1], hidden_size),
            device=input.device,
            dtype=input.dtype,
        )

    y2d = out.view(N, hidden_size)

    BLOCK = triton.next_power_of_2(hidden_size)

    grid = (N,)

    silu_and_mul_kernel[grid](
        x2d,
        y2d,
        x2d.stride(0),
        y2d.stride(0),
        H=hidden_size,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 128 else 8,
        num_stages=2,
        reverse=reverse,
    )

    return out


def masked_fp8_block_quant_triton(
    y: torch.Tensor,  # (E, T, H)
    tokens_per_expert: torch.Tensor,  # (E,)
    block_size: int = 128,
):
    assert y.is_cuda and y.ndim == 3
    assert y.dtype in (torch.bfloat16, torch.float16)
    assert y.shape[-1] % block_size == 0
    E, T, H = y.shape
    assert tokens_per_expert.is_cuda and tokens_per_expert.ndim == 1
    assert tokens_per_expert.shape[0] == E

    y_q = torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device=y.device)
    G = H // block_size
    y_s = torch.empty_strided(
        (E, T, G),
        (T * G, 1, T),
        dtype=torch.float32,
        device=y.device,
    )

    block_m = 8 if T <= 256 else 16
    grid = (E, triton.cdiv(T, block_m), G)
    finfo = torch.finfo(torch.float8_e4m3fn)
    masked_fp8_block_quant_kernel[grid](
        y,
        tokens_per_expert.to(dtype=torch.int32, device=y.device),
        y_q,
        y_s,
        E,
        T,
        H,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y_q.stride(0),
        y_q.stride(1),
        y_q.stride(2),
        y_s.stride(0),
        y_s.stride(1),
        y_s.stride(2),
        tokens_per_expert.stride(0),
        finfo.min,
        finfo.max,
        1e-10,
        BLOCK=block_size,
        BLOCK_M=block_m,
        num_warps=8 if block_m >= 16 else 4,
        num_stages=2,
    )
    return y_q, y_s


def per_token_group_quant_fp8_channelwise_triton(
    x: torch.Tensor,
    fp8_min: float,
    fp8_max: float,
    min_scaling_factor: float,
):
    assert x.is_cuda and x.ndim == 2
    assert x.dtype in (torch.bfloat16, torch.float16)

    m, n = x.shape
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    block_n = triton.next_power_of_2(n)
    num_warps = 4 if block_n <= 4096 else 8

    per_token_fp8_channelwise_quant_kernel[(m,)](
        x_contig,
        out,
        scale,
        m,
        n,
        x_contig.stride(0),
        x_contig.stride(1),
        out.stride(0),
        out.stride(1),
        scale.stride(0),
        fp8_min,
        fp8_max,
        min_scaling_factor,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return out, scale


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the attention
    # backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)
    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        head_size,
        padded_head_size,
        output_lse is not None,
    )


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N, "IS_BASE_E": is_lse_base_on_e}
    ctx.call_kernel(correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def causal_attn_uniform_q_len_suffix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    prefix_out: torch.Tensor | None = None,
    prefix_lse: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.float16, torch.bfloat16)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, q_len, num_q_heads, qk_dim = q.shape
    num_kv_heads = k.shape[2]
    assert num_kv_heads == 1, (
        "causal_attn_uniform_q_len_suffix currently requires kv_head == 1"
    )
    v_dim = v.shape[-1]
    if sm_scale is None:
        sm_scale = qk_dim**-0.5

    has_prefix = prefix_out is not None
    if has_prefix:
        assert prefix_lse is not None
        assert prefix_out.shape == (batch_size, q_len, num_q_heads, v_dim)
        assert prefix_lse.shape == (batch_size, q_len, num_q_heads)
        prefix_out = prefix_out.contiguous()
        prefix_lse = prefix_lse.contiguous()
    else:
        assert prefix_lse is None

    if out is None:
        out = torch.empty(
            (batch_size, q_len, num_q_heads, v_dim),
            device=q.device,
            dtype=v.dtype,
        )
    if lse is None:
        lse = torch.empty(
            (batch_size, q_len, num_q_heads), device=q.device, dtype=torch.float32
        )

    block_dqk = 128 if qk_dim >= 128 else 64
    if q_len == 2 and 96 <= batch_size < 192:
        block_dv = 256
        num_warps = 4
        num_stages = 3
    elif q_len <= 4:
        if batch_size >= 128:
            block_dv = 128
            num_warps = 2
        else:
            block_dv = 256
            num_warps = 4
        num_stages = 3
    elif q_len <= 8 and batch_size >= 128:
        block_dv = 128
        num_warps = 2
        num_stages = 3
    elif batch_size >= 256:
        block_dv = 32
        num_warps = 4
        num_stages = 2
    elif batch_size >= 128:
        block_dv = 64
        num_warps = 4
        num_stages = 2
    else:
        block_dv = 128 if v_dim >= 128 else 64
        num_warps = 4
        num_stages = 2
    if num_q_heads <= 8:
        block_g = 8
    elif q_len == 2:
        if batch_size < 96 or batch_size >= 384:
            block_g = 8
        else:
            block_g = 32
    elif q_len == 3:
        block_g = 8 if batch_size <= 48 or batch_size >= 192 else 16
    elif q_len == 4:
        block_g = 32 if 64 <= batch_size <= 128 else 8
    elif q_len <= 8 and batch_size >= 128:
        block_g = 8
    else:
        block_g = 16 if num_q_heads > 16 else 32
    common_args = (
        q,
        k,
        v,
        prefix_out,
        prefix_lse,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        0 if prefix_out is None else prefix_out.stride(0),
        0 if prefix_out is None else prefix_out.stride(1),
        0 if prefix_out is None else prefix_out.stride(2),
        0 if prefix_out is None else prefix_out.stride(3),
        0 if prefix_lse is None else prefix_lse.stride(0),
        0 if prefix_lse is None else prefix_lse.stride(1),
        0 if prefix_lse is None else prefix_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
    )
    max_q = 4 if q_len <= 4 else 8 if q_len <= 8 else 16 if q_len <= 16 else 32
    head_tiles = triton.cdiv(num_q_heads, block_g)
    causal_attn_uniform_q_len_suffix_kernel[(batch_size, q_len, head_tiles)](
        *common_args,
        num_q_heads,
        q_len,
        qk_dim,
        v_dim,
        sm_scale,
        BLOCK_DQK=block_dqk,
        BLOCK_DV=block_dv,
        MAX_Q=max_q,
        BLOCK_G=block_g,
        HAS_PREFIX=has_prefix,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse
