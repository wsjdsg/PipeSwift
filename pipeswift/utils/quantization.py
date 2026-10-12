from typing import Tuple

import pipeswift_kernels
import torch

from .triton_kernels import (
    fp8_block_dequant_kernel,
    masked_fp8_block_quant_triton,
    per_token_group_quant_fp8_channelwise_triton,
    silu_and_mul,
)

BLOCK_QUANT_SIZE = 128


def get_fp8_min_max() -> Tuple[float, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return finfo.min, finfo.max


_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def per_token_group_quant_fp8_channelwise_triton_impl(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return per_token_group_quant_fp8_channelwise_triton(
        x, _FP8_MIN, _FP8_MAX, _FP8_MIN_SCALING_FACTOR
    )


# copied from vllm
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        # If tensor has fewer dimensions than target shape, treat missing
        # dimensions as size 1 (standard PyTorch broadcasting behavior)
        t_dim_size = t.shape[i] if i < t.ndim else 1
        if t_dim_size != s and t_dim_size != 1:
            assert s % t_dim_size == 0
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t_dim_size, *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return t


# Keep the public API stable while routing channelwise FP8 quantization through Triton.
def per_token_group_quant_fp8_channelwise_torchnative(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return per_token_group_quant_fp8_channelwise_triton_impl(x)


def per_token_group_quant_fp8_blockwise_vllm(
    x: torch.Tensor,
    group_size: int = BLOCK_QUANT_SIZE,
    eps: float = 1e-10,
    use_ue8m0: bool = False,
    column_major_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.shape[-1] % group_size == 0, (
        f"Last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    if column_major_scales:
        num_groups_per_row = x.shape[-1] // group_size
        shape = x.shape[:-2] + (num_groups_per_row, x.shape[-2])
        x_scale = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
            -1, -2
        )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_scale = torch.empty(shape, device=x.device, dtype=torch.float32)

    try:
        pipeswift_kernels.per_token_group_fp8_quant(
            x,
            x_fp8,
            x_scale,
            group_size,
            eps,
            _FP8_MIN,
            _FP8_MAX,
            use_ue8m0,
        )
    except (RuntimeError, AttributeError, ImportError) as e:
        raise RuntimeError(
            f"Failed to call vLLM kernel. Make sure pipeswift is properly installed. "
            f"Error: {e}\n"
            f"Try: pip install -e . (from pipeswift root directory)\n"
            f"Fallback: Use DeepGEMM's per_token_cast_to_fp8 instead."
        )

    return x_fp8, x_scale


# copied from vllm
def vllm_cutlass_fp8_moe(
    a1q: torch.Tensor,
    a1q_scale: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    out: torch.Tensor,
    ab_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    global_num_experts,
    expert_map: torch.Tensor = None,
):
    assert out.dtype in [torch.bfloat16, torch.float16]
    M = a1q.size(0)  # non batched expert M
    _, K, N = w2.shape
    device = a1q.device

    topk = topk_idx.size(1)
    local_E = w1.size(0)
    workspace13 = torch.empty(
        (M * topk, max(a1q.shape[-1], 2 * N)), dtype=out.dtype, device=device
    )
    workspace2 = torch.empty(
        (M * topk, max(a1q.shape[-1], N)), dtype=out.dtype, device=device
    )
    # create out buffer
    a1q_perm = _resize_cache(workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K))
    mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
    act_out = _resize_cache(workspace2, (M * topk, N))
    mm2_out = _resize_cache(workspace2, (M * topk, K))

    # permuted a1q reuses workspace2
    problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
    num_expert = global_num_experts

    a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
        a1q,
        a1q_scale,
        topk_idx,
        num_expert,
        local_E,
        expert_map=expert_map,
        permuted_hidden_states=a1q_perm,
    )
    # swap_ab is a CUTLASS grouped-GEMM optimization (M <= 64 reduces padding).
    swap_ab = a1q.size(0) <= 64
    pipeswift_kernels.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab
    )
    expert_offsets = expert_first_token_offset[:-1]
    pipeswift_kernels.cutlass_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_scale,
        expert_offsets,
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
        per_act_token=True,
        per_out_ch=True,
    )
    act_out = silu_and_mul(mm1_out, out=act_out, reverse=True)
    a2q, a2q_scale = per_token_group_quant_fp8_channelwise_torchnative(act_out)
    pipeswift_kernels.cutlass_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_scale,
        expert_offsets,
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
        per_act_token=True,
        per_out_ch=True,
    )

    moe_unpermute(
        out=out,
        permuted_hidden_states=mm2_out,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_perm,
        expert_first_token_offset=expert_first_token_offset,
    )


from math import prod


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(), (
        f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    )  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    align_block_size: int | None = None,
    fill_invalid_expert: int = -1,
    permuted_hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_token, n_hidden = hidden_states.size()
    topk = topk_ids.size(1)
    assert (n_hidden * hidden_states.element_size()) % 16 == 0, (
        "permue kernel need hidden dim align to 16B"
    )
    permuted_row_size = n_token * topk
    if align_block_size is not None:
        permuted_row_size = (
            (
                permuted_row_size
                + n_expert * (align_block_size - 1)
                + align_block_size
                - 1
            )
            // align_block_size
            * align_block_size
        )
    if n_local_expert == -1:
        n_local_expert = n_expert
    if permuted_hidden_states is None:
        permuted_hidden_states = torch.empty(
            (permuted_row_size, n_hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    assert permuted_hidden_states.size() == (permuted_row_size, n_hidden), (
        f"Expected permuted hidden states to be {(permuted_row_size, n_hidden)}"
        f" but got {permuted_hidden_states.size()}"
    )

    token_expert_indices = torch.arange(
        0, n_token * topk, dtype=torch.int32, device=hidden_states.device
    ).reshape((n_token, topk))

    m_indices = torch.full(
        (permuted_row_size,),
        fill_invalid_expert,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    expert_first_token_offset = torch.empty(
        n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    permuted_idx = torch.full(
        (permuted_row_size,),
        n_token * topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    inv_permuted_idx = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    topk_ids = topk_ids.to(torch.int32)

    pipeswift_kernels.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        expert_map,
        n_expert,
        n_local_expert,
        topk,
        align_block_size,
        permuted_hidden_states,
        expert_first_token_offset,
        inv_permuted_idx,
        permuted_idx,
        m_indices,
    )

    if a1q_scale is not None and a1q_scale.dim() > 1:
        a1q_scale = a1q_scale[permuted_idx.clamp(max=n_token * topk - 1) // topk]
    return (
        permuted_hidden_states,
        a1q_scale,
        expert_first_token_offset,
        inv_permuted_idx.flatten(),
        m_indices,
    )


def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None:
    topk = topk_weights.size(1)
    n_hidden = permuted_hidden_states.size(-1)
    assert (n_hidden * permuted_hidden_states.element_size()) % 16 == 0, (
        "unpermue kernel need hidden dim align to 16B"
    )

    pipeswift_kernels.moe_unpermute(
        permuted_hidden_states,
        topk_weights,
        inv_permuted_idx,
        expert_first_token_offset,
        topk,
        out,
    )


def fp8_block_dequant(hidden_fp8, scale_fp32, block_size=128):
    B, D = hidden_fp8.shape
    n_blocks = (D + block_size - 1) // block_size

    assert hidden_fp8.dtype == torch.float8_e4m3fn
    assert scale_fp32.shape == (B, n_blocks)

    out = torch.empty((B, D), device=hidden_fp8.device, dtype=torch.bfloat16)

    grid = (B, n_blocks)
    fp8_block_dequant_kernel[grid](
        hidden_fp8,
        scale_fp32,
        out,
        B,
        D,
        n_blocks,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )
    return out


# masked fp8 block quantization for DeepEP's batched format activation
def masked_fp8_block_quant(
    y: torch.Tensor,  # (E, T, H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
) -> tuple[torch.Tensor, torch.Tensor]:
    assert y.ndim == 3, "y must be (E, T, H)"
    assert y.is_cuda, "y must be CUDA tensor"
    assert y.dtype in (torch.bfloat16, torch.float16), "y must be bf16/fp16"
    E, T, H = y.shape
    assert H % BLOCK_QUANT_SIZE == 0, "H must be divisible by 128"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, (
        "tokens_per_expert must be (E,)"
    )
    if tokens_per_expert.device != y.device or tokens_per_expert.dtype != torch.int32:
        tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)
    return masked_fp8_block_quant_triton(
        y,
        tokens_per_expert,
        block_size=BLOCK_QUANT_SIZE,
    )


# copied from vllm
def fused_masked_m_silu_mul_fp8_block_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    y_q: torch.Tensor | None = None,
    y_s: torch.Tensor | None = None,
    reverse: bool = False,  # False: [gate, up], True: [up, gate]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1. fused silu_mul activation and fp8 block quantization
    2. include mask into considertion, i.e., in batched-format, we activate & quant only actual received number of tokens.
    By skipping padded tokens, we eliminate the redundant computation.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    assert y.is_cuda, "y must be CUDA tensor"
    assert y.dtype == torch.bfloat16, "fused masked kernel currently expects bf16 input"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + BLOCK_QUANT_SIZE - 1) // BLOCK_QUANT_SIZE
    assert H % BLOCK_QUANT_SIZE == 0, "H must be divisible by 128"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    if tokens_per_expert.device != y.device or tokens_per_expert.dtype != torch.int32:
        tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    if y_q is None:
        y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)
    else:
        assert y_q.shape == (E, T, H), "y_q shape must be (E, T, H)"
        assert y_q.dtype == fp8_dtype and y_q.is_cuda
    if y_s is None:
        # DeepGEMM requires column major
        ys_shape, ys_strides, ys_dtype = (E, T, G), (T * G, 1, T), torch.float32
        y_s = torch.empty_strided(
            ys_shape,
            ys_strides,
            dtype=ys_dtype,
            device=y.device,
        )
    else:
        assert y_s.shape == (E, T, G), "y_s shape must be (E, T, G)"
        assert y_s.dtype == torch.float32 and y_s.is_cuda

    pipeswift_kernels.persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        y_q,
        y_s,
        False,
        reverse=reverse,  # do not use e8m0
    )
    return y_q, y_s


def dequant_weight_fp8_block(
    w_fp8: torch.Tensor,  # [out_dim, inp_dim], torch.float8_e4m3fn/e4m3fnuz
    s: torch.Tensor,  # [out_dim//128, inp_dim//128]
    block_n: int = 128,
    block_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,  # or torch.float16
) -> torch.Tensor:
    out_dim, inp_dim = w_fp8.shape

    # [Ob, Ib] -> [Ob, 1, Ib, 1] -> [Ob, 128, Ib, 128] -> [out_dim_pad, inp_dim_pad]
    s_full = (
        s[:, None, :, None]
        .expand(s.shape[0], block_n, s.shape[1], block_k)
        .reshape(s.shape[0] * block_n, s.shape[1] * block_k)
    )
    s_full = s_full[:out_dim, :inp_dim]
    w = (w_fp8.to(torch.float32) * s_full.to(torch.float32)).to(out_dtype)
    return w
