import triton
import triton.language as tl


# DeepGEMM act
@triton.jit
def mul_silu_masked_kernel(
    hidden_ptr,
    out_ptr,
    masked_m_ptr,
    stride_h0,
    stride_h1,
    stride_h2,
    stride_o0,
    stride_o1,
    stride_o2,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)

    m_valid = tl.load(masked_m_ptr + pid_b)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < m_valid
    h_mask = offs_h < H
    mask = m_mask[:, None] & h_mask[None, :]

    ptr_x = (
        hidden_ptr
        + pid_b * stride_h0
        + offs_m[:, None] * stride_h1
        + offs_h[None, :] * stride_h2
    )

    ptr_y = (
        hidden_ptr
        + pid_b * stride_h0
        + offs_m[:, None] * stride_h1
        + (offs_h[None, :] + H) * stride_h2
    )

    x = tl.load(ptr_x, mask=mask, other=0.0)
    y = tl.load(ptr_y, mask=mask, other=0.0)

    y_fp32 = y.to(tl.float32)
    silu_y = y_fp32 * tl.sigmoid(y_fp32)
    silu_y = silu_y.to(x.dtype)

    out = x * silu_y

    ptr_out = (
        out_ptr
        + pid_b * stride_o0
        + offs_m[:, None] * stride_o1
        + offs_h[None, :] * stride_o2
    )

    tl.store(ptr_out, out, mask=mask)


# silu act
@triton.jit
def silu_and_mul_kernel(
    X,  # [N, 2*H]
    Y,  # [N, H]
    stride_xn,
    stride_yn,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    reverse: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < H

    if reverse:
        x1_ptr = X + pid * stride_xn + offs + H
        x2_ptr = X + pid * stride_xn + offs
    else:
        x1_ptr = X + pid * stride_xn + offs
        x2_ptr = X + pid * stride_xn + offs + H

    x1 = tl.load(x1_ptr, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr, mask=mask, other=0.0)

    x1_fp32 = x1.to(tl.float32)
    silu_fp32 = x1_fp32 * tl.sigmoid(x1_fp32)

    out = silu_fp32.to(x1.dtype) * x2

    y_ptr = Y + pid * stride_yn + offs
    tl.store(y_ptr, out, mask=mask)


# For Decode Context Parallel
@triton.jit
def cp_merge_attn_out_kernel(
    attn_out_ptr,  # [N, B, H, D]
    lse_ptr,  # [N, B, H]
    out_ptr,  # [B, H, D]
    lse_out_ptr,  # [B, H]
    stride_ON,
    stride_OB,
    stride_OH,
    stride_OD,
    stride_LN,
    stride_LB,
    stride_LH,
    stride_OB_out,
    stride_OH_out,
    stride_OD_out,
    stride_LB_out,
    stride_LH_out,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    d = tl.arange(0, HEAD_DIM)

    # -------------------------------------------------
    # 1. compute lse_final = logsumexp_n(lse_n)
    # -------------------------------------------------
    lse_max = -float("inf")

    # first pass: max
    for n in tl.static_range(0, N):
        lse_offset = n * stride_LN + b * stride_LB + h * stride_LH
        lse_n = tl.load(lse_ptr + lse_offset)
        lse_n = tl.where(
            (lse_n != lse_n) | (lse_n == float("inf")),
            -float("inf"),
            lse_n,
        )
        lse_max = tl.maximum(lse_max, lse_n)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # second pass: sum exp
    lse_sum = 0.0
    for n in tl.static_range(0, N):
        lse_offset = n * stride_LN + b * stride_LB + h * stride_LH
        lse_n = tl.load(lse_ptr + lse_offset)
        lse_n = tl.where(
            (lse_n != lse_n) | (lse_n == float("inf")),
            -float("inf"),
            lse_n,
        )
        diff = lse_n - lse_max
        lse_sum += tl.exp(diff) if IS_BASE_E else tl.exp2(diff)

    lse_final = (tl.log(lse_sum) if IS_BASE_E else tl.log2(lse_sum)) + lse_max

    # store final lse
    tl.store(
        lse_out_ptr + b * stride_LB_out + h * stride_LH_out,
        lse_final,
    )

    # -------------------------------------------------
    # 2. compute attn_out_final
    # -------------------------------------------------
    out_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(0, N):
        # load lse_n
        lse_offset = n * stride_LN + b * stride_LB + h * stride_LH
        lse_n = tl.load(lse_ptr + lse_offset)
        diff = lse_n - lse_final
        diff = tl.where(
            (diff != diff) | (diff == float("inf")),
            -float("inf"),
            diff,
        )
        factor = tl.exp(diff) if IS_BASE_E else tl.exp2(diff)

        # load out_n
        out_offset = n * stride_ON + b * stride_OB + h * stride_OH + d * stride_OD
        out_n = tl.load(attn_out_ptr + out_offset)

        out_acc += out_n * factor

    # store output
    out_offset = b * stride_OB_out + h * stride_OH_out + d * stride_OD_out
    tl.store(out_ptr + out_offset, out_acc)


@triton.jit
def correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )


@triton.jit
def fp8_block_dequant_kernel(
    hidden_ptr,
    scale_ptr,
    out_ptr,
    B,
    D,
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    cols = pid_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    hidden_ptrs = hidden_ptr + pid_row * D + cols
    out_ptrs = out_ptr + pid_row * D + cols
    scale_idx = pid_row * N_BLOCKS + pid_blk

    h = tl.load(hidden_ptrs, mask=mask, other=0.0).to(tl.bfloat16)
    s = tl.load(scale_ptr + scale_idx).to(tl.bfloat16)
    o = h * s
    tl.store(out_ptrs, o, mask=mask)


@triton.jit
def masked_fp8_block_quant_kernel(
    input_ptr,  # (E, T, H)
    counts_ptr,  # (E,)
    y_q_ptr,  # (E, T, H) fp8
    y_s_ptr,  # (E, T, G) float32, stride expected (T*G, 1, T)
    E,
    T,
    H,
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_q_e,
    stride_q_t,
    stride_q_h,
    stride_s_e,
    stride_s_t,
    stride_s_g,
    stride_c_e,
    fp8_min,
    fp8_max,
    eps,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_mt = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_mt * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_g * BLOCK + tl.arange(0, BLOCK)
    valid_t = tl.load(counts_ptr + pid_e * stride_c_e)
    m_mask = offs_m < valid_t
    k_mask = offs_k < H
    mask = m_mask[:, None] & k_mask[None, :]

    in_ptrs = (
        input_ptr
        + pid_e * stride_i_e
        + offs_m[:, None] * stride_i_t
        + offs_k[None, :] * stride_i_h
    )
    x = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(absmax / fp8_max, eps)
    scale = tl.where(m_mask, scale, 0.0)
    scale_safe = tl.where(scale > 0, scale, 1.0)

    q = tl.clamp(x / scale_safe[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    out_ptrs = (
        y_q_ptr
        + pid_e * stride_q_e
        + offs_m[:, None] * stride_q_t
        + offs_k[None, :] * stride_q_h
    )
    tl.store(out_ptrs, q, mask=mask)

    s_ptr = y_s_ptr + pid_e * stride_s_e + offs_m * stride_s_t + pid_g * stride_s_g
    tl.store(s_ptr, scale, mask=m_mask)


@triton.jit
def per_token_fp8_channelwise_quant_kernel(
    input_ptr,
    out_ptr,
    scale_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    stride_scale_m,
    fp8_min,
    fp8_max,
    min_scaling_factor,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)

    input_ptrs = input_ptr + pid_m * stride_in_m + offs_n * stride_in_n
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax / fp8_max, min_scaling_factor)

    q = tl.clamp(x / scale, fp8_min, fp8_max).to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + pid_m * stride_out_m + offs_n * stride_out_n
    tl.store(out_ptrs, q, mask=mask)
    tl.store(scale_ptr + pid_m * stride_scale_m, scale, mask=pid_m < M)


@triton.jit
def causal_attn_uniform_q_len_suffix_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kq,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vq,
    stride_vh,
    stride_vd,
    stride_pob,
    stride_poq,
    stride_poh,
    stride_pod,
    stride_plb,
    stride_plq,
    stride_plh,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lq,
    stride_lh,
    num_q_heads,
    q_len,
    qk_dim,
    v_dim,
    sm_scale,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    MAX_Q: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    head_tile_idx = tl.program_id(2)

    offs_g = tl.arange(0, BLOCK_G)
    q_head_idx = head_tile_idx * BLOCK_G + offs_g
    mask_g = q_head_idx < num_q_heads

    offs_q = tl.arange(0, MAX_Q)
    valid_q = offs_q < q_len

    q_base = q_ptr + batch_idx * stride_qb + q_idx * stride_qq
    k_base = k_ptr + batch_idx * stride_kb
    v_base = v_ptr + batch_idx * stride_vb

    causal_q = offs_q <= q_idx
    valid_k = valid_q & causal_q

    logits = tl.zeros((MAX_Q, BLOCK_G), dtype=tl.float32)
    for start_d in tl.range(0, qk_dim, BLOCK_DQK):
        offs_d = start_d + tl.arange(0, BLOCK_DQK)
        mask_d = offs_d < qk_dim
        q = tl.load(
            q_base + q_head_idx[:, None] * stride_qh + offs_d[None, :] * stride_qd,
            mask=mask_g[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_base + offs_q[:, None] * stride_kq + offs_d[None, :] * stride_kd,
            mask=valid_q[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        logits += tl.dot(k, tl.trans(q))

    logits *= sm_scale
    logits = tl.where(valid_k[:, None] & mask_g[None, :], logits, -float("inf"))
    m_i = tl.max(logits, axis=0)
    p = tl.exp(logits - m_i[None, :])
    l_i = tl.sum(p, axis=0)
    suffix_lse = tl.log(l_i) + m_i
    if HAS_PREFIX:
        prefix_lse = tl.load(
            prefix_lse_ptr
            + batch_idx * stride_plb
            + q_idx * stride_plq
            + q_head_idx * stride_plh,
            mask=mask_g,
            other=-float("inf"),
        )
        prefix_lse = tl.where(
            (prefix_lse != prefix_lse) | (prefix_lse == float("inf")),
            -float("inf"),
            prefix_lse,
        )
        suffix_lse = tl.where(
            (suffix_lse != suffix_lse) | (suffix_lse == float("inf")),
            -float("inf"),
            suffix_lse,
        )
        max_lse = tl.maximum(prefix_lse, suffix_lse)
        lse = max_lse + tl.log(
            tl.exp(prefix_lse - max_lse) + tl.exp(suffix_lse - max_lse)
        )
        prefix_scale = tl.exp(prefix_lse - lse)
        suffix_scale = tl.exp(suffix_lse - lse)
    else:
        lse = suffix_lse
        prefix_scale = 0.0
        suffix_scale = 1.0
    tl.store(
        lse_ptr + batch_idx * stride_lb + q_idx * stride_lq + q_head_idx * stride_lh,
        lse,
        mask=mask_g,
    )
    p = p / l_i[None, :]

    for start_dv in tl.range(0, v_dim, BLOCK_DV):
        offs_dv = start_dv + tl.arange(0, BLOCK_DV)
        mask_dv = offs_dv < v_dim
        v = tl.load(
            v_base + offs_q[:, None] * stride_vq + offs_dv[None, :] * stride_vd,
            mask=valid_k[:, None] & mask_dv[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(p[:, :, None] * v[:, None, :], axis=0)
        if HAS_PREFIX:
            prefix_out = tl.load(
                prefix_out_ptr
                + batch_idx * stride_pob
                + q_idx * stride_poq
                + q_head_idx[:, None] * stride_poh
                + offs_dv[None, :] * stride_pod,
                mask=mask_g[:, None] & mask_dv[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = prefix_out * prefix_scale[:, None] + acc * suffix_scale[:, None]
        tl.store(
            out_ptr
            + batch_idx * stride_ob
            + q_idx * stride_oq
            + q_head_idx[:, None] * stride_oh
            + offs_dv[None, :] * stride_od,
            acc,
            mask=mask_g[:, None] & mask_dv[None, :],
        )
