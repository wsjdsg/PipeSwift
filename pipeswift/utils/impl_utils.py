from typing import Tuple

import torch

cos_sin_cache: torch.Tensor = None
import math

import flashinfer
import numpy as np
import torch.distributed as dist

from .metadata import KVCacheInfo
from .triton_kernels import *


# copied from sglang
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _compute_inv_freq_llama2(rope_theta, head_dim):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float, device="cpu") / head_dim)
    )
    return inv_freq


def _compute_inv_freq_llama3(rope_theta, head_dim, rope_scaling) -> torch.Tensor:
    inv_freqs = _compute_inv_freq_llama2(rope_theta, head_dim)
    orig_max_position = rope_scaling["original_max_position_embeddings"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    scaling_factor = rope_scaling["factor"]
    low_freq_wavelen = orig_max_position / low_freq_factor
    high_freq_wavelen = orig_max_position / high_freq_factor

    wave_len = 2 * math.pi / inv_freqs
    if low_freq_factor != high_freq_factor:
        smooth = (orig_max_position / wave_len - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = 0
    new_freqs = torch.where(
        wave_len < high_freq_wavelen,
        inv_freqs,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freqs / scaling_factor,
            (1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs,
        ),
    )
    return new_freqs


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device = None
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _compute_inv_freq_dpsk(
    rope_theta, head_dim, scaling_factor: float, rope_scaling, extrapolation_factor=1.0
) -> torch.Tensor:
    pos_freqs = rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = _yarn_find_correction_range(
        rope_scaling["beta_fast"],
        rope_scaling["beta_slow"],
        head_dim,
        rope_theta,
        rope_scaling["original_max_position_embeddings"],
    )
    # Get n-d rotational scaling corrected for extrapolation
    inv_freq_mask = (
        1 - _yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float)
    ) * extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    return inv_freq


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_and_set_cos_sin_cache(
    rotary_dim, max_position_embedding, rope_theta, rope_scaling=None
):
    """Compute the cos and sin cache."""
    if rope_scaling is None:
        inv_freq = _compute_inv_freq_llama2(rope_theta, rotary_dim)
        t = torch.arange(max_position_embedding, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
    else:
        _rope_type_key = "type" if "type" in rope_scaling.keys() else "rope_type"
        if rope_scaling[_rope_type_key] == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            mscale, mscale_alldim = (
                rope_scaling["mscale"],
                rope_scaling["mscale_all_dim"],
            )
            attn_factor = getattr(rope_scaling, "attn_factor", 1.0)
            mscale = float(
                yarn_get_mscale(scaling_factor, float(mscale))
                / yarn_get_mscale(scaling_factor, float(mscale_alldim))
                * attn_factor
            )
            inv_freq = _compute_inv_freq_dpsk(
                rope_theta, rotary_dim, scaling_factor, rope_scaling
            )
            t = torch.arange(
                original_max_position * scaling_factor,
                dtype=torch.float32,
            )
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos() * mscale
            sin = freqs.sin() * mscale
            cache = torch.cat((cos, sin), dim=-1)
        elif rope_scaling[_rope_type_key] == "llama3":
            inv_freq = _compute_inv_freq_llama3(rope_theta, rotary_dim, rope_scaling)
            t = torch.arange(max_position_embedding, dtype=torch.float)

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
        else:
            raise NotImplementedError(
                f"Rope scaling type {rope_scaling['type']} not implemented"
            )
    global cos_sin_cache
    cos_sin_cache = cache.cuda()  # HF impl requires cache.to(x.dtype). However flashinfer's kernel necessitate cache to be fp32


def topk_routing(
    gating_output: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = None,
):
    routing_weights, selected_experts = fused_topk(
        gating_output=gating_output,
        topk=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        bias=None,
    )
    if routed_scaling_factor is not None:
        routing_weights *= routed_scaling_factor
    return routing_weights, selected_experts


def topk_grouped_routing(
    gating_output: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    n_group: int = None,
    topk_group: int = None,
    routed_scaling_factor: float = None,
    e_score_correction_bias: torch.Tensor = None,  # for GLM MoE
):
    if n_group <= 32 and top_k <= 32 and e_score_correction_bias is not None:
        return fused_grouped_topk(
            gating_output=gating_output,
            topk=top_k,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=n_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, n_group, -1).max(dim=-1).values
        )  # [n, n_group]
    use_sorted = False
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, n_group, scores.size(-1) // n_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=top_k, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def routing(
    gating_output: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    topk_method: str = "greedy",
    n_group: int = None,
    topk_group: int = None,
    routed_scaling_factor: float = None,
    e_score_correction_bias: torch.Tensor = None,  # for GLM MoE
):
    if topk_method == "greedy":  # Qwen3MoE, DeepSeek-V2
        return topk_routing(
            gating_output,
            top_k=top_k,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
    elif topk_method == "noaux_tc":  # DeepSeek-V3 & GLM4.7
        # copied from transformers
        return topk_grouped_routing(
            gating_output=gating_output,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
    else:
        raise NotImplemented(f"{topk_method} routing method not implemented yet!")


def append_kv_cache(key, value, kv_cache_info: KVCacheInfo = None, layer_idx: int = 0):
    if key.shape[0] == 0:
        return
    triton_reshape_and_cache_flash(
        key,
        value,
        kv_cache_info.kv_cache[0][layer_idx],
        kv_cache_info.kv_cache[1][layer_idx],
        kv_cache_info.slot_mapping,
    )


def append_mla_kv_cache(
    ckv, kpe, kv_cache_info: KVCacheInfo = None, layer_idx: int = 0
):
    if ckv.shape[0] == 0:
        return
    triton_reshape_and_cache_flash_diffkv(
        ckv.unsqueeze(1),
        kpe.unsqueeze(1),
        kv_cache_info.mla_cache_tensor[layer_idx].unsqueeze(2),
        kv_cache_info.slot_mapping,
    )


def apply_rotary_embedding(postion_ids, query, key, head_dim, is_neox=False):
    global cos_sin_cache
    flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
        positions=postion_ids,
        query=query,
        key=key,
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def coalesced_to_cuda(arr_list):
    for i in range(len(arr_list)):
        if not isinstance(arr_list[i], np.ndarray):
            if isinstance(arr_list[i], torch.Tensor):
                arr_list[i] = arr_list[i].numpy()
            else:
                arr_list[i] = np.array(arr_list[i], dtype=np.int32)
    sizes = [t.size for t in arr_list]
    cpu_np_arrary = np.concat(
        arr_list
    )  # use np.concat rather than torch.concat as the latter degrades efficiency
    big_cpu = torch.from_numpy(cpu_np_arrary)
    big_gpu = big_cpu.to("cuda", non_blocking=True)
    gpu_tensors = []
    offset = 0
    for n in sizes:
        gpu_tensors.append(big_gpu[offset : offset + n])
        offset += n
    return gpu_tensors


def gather_query_along_head_across_dcp(query: torch.Tensor):
    from .mpu import parallel_state

    query_gather_head_along_dcp = torch.empty(
        (parallel_state.dcp_deg,) + query.shape, dtype=query.dtype, device="cuda"
    )
    # [N, S, H, D]
    torch.distributed.all_gather_into_tensor(
        query_gather_head_along_dcp, query.contiguous(), parallel_state.dcp_group
    )
    query_gather_head_along_dcp = (
        query_gather_head_along_dcp.transpose(0, 1)
        .contiguous()
        .view(query.shape[0], -1, query.shape[-1])
    )
    return query_gather_head_along_dcp


def ag_lse_correct_attn_and_rs(
    attn_out: torch.Tensor, lse: torch.Tensor, ctx: CPTritonContext | None = None
):
    # attn_out: [B, H, D], lse: [B, H]
    from .mpu import parallel_state

    assert lse.dtype == torch.float32

    if ctx is None:
        ctx = CPTritonContext()
    gathered_lse = torch.empty(
        (parallel_state.dcp_deg,) + lse.shape, device="cuda", dtype=lse.dtype
    )
    dist.all_gather_into_tensor(gathered_lse, lse, parallel_state.dcp_group)
    final_out, final_lse = correct_attn_out(
        attn_out, gathered_lse, parallel_state.dcp_rank, ctx, is_lse_base_on_e=True
    )
    final_lse = torch.tensor_split(final_lse, parallel_state.dcp_deg, 1)[
        parallel_state.dcp_rank
    ]
    # [B, H, D] -> [B, H // dcp, D]
    # NOTE(wsj): needs reduce local head results
    final_out = final_out.transpose(0, 1).contiguous()
    out_rs = torch.empty(
        (final_out.shape[0] // parallel_state.dcp_deg,) + final_out.shape[1:],
        dtype=final_out.dtype,
        device="cuda",
    )
    dist.reduce_scatter_tensor(
        out_rs, final_out, op=dist.ReduceOp.SUM, group=parallel_state.dcp_group
    )
    return out_rs.transpose(0, 1).contiguous(), final_lse


def ag_lse_and_correct_attn(
    attn_out: torch.Tensor, lse: torch.Tensor, ctx: CPTritonContext | None = None
):
    # attn_out: [B, H, D], lse: [B, H]
    from .mpu import parallel_state

    assert lse.dtype == torch.float32

    if ctx is None:
        ctx = CPTritonContext()

    gathered_lse = torch.empty(
        (parallel_state.dcp_deg,) + lse.shape, device="cuda", dtype=lse.dtype
    )
    dist.all_gather_into_tensor(gathered_lse, lse, parallel_state.dcp_group)
    final_out, final_lse = correct_attn_out(
        attn_out, gathered_lse, parallel_state.dcp_rank, ctx, is_lse_base_on_e=True
    )
    final_lse = torch.tensor_split(final_lse, parallel_state.dcp_deg, 1)[
        parallel_state.dcp_rank
    ]
    return final_out, final_lse
