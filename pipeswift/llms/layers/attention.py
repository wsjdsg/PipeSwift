import flashinfer
import torch

from pipeswift.utils import (
    KVCacheInfo,
    ag_lse_and_correct_attn,
    ag_lse_correct_attn_and_rs,
    gather_query_along_head_across_dcp,
    merge_attn_states,
)
from pipeswift.utils.triton_kernels.common import causal_attn_uniform_q_len_suffix

flashinfer_workspace_buffer: torch.Tensor = None
flashinfer_mla_decode_wrapper: flashinfer.mla.BatchMLAPagedAttentionWrapper = (
    None  # only used for prefill
)
flash_mla_tile_sche_meta, flash_mla_num_splits = None, None
cg_flash_mla_tile_scheduler_metadata, cg_flash_mla_num_splits = None, None


def set_flashmla_cg_meta(max_bsz):
    global cg_flash_mla_tile_scheduler_metadata, cg_flash_mla_num_splits
    device_properties = torch.cuda.get_device_properties("cuda")
    num_sms = device_properties.multi_processor_count
    cg_flash_mla_tile_scheduler_metadata = torch.zeros(
        (num_sms, 8), device="cuda", dtype=torch.int32
    )
    cg_flash_mla_num_splits = torch.empty(max_bsz + 1, device="cuda", dtype=torch.int32)


def set_flash_mla_info(
    kv_cache_info: KVCacheInfo, num_q_head, is_decode=False, cg_bsz: int = -1
):
    from flash_mla import get_mla_metadata

    global flash_mla_tile_sche_meta, flash_mla_num_splits, cg_flash_mla_num_splits
    if is_decode:
        assert kv_cache_info is not None
        assert kv_cache_info.actual_lens_q_cpu is not None
        q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())
        cache_seqlens = kv_cache_info.actual_lens_kv
        if cg_bsz > 0 and cache_seqlens.size(0) < cg_bsz:
            padded_cache_seqlens = torch.zeros(
                (cg_bsz,), device=cache_seqlens.device, dtype=cache_seqlens.dtype
            )
            padded_cache_seqlens[: cache_seqlens.size(0)].copy_(
                kv_cache_info.actual_lens_kv
            )
            cache_seqlens = padded_cache_seqlens
        flash_mla_tile_sche_meta, flash_mla_num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_q_tokens_per_head_k=q_len * num_q_head,
            num_heads_k=1,
            num_heads_q=num_q_head,
            is_fp8_kvcache=False,
        )
        if cg_bsz > 0:
            sm_parts = flash_mla_tile_sche_meta.size(0)
            assert sm_parts <= cg_flash_mla_tile_scheduler_metadata.size(0)
            cg_flash_mla_tile_scheduler_metadata[:sm_parts].copy_(
                flash_mla_tile_sche_meta
            )
            n = flash_mla_num_splits.size(0)
            assert n <= cg_flash_mla_num_splits.size(0)
            cg_flash_mla_num_splits[:n].copy_(flash_mla_num_splits)
            flash_mla_tile_sche_meta, flash_mla_num_splits = (
                cg_flash_mla_tile_scheduler_metadata[:sm_parts],
                cg_flash_mla_num_splits[:n],
            )


def _reshape_decode_query(query: torch.Tensor, kv_cache_info: KVCacheInfo):
    assert kv_cache_info.actual_lens_q_cpu is not None
    q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())
    batch_size = query.shape[0] // q_len
    return query.contiguous().view(batch_size, q_len, query.shape[1], query.shape[2])


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        qkhead_dim,
        attention_backend="flash_attn",
        causal=True,
        use_mla=False,
        layer_id: int = 0,
        gqa_group_size: int = 1,
        rope_scaling: dict = None,
    ):
        super().__init__()

        # `d ** -0.5`, not `1.0 / (d ** 0.5)`: the two differ by 1 ulp in
        # binary64 (for d=128, ...845 vs ...843), and vLLM uses the former
        # (`self.scaling = self.head_dim**-0.5` in its model definitions). The
        # numerical difference is negligible on its own; matching it means the
        # softmax_scale handed to FlashAttention is bit-identical to vLLM's, so
        # a divergence between the two engines can never be blamed on this.
        self.scale = qkhead_dim**-0.5
        # YaRN attention-scale correction (DeepSeek-V2/V3/R1 MLA).
        # HF reference: softmax_scale *= yarn_get_mscale(factor, mscale_all_dim) ** 2
        # when mscale_all_dim is set. PipeSwift's cos/sin cache already handles the
        # RoPE-side mscale ratio; this is the separate softmax-side factor.
        # Toggle off with PIPESWIFT_DISABLE_YARN_ATTN_MSCALE=1 for A/B testing.
        import os

        if (
            use_mla
            and isinstance(rope_scaling, dict)
            and os.getenv("PIPESWIFT_DISABLE_YARN_ATTN_MSCALE", "0") != "1"
        ):
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if rope_type == "yarn":
                mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
                if mscale_all_dim:
                    from pipeswift.utils.impl_utils import yarn_get_mscale

                    scaling_factor = rope_scaling["factor"]
                    mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                    self.scale *= mscale * mscale
                    if layer_id == 0:
                        print(
                            f"[SelfAttention] YaRN attn mscale²={mscale * mscale:.6f} "
                            f"applied: softmax_scale={self.scale:.6f}",
                            flush=True,
                        )
        self.causal = causal
        self.attention_backend = attention_backend
        self.use_mla = use_mla
        self.layer_idx = layer_id
        self.gqa_group_size = gqa_group_size
        if self.attention_backend == "flash_mla":
            import flashinfer

            global flashinfer_workspace_buffer, flashinfer_mla_decode_wrapper
            if flashinfer_workspace_buffer == None:
                flashinfer_workspace_buffer = torch.zeros(
                    256 * 1024 * 1024, dtype=torch.int8, device="cuda"
                )

            if use_mla and flashinfer_mla_decode_wrapper is None:
                flashinfer_mla_decode_wrapper = (
                    flashinfer.mla.BatchMLAPagedAttentionWrapper(
                        flashinfer_workspace_buffer,
                        backend="fa3" if "H" in torch.cuda.get_device_name() else "fa2",
                    )
                )

    def _fp8_paged_attn(
        self,
        query,
        kv_cache_info: KVCacheInfo,
        q_len: int,
        causal: bool,
        return_softmax_lse: bool = False,
    ):
        """Paged GQA attention over an FP8 e4m3 KV cache, via vLLM's FA3 fork.

        Mirrors vLLM's FlashAttentionImpl for kv_cache_dtype="fp8": the query is
        quantized to e4m3 too (the kernel takes one dtype for all three operands)
        and all descales are 1.0. ``max_seqlen_k=None`` follows upstream FA3's own
        ``flash_attn_with_kvcache`` convention -- the kernel derives it from
        ``seqused_k`` on device, which keeps the call CUDA-graph safe.
        """
        from pipeswift.utils.fp8_kv import attn_varlen_func, quantize, unit_descale

        k_cache = kv_cache_info.kv_cache[0][self.layer_idx]
        v_cache = kv_cache_info.kv_cache[1][self.layer_idx]
        descale = unit_descale(
            kv_cache_info.cu_seqlens_q.shape[0] - 1, k_cache.shape[2], query.device
        )
        return attn_varlen_func()(
            q=quantize(query),
            k=k_cache,
            v=v_cache,
            max_seqlen_q=q_len,
            cu_seqlens_q=kv_cache_info.cu_seqlens_q,
            max_seqlen_k=None,
            seqused_k=kv_cache_info.actual_lens_kv,
            block_table=kv_cache_info.page_table,
            softmax_scale=self.scale,
            causal=causal,
            fa_version=3,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            return_softmax_lse=return_softmax_lse,
        )

    def forward_gqa_attn(self, query, kv_cache_info: KVCacheInfo = None):
        from pipeswift.utils.fp8_kv import is_enabled as fp8_kv_enabled

        assert kv_cache_info is not None
        assert kv_cache_info.actual_lens_q_cpu is not None
        q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())

        if fp8_kv_enabled():
            return self._fp8_paged_attn(query, kv_cache_info, q_len, self.causal)

        from flash_attn_interface import flash_attn_with_kvcache

        return flash_attn_with_kvcache(
            query,
            kv_cache_info.kv_cache[0][self.layer_idx],
            kv_cache_info.kv_cache[1][self.layer_idx],
            cu_seqlens_q=kv_cache_info.cu_seqlens_q,
            cache_seqlens=kv_cache_info.actual_lens_kv,
            max_seqlen_q=q_len,
            softmax_scale=self.scale,
            causal=self.causal,
            page_table=kv_cache_info.page_table,
        )

    def forward_mla_attn(
        self, q_nope, q_pe, kv_cache_info: KVCacheInfo, ckv_head_dim: int, is_decode
    ):
        assert self.attention_backend != "flash_attn", (
            "flash_attn doesn't support mla decode"
        )
        if not is_decode:
            ckv_cache, kpe_cache = (
                kv_cache_info.kv_cache[0][self.layer_idx],
                kv_cache_info.kv_cache[1][self.layer_idx],
            )
            return flashinfer_mla_decode_wrapper.run(
                q_nope=q_nope,
                q_pe=q_pe,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
            )
        else:
            from flash_mla import flash_mla_with_kvcache

            q = torch.concat((q_nope, q_pe), dim=-1)
            q = _reshape_decode_query(q, kv_cache_info)
            o, _ = flash_mla_with_kvcache(
                q=q,
                k_cache=kv_cache_info.mla_cache_tensor[self.layer_idx].unsqueeze(2),
                block_table=kv_cache_info.page_table,
                cache_seqlens=kv_cache_info.actual_lens_kv,
                head_dim_v=ckv_head_dim,
                tile_scheduler_metadata=flash_mla_tile_sche_meta,
                num_splits=flash_mla_num_splits,
                softmax_scale=self.scale,
                causal=True,
                is_fp8_kvcache=False,
                indices=None,
            )
            return o.contiguous().view(-1, o.shape[2], o.shape[3])

    def forward_gqa_dcp(
        self,
        query,
        key,
        value,
        kv_cache_info: KVCacheInfo = None,
        is_decode: bool = False,
    ):
        from flash_attn_interface import (
            flash_attn_with_kvcache,  # use fa3 for hopper GPU
        )
        from flash_attn_interface import flash_attn_varlen_func

        from pipeswift.utils import parallel_state
        from pipeswift.utils.fp8_kv import is_enabled as fp8_kv_enabled

        assert kv_cache_info is not None
        assert kv_cache_info.actual_lens_q_cpu is not None
        q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())
        if not is_decode:
            query_gathered_along_dcp = gather_query_along_head_across_dcp(query)
        else:
            query_gathered_along_dcp = query

        # [B, H, D], [H, B]
        # The prefix half reads the paged cache, so it follows the cache dtype;
        # the suffix half below runs on raw bf16 k/v either way, and the two are
        # combined through the fp32 LSE merge.
        if fp8_kv_enabled():
            prefix_out, prefix_lse = self._fp8_paged_attn(
                query_gathered_along_dcp,
                kv_cache_info,
                q_len,
                causal=False,
                return_softmax_lse=True,
            )
        else:
            prefix_out, prefix_lse, _, _ = flash_attn_with_kvcache(
                query_gathered_along_dcp,
                kv_cache_info.kv_cache[0][self.layer_idx],
                kv_cache_info.kv_cache[1][self.layer_idx],
                cu_seqlens_q=kv_cache_info.cu_seqlens_q,
                cache_seqlens=kv_cache_info.actual_lens_kv,
                max_seqlen_q=q_len,
                softmax_scale=self.scale,
                causal=False,
                page_table=kv_cache_info.page_table,
                return_softmax_lse=True,
            )
        prefix_lse = prefix_lse.t().contiguous()

        if is_decode:
            if q_len > 1 and parallel_state.dcp_rank == parallel_state.dcp_deg - 1:
                suffix_out, suffix_lse = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens_q=kv_cache_info.cu_seqlens_q,
                    cu_seqlens_k=kv_cache_info.cu_seqlens_q,
                    max_seqlen_q=q_len,
                    max_seqlen_k=q_len,
                    softmax_scale=self.scale,
                    causal=self.causal,
                    return_attn_probs=True,
                )
                merged_out = torch.empty_like(prefix_out)
                merged_lse = torch.empty(
                    (prefix_out.shape[1], prefix_out.shape[0]),
                    device=prefix_lse.device,
                    dtype=prefix_lse.dtype,
                )
                merge_attn_states(
                    merged_out,
                    prefix_out,
                    prefix_lse.t().contiguous(),
                    suffix_out,
                    suffix_lse,
                    output_lse=merged_lse,
                )
                prefix_out = merged_out
                prefix_lse = merged_lse.t().contiguous()
            prefix_out, prefix_lse = ag_lse_and_correct_attn(prefix_out, prefix_lse)
            return prefix_out

        prefix_out, prefix_lse = ag_lse_correct_attn_and_rs(prefix_out, prefix_lse)

        # prefill suffix attention
        prefix_lse = prefix_lse.transpose(0, 1).contiguous()
        suffix_out, suffix_lse = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=kv_cache_info.cu_seqlens_q,
            cu_seqlens_k=kv_cache_info.cu_seqlens_q,
            max_seqlen_q=kv_cache_info.cu_seqlens_q_cpu.max(),
            max_seqlen_k=kv_cache_info.cu_seqlens_q_cpu.max(),
            softmax_scale=self.scale,
            causal=self.causal,
            return_attn_probs=True,
        )
        out = torch.empty_like(query)
        merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)
        return out

    def forward_mla_dcp(
        self,
        q_nope,
        q_pe,
        q,
        k,
        v,
        ckv,
        k_pe,
        ckv2v_weight: torch.Tensor,
        kv_cache_info: KVCacheInfo,
        ckv_head_dim: int,
        is_decode,
    ):
        from pipeswift.utils import parallel_state

        if not is_decode:  # two stages, kv cache and k, v
            from flash_attn_interface import flash_attn_varlen_func

            suffix_out, suffix_lse = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=kv_cache_info.cu_seqlens_q,
                cu_seqlens_k=kv_cache_info.cu_seqlens_q,
                max_seqlen_q=kv_cache_info.actual_lens_q_cpu.max().item(),
                max_seqlen_k=kv_cache_info.actual_lens_q_cpu.max().item(),
                softmax_scale=self.scale,
                causal=self.causal,
                return_attn_probs=True,
            )
            if torch.sum(kv_cache_info.actual_lens_kv_cpu).item() == 0:
                return suffix_out
            else:
                _q = torch.concat((q_nope, q_pe), dim=-1)
                q_gather_head_along_dcp = gather_query_along_head_across_dcp(_q)
                q_nope_gathered, q_pe_gathered = q_gather_head_along_dcp.split(
                    [q_nope.shape[-1], q_pe.shape[-1]], dim=-1
                )
                ckv_cache, kpe_cache = (
                    kv_cache_info.kv_cache[0][self.layer_idx],
                    kv_cache_info.kv_cache[1][self.layer_idx],
                )
                prefix_out, prefix_lse = flashinfer_mla_decode_wrapper.run(
                    q_nope=q_nope_gathered,
                    q_pe=q_pe_gathered,
                    ckv_cache=ckv_cache,
                    kpe_cache=kpe_cache,
                    return_lse=True,
                )
                prefix_out, prefix_lse = ag_lse_correct_attn_and_rs(
                    prefix_out, prefix_lse
                )
                _ckv2v_weight = ckv2v_weight.tensor_split(parallel_state.dcp_deg, 0)[
                    parallel_state.dcp_rank
                ]
                prefix_out = torch.einsum("bhr,hdr->bhd", prefix_out, _ckv2v_weight)
                prefix_out = prefix_out.contiguous()
                out = torch.empty_like(v)
                # Transpose LSE from (B, H) to (H, B) for merge_attn_states
                merge_attn_states(
                    out,
                    prefix_out,
                    prefix_lse.t().contiguous(),
                    suffix_out,
                    suffix_lse,
                )
                return out
        else:  # one stage, only access kv cache
            from flash_mla import flash_mla_with_kvcache

            assert kv_cache_info.actual_lens_q_cpu is not None
            q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())
            _q = torch.concat((q_nope, q_pe), dim=-1)
            _q = _reshape_decode_query(_q, kv_cache_info)
            prefix_out, prefix_lse = flash_mla_with_kvcache(
                q=_q,
                k_cache=kv_cache_info.mla_cache_tensor[self.layer_idx].unsqueeze(2),
                block_table=kv_cache_info.page_table,
                cache_seqlens=kv_cache_info.actual_lens_kv,
                head_dim_v=ckv_head_dim,
                tile_scheduler_metadata=flash_mla_tile_sche_meta,
                num_splits=flash_mla_num_splits,
                softmax_scale=self.scale,
                causal=False,
                is_fp8_kvcache=False,
                indices=None,
            )
            prefix_lse = prefix_lse.permute(0, 2, 1).contiguous()
            if q_len > 1 and parallel_state.dcp_rank == parallel_state.dcp_deg - 1:
                q_suffix = _q
                batch_size = ckv.shape[0] // q_len
                ckv_suffix = ckv.contiguous().view(batch_size, q_len, 1, ckv.shape[-1])
                k_pe_suffix = k_pe.contiguous().view(
                    batch_size, q_len, 1, k_pe.shape[-1]
                )
                k_suffix = torch.concat((ckv_suffix, k_pe_suffix), dim=-1)
                v_suffix = ckv_suffix
                prefix_out, prefix_lse = causal_attn_uniform_q_len_suffix(
                    q=q_suffix,
                    k=k_suffix,
                    v=v_suffix,
                    sm_scale=self.scale,
                    prefix_out=prefix_out,
                    prefix_lse=prefix_lse,
                )
            prefix_out = prefix_out.contiguous().view(
                -1, prefix_out.shape[2], prefix_out.shape[3]
            )
            prefix_lse = prefix_lse.view(prefix_out.shape[0], prefix_out.shape[1])
            prefix_out, prefix_lse = ag_lse_and_correct_attn(prefix_out, prefix_lse)

            prefix_out = torch.einsum("bhr,hdr->bhd", prefix_out, ckv2v_weight)
            prefix_out = prefix_out.contiguous()
            return prefix_out
