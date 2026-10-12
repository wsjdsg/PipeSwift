"""
In this file, we provide a standard implementation of Llama-2's Transformer Block, using the most advanced kernel to achieve SOTA performance
kernel: use PyTorch Native kernels except PagedAttn, FusedAddRMSNorm, FusedActAndMul, RoPE
We target reusing this file's code for implementation of numerous models
NOTE(wsj): for other models differ from Llama-2, please replace with your specific implementation if not applicable
"""

from typing import Dict

import pipeswift_kernels
import torch
import torch.distributed as dist
from torch.nn import Parameter

from pipeswift.utils import (
    BLOCK_QUANT_SIZE,
    KVCacheInfo,
    QuantType,
    append_kv_cache,
    append_mla_kv_cache,
    apply_rotary_embedding,
    coalesced_load_params,
    routing,
    silu_and_mul,
)

from .attention import SelfAttention
from .linear import QuantExperts, QuantLinear, quant_matmul

# FP8 block fallback: when ParallelMLP per-rank intermediate is not 128-aligned
# (e.g. shared_experts at TP=8: 1536/8=192), demote that MLP to BF16 and
# dequant the checkpoint's FP8 weights on load.
_FP8_BLOCK_FALLBACK_WARNED = False


def _warn_fp8_block_fallback_once(layer_idx, intermediate, split_deg):
    global _FP8_BLOCK_FALLBACK_WARNED
    from pipeswift.utils import parallel_state

    if _FP8_BLOCK_FALLBACK_WARNED or parallel_state.global_rank != 0:
        _FP8_BLOCK_FALLBACK_WARNED = True
        return
    _FP8_BLOCK_FALLBACK_WARNED = True
    print(
        f"[\033[33mWarning\033[0m] FP8 block fallback to BF16 for ParallelMLP "
        f"(layer {layer_idx}: intermediate={intermediate}, split_deg={split_deg}, "
        f"per-rank={intermediate // split_deg} not divisible by {BLOCK_QUANT_SIZE}).",
        flush=True,
    )


class ParallelVocabEmbedding(torch.nn.Module):
    # simplest embed
    def __init__(self, hf_config, dtype=torch.bfloat16):
        super().__init__()
        from pipeswift.utils import parallel_state

        vocab_size = hf_config["vocab_size"]
        hidden_size = hf_config["hidden_size"]

        assert vocab_size % parallel_state.tp_deg == 0
        self.weight = Parameter(
            torch.empty(
                vocab_size // parallel_state.tp_deg,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            ),
        )
        self.tot_vocab_size = vocab_size
        self.vocab_start_index = (
            self.tot_vocab_size // parallel_state.tp_deg * parallel_state.tp_rank
        )
        self.vocab_end_index = (
            self.tot_vocab_size // parallel_state.tp_deg * (parallel_state.tp_rank + 1)
        )

    def forward(self, hidden_states):
        from pipeswift.utils import parallel_state

        def get_masked_input_and_mask(hidden_states, vocab_start, vocab_end):
            mask = (hidden_states >= vocab_start) & (hidden_states < vocab_end)
            shifted = torch.where(
                mask,
                hidden_states - vocab_start,
                torch.zeros_like(hidden_states),
            )
            return shifted, ~mask

        if parallel_state.tp_deg > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                hidden_states, self.vocab_start_index, self.vocab_end_index
            )
            output_parallel = self.weight[masked_input]
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            output_parallel = parallel_state.adaptive_ar_tp.forward(output_parallel)
            return output_parallel
        else:
            return self.weight[hidden_states]

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        vocab_embed_str = "model.embed_tokens.weight"
        weight_tensor = safetensor_dict[safetensor_index[vocab_embed_str]][
            vocab_embed_str
        ]
        vocab_dim_weight_slices = weight_tensor[
            self.vocab_start_index : self.vocab_end_index, :
        ]
        assert self.weight.shape == vocab_dim_weight_slices.shape
        self.weight.copy_(vocab_dim_weight_slices)


class RMSNorm(torch.nn.Module):
    def __init__(self, hf_config, qk_head_dim: int = None, dtype=torch.bfloat16):
        super().__init__()
        norm_dim = hf_config["hidden_size"] if qk_head_dim is None else qk_head_dim
        self.weight = Parameter(
            torch.empty(norm_dim, device=torch.cuda.current_device(), dtype=dtype)
        )
        self.eps = hf_config["rms_norm_eps"]
        self.norm_size = norm_dim

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        pipeswift_kernels.rms_norm(out, x, self.weight, self.eps)
        return out

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pipeswift_kernels.fused_add_rms_norm(x, residual, self.weight, self.eps)
        return x, residual

    def forward(self, hidden_states, residual: torch.Tensor = None):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states
        if residual is not None:
            hidden_states, residual = self.add_rms_forward(hidden_states, residual)
            return hidden_states, residual
        else:
            residual = hidden_states
            hidden_states = self.rms_forward(hidden_states)
            return hidden_states, residual

    def _load_weight(self, weight_tensor: torch.Tensor):
        self.weight.copy_(weight_tensor)


# Llama-2's lm_head share weights with VocabEmbedding, use tie_weight function
class ParallelLMHead(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        vocab_embedding_weight: torch.nn.Parameter = None,
        dtype=torch.bfloat16,
    ):
        from pipeswift.utils import parallel_state

        vocab_size, hidden_size, tie_word_embedding = (
            hf_config["vocab_size"],
            hf_config["hidden_size"],
            hf_config["tie_word_embeddings"],
        )
        super().__init__()
        # NOTE(wsj): Seems almost no model uses bias in linear layer?
        assert vocab_size % parallel_state.tp_deg == 0
        if not tie_word_embedding:
            self.weight = Parameter(
                torch.empty(
                    vocab_size // parallel_state.tp_deg,
                    hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=dtype,
                )
            )
        else:
            assert vocab_embedding_weight is not None
            self.weight = vocab_embedding_weight
        self.tie_word_embedding = tie_word_embedding
        self.vocab_size = vocab_size

    def forward(self, hidden_states):
        from pipeswift.utils import parallel_state

        if parallel_state.tp_deg == 1:
            return torch.nn.functional.linear(hidden_states, self.weight, bias=None)

        logits_sharded = torch.nn.functional.linear(
            hidden_states, self.weight, bias=None
        )
        all_logits = torch.empty(
            torch.Size((parallel_state.tp_deg,)) + logits_sharded.shape,
            dtype=logits_sharded.dtype,
            device="cuda",
        )
        dist.all_gather_into_tensor(
            all_logits, logits_sharded, group=parallel_state.tp_group
        )
        all_logits = all_logits.transpose(0, 1).contiguous().view(-1, self.vocab_size)
        # if parallel_state.tp_rank != 0:
        #     all_logits = None
        return all_logits

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        from pipeswift.utils import parallel_state

        lmhead_str = "lm_head.weight"
        if self.tie_word_embedding:
            lmhead_str = "model.embed_tokens.weight"
        lmhead_tensor = safetensor_dict[safetensor_index[lmhead_str]][lmhead_str]
        tp_deg, tp_rank = parallel_state.tp_deg, parallel_state.tp_rank
        vocab_dim_per_device = self.vocab_size // tp_deg
        vocab_dim_weight_slices = lmhead_tensor[
            vocab_dim_per_device * tp_rank : vocab_dim_per_device * (tp_rank + 1), :
        ]
        assert self.weight.shape == vocab_dim_weight_slices.shape
        self.weight.copy_(vocab_dim_weight_slices)


class ParallelDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        attention_backend="flash_attn",
        fp8_gemm_backend="vllm_cutlass",
        layer_id: int = 0,
        qk_norm=False,
        include_e_score_correction_bias=False,
        use_mla=False,
        layer_start_idx=0,
        weight_prefix: str = "model.layers",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.global_layer_idx = self.layer_id + layer_start_idx
        self.weight_prefix = weight_prefix
        if not use_mla:
            self.attention = ParallelAttention(
                hf_config,
                dtype=dtype,
                quant_type=quant_type,
                attention_backend=attention_backend,
                fp8_gemm_backend=fp8_gemm_backend,
                qk_norm=qk_norm,
                layer_id=layer_id,
                global_layer_idx=self.global_layer_idx,
                weight_prefix=weight_prefix,
            )
        else:
            self.attention = ParallelMLA(
                hf_config,
                dtype=dtype,
                quant_type=quant_type,
                fp8_gemm_backend=fp8_gemm_backend,
                layer_id=layer_id,
                global_layer_idx=self.global_layer_idx,
            )
        if "moe_intermediate_size" not in hf_config.keys() or (
            "first_k_dense_replace" in hf_config.keys()
            and self.global_layer_idx < hf_config["first_k_dense_replace"]
        ):
            self.ffn = ParallelMLP(
                hf_config,
                dtype=dtype,
                quant_type=quant_type,
                fp8_gemm_backend=fp8_gemm_backend,
                global_layer_idx=self.global_layer_idx,
                weight_prefix=weight_prefix,
            )
        else:
            self.ffn = ParallelMoE(
                hf_config,
                include_e_score_correction_bias=include_e_score_correction_bias,
                dtype=dtype,
                quant_type=quant_type,
                fp8_gemm_backend=fp8_gemm_backend,
                global_layer_idx=self.global_layer_idx,
                weight_prefix=weight_prefix,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: torch.Tensor = None,
        kv_cache_info=None,
        is_decode=False,
    ):
        hidden_states, residual = self.attention(
            hidden_states, position_ids, residual, kv_cache_info, is_decode
        )
        if isinstance(self.ffn, ParallelMoE):
            hidden_states, residual = self.ffn(hidden_states, residual, is_decode)
        else:
            hidden_states, residual = self.ffn(hidden_states, residual)
        return hidden_states, residual

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        self.attention._load_weight_with_index(safetensor_dict, safetensor_index)
        self.ffn._load_weight_with_index(safetensor_dict, safetensor_index)


class ParallelAttention(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        attention_backend="flash_attn",
        fp8_gemm_backend="vllm_cutlass",
        qk_norm=False,
        layer_id: int = 0,
        global_layer_idx: int = 0,
        attn_output_gate: bool = False,
        weight_prefix: str = "model.layers",
    ):
        from pipeswift.utils import parallel_state

        super().__init__()
        self.dtype = dtype
        self.quant_type = quant_type
        if (
            self.quant_type != QuantType.NO_QUANT
            and "quantization_config" in hf_config.keys()
        ):
            quant_config = hf_config["quantization_config"]
            quant_ignore_modules = (
                quant_config["modules_to_not_convert"]
                if "modules_to_not_convert" in quant_config.keys()
                else quant_config["ignore"]
                if "ignore" in quant_config.keys()
                else None
            )
            if quant_ignore_modules:
                if (
                    f"{weight_prefix}.{global_layer_idx}.self_attn.q_proj"
                    in quant_ignore_modules
                ):
                    self.quant_type = (
                        QuantType.NO_QUANT
                    )  # only support layer-granularity partially quantization ignore

        self.n_head, self.n_kv_head = (
            hf_config["num_attention_heads"],
            hf_config["num_key_value_heads"],
        )
        self.hidden_size = hf_config["hidden_size"]
        self.head_dim = (
            self.hidden_size // self.n_head
            if "head_dim" not in hf_config.keys()
            else hf_config["head_dim"]
        )
        # tp_deg cannot surpass n_kv_head?
        assert self.n_head % parallel_state.tp_deg == 0
        if parallel_state.dcp_deg > 1:
            self.head_per_device = (
                self.n_head // parallel_state.tp_deg * parallel_state.dcp_deg
            )
        else:
            self.head_per_device = self.n_head // parallel_state.tp_deg
        self.gqa_group_size = self.n_head // self.n_kv_head
        import math

        if "attention_bias" in hf_config.keys():
            self.attention_bias = hf_config["attention_bias"]
        elif "bias" in hf_config.keys():
            self.attention_bias = hf_config["bias"]
        else:
            self.attention_bias = False

        self.kv_head_per_device = math.ceil(self.head_per_device / self.gqa_group_size)
        if parallel_state.dcp_deg > 1:
            assert self.kv_head_per_device == 1, (
                "The number of kv heads per device must be 1 when dcp is enabled!"
            )
        # attn_output_gate doubles the q output (q + gate fused channel-wise).
        self.attn_output_gate = attn_output_gate
        self.weight_prefix = weight_prefix
        q_factor = 2 if attn_output_gate else 1
        self.qkv_proj = QuantLinear(
            self.hidden_size,
            (self.head_per_device * q_factor + 2 * self.kv_head_per_device)
            * self.head_dim,
            with_bias=self.attention_bias,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        self.rms_norm = RMSNorm(hf_config, dtype=dtype)
        self.o_proj = QuantLinear(
            self.head_per_device * self.head_dim,
            self.hidden_size,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(hf_config, self.head_dim, dtype=dtype)
            self.k_norm = RMSNorm(hf_config, self.head_dim, dtype=dtype)

        self.self_attn = SelfAttention(
            self.head_dim,
            attention_backend=attention_backend,
            layer_id=layer_id,
            gqa_group_size=self.gqa_group_size,
        )
        self.rope_is_neox_style = (
            True
            if "rope_is_neox_style" not in hf_config.keys()
            else hf_config["rope_is_neox_style"]
        )
        self.max_position_embedding = hf_config["max_position_embeddings"]

        self.layer_idx = layer_id
        self.global_layer_idx = global_layer_idx
        partial_rotary_factor = (
            hf_config["partial_rotary_factor"]
            if "partial_rotary_factor" in hf_config.keys()
            else 1
        )
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)

        if parallel_state.dcp_deg > 1:
            if self.quant_type == QuantType.NO_QUANT:
                self._sliced_o_proj_weight = self.o_proj.weight.tensor_split(
                    parallel_state.dcp_deg, -1
                )[parallel_state.dcp_rank]
            else:
                self._sliced_o_proj_weight = self.o_proj.weight.tensor_split(
                    parallel_state.dcp_deg, -1
                )[parallel_state.dcp_rank].clone()
                self._sliced_o_proj_weight = torch.nn.Parameter(
                    self._sliced_o_proj_weight
                )

    def forward(
        self,
        hidden_states,
        position_ids,
        residual,
        kv_cache_info: KVCacheInfo = None,
        is_decode: bool = False,
    ):
        from pipeswift.utils import parallel_state

        if len(hidden_states) == 0:
            return hidden_states, hidden_states
        hidden_states, residual = self.rms_norm(hidden_states, residual)

        if parallel_state.dcp_deg > 1 and not is_decode:
            assert self.kv_head_per_device == 1, (
                "each device should only has one kv_head in dcp"
            )
            # attn_output_gate doubles each head's q region ([q || gate]); the
            # DCP slice must keep BOTH halves for each of this rank's heads,
            # otherwise the gate is dropped and the q reshape mismatches.
            q_factor = 2 if self.attn_output_gate else 1
            q_dim_slice = (
                self.head_per_device
                // parallel_state.dcp_deg
                * self.head_dim
                * q_factor
            )
            _q_proj_weight = self.qkv_proj.weight[
                q_dim_slice * parallel_state.dcp_rank : q_dim_slice
                * (parallel_state.dcp_rank + 1)
            ]
            _kv_proj_weight = self.qkv_proj.weight[
                -2 * self.kv_head_per_device * self.head_dim :
            ]
            _q_proj_bias, _kv_proj_bias = None, None
            if self.attention_bias:
                _q_proj_bias = self.qkv_proj.bias[
                    q_dim_slice * parallel_state.dcp_rank : q_dim_slice
                    * (parallel_state.dcp_rank + 1)
                ]
                _kv_proj_bias = self.qkv_proj.bias[
                    -2 * self.kv_head_per_device * self.head_dim :
                ]
            if self.quant_type == QuantType.NO_QUANT:
                query = quant_matmul(
                    hidden_states,
                    _q_proj_weight,
                    self.dtype,
                    self.quant_type,
                    _q_proj_bias,
                    self.qkv_proj.fp8_gemm_backend,
                )
                kv = quant_matmul(
                    hidden_states,
                    _kv_proj_weight,
                    self.dtype,
                    self.quant_type,
                    _kv_proj_bias,
                    self.qkv_proj.fp8_gemm_backend,
                )
            else:
                _quant_block = (
                    1 if self.quant_type == QuantType.FP8_PER_CHANNEL else 128
                )
                _q_proj_scale = self.qkv_proj.weight_scale[
                    q_dim_slice * parallel_state.dcp_rank // _quant_block : q_dim_slice
                    * (parallel_state.dcp_rank + 1)
                    // _quant_block
                ]
                _kv_proj_scale = self.qkv_proj.weight_scale[
                    -2 * self.kv_head_per_device * self.head_dim // _quant_block :
                ]
                query = quant_matmul(
                    hidden_states,
                    (_q_proj_weight, _q_proj_scale),
                    self.dtype,
                    self.quant_type,
                    _q_proj_bias,
                    self.qkv_proj.fp8_gemm_backend,
                )
                kv = quant_matmul(
                    hidden_states,
                    (_kv_proj_weight, _kv_proj_scale),
                    self.dtype,
                    self.quant_type,
                    _kv_proj_bias,
                    self.qkv_proj.fp8_gemm_backend,
                )
            key, value = kv.tensor_split(2, dim=-1)
        else:
            qkv = self.qkv_proj(hidden_states)
            q_factor = 2 if self.attn_output_gate else 1
            q_size = self.head_dim * self.head_per_device * q_factor
            kv_size = self.head_dim * self.kv_head_per_device
            query, key, value = (
                qkv[:, :q_size],
                qkv[:, q_size : q_size + kv_size],
                qkv[:, q_size + kv_size : q_size + 2 * kv_size],
            )

        gate = None
        if self.attn_output_gate:
            # query holds [B, 2 * n_q_heads * head_dim] interleaved per-head as
            # [head, 2, head_dim] (q || gate concatenated along the last axis).
            # Under DCP prefill, q was sliced to a subset of heads, so derive the
            # head count from the runtime shape rather than full head_per_device.
            n_q_heads = query.shape[1] // (2 * self.head_dim)
            q_view = query.view(query.shape[0], n_q_heads, 2, self.head_dim)
            query = q_view[:, :, 0, :].contiguous().view(query.shape[0], -1)
            gate = q_view[:, :, 1, :].contiguous().view(query.shape[0], -1)

        if self.qk_norm:
            query, _ = self.q_norm(query.view(query.shape[0], -1, self.head_dim))
            query = query.view(query.shape[0], -1)
            key, _ = self.k_norm(key.view(key.shape[0], -1, self.head_dim))
            key = key.view(key.shape[0], -1)

        query, key, value = (
            query.view(query.shape[0], -1, self.head_dim),
            key.view(key.shape[0], -1, self.head_dim),
            value.view(value.shape[0], -1, self.head_dim),
        )

        apply_rotary_embedding(
            position_ids,
            query[..., : self.rotary_dim],
            key[..., : self.rotary_dim],
            self.rotary_dim,
            self.rope_is_neox_style,
        )
        append_kv_cache(key, value, kv_cache_info, layer_idx=self.layer_idx)
        if parallel_state.dcp_deg == 1:
            hidden_states = self.self_attn.forward_gqa_attn(query, kv_cache_info)
        else:
            hidden_states = self.self_attn.forward_gqa_dcp(
                query, key, value, kv_cache_info, is_decode=is_decode
            )

        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        if self.attn_output_gate and gate is not None:
            hidden_states = hidden_states * torch.sigmoid(gate)
        if parallel_state.dcp_deg > 1 and not is_decode:
            if self.quant_type == QuantType.FP8_PER_CHANNEL:
                _o_proj_scale = self.o_proj.weight_scale
            elif self.quant_type == QuantType.FP8_PER_BLOCK:
                _o_proj_scale = self.o_proj.weight_scale.tensor_split(
                    parallel_state.dcp_deg, -1
                )[parallel_state.dcp_rank]
            hidden_states = quant_matmul(
                hidden_states,
                self._sliced_o_proj_weight
                if self.quant_type == QuantType.NO_QUANT
                else (self._sliced_o_proj_weight, _o_proj_scale.contiguous()),
                dtype=self.dtype,
                quant_type=self.quant_type,
                bias=None,
                fp8_gemm_backend=self.o_proj.fp8_gemm_backend,
            )
        else:
            hidden_states = self.o_proj(hidden_states)

        if parallel_state.tp_deg > 1:
            hidden_states = parallel_state.adaptive_ar_tp.forward(hidden_states)
        return hidden_states, residual

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        from pipeswift.utils import parallel_state

        prefix_str = (
            f"{self.weight_prefix}.{self.global_layer_idx}."  # configurable HF prefix
        )
        suffix_str = ".weight"
        input_ln_str = prefix_str + "input_layernorm" + suffix_str
        attn_q_proj_str = prefix_str + "self_attn.q_proj" + suffix_str
        attn_k_proj_str = prefix_str + "self_attn.k_proj" + suffix_str
        attn_v_proj_str = prefix_str + "self_attn.v_proj" + suffix_str
        attn_o_proj_str = prefix_str + "self_attn.o_proj" + suffix_str

        tp_rank = parallel_state.tp_rank
        input_ln_tensor = safetensor_dict[safetensor_index[input_ln_str]][input_ln_str]
        # Gemma-style RMSNorm: callers patch self.rms_norm._load_weight to add +1.
        # For standard RMSNorm we just copy.  Either way honour the loader hook
        # if the module defines one.
        if hasattr(self.rms_norm, "_load_weight"):
            self.rms_norm._load_weight(input_ln_tensor.to(self.rms_norm.weight.dtype))
        else:
            assert self.rms_norm.weight.shape == input_ln_tensor.shape
            self.rms_norm.weight.copy_(input_ln_tensor)

        q_factor = 2 if self.attn_output_gate else 1
        q_proj_tensor, k_proj_tensor, v_proj_tensor, o_proj_tensor = (
            torch.empty(
                self.head_dim * self.n_head * q_factor,
                self.hidden_size,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.head_dim * self.n_kv_head,
                self.hidden_size,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.head_dim * self.n_kv_head,
                self.hidden_size,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.hidden_size,
                self.head_dim * self.n_head,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
        )
        coalesced_load_params(
            [attn_q_proj_str, attn_k_proj_str, attn_v_proj_str, attn_o_proj_str],
            safetensor_dict,
            safetensor_index,
            [q_proj_tensor, k_proj_tensor, v_proj_tensor, o_proj_tensor],
            is_tp=True,
        )

        split_rank = (
            tp_rank // parallel_state.dcp_deg if parallel_state.dcp_deg > 1 else tp_rank
        )

        if parallel_state.dcp_deg > 1:
            kv_st_idx = (
                self.head_per_device
                // parallel_state.dcp_deg
                * tp_rank
                // self.gqa_group_size
                * self.head_dim
            )
        else:
            kv_st_idx = (
                self.head_per_device * tp_rank // self.gqa_group_size * self.head_dim
            )
        kv_slice = self.kv_head_per_device * self.head_dim
        q_slice = self.head_per_device * self.head_dim * q_factor
        # ``o_slice`` is the per-rank input width of o_proj — head_per_device *
        # head_dim, **without** the attn_output_gate doubling (the gate sits on
        # the QKV input side only, not on attention output).
        o_slice = self.head_per_device * self.head_dim
        q_proj_tensor = q_proj_tensor[
            q_slice * split_rank : q_slice * (split_rank + 1), :
        ]
        k_proj_tensor = k_proj_tensor[kv_st_idx : kv_st_idx + kv_slice, :]
        v_proj_tensor = v_proj_tensor[kv_st_idx : kv_st_idx + kv_slice, :]
        o_proj_tensor = o_proj_tensor[
            :, o_slice * split_rank : o_slice * (split_rank + 1)
        ]
        qkv_proj_tensor = torch.concat(
            (q_proj_tensor, k_proj_tensor, v_proj_tensor), dim=0
        )
        global_layer_idx = self.global_layer_idx
        assert self.qkv_proj.weight.shape == qkv_proj_tensor.shape
        self.qkv_proj.weight.copy_(qkv_proj_tensor)
        assert self.o_proj.weight.shape == o_proj_tensor.shape
        self.o_proj.weight.copy_(o_proj_tensor)

        if parallel_state.dcp_deg > 1 and self.quant_type != QuantType.NO_QUANT:
            self._sliced_o_proj_weight.copy_(
                self.o_proj.weight.tensor_split(parallel_state.dcp_deg, -1)[
                    parallel_state.dcp_rank
                ]
            )

        if self.qk_norm:
            _q_norm_key = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.q_norm.weight"
            )
            _k_norm_key = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.k_norm.weight"
            )
            _safetensor_q = safetensor_dict[safetensor_index[_q_norm_key]]
            q_norm_weight = _safetensor_q[_q_norm_key]
            _safetensor_k = safetensor_dict[safetensor_index[_k_norm_key]]
            k_norm_weight = _safetensor_k[_k_norm_key]
            assert self.q_norm.weight.shape == q_norm_weight.shape
            assert self.k_norm.weight.shape == k_norm_weight.shape
            # Go through ``_load_weight`` so GemmaRMSNorm's +1 offset is
            # applied where needed; plain RMSNorm's ``_load_weight`` is just
            # a copy_, so non-Gemma callers are unaffected.
            self.q_norm._load_weight(q_norm_weight)
            self.k_norm._load_weight(k_norm_weight)

        # for glm4
        if self.attention_bias:
            q_proj_bias_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.q_proj.bias"
            )
            k_proj_bias_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.k_proj.bias"
            )
            v_proj_bias_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.v_proj.bias"
            )
            q_proj_bias, k_proj_bias, v_proj_bias = (
                safetensor_dict[safetensor_index[q_proj_bias_str]][q_proj_bias_str][
                    q_slice * split_rank : q_slice * (split_rank + 1)
                ],
                safetensor_dict[safetensor_index[k_proj_bias_str]][k_proj_bias_str][
                    kv_st_idx : kv_st_idx + kv_slice
                ],
                safetensor_dict[safetensor_index[v_proj_bias_str]][v_proj_bias_str][
                    kv_st_idx : kv_st_idx + kv_slice
                ],
            )
            qkv_proj_bias = torch.concat([q_proj_bias, k_proj_bias, v_proj_bias])
            self.qkv_proj.bias.copy_(qkv_proj_bias)

        if self.quant_type in [QuantType.FP8_PER_BLOCK, QuantType.FP8_PER_CHANNEL]:
            q_proj_scale_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.q_proj.weight_scale"
            )
            k_proj_scale_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.k_proj.weight_scale"
            )
            v_proj_scale_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.v_proj.weight_scale"
            )
            o_proj_scale_str = (
                f"{self.weight_prefix}.{global_layer_idx}.self_attn.o_proj.weight_scale"
            )
            try:
                safetensor_dict[safetensor_index[q_proj_scale_str]][q_proj_scale_str]
            except:
                q_proj_scale_str += "_inv"
                k_proj_scale_str += "_inv"
                v_proj_scale_str += "_inv"
                o_proj_scale_str += "_inv"

            div_block = (
                BLOCK_QUANT_SIZE if self.quant_type == QuantType.FP8_PER_BLOCK else 1
            )
            q_proj_scale, k_proj_scale, v_proj_scale = (
                safetensor_dict[safetensor_index[q_proj_scale_str]][q_proj_scale_str][
                    q_slice * split_rank // div_block : q_slice
                    * (split_rank + 1)
                    // div_block
                ],
                safetensor_dict[safetensor_index[k_proj_scale_str]][k_proj_scale_str][
                    kv_st_idx // div_block : (kv_st_idx + kv_slice) // div_block
                ],
                safetensor_dict[safetensor_index[v_proj_scale_str]][v_proj_scale_str][
                    kv_st_idx // div_block : (kv_st_idx + kv_slice) // div_block
                ],
            )
            if self.quant_type == QuantType.FP8_PER_BLOCK:
                o_proj_scale = safetensor_dict[safetensor_index[o_proj_scale_str]][
                    o_proj_scale_str
                ][
                    :,
                    o_slice * split_rank // div_block : o_slice
                    * (split_rank + 1)
                    // div_block,
                ]
            else:
                o_proj_scale = safetensor_dict[safetensor_index[o_proj_scale_str]][
                    o_proj_scale_str
                ]

            self.qkv_proj.weight_scale.copy_(
                torch.concat((q_proj_scale, k_proj_scale, v_proj_scale), dim=0)
            )
            self.o_proj.weight_scale.copy_(o_proj_scale)


class ParallelMLP(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        fp8_gemm_backend="vllm_cutlass",
        is_shared_expert=False,
        global_layer_idx: int = 0,
        weight_prefix: str = "model.layers",
        shared_expert_name: str = "shared_experts",
    ):
        super().__init__()
        from pipeswift.utils import parallel_state

        self.weight_prefix = weight_prefix
        self.shared_expert_name = shared_expert_name
        self.dtype = dtype
        self.quant_type = quant_type
        if (
            not is_shared_expert
            and self.quant_type != QuantType.NO_QUANT
            and "quantization_config" in hf_config.keys()
        ):
            quant_config = hf_config["quantization_config"]
            quant_ignore_modules = (
                quant_config["modules_to_not_convert"]
                if "modules_to_not_convert" in quant_config.keys()
                else quant_config["ignore"]
                if "ignore" in quant_config.keys()
                else None
            )
            if quant_ignore_modules:
                if (
                    f"{weight_prefix}.{global_layer_idx}.mlp.gate_proj"
                    in quant_ignore_modules
                ):
                    self.quant_type = (
                        QuantType.NO_QUANT
                    )  # only support module-granularity partially quantization ignoring

        self.is_shared_expert = is_shared_expert
        self.hidden_size, self.intermediate_size = (
            hf_config["hidden_size"],
            hf_config["intermediate_size"],
        )
        if self.is_shared_expert:
            self.intermediate_size = (
                hf_config["moe_intermediate_size"] * hf_config["n_shared_experts"]
            )
        split_deg = (
            parallel_state.tp_deg
            if (not is_shared_expert or parallel_state.deepep is None)
            else parallel_state.etp_deg
        )
        assert self.intermediate_size % split_deg == 0

        # FP8 block fallback: per-rank intermediate must be 128-aligned.
        self._fp8_block_fallback = (
            self.quant_type == QuantType.FP8_PER_BLOCK
            and (self.intermediate_size // split_deg) % BLOCK_QUANT_SIZE != 0
        )
        if self._fp8_block_fallback:
            _warn_fp8_block_fallback_once(
                global_layer_idx, self.intermediate_size, split_deg
            )
            self.quant_type = QuantType.NO_QUANT

        self.gate_up_proj = QuantLinear(
            self.hidden_size,
            self.intermediate_size * 2 // split_deg,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        self.down_proj = QuantLinear(
            self.intermediate_size // split_deg,
            self.hidden_size,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        assert "hidden_act" in hf_config.keys()
        self.act = silu_and_mul
        # https://docs.flashinfer.ai/generated/flashinfer.norm.gemma_fused_add_rmsnorm.html
        if not self.is_shared_expert:
            self.rms_norm = RMSNorm(hf_config, dtype=dtype)

        self.global_layer_idx = global_layer_idx

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None):
        from pipeswift.utils import parallel_state

        if not self.is_shared_expert and len(hidden_states) > 0:
            hidden_states, residual = self.rms_norm(hidden_states, residual)

        if len(hidden_states) == 0:
            return hidden_states, hidden_states
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        if not self.is_shared_expert:
            if parallel_state.tp_deg > 1:
                hidden_states = parallel_state.adaptive_ar_tp.forward(hidden_states)
            return hidden_states, residual
        else:
            return hidden_states

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        from pipeswift.utils import parallel_state

        prefix_str = f"{self.weight_prefix}.{self.global_layer_idx}."
        suffix_str = ".weight"
        mlp_gate_proj_str = (
            prefix_str
            + "mlp."
            + (f"{self.shared_expert_name}." if self.is_shared_expert else "")
            + "gate_proj"
            + suffix_str
        )
        mlp_up_proj_str = (
            prefix_str
            + "mlp."
            + (f"{self.shared_expert_name}." if self.is_shared_expert else "")
            + "up_proj"
            + suffix_str
        )
        mlp_down_proj_str = (
            prefix_str
            + "mlp."
            + (f"{self.shared_expert_name}." if self.is_shared_expert else "")
            + "down_proj"
            + suffix_str
        )
        if self.is_shared_expert and parallel_state.deepep is not None:
            split_rank, split_deg = parallel_state.etp_rank, parallel_state.etp_deg
        else:
            split_rank, split_deg = parallel_state.tp_rank, parallel_state.tp_deg
        dim_slice = self.intermediate_size // split_deg

        # When falling back, the checkpoint still has FP8 weights — load them
        # as FP8 and dequant in place before the existing slice/concat path.
        load_fp8 = self.quant_type != QuantType.NO_QUANT or self._fp8_block_fallback
        load_dtype = torch.float8_e4m3fn if load_fp8 else self.dtype

        mlp_down_proj_tensor = torch.empty(
            self.hidden_size,
            self.intermediate_size,
            dtype=load_dtype,
            device="cuda",
        )
        mlp_gate_tensor, mlp_up_tensor = (
            torch.empty(
                self.intermediate_size,
                self.hidden_size,
                dtype=load_dtype,
                device="cuda",
            ),
            torch.empty(
                self.intermediate_size,
                self.hidden_size,
                dtype=load_dtype,
                device="cuda",
            ),
        )
        coalesced_load_params(
            [mlp_gate_proj_str, mlp_up_proj_str, mlp_down_proj_str],
            safetensor_dict,
            safetensor_index,
            [mlp_gate_tensor, mlp_up_tensor, mlp_down_proj_tensor],
            is_tp=not self.is_shared_expert or parallel_state.deepep is None,
        )

        if self._fp8_block_fallback:
            from pipeswift.utils import dequant_weight_fp8_block

            scale_g = mlp_gate_proj_str[: -len(".weight")] + ".weight_scale"
            scale_u = mlp_up_proj_str[: -len(".weight")] + ".weight_scale"
            scale_d = mlp_down_proj_str[: -len(".weight")] + ".weight_scale"
            try:
                safetensor_dict[safetensor_index[scale_g]][scale_g]
            except Exception:
                scale_g += "_inv"
                scale_u += "_inv"
                scale_d += "_inv"
            mlp_gate_tensor = dequant_weight_fp8_block(
                mlp_gate_tensor,
                safetensor_dict[safetensor_index[scale_g]][scale_g].cuda(),
                out_dtype=self.dtype,
            )
            mlp_up_tensor = dequant_weight_fp8_block(
                mlp_up_tensor,
                safetensor_dict[safetensor_index[scale_u]][scale_u].cuda(),
                out_dtype=self.dtype,
            )
            mlp_down_proj_tensor = dequant_weight_fp8_block(
                mlp_down_proj_tensor,
                safetensor_dict[safetensor_index[scale_d]][scale_d].cuda(),
                out_dtype=self.dtype,
            )

        mlp_gate_tensor = mlp_gate_tensor[
            dim_slice * split_rank : dim_slice * (split_rank + 1), :
        ]
        mlp_up_tensor = mlp_up_tensor[
            dim_slice * split_rank : dim_slice * (split_rank + 1), :
        ]
        mlp_down_proj_tensor = mlp_down_proj_tensor[
            :, dim_slice * split_rank : dim_slice * (split_rank + 1)
        ]

        mlp_gateup_proj_tensor = torch.concat((mlp_gate_tensor, mlp_up_tensor), dim=0)
        assert self.gate_up_proj.weight.shape == mlp_gateup_proj_tensor.shape
        self.gate_up_proj.weight.copy_(mlp_gateup_proj_tensor)
        assert self.down_proj.weight.shape == mlp_down_proj_tensor.shape
        self.down_proj.weight.copy_(mlp_down_proj_tensor)
        if not self.is_shared_expert:
            post_attn_ln_str = prefix_str + "post_attention_layernorm" + suffix_str
            post_attn_ln_tensor = safetensor_dict[safetensor_index[post_attn_ln_str]][
                post_attn_ln_str
            ]
            assert self.rms_norm.weight.shape == post_attn_ln_tensor.shape
            self.rms_norm.weight.copy_(post_attn_ln_tensor)

        if self.quant_type in [QuantType.FP8_PER_BLOCK, QuantType.FP8_PER_CHANNEL]:
            _prefix_str = (
                prefix_str
                + "mlp."
                + (f"{self.shared_expert_name}." if self.is_shared_expert else "")
            )
            mlp_gate_proj_scale_str = _prefix_str + "gate_proj.weight_scale"
            mlp_up_proj_scale_str = _prefix_str + "up_proj.weight_scale"
            mlp_down_proj_scale_str = _prefix_str + "down_proj.weight_scale"
            try:
                safetensor_dict[safetensor_index[mlp_gate_proj_scale_str]][
                    mlp_gate_proj_scale_str
                ]
            except:
                mlp_gate_proj_scale_str += "_inv"
                mlp_up_proj_scale_str += "_inv"
                mlp_down_proj_scale_str += "_inv"

            div_block = (
                BLOCK_QUANT_SIZE if self.quant_type == QuantType.FP8_PER_BLOCK else 1
            )
            gate_proj_scale, up_proj_scale = (
                safetensor_dict[safetensor_index[mlp_gate_proj_scale_str]][
                    mlp_gate_proj_scale_str
                ][
                    dim_slice * split_rank // div_block : dim_slice
                    * (split_rank + 1)
                    // div_block,
                    :,
                ],
                safetensor_dict[safetensor_index[mlp_up_proj_scale_str]][
                    mlp_up_proj_scale_str
                ][
                    dim_slice * split_rank // div_block : dim_slice
                    * (split_rank + 1)
                    // div_block,
                    :,
                ],
            )
            if self.quant_type == QuantType.FP8_PER_BLOCK:
                down_proj_scale = safetensor_dict[
                    safetensor_index[mlp_down_proj_scale_str]
                ][mlp_down_proj_scale_str][
                    :,
                    dim_slice * split_rank // BLOCK_QUANT_SIZE : dim_slice
                    * (split_rank + 1)
                    // BLOCK_QUANT_SIZE,
                ]
            else:
                down_proj_scale = safetensor_dict[
                    safetensor_index[mlp_down_proj_scale_str]
                ][mlp_down_proj_scale_str]

            mlp_gate_up_proj_scale = torch.concat(
                (gate_proj_scale, up_proj_scale), dim=0
            )
            self.gate_up_proj.weight_scale.copy_(mlp_gate_up_proj_scale)
            self.down_proj.weight_scale.copy_(down_proj_scale)


class ParallelMoE(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        include_e_score_correction_bias=False,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        fp8_gemm_backend="vllm_cutlass",
        global_layer_idx: int = 0,
        weight_prefix: str = "model.layers",
    ):
        super().__init__()
        self.weight_prefix = weight_prefix
        self.hidden_size = hf_config["hidden_size"]
        self.num_experts = (
            hf_config["num_experts"]
            if "num_experts" in hf_config.keys()
            else hf_config["n_routed_experts"]
        )
        self.model_type = hf_config.get("model_type")
        self.use_fp32_gate = isinstance(
            self.model_type, str
        ) and self.model_type.startswith(
            "glm4_moe"
        )  # for GLM series models, use fp32 gate proj
        self.dtype = dtype
        self.quant_type = quant_type
        if (
            self.quant_type != QuantType.NO_QUANT
            and "quantization_config" in hf_config.keys()
        ):
            quant_config = hf_config["quantization_config"]
            quant_ignore_modules = (
                quant_config["modules_to_not_convert"]
                if "modules_to_not_convert" in quant_config.keys()
                else quant_config["ignore"]
                if "ignore" in quant_config.keys()
                else None
            )
            if quant_ignore_modules:
                if (
                    f"{weight_prefix}.{global_layer_idx}.mlp.experts.0.gate_proj"
                    in quant_ignore_modules
                ):
                    self.quant_type = (
                        QuantType.NO_QUANT
                    )  # only support module-granularity partially quantization ignoring

        self.gate = torch.nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            device="cuda",
            dtype=torch.float32 if self.use_fp32_gate else dtype,
        )  # NOTE(wsj): gate is usually not quantized
        self.top_k = hf_config["num_experts_per_tok"]
        from pipeswift.utils import parallel_state

        assert self.num_experts % parallel_state.ep_deg == 0

        self.num_experts_per_device = self.num_experts // parallel_state.ep_deg
        self.expert_st_idx, self.expert_ed_idx = (
            self.num_experts_per_device * parallel_state.ep_rank,
            self.num_experts_per_device * (parallel_state.ep_rank + 1),
        )
        self.global_expert_indices = range(self.expert_st_idx, self.expert_ed_idx)
        self.moe_intermediate_size = hf_config["moe_intermediate_size"]
        assert hf_config["hidden_act"] == "silu", (
            "Only silu is supported for Flashinfer's FusedMoE"
        )

        if self.quant_type == QuantType.FP8_PER_BLOCK:
            assert self.moe_intermediate_size // parallel_state.etp_deg % 128 == 0, (
                "dimension must be block aligned!"
            )

        self.fused_moe = QuantExperts(
            num_experts=self.num_experts_per_device,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            dtype=dtype,
            quant_type=self.quant_type,
        )

        self.include_e_score_correction_bias = include_e_score_correction_bias
        if self.include_e_score_correction_bias:
            self.e_score_correction_bias = torch.zeros(
                self.num_experts, dtype=torch.float32, device="cuda"
            )
        self.rms_norm = RMSNorm(hf_config, dtype=dtype)
        self.etp_group = parallel_state.etp_group
        # for DeepSeek
        self.norm_topk_prob = hf_config["norm_topk_prob"]
        self.topk_method = (
            hf_config["topk_method"] if "topk_method" in hf_config.keys() else "greedy"
        )
        self.routed_scaling_factor = (
            hf_config["routed_scaling_factor"]
            if "routed_scaling_factor" in hf_config.keys()
            else None
        )
        self.n_group = hf_config["n_group"] if "n_group" in hf_config.keys() else None
        self.topk_group = (
            hf_config["topk_group"] if "topk_group" in hf_config.keys() else None
        )
        self.n_shared_experts = (
            hf_config["n_shared_experts"]
            if "n_shared_experts" in hf_config.keys()
            else 0
        )
        if self.n_shared_experts > 0:
            self.shared_experts = ParallelMLP(
                hf_config,
                dtype,
                is_shared_expert=True,
                quant_type=self.quant_type,
                fp8_gemm_backend=fp8_gemm_backend,
                global_layer_idx=global_layer_idx,
                weight_prefix=weight_prefix,
            )

        # Qwen2-MoE style shared expert: a *single* MLP loaded under
        # the singular ``shared_expert.*`` key, gated by sigmoid(W @ x) from a
        # tiny ``shared_expert_gate`` linear.  Activated when the HF config
        # exposes ``shared_expert_intermediate_size`` AND ``n_shared_experts``
        # is unset (so existing DeepSeek-V3 paths are untouched).
        sez = hf_config.get("shared_expert_intermediate_size")
        self.has_qwen_shared_expert = self.n_shared_experts == 0 and bool(sez)
        if self.has_qwen_shared_expert:
            _shared_cfg = dict(hf_config)
            _shared_cfg["n_shared_experts"] = 1
            _shared_cfg["moe_intermediate_size"] = sez
            self.shared_experts = ParallelMLP(
                _shared_cfg,
                dtype,
                is_shared_expert=True,
                quant_type=self.quant_type,
                fp8_gemm_backend=fp8_gemm_backend,
                global_layer_idx=global_layer_idx,
                weight_prefix=weight_prefix,
                shared_expert_name="shared_expert",
            )
            self.shared_expert_gate = QuantLinear(
                hf_config["hidden_size"],
                1,
                with_bias=False,
                dtype=dtype,
                quant_type=QuantType.NO_QUANT,
                fp8_gemm_backend=fp8_gemm_backend,
            )

        self.global_layer_idx = global_layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor = None,
        is_decode=False,
    ):
        from pipeswift.utils import parallel_state

        if hidden_states.shape[0] > 0:  # retain!
            hidden_states, residual = self.rms_norm(hidden_states, residual)

        if parallel_state.deepep is not None:
            assert parallel_state.ep_deg > 1
            return self._forward_all2all(hidden_states, is_decode), residual
        else:
            return self._forward_normal(hidden_states), residual

    def _forward_normal(
        self,
        hidden_states: torch.Tensor,
    ):
        from pipeswift.utils import parallel_state

        if self.n_shared_experts > 0 or self.has_qwen_shared_expert:
            _hidden_states = hidden_states.clone()

        topk_idx, topk_weights = self.topk_routing(hidden_states)

        hidden_states = self.fused_moe.forward_contiguous(
            hidden_states, topk_idx, topk_weights
        )

        if hidden_states.shape[0] > 0:
            if self.has_qwen_shared_expert:
                shared_out = self.shared_experts(_hidden_states)
                shared_gate = torch.sigmoid(self.shared_expert_gate(_hidden_states))
                hidden_states = hidden_states + shared_gate * shared_out
            elif self.n_shared_experts > 0:
                hidden_states += self.shared_experts(_hidden_states)
        if parallel_state.tp_deg > 1:
            hidden_states = parallel_state.adaptive_ar_tp.forward(hidden_states)

        return hidden_states

    def _forward_all2all(self, hidden_states: torch.Tensor, is_decode: bool):
        from pipeswift.utils import parallel_state

        assert parallel_state.deepep is not None

        # redistribute activation from tp format to etp format
        if parallel_state.tp_deg != parallel_state.etp_deg:
            hidden_states, scatter_start_idx, scatter_end_idx, _token_nums = (
                self._redistribute_act_tp2etp(hidden_states)
            )

        if self.n_shared_experts > 0 or self.has_qwen_shared_expert:
            _hidden_states = hidden_states.clone()

        topk_idx, topk_weights = self.topk_routing(hidden_states)

        # deepep dispatch
        hidden_states, _maybe_topk_idx, _maybe_topk_weight = (
            parallel_state.deepep.dispatch(
                hidden_states,
                topk_idx=topk_idx.long(),
                topk_weights=topk_weights,
                is_decode=is_decode,
            )
        )

        # MoE Core
        if (
            not is_decode or parallel_state.deepep.disable_deepep_ll
        ):  # DeepEP Normal uses contiguous layout
            topk_idx, topk_weights = _maybe_topk_idx, _maybe_topk_weight
            topk_idx += parallel_state.ep_rank * self.num_experts_per_device
            hidden_states = self.fused_moe.forward_contiguous(
                hidden_states, topk_idx, topk_weights
            )
        else:  # DeepEP Low Latency uses masked format
            masked_m, expected_m = _maybe_topk_idx, _maybe_topk_weight
            hidden_states = self.fused_moe.forward_masked(
                hidden_states, masked_m, expected_m
            )

        # deepep combine
        assert hidden_states.dtype == torch.bfloat16, (
            "DeepEP only supports bf16 combine!"
        )
        hidden_states = parallel_state.deepep.combine(
            hidden_states, topk_idx.long(), topk_weights, is_decode
        )
        if hidden_states.shape[0] > 0:
            if self.has_qwen_shared_expert:
                shared_out = self.shared_experts(_hidden_states)
                shared_gate = torch.sigmoid(self.shared_expert_gate(_hidden_states))
                hidden_states = hidden_states + shared_gate * shared_out
            elif self.n_shared_experts > 0:
                hidden_states += self.shared_experts(_hidden_states)

        if parallel_state.etp_deg > 1:
            hidden_states = parallel_state.adaptive_ar_etp.forward(hidden_states)

        # redistribute activation from etp format to tp format back
        if parallel_state.tp_deg != parallel_state.etp_deg:
            hidden_states = self._redistribute_act_etp2tp(
                hidden_states, scatter_start_idx, scatter_end_idx, _token_nums
            )
        return hidden_states

    def topk_routing(self, hidden_states: torch.Tensor):
        gating_output = self.gate(
            hidden_states.to(torch.float32) if self.use_fp32_gate else hidden_states
        )
        if gating_output.shape[0] > 0:
            assert self.scoring_func is not None and self.scoring_func in [
                "sigmoid",
                "softmax",
            ]
            topk_weights, topk_idx = routing(
                gating_output=gating_output,
                top_k=self.top_k,
                scoring_func=self.scoring_func,
                renormalize=self.norm_topk_prob,
                topk_method=self.topk_method,
                n_group=self.n_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias
                if self.include_e_score_correction_bias
                else None,
            )
        else:
            topk_weights, topk_idx = (
                torch.empty((0, self.top_k), dtype=torch.float32, device="cuda"),
                torch.empty((0, self.top_k), dtype=torch.int64, device="cuda"),
            )
        return topk_idx, topk_weights

    def _redistribute_act_tp2etp(self, hidden_states: torch.Tensor):
        from pipeswift.utils import parallel_state

        if parallel_state.tp_deg != parallel_state.etp_deg:
            if parallel_state.tp_deg > 1:
                hidden_states, _token_nums = (
                    parallel_state.act_redistributor.scatter_act(
                        hidden_states, parallel_state.tp_group
                    )
                )
            if parallel_state.etp_deg > 1:
                tot_token_num = hidden_states.shape[0]
                expanded_token_nums = [tot_token_num // parallel_state.tp_deg + 1] * (
                    tot_token_num % parallel_state.tp_deg
                ) + [tot_token_num // parallel_state.tp_deg] * (
                    parallel_state.tp_deg - tot_token_num % parallel_state.tp_deg
                )
                tp_first_rank = dist.get_process_group_ranks(parallel_state.tp_group)[0]
                _token_nums_etp = [
                    expanded_token_nums[_ - tp_first_rank]
                    for _ in dist.get_process_group_ranks(parallel_state.etp_group)
                ]

                hidden_states, scatter_start_idx, scatter_end_idx = (
                    parallel_state.act_redistributor.allgather_act(
                        hidden_states,
                        parallel_state.etp_group,
                        _token_nums=_token_nums_etp,
                    )
                )
                return hidden_states, scatter_start_idx, scatter_end_idx, _token_nums
            else:
                return hidden_states, None, None, _token_nums
        else:
            return hidden_states, None, None, None

    def _redistribute_act_etp2tp(
        self,
        hidden_states: torch.Tensor,
        scatter_start_idx,
        scatter_end_idx,
        _token_nums,
    ):
        from pipeswift.utils import parallel_state

        if parallel_state.tp_deg != parallel_state.etp_deg:
            if parallel_state.etp_deg > 1:
                hidden_states, _ = parallel_state.act_redistributor.scatter_act(
                    hidden_states,
                    parallel_state.etp_group,
                    scatter_start_idx,
                    scatter_end_idx,
                )

            if parallel_state.tp_deg > 1:
                hidden_states, _, _ = parallel_state.act_redistributor.allgather_act(
                    hidden_states, parallel_state.tp_group, _token_nums
                )
        return hidden_states

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        from pipeswift.utils import parallel_state

        prefix_str = f"{self.weight_prefix}.{self.global_layer_idx}."
        suffix_str = ".weight"
        post_attn_ln_str = prefix_str + "post_attention_layernorm" + suffix_str
        gate_str = prefix_str + "mlp.gate" + suffix_str

        etp_rank, etp_deg = parallel_state.etp_rank, parallel_state.etp_deg
        dim_slice = self.moe_intermediate_size // etp_deg
        _local_expert_idx = 0
        for _expert_idx in self.global_expert_indices:
            mlp_gate_proj_str = (
                prefix_str + f"mlp.experts.{_expert_idx}.gate_proj" + suffix_str
            )
            mlp_up_proj_str = (
                prefix_str + f"mlp.experts.{_expert_idx}.up_proj" + suffix_str
            )
            mlp_down_proj_str = (
                prefix_str + f"mlp.experts.{_expert_idx}.down_proj" + suffix_str
            )
            mlp_gate_tensor, mlp_up_tensor, mlp_down_proj_tensor = (
                torch.empty(
                    self.moe_intermediate_size,
                    self.hidden_size,
                    dtype=self.dtype
                    if self.quant_type == QuantType.NO_QUANT
                    else torch.float8_e4m3fn,
                    device="cuda",
                ),
                torch.empty(
                    self.moe_intermediate_size,
                    self.hidden_size,
                    dtype=self.dtype
                    if self.quant_type == QuantType.NO_QUANT
                    else torch.float8_e4m3fn,
                    device="cuda",
                ),
                torch.empty(
                    self.hidden_size,
                    self.moe_intermediate_size,
                    dtype=self.dtype
                    if self.quant_type == QuantType.NO_QUANT
                    else torch.float8_e4m3fn,
                    device="cuda",
                ),
            )

            coalesced_load_params(
                [mlp_gate_proj_str, mlp_up_proj_str, mlp_down_proj_str],
                safetensor_dict,
                safetensor_index,
                [mlp_gate_tensor, mlp_up_tensor, mlp_down_proj_tensor],
                is_tp=False,
            )

            mlp_gate_tensor = mlp_gate_tensor[
                dim_slice * etp_rank : dim_slice * (etp_rank + 1), :
            ]
            mlp_up_tensor = mlp_up_tensor[
                dim_slice * etp_rank : dim_slice * (etp_rank + 1), :
            ]
            mlp_down_proj_tensor = mlp_down_proj_tensor[
                :, dim_slice * etp_rank : dim_slice * (etp_rank + 1)
            ]

            mlp_upgate_proj_tensor = torch.concat(
                (mlp_up_tensor, mlp_gate_tensor), dim=0
            )  # NOTE(wsj): flashinfer cutlass_fuse_moe requires upgate instead of gateup
            self.fused_moe.w1[_local_expert_idx].copy_(mlp_upgate_proj_tensor)
            self.fused_moe.w2[_local_expert_idx].copy_(mlp_down_proj_tensor)
            _local_expert_idx += 1
        if self.n_shared_experts > 0 or self.has_qwen_shared_expert:
            self.shared_experts._load_weight_with_index(
                safetensor_dict, safetensor_index
            )
        if self.has_qwen_shared_expert:
            gate_key = (
                f"{self.weight_prefix}.{self.global_layer_idx}"
                f".mlp.shared_expert_gate.weight"
            )
            gate_tensor = (
                safetensor_dict[safetensor_index[gate_key]][gate_key]
                .to("cuda")
                .to(self.shared_expert_gate.weight.dtype)
            )
            assert self.shared_expert_gate.weight.shape == gate_tensor.shape, (
                f"shared_expert_gate shape mismatch: "
                f"param {self.shared_expert_gate.weight.shape} vs ckpt {gate_tensor.shape}"
            )
            self.shared_expert_gate.weight.copy_(gate_tensor)
        post_attn_ln_tensor = safetensor_dict[safetensor_index[post_attn_ln_str]][
            post_attn_ln_str
        ]
        if hasattr(self.rms_norm, "_load_weight"):
            self.rms_norm._load_weight(
                post_attn_ln_tensor.to(self.rms_norm.weight.dtype)
            )
        else:
            assert self.rms_norm.weight.shape == post_attn_ln_tensor.shape
            self.rms_norm.weight.copy_(post_attn_ln_tensor)
        gate_tensor = safetensor_dict[safetensor_index[gate_str]][gate_str]
        assert self.gate.weight.shape == gate_tensor.shape
        self.gate.weight.copy_(gate_tensor)
        if self.include_e_score_correction_bias:
            e_score_corr_bias_str = prefix_str + "mlp.gate.e_score_correction_bias"
            e_scroe_bias_tensor = safetensor_dict[
                safetensor_index[e_score_corr_bias_str]
            ][e_score_corr_bias_str]
            self.e_score_correction_bias.copy_(e_scroe_bias_tensor)

        if self.quant_type in [QuantType.FP8_PER_BLOCK, QuantType.FP8_PER_CHANNEL]:
            _local_expert_idx = 0
            for _expert_idx in self.global_expert_indices:
                _prefix_str = prefix_str + f"mlp.experts.{_expert_idx}."
                mlp_gate_proj_scale_str = _prefix_str + "gate_proj.weight_scale"
                mlp_up_proj_scale_str = _prefix_str + "up_proj.weight_scale"
                mlp_down_proj_scale_str = _prefix_str + "down_proj.weight_scale"

                try:
                    safetensor_dict[safetensor_index[mlp_gate_proj_scale_str]][
                        mlp_gate_proj_scale_str
                    ]
                except:
                    mlp_gate_proj_scale_str += "_inv"
                    mlp_up_proj_scale_str += "_inv"
                    mlp_down_proj_scale_str += "_inv"

                div_block = (
                    BLOCK_QUANT_SIZE
                    if self.quant_type == QuantType.FP8_PER_BLOCK
                    else 1
                )
                gate_proj_scale, up_proj_scale = (
                    safetensor_dict[safetensor_index[mlp_gate_proj_scale_str]][
                        mlp_gate_proj_scale_str
                    ][
                        dim_slice * etp_rank // div_block : dim_slice
                        * (etp_rank + 1)
                        // div_block,
                        :,
                    ],
                    safetensor_dict[safetensor_index[mlp_up_proj_scale_str]][
                        mlp_up_proj_scale_str
                    ][
                        dim_slice * etp_rank // div_block : dim_slice
                        * (etp_rank + 1)
                        // div_block,
                        :,
                    ],
                )
                if self.quant_type == QuantType.FP8_PER_BLOCK:
                    down_proj_scale = safetensor_dict[
                        safetensor_index[mlp_down_proj_scale_str]
                    ][mlp_down_proj_scale_str][
                        :,
                        dim_slice * etp_rank // BLOCK_QUANT_SIZE : dim_slice
                        * (etp_rank + 1)
                        // BLOCK_QUANT_SIZE,
                    ]
                else:
                    down_proj_scale = safetensor_dict[
                        safetensor_index[mlp_down_proj_scale_str]
                    ][mlp_down_proj_scale_str]

                mlp_up_gate_proj_scale = torch.concat(
                    (up_proj_scale, gate_proj_scale), dim=0
                )  # NOTE(wsj), should inverse
                self.fused_moe.w1_scale[_local_expert_idx].copy_(mlp_up_gate_proj_scale)
                self.fused_moe.w2_scale[_local_expert_idx].copy_(down_proj_scale)
                _local_expert_idx += 1

            # crucial for non-E8M0 block scale
            if self.quant_type == QuantType.FP8_PER_BLOCK:
                _FI_CUTLASS_MIN_BLOCK_SCALE = 1e-10
                self.fused_moe.w1_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)
                self.fused_moe.w2_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)


class ParallelMLA(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        attention_backend="flash_mla",
        fp8_gemm_backend="vllm_cutlass",
        layer_id: int = 0,
        global_layer_idx: int = 0,
    ):
        from pipeswift.utils import parallel_state

        super().__init__()
        self.hidden_size = hf_config["hidden_size"]
        self.qk_nope_head_dim = hf_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = hf_config["qk_rope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = hf_config["v_head_dim"]
        self.q_lora_rank = hf_config["q_lora_rank"]
        self.kv_lora_rank = hf_config["kv_lora_rank"]
        self.n_head = hf_config["num_attention_heads"]
        assert self.n_head % parallel_state.tp_deg == 0
        self.head_per_device = self.n_head // parallel_state.tp_deg
        if parallel_state.dcp_deg > 1:
            self.head_per_device *= parallel_state.dcp_deg

        self.dtype = dtype
        self.quant_type = quant_type
        self.fp8_gemm_backend = fp8_gemm_backend
        if (
            self.quant_type != QuantType.NO_QUANT
            and "quantization_config" in hf_config.keys()
        ):
            quant_config = hf_config["quantization_config"]
            quant_ignore_modules = (
                quant_config["modules_to_not_convert"]
                if "modules_to_not_convert" in quant_config.keys()
                else quant_config["ignore"]
                if "ignore" in quant_config.keys()
                else None
            )
            if quant_ignore_modules:
                if (
                    f"model.layers.{global_layer_idx}.self_attn.kv_a_proj_with_mqa"
                    in quant_ignore_modules
                ):
                    self.quant_type = (
                        QuantType.NO_QUANT
                    )  # only support module-granularity partially quantization ignoring

        self.attention_backend = attention_backend
        self.rms_norm = RMSNorm(hf_config, dtype=dtype)
        self.qkv_a_proj_with_mqa = QuantLinear(
            self.hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        self.q_a_layer_norm = RMSNorm(hf_config, self.q_lora_rank, dtype=dtype)
        self.q_b_proj = QuantLinear(
            self.q_lora_rank,
            self.head_per_device * self.qk_head_dim,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )  # duplicate in dcp_group

        self.kv_a_layernorm = RMSNorm(hf_config, self.kv_lora_rank, dtype=dtype)
        self.kv_b_proj = QuantLinear(
            self.kv_lora_rank,
            self.head_per_device * (self.qk_nope_head_dim + self.v_head_dim),
            with_bias=False,
            dtype=dtype,
            # NOTE(wsj): currently lack of fp8 channel-wise batched gemm kernels, so retain this parameter in bf16 format
            quant_type=QuantType.NO_QUANT,
        )  # replicate in dcp_group

        # used for weight obsorbation's einsum operation
        self.ckv2k_weight = self.kv_b_proj.weight.view(
            self.head_per_device, -1, self.kv_lora_rank
        )[:, : self.qk_nope_head_dim, :]
        self.ckv2v_weight = self.kv_b_proj.weight.view(
            self.head_per_device, -1, self.kv_lora_rank
        )[:, self.qk_nope_head_dim :, :]

        self.o_proj = QuantLinear(
            self.head_per_device * self.v_head_dim,
            self.hidden_size,
            with_bias=False,
            dtype=dtype,
            quant_type=self.quant_type,
            fp8_gemm_backend=fp8_gemm_backend,
        )  # replicate in dcp_group
        self.self_attn = SelfAttention(
            self.qk_head_dim,
            attention_backend=attention_backend,
            use_mla=True,
            layer_id=layer_id,
            rope_scaling=hf_config.get("rope_scaling"),
        )
        self.rope_is_neox_style = (
            False
            if "rope_is_neox_style" not in hf_config.keys()
            else hf_config["rope_is_neox_style"]
        )
        self.layer_idx = layer_id
        self.global_layer_idx = global_layer_idx

        if (
            parallel_state.dcp_deg > 1 and self.quant_type != QuantType.NO_QUANT
        ):  # deepcopy
            self._sliced_o_proj_weight = self.o_proj.weight.tensor_split(
                parallel_state.dcp_deg, -1
            )[parallel_state.dcp_rank].clone()
            self._sliced_o_proj_weight = torch.nn.Parameter(self._sliced_o_proj_weight)
        else:  # shallow copy
            self._sliced_o_proj_weight = self.o_proj.weight.tensor_split(
                parallel_state.dcp_deg, -1
            )[parallel_state.dcp_rank]

    def forward(
        self,
        hidden_states,
        position_ids,
        residual,
        kv_cache_info: KVCacheInfo = None,
        is_decode: bool = False,
    ):
        if len(hidden_states) == 0:
            return hidden_states, hidden_states
        from pipeswift.utils import parallel_state

        hidden_states, residual = self.rms_norm(hidden_states, residual)

        if parallel_state.dcp_deg == 1:
            hidden_states = self._forward(
                hidden_states, position_ids, kv_cache_info, is_decode
            )
        else:
            hidden_states = self._forward_dcp(
                hidden_states, position_ids, kv_cache_info, is_decode
            )

        if parallel_state.tp_deg > 1:
            hidden_states = parallel_state.adaptive_ar_tp.forward(hidden_states)
        return hidden_states, residual

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo,
        is_decode: bool,
    ):
        q, latent_cache = self.qkv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q, _ = self.q_a_layer_norm(q)
        q = self.q_b_proj(q).view(-1, self.head_per_device, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        ckv, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        ckv, _ = self.kv_a_layernorm(ckv)
        apply_rotary_embedding(
            position_ids, q_pe, k_pe, self.qk_rope_head_dim, self.rope_is_neox_style
        )
        append_mla_kv_cache(ckv, k_pe, kv_cache_info, self.layer_idx)
        q_nope = torch.einsum(
            "bhd,hdr->bhr", q_nope, self.ckv2k_weight
        )  # NOTE(wsj): always bf16/fp16 einsum
        hidden_states = self.self_attn.forward_mla_attn(
            q_nope, q_pe, kv_cache_info, self.kv_lora_rank, is_decode
        )  # [bsz, n_head, kv_lora_rank]
        hidden_states = torch.einsum(
            "bhr,hdr->bhd", hidden_states, self.ckv2v_weight
        ).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = self.o_proj(hidden_states)
        return hidden_states

    def _forward_dcp(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo,
        is_decode: bool,
    ):
        from pipeswift.utils import parallel_state

        q, latent_cache = self.qkv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q, _ = self.q_a_layer_norm(q)
        # prefill same as normal dcp
        if not is_decode:
            _q_b_proj_weight = self.q_b_proj.weight.tensor_split(
                parallel_state.dcp_deg, 0
            )[parallel_state.dcp_rank]
            if self.quant_type == QuantType.NO_QUANT:
                q = quant_matmul(q, _q_b_proj_weight, self.dtype, self.quant_type)
            else:
                _q_b_proj_scale = self.q_b_proj.weight_scale.tensor_split(
                    parallel_state.dcp_deg, 0
                )[parallel_state.dcp_rank]
                q = quant_matmul(
                    q,
                    (_q_b_proj_weight, _q_b_proj_scale),
                    self.dtype,
                    self.quant_type,
                    fp8_gemm_backend=self.q_b_proj.fp8_gemm_backend,
                )
            q = q.view(
                -1, self.head_per_device // parallel_state.dcp_deg, self.qk_head_dim
            )
        else:
            q = self.q_b_proj(q).view(-1, self.head_per_device, self.qk_head_dim)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        ckv, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        ckv, _ = self.kv_a_layernorm(ckv)
        apply_rotary_embedding(
            position_ids, q_pe, k_pe, self.qk_rope_head_dim, self.rope_is_neox_style
        )
        append_mla_kv_cache(ckv, k_pe, kv_cache_info, self.layer_idx)
        if not is_decode:
            _ckv2k_weight = self.ckv2k_weight.tensor_split(parallel_state.dcp_deg, 0)[
                parallel_state.dcp_rank
            ]
            q_nope = torch.einsum("bhd,hdr->bhr", q_nope, _ckv2k_weight)
        else:
            q_nope = torch.einsum("bhd,hdr->bhr", q_nope, self.ckv2k_weight)

        if not is_decode:
            _kv_b_proj_weight = self.kv_b_proj.weight.tensor_split(
                parallel_state.dcp_deg, 0
            )[parallel_state.dcp_rank]
            k_nope_v = torch.matmul(ckv, _kv_b_proj_weight.t())
            k_nope_v = k_nope_v.view(
                ckv.shape[0], self.head_per_device // parallel_state.dcp_deg, -1
            )
            k_nope, v = k_nope_v.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.concat(
                (k_nope, k_pe.unsqueeze(-2).expand(-1, k_nope.shape[1], -1)), dim=-1
            )
        else:
            k = v = None
        hidden_states = self.self_attn.forward_mla_dcp(
            q_nope,
            q_pe,
            q,
            k,
            v,
            ckv,
            k_pe,
            self.ckv2v_weight,
            kv_cache_info,
            self.kv_lora_rank,
            is_decode,
        )

        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        if not is_decode:
            if self.quant_type == QuantType.FP8_PER_CHANNEL:
                _o_proj_scale = self.o_proj.weight_scale
            elif self.quant_type == QuantType.FP8_PER_BLOCK:
                _o_proj_scale = self.o_proj.weight_scale.tensor_split(
                    parallel_state.dcp_deg, -1
                )[parallel_state.dcp_rank]
            hidden_states = quant_matmul(
                hidden_states,
                self._sliced_o_proj_weight
                if self.quant_type == QuantType.NO_QUANT
                else (self._sliced_o_proj_weight, _o_proj_scale.contiguous()),
                dtype=self.dtype,
                quant_type=self.quant_type,
                bias=None,
                fp8_gemm_backend=self.o_proj.fp8_gemm_backend,
            )
        else:
            hidden_states = self.o_proj(hidden_states)

        return hidden_states

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        from pipeswift.utils import parallel_state

        tp_rank = parallel_state.tp_rank
        prefix_str = f"model.layers.{self.global_layer_idx}."  # same as huggingface
        suffix_str = ".weight"
        kv_a_proj_with_mqa_str = (
            prefix_str + "self_attn.kv_a_proj_with_mqa" + suffix_str
        )
        kv_b_proj_str = prefix_str + "self_attn.kv_b_proj" + suffix_str
        o_proj_str = prefix_str + "self_attn.o_proj" + suffix_str
        q_a_proj_str = prefix_str + "self_attn.q_a_proj" + suffix_str
        q_b_proj_str = prefix_str + "self_attn.q_b_proj" + suffix_str

        (
            kv_a_proj_with_mqa_tensor,
            kv_b_proj_tensor,
            o_proj_tensor,
            q_a_proj_tensor,
            q_b_proj_tensor,
        ) = (
            torch.empty(
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.hidden_size,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.hidden_size,
                self.n_head * self.v_head_dim,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.q_lora_rank,
                self.hidden_size,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.empty(
                self.n_head * self.qk_head_dim,
                self.q_lora_rank,
                dtype=self.dtype
                if self.quant_type == QuantType.NO_QUANT
                else torch.float8_e4m3fn,
                device="cuda",
            ),
        )
        coalesced_load_params(
            [
                kv_a_proj_with_mqa_str,
                kv_b_proj_str,
                o_proj_str,
                q_a_proj_str,
                q_b_proj_str,
            ],
            safetensor_dict,
            safetensor_index,
            [
                kv_a_proj_with_mqa_tensor,
                kv_b_proj_tensor,
                o_proj_tensor,
                q_a_proj_tensor,
                q_b_proj_tensor,
            ],
            is_tp=True,
        )

        slice_rank = (
            tp_rank // parallel_state.dcp_deg if parallel_state.dcp_deg > 1 else tp_rank
        )

        o_slice = self.head_per_device * self.v_head_dim
        self.o_proj.weight.copy_(
            o_proj_tensor[:, o_slice * slice_rank : o_slice * (slice_rank + 1)]
        )
        if parallel_state.dcp_deg > 1 and self.quant_type != QuantType.NO_QUANT:
            self._sliced_o_proj_weight.copy_(
                self.o_proj.weight.tensor_split(parallel_state.dcp_deg, -1)[
                    parallel_state.dcp_rank
                ]
            )

        kv_b_slice = self.head_per_device * (self.qk_nope_head_dim + self.v_head_dim)
        kv_b_proj_tensor = kv_b_proj_tensor[
            slice_rank * kv_b_slice : (slice_rank + 1) * kv_b_slice
        ]

        qkv_proj_tensor = torch.concat(
            [q_a_proj_tensor, kv_a_proj_with_mqa_tensor], dim=0
        )
        self.qkv_a_proj_with_mqa.weight.copy_(qkv_proj_tensor)
        q_b_slice = self.head_per_device * self.qk_head_dim
        self.q_b_proj.weight.copy_(
            q_b_proj_tensor[q_b_slice * slice_rank : q_b_slice * (slice_rank + 1), :]
        )

        input_ln_str = prefix_str + "input_layernorm" + suffix_str
        input_ln_tensor = safetensor_dict[safetensor_index[input_ln_str]][input_ln_str]
        assert self.rms_norm.weight.shape == input_ln_tensor.shape
        self.rms_norm.weight.copy_(input_ln_tensor)
        kv_a_layernorm_weight_str = prefix_str + "self_attn.kv_a_layernorm" + suffix_str
        kv_a_layernorm_weight = safetensor_dict[
            safetensor_index[kv_a_layernorm_weight_str]
        ][kv_a_layernorm_weight_str]
        assert self.kv_a_layernorm.weight.shape == kv_a_layernorm_weight.shape
        self.kv_a_layernorm.weight.copy_(kv_a_layernorm_weight)

        q_a_layernorm_tensor_str = prefix_str + "self_attn.q_a_layernorm" + suffix_str
        q_a_layernorm_tensor = safetensor_dict[
            safetensor_index[q_a_layernorm_tensor_str]
        ][q_a_layernorm_tensor_str]
        self.q_a_layer_norm.weight.copy_(q_a_layernorm_tensor)

        if self.quant_type in [QuantType.FP8_PER_BLOCK, QuantType.FP8_PER_CHANNEL]:
            q_a_proj_scale_str = (
                prefix_str + "self_attn." + "q_a_proj" + ".weight_scale"
            )
            kv_a_proj_with_mqa_scale_str = (
                prefix_str + "self_attn." + "kv_a_proj_with_mqa" + ".weight_scale"
            )
            kv_b_proj_scale_str = (
                prefix_str + "self_attn." + "kv_b_proj" + ".weight_scale"
            )
            q_b_proj_scale_str = (
                prefix_str + "self_attn." + "q_b_proj" + ".weight_scale"
            )
            o_proj_scale_str = prefix_str + "self_attn." + "o_proj" + ".weight_scale"

            try:
                safetensor_dict[safetensor_index[q_a_proj_scale_str]][
                    q_a_proj_scale_str
                ]
            except:
                q_a_proj_scale_str += "_inv"
                kv_a_proj_with_mqa_scale_str += "_inv"
                kv_b_proj_scale_str += "_inv"
                q_b_proj_scale_str += "_inv"
                o_proj_scale_str += "_inv"

            q_a_proj_scale = safetensor_dict[safetensor_index[q_a_proj_scale_str]][
                q_a_proj_scale_str
            ]
            kv_a_proj_with_mqa_scale = safetensor_dict[
                safetensor_index[kv_a_proj_with_mqa_scale_str]
            ][kv_a_proj_with_mqa_scale_str]
            div_block = (
                BLOCK_QUANT_SIZE if self.quant_type == QuantType.FP8_PER_BLOCK else 1
            )
            dim_slice_q_b, dim_slice_kv_b = (
                self.head_per_device * self.qk_head_dim,
                self.head_per_device * (self.qk_nope_head_dim + self.v_head_dim),
            )
            q_b_proj_scale, kv_b_proj_scale = (
                safetensor_dict[safetensor_index[q_b_proj_scale_str]][
                    q_b_proj_scale_str
                ][
                    dim_slice_q_b * slice_rank // div_block : dim_slice_q_b
                    * (slice_rank + 1)
                    // div_block,
                    :,
                ],
                safetensor_dict[safetensor_index[kv_b_proj_scale_str]][
                    kv_b_proj_scale_str
                ][
                    dim_slice_kv_b * slice_rank // div_block : dim_slice_kv_b
                    * (slice_rank + 1)
                    // div_block,
                    :,
                ],
            )
            o_slice = self.head_per_device * self.v_head_dim
            if self.quant_type == QuantType.FP8_PER_BLOCK:
                o_proj_scale = safetensor_dict[safetensor_index[o_proj_scale_str]][
                    o_proj_scale_str
                ][
                    :,
                    o_slice * slice_rank // div_block : o_slice
                    * (slice_rank + 1)
                    // div_block,
                ]
            else:
                o_proj_scale = safetensor_dict[safetensor_index[o_proj_scale_str]][
                    o_proj_scale_str
                ]

            self.qkv_a_proj_with_mqa.weight_scale.copy_(
                torch.concat((q_a_proj_scale, kv_a_proj_with_mqa_scale), dim=0)
            )
            self.q_b_proj.weight_scale.copy_(q_b_proj_scale)
            self.o_proj.weight_scale.copy_(o_proj_scale)

        kv_b_proj_tensor_dequantized = kv_b_proj_tensor
        if self.quant_type == QuantType.FP8_PER_CHANNEL:
            # dequant to self.dtype
            kv_b_proj_tensor_dequantized = (
                kv_b_proj_tensor.to(torch.float32) * kv_b_proj_scale.cuda()
            ).to(self.dtype)
        elif self.quant_type == QuantType.FP8_PER_BLOCK:
            from pipeswift.utils import dequant_weight_fp8_block

            kv_b_proj_tensor_dequantized = dequant_weight_fp8_block(
                kv_b_proj_tensor, kv_b_proj_scale.cuda(), out_dtype=self.dtype
            )
        self.kv_b_proj.weight.copy_(kv_b_proj_tensor_dequantized)
