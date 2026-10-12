from typing import Dict

import torch

from pipeswift.utils import KVCacheInfo, QuantType

from .linear import QuantLinear
from .transformer import ParallelDecoderLayer, ParallelVocabEmbedding, RMSNorm


class ParallelMTP(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        dtype=torch.bfloat16,
        quant_type: QuantType = QuantType.NO_QUANT,
        attention_backend="flash_attn",
        fp8_gemm_backend="vllm_cutlass",
        qk_norm=False,
        include_e_score_correction_bias=False,
        use_mla=False,
        cache_layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hf_config["hidden_size"]
        self.global_layer_idx = hf_config["num_hidden_layers"]
        self.cache_layer_id = cache_layer_id

        self.hnorm = RMSNorm(hf_config, dtype=dtype)
        self.enorm = RMSNorm(hf_config, dtype=dtype)
        self.eh_proj = QuantLinear(
            self.hidden_size * 2,
            self.hidden_size,
            with_bias=False,
            dtype=dtype,
            quant_type=QuantType.NO_QUANT,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        self.embed_tokens = ParallelVocabEmbedding(hf_config, dtype=dtype)
        self.decoder = ParallelDecoderLayer(
            hf_config,
            dtype=dtype,
            quant_type=quant_type,
            attention_backend=attention_backend,
            fp8_gemm_backend=fp8_gemm_backend,
            layer_id=cache_layer_id,
            qk_norm=qk_norm,
            include_e_score_correction_bias=include_e_score_correction_bias,
            use_mla=use_mla,
            layer_start_idx=self.global_layer_idx - cache_layer_id,
        )
        self.shared_head_norm = RMSNorm(hf_config, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo = None,
        is_decode=False,
    ) -> torch.Tensor:
        hidden_states, _ = self.hnorm(hidden_states)
        input_embeds = self.embed_tokens(input_ids)
        input_embeds, _ = self.enorm(input_embeds)
        hidden_states = self.eh_proj(torch.cat((input_embeds, hidden_states), dim=-1))
        hidden_states, residual = self.decoder(
            hidden_states,
            position_ids,
            residual=None,
            kv_cache_info=kv_cache_info,
            is_decode=is_decode,
        )
        hidden_states, _ = self.shared_head_norm(hidden_states, residual)
        return hidden_states

    def _load_weight_with_index(
        self, safetensor_dict, safetensor_index: Dict[str, str]
    ):
        prefix_str = f"model.layers.{self.global_layer_idx}."
        eh_proj_str = prefix_str + "eh_proj.weight"
        hnorm_str = prefix_str + "hnorm.weight"
        enorm_str = prefix_str + "enorm.weight"
        embed_tokens_str = prefix_str + "embed_tokens.weight"
        shared_head_norm_str = prefix_str + "shared_head.norm.weight"

        self.hnorm._load_weight(safetensor_dict[safetensor_index[hnorm_str]][hnorm_str])
        self.enorm._load_weight(safetensor_dict[safetensor_index[enorm_str]][enorm_str])
        self.eh_proj.weight.copy_(
            safetensor_dict[safetensor_index[eh_proj_str]][eh_proj_str]
        )
        embed_weight = safetensor_dict[safetensor_index[embed_tokens_str]][
            embed_tokens_str
        ]
        self.embed_tokens.weight.copy_(
            embed_weight[
                self.embed_tokens.vocab_start_index : self.embed_tokens.vocab_end_index,
                :,
            ]
        )
        self.decoder._load_weight_with_index(safetensor_dict, safetensor_index)
        self.shared_head_norm._load_weight(
            safetensor_dict[safetensor_index[shared_head_norm_str]][
                shared_head_norm_str
            ]
        )
