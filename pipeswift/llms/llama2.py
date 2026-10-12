import torch

from pipeswift.utils import KVCacheInfo, compute_and_set_cos_sin_cache

from .layers import *
from .llm import LLM


class Llama2(LLM):
    def __init__(self, args):
        from pipeswift.utils import parallel_state

        super().__init__(args)
        hf_config = args.hf_config
        if parallel_state.pp_prev_rank is None or parallel_state.pp_next_rank is None:
            self.vocab_embed = ParallelVocabEmbedding(hf_config, dtype=self.dtype)

        self.use_mla = "qk_rope_head_dim" in hf_config.keys()
        self.layers = torch.nn.ModuleList(
            [
                ParallelDecoderLayer(
                    hf_config,
                    dtype=self.dtype,
                    quant_type=self.quant_type,
                    attention_backend=args.attention_backend,
                    fp8_gemm_backend=args.fp8_gemm_backend,
                    layer_id=layer_idx,
                    qk_norm=args.qk_norm if hasattr(args, "qk_norm") else False,
                    include_e_score_correction_bias=args.include_e_score_correction_bias
                    if hasattr(args, "include_e_score_correction_bias")
                    else False,
                    use_mla=self.use_mla,
                    layer_start_idx=self.start_layer_idx,
                )
                for layer_idx in range(self.n_layer_per_device)
            ]
        )
        if parallel_state.pp_next_rank is None:
            self.pre_ln = RMSNorm(hf_config, dtype=self.dtype)
            self.lm_head = ParallelLMHead(
                hf_config, self.vocab_embed.weight, dtype=self.dtype
            )
            if parallel_state.pp_deg > 1:
                del self.vocab_embed

        rope_parameters = hf_config.get("rope_parameters")
        self.rope_theta = hf_config.get("rope_theta")
        if self.rope_theta is None and isinstance(rope_parameters, dict):
            self.rope_theta = rope_parameters.get("rope_theta")
        if self.rope_theta is None:
            self.rope_theta = 10000
        rope_head_dim = (
            (
                hf_config["hidden_size"] // hf_config["num_attention_heads"]
                if "head_dim" not in hf_config.keys()
                else hf_config["head_dim"]
            )
            if not self.use_mla
            else hf_config["qk_rope_head_dim"]
        )
        partial_rotary_factor = hf_config.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = 1
        if "qk_rope_head_dim" in hf_config.keys():
            # NOTE(wsj): Seems no combination of MLA and partial rope?
            assert partial_rotary_factor == 1
        rope_head_dim *= partial_rotary_factor

        self.rope_scaling = hf_config.get("rope_scaling")
        if self.rope_scaling is None and isinstance(rope_parameters, dict):
            rope_type = rope_parameters.get("rope_type")
            if rope_type is not None and rope_type != "default":
                self.rope_scaling = dict(rope_parameters)
        compute_and_set_cos_sin_cache(
            rope_head_dim,
            hf_config["max_position_embeddings"],
            self.rope_theta,
            self.rope_scaling,
        )

    def _forward_before_lmhead(
        self,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo = None,
        is_decode=False,
    ) -> torch.Tensor:
        from pipeswift.utils import parallel_state

        if parallel_state.pp_prev_rank is None:
            hidden_states = self.vocab_embed(input_tensor)
        else:
            hidden_states = input_tensor
        residual = None
        layer_idx = 0
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, position_ids, residual, kv_cache_info, is_decode
            )
            layer_idx += 1

        if parallel_state.pp_next_rank is None:
            hidden_states, _ = self.pre_ln(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states + residual
