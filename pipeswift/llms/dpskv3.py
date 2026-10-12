from .layers import ParallelMoE, ParallelMTP
from .llama2 import Llama2
from .module_mtp import ModuleMTP


class DeepSeekV3(ModuleMTP, Llama2):
    def __init__(self, args):
        args.include_e_score_correction_bias = True
        super().__init__(args)
        for _decoder_layer in self.layers:
            if isinstance(_decoder_layer.ffn, ParallelMoE):
                _decoder_layer.ffn.scoring_func = "sigmoid"

        self.num_spec_tokens = args.num_spec_tokens

        from pipeswift.utils import parallel_state

        if self.num_spec_tokens > 0 and parallel_state.pp_next_rank is None:
            self.mtp = ParallelMTP(
                hf_config=args.hf_config,
                dtype=self.dtype,
                quant_type=self.quant_type,
                attention_backend=args.attention_backend,
                fp8_gemm_backend=args.fp8_gemm_backend,
                qk_norm=args.qk_norm if hasattr(args, "qk_norm") else False,
                include_e_score_correction_bias=True,
                use_mla=self.use_mla,
                cache_layer_id=self.n_layer_per_device,
            )
            if isinstance(self.mtp.decoder.ffn, ParallelMoE):
                self.mtp.decoder.ffn.scoring_func = "sigmoid"
