from .layers import ParallelMoE
from .llama2 import Llama2


class Qwen3(Llama2):
    def __init__(self, args):
        args.qk_norm = True
        if "decoder_sparse_step" in args.hf_config.keys():
            assert args.hf_config["decoder_sparse_step"] == 1, (
                "Seems that all qwen3moe series adopts fully sparse ffn"
            )
        super().__init__(args)
        for _decoder_layer in self.layers:
            if isinstance(_decoder_layer.ffn, ParallelMoE):
                _decoder_layer.ffn.scoring_func = "softmax"
