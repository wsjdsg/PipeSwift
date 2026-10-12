from .dpskv3 import DeepSeekV3


class GLM4(DeepSeekV3):
    def __init__(self, args):
        # GLM-4.7 sets ``use_qk_norm`` in its config; GLM-4.7-Flash does not.
        # GLM-4.7 also provides ``e_score_correction_bias`` but it harms
        # generation quality — both variants use ``noaux_tc`` routing here.
        if args.hf_config.get("use_qk_norm", False):
            args.qk_norm = True
        args.hf_config["topk_method"] = "noaux_tc"
        super().__init__(args)
