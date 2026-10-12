import argparse
import json
import os
import socket
from typing import List

from .metadata import QuantType, SamplingParams


def parallel_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Parallel Arguments")
    group.add_argument(
        "--tp_deg", "--tp", type=int, default=1, help="Tensor parallel degree."
    )
    group.add_argument(
        "--dcp_deg",
        "--dcp",
        type=int,
        default=1,
        help="Decode context parallel degree.",
    )
    group.add_argument(
        "--ep_deg", "--ep", type=int, default=1, help="Expert parallel degree."
    )
    group.add_argument(
        "--pp_deg", "--pp", type=int, default=1, help="Pipeline parallel degree."
    )
    group.add_argument(
        "--use_zigzag_schedule_decode",
        action="store_true",
        default=False,
        help="Partition decode seqs into PP micro-batches balanced by context length via zigzag.",
    )
    group.add_argument(
        "--ip_addrs",
        type=str,
        default="localhost",
        help=(
            "Comma-separated cluster node addresses. The first address is the "
            "launch node. Use localhost for a single-node run."
        ),
    )
    group.add_argument("--master_port", type=int, default=7777)
    group.add_argument("--node_rank", type=int, default=0)
    return parser


def sampling_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Sampling Arguments")
    group.add_argument(
        "--n_seqs", type=int, default=1, help="Number of sequences generated."
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=-1,
        help="Sampling temperature. Greedy decoding if set to 0.",
    )
    group.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
    )
    group.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Controls the number of top tokens to consider. Set to 0 to consider all tokens.",
    )
    group.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Whether to ignore the EOS token and continue generating tokens after EOS is generated.",
    )
    group.add_argument(
        "--max_gen_tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens to generate (decode steps).",
    )
    group.add_argument(
        "--max_context_len",
        type=int,
        default=-1,
        help="Maximum context len, default set from hf_config.",
    )
    return parser


def model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Model Arguments")
    group.add_argument(
        "--model_name",
        "--model",
        type=str,
        default="qwen3",
        choices=[
            "llama2",
            "llama3",
            "qwen3",
            "qwen3coder",
            "glm4",
            "dpskv3",
            "dpsk-distill-qwen",
        ],
        help="Model family. Architecture details are read from the config "
        "file at --weight_path.",
    )
    group.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="Storage path of model weights (must contain config.json, "
        "tokenizer.json, and safetensors).",
    )
    group.add_argument(
        "--quant_type",
        type=str,
        default="no_quant",
        choices=["no_quant", "fp8_per_channel", "fp8_per_block"],
        help="Quantization method on linear layers, default no quant",
    )
    return parser


def inference_args(parser: argparse.ArgumentParser | None = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="PipeSwift Inference Arguments")
    group.add_argument(
        "--page_size",
        type=int,
        default=128,
        help="Number of tokens in a page of kv cache",
    )
    group.add_argument(
        "--dcp_interleave_size",
        type=int,
        default=1,
        help="Every which number of token's kv that are written interleaved in DCP group, for decode context parallel",
    )
    group.add_argument(
        "--mem_util_threshold",
        type=float,
        default=0.9,
        help="Memory capacity utilization threshold",
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Inference Data Type",
        choices=["bfloat16", "float16"],
    )
    group.add_argument(
        "--enable_fp8_kv",
        action="store_true",
        help="Store the KV cache in FP8 e4m3 and run attention in FP8, matching "
        "vLLM's --kv-cache-dtype fp8 (per-tensor scale 1.0, uncalibrated). "
        "Halves KV cache memory. GQA/MHA models on SM90+ only; MLA is rejected. "
        "Requires vLLM's FlashAttention fork (see scripts/install_fp8_kv_backend.sh)",
    )
    group.add_argument(
        "--fp8_gemm_backend",
        type=str,
        default="deep_gemm",
        choices=["deep_gemm", "vllm_cutlass", "torch_native"],
    )
    group.add_argument(
        "--cuda_graph_max_batchsize",
        type=int,
        default=1024,
        help="Maximum capturing batches of CUDA Graph",
    )
    # NOTE(wsj): important! max_gen_tokens will be clipped if prefill + decode exceeds
    group.add_argument(
        "--max_batched_tokens",
        type=int,
        default=16384,
        help="Maximum tokens in batch, impact peak activation memory assessment",
    )
    group.add_argument(
        "--max_batchsize",
        type=int,
        default=512,
        help="Maximum batchsize, impact peak activation memory assessment",
    )
    group.add_argument(
        "--verbose", action="store_true", help="whether to print detailed runtime log"
    )
    group.add_argument(
        "--enforce_eager",
        action="store_true",
        help="whether to enforce using eager mode for low batch decode",
    )
    group.add_argument(
        "-o",
        type=str,
        default=None,
        help="The name of output file in ./output directory",
    )
    group.add_argument(
        "--disable_vllm_custom_ar",
        action="store_true",
        default=False,
        help="Whether to disable vllm_custom_ar.",
    )
    group.add_argument(
        "--use_deepep",
        action="store_true",
        default=False,
        help="Whether to use deepep.",
    )
    group.add_argument(
        "--enable_fp8_dispatch",
        action="store_true",
        default=False,
        help="Whether to use fp8 for EP dispatch.",
    )
    group.add_argument(
        "--disable_deepep_ll",
        action="store_true",
        default=False,
        help="Whether to disable DeepEP's low latency kernel.",
    )
    group.add_argument(
        "--disable_overlap_schedule",
        action="store_true",
        default=False,
        help="Whether to disable multi-step schedule to hide the CPU overhead.",
    )
    group.add_argument(
        "--num_spec_tokens",
        type=int,
        default=0,
        help="Number of draft tokens",
    )
    group.add_argument(
        "--draft_vs_layer",
        type=float,
        default=2,
        help="Indicates a draft step's workload, w.r.t. a layer",
    )
    group.add_argument(
        "--spec_rejection_mode",
        type=str,
        default="probabilistic",
        choices=["strict", "probabilistic"],
        help="Speculative decoding verification mode.",
    )
    group.add_argument("--log_interval", type=int, default=512)
    group.add_argument(
        "--control_interval",
        type=int,
        default=8,
        help="Engine control-plane polling interval in scheduler steps. Also "
        "sizes the scheduler's pull cycle (2x this) and therefore the prefill "
        "wave: one prefill wave per engine pull, which is how often the waiting "
        "queue could grow before L2 existed.",
    )
    group.add_argument(
        "--l2_cache_gb",
        type=float,
        default=0.0,
        help="CPU L2 KV cache budget per rank, in GiB. 0 disables L2.",
    )
    group.add_argument(
        "--l2_step_ms",
        type=float,
        default=10.0,
        help="Scheduler step duration in milliseconds used to size L2 copy deadlines.",
    )
    parser = parallel_args(parser)
    parser = sampling_args(parser)
    parser = model_args(parser)
    return parser


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr, port = s.getsockname()
    s.close()
    return port


def parse_args(args=None):
    if args is None:
        args = inference_args().parse_args()
    if args.master_port is None:
        args.master_port = find_free_port()

    root_dir_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(args.weight_path, "config.json")
    gen_config_path = os.path.join(args.weight_path, "generation_config.json")
    assert os.path.isfile(config_path), (
        f"--weight_path={args.weight_path} must contain config.json"
    )
    assert os.path.isfile(gen_config_path), (
        f"--weight_path={args.weight_path} must contain generation_config.json"
    )
    with open(config_path) as f:
        hf_config = json.load(f)
    # Multimodal nested configs expose model details under ``text_config``.
    if "text_config" in hf_config and isinstance(hf_config["text_config"], dict):
        for k, v in hf_config["text_config"].items():
            hf_config.setdefault(k, v)
    with open(gen_config_path) as f:
        gen_hf_config = json.load(f)
    args.eos_token_ids = gen_hf_config["eos_token_id"]
    if not isinstance(args.eos_token_ids, List):
        args.eos_token_ids = [args.eos_token_ids]
    if args.top_k == -1:
        if "top_k" in gen_hf_config.keys():
            args.top_k = gen_hf_config["top_k"]
        else:
            args.top_k = hf_config["vocab_size"]
    elif args.top_k == 0:
        args.top_k = hf_config["vocab_size"]

    if args.top_p == 0.0:
        if "top_p" in gen_hf_config.keys():
            args.top_p = gen_hf_config["top_p"]
        else:
            args.top_p = 1.0

    if args.temperature == -1:
        if "temperature" in gen_hf_config.keys():
            args.temperature = gen_hf_config["temperature"]
        else:
            args.temperature = 1

    if (
        "num_experts" not in hf_config.keys()
        and "n_routed_experts" not in hf_config.keys()
    ):
        assert args.ep_deg == 1, "Dense LLM doesn't support EP!"

    if args.dcp_deg > 1:
        args.tp_deg = args.dcp_deg if args.tp_deg == 1 else args.tp_deg
        assert args.tp_deg >= args.dcp_deg and args.tp_deg % args.dcp_deg == 0

    if args.dcp_deg > 1 and args.dcp_interleave_size > args.page_size:
        raise ValueError(
            f"dcp_interleave_size ({args.dcp_interleave_size}) must be"
            f" <= page_size ({args.page_size}) to avoid slot_mapping"
            f" collisions in the DCP path. Use --dcp_interleave_size <="
            f" {args.page_size} or increase --page_size."
        )
    if args.l2_cache_gb < 0:
        raise ValueError("--l2_cache_gb must be >= 0")

    if args.ep_deg > args.tp_deg:
        args.tp_deg = args.ep_deg
    args.etp_deg = args.tp_deg // args.ep_deg

    # Validate DCP configuration
    if args.dcp_deg > 1:
        assert args.tp_deg % args.dcp_deg == 0, (
            f"Tensor parallel degree ({args.tp_deg}) must be divisible by "
            f"decode context parallel degree ({args.dcp_deg})"
        )
        if "qk_nope_head_dim" in hf_config.keys():
            assert args.tp_deg > 1, (
                "There is not motivation to enable decode context parallel when tp_deg == 1 for MLA"
            )
        else:
            assert (
                hf_config["num_key_value_heads"] <= (args.tp_deg // args.dcp_deg)
                and args.tp_deg % hf_config["num_key_value_heads"] == 0
            ), (
                "There is not motivation to enable decode context parallel when kv_head_per_device > 1"
            )

    if args.use_deepep:
        assert args.ep_deg > 1, (
            "Expert Parallelism not enabled! Please set --ep_deg to a number greater than 1"
        )
        if args.max_batchsize // args.tp_deg * args.etp_deg > 128:
            print(
                f"[\033[31mWarning\033[0m] DeepEP supports maximum batchsize 128 per rank, turning max_batchsize from {args.max_batchsize} to {128 * args.tp_deg // args.etp_deg}!"
            )
            args.max_batchsize = 128 * args.tp_deg // args.etp_deg
            args.cuda_graph_max_batchsize = max(
                args.cuda_graph_max_batchsize, args.max_batchsize
            )

    args.ip_addrs = args.ip_addrs.split(",")
    total_gpus = args.tp_deg * args.pp_deg
    assert total_gpus % len(args.ip_addrs) == 0, (
        f"Total workers ({total_gpus}) must be divisible by the number of nodes "
        f"({len(args.ip_addrs)})."
    )
    args.num_gpus_per_node = total_gpus // len(args.ip_addrs)
    if args.ep_deg > args.num_gpus_per_node and args.use_deepep:
        assert args.ep_deg % 8 == 0 and args.num_gpus_per_node == 8, (
            "DeepEP requires ep_deg to be 8 * num_nodes when deployed in multiple nodes!"
        )

    if args.quant_type != "no_quant":
        if args.quant_type == "fp8_per_block":
            if args.fp8_gemm_backend not in ["deep_gemm", "vllm_cutlass"]:
                args.fp8_gemm_backend = "deep_gemm"
                print(
                    f"[INFO] Switch fp8_gemm_backend to {args.fp8_gemm_backend} for fp8_per_block quantization!"
                )
        elif args.quant_type == "fp8_per_channel":
            if args.fp8_gemm_backend not in ["vllm_cutlass", "torch_native"]:
                args.fp8_gemm_backend = "vllm_cutlass"
                print(
                    f"[INFO] Switch fp8_gemm_backend to {args.fp8_gemm_backend} for fp8_per_channel quantization!"
                )
        assert "quantization_config" in hf_config.keys(), (
            "Detected quantization mismatch! Please check --weight_path, --quant_type, --model are correct"
        )
    else:
        assert "quantization_config" not in hf_config.keys(), (
            "Detected quantization mismatch! Please check --weight_path, --quant_type, --model are correct"
        )

    if args.enable_fp8_dispatch:
        assert (
            args.use_deepep and args.ep_deg > 1 and args.quant_type == "fp8_per_block"
        )

    if args.use_deepep:
        if args.disable_deepep_ll:
            assert args.ep_deg <= args.num_gpus_per_node, (
                "DeepEP Normal Kernel only supports intra-node cuda-graph"
            )
        else:
            assert args.quant_type != "fp8_per_channel", (
                "DeepEP low latency kernel forces fp8 blockwise quantization"
            )

    assert (
        hf_config["num_key_value_heads"] % (args.tp_deg // args.dcp_deg) == 0
        or (args.tp_deg // args.dcp_deg) % hf_config["num_key_value_heads"] == 0
    )
    import torch

    sampling_params = SamplingParams()
    sampling_params.set_from_args(args)

    if "kv_lora_rank" in hf_config.keys():
        args.attention_backend = "flash_mla"
        print(
            "[INFO] Select flashinfer backend for MLA prefill and flashmla backend for MLA decode"
        )
    else:
        args.attention_backend = "flash_attn"
        print("[INFO] Select FA3 backend for GQA/MHA attention")

    if args.attention_backend == "flash_mla" and args.page_size != 64:
        args.page_size = 64
        print("[\033[31mWarning\033[0m] flash_mla only supports page_size 64!")

    if args.enable_fp8_kv:
        # MLA keeps a single latent cache that flash_mla reads directly, and its
        # FP8 path is a different quantization scheme (per-128-element tiles with
        # scales appended to the cache entry), not this one. Reject rather than
        # silently fall back to bf16.
        assert "kv_lora_rank" not in hf_config.keys(), (
            "--enable_fp8_kv does not support MLA models "
            f"({args.model_name} has kv_lora_rank in its config). FP8 KV cache is "
            "implemented for GQA/MHA attention only."
        )
        assert args.attention_backend == "flash_attn", (
            f"--enable_fp8_kv requires the flash_attn backend, got "
            f"{args.attention_backend}"
        )
        from pipeswift.utils.fp8_kv import check_available

        check_available()
        print("[INFO] FP8 e4m3 KV cache enabled (per-tensor scale 1.0, vLLM-aligned)")

    num_experts = (
        hf_config["num_experts"]
        if "num_experts" in hf_config.keys()
        else hf_config["n_routed_experts"]
        if "n_routed_experts" in hf_config.keys()
        else None
    )
    if args.o is None:
        from datetime import datetime

        args.o = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}-{args.model_name}"
    output_dir = os.path.join(root_dir_name, "output")
    os.makedirs(output_dir, exist_ok=True)
    args.o = os.path.abspath(os.path.join(output_dir, f"{args.o}.jsonl"))
    with open(args.o, "+a") as f:
        f.write(f"Arguments: {json.dumps(vars(args), indent=2)}\n")

    if args.num_spec_tokens != 0:
        assert args.num_spec_tokens > 0, (
            "--num_spec_tokens must be 0 (disabled) or greater than 0."
        )
        if args.spec_rejection_mode == "probabilistic":
            assert args.temperature != 0.0 and args.top_k > 1, (
                "Probabilistic MTP speculative decoding requires temperature != 0 and top_k > 1."
            )
        assert args.model_name in {"dpskv3", "glm4"}, (
            f"{args.model_name} does not support MTP speculative decoding."
        )
        assert args.disable_overlap_schedule, (
            "MTP speculative decoding currently requires --disable_overlap_schedule."
        )

    args.hf_config = hf_config

    if args.max_context_len == -1:
        args.max_context_len = hf_config["max_position_embeddings"]
    assert args.max_context_len >= args.max_gen_tokens, (
        f"Maximum generation tokens {args.max_gen_tokens} exceeds {args.model_name}'s maximum context length: {args.max_context_len}!"
    )

    args.sampling_params = sampling_params
    args.dtype = getattr(torch, args.dtype)
    # Dtype of the paged KV cache itself, which diverges from args.dtype only
    # under --enable_fp8_kv. Activations and weights stay at args.dtype.
    if args.enable_fp8_kv:
        from pipeswift.utils.fp8_kv import FP8_KV_DTYPE

        args.kv_cache_dtype = FP8_KV_DTYPE
    else:
        args.kv_cache_dtype = args.dtype
    args.quant_type = (
        QuantType.NO_QUANT
        if args.quant_type == "no_quant"
        else QuantType.FP8_PER_BLOCK
        if args.quant_type == "fp8_per_block"
        else QuantType.FP8_PER_CHANNEL
        if args.quant_type == "fp8_per_channel"
        else None
    )
    return args
