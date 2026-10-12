from collections import defaultdict
from contextlib import nullcontext
from typing import List

import flashinfer
import torch
import torch.distributed as dist
from tqdm import tqdm

from pipeswift.utils import (
    KVCacheInfo,
    freeze_gc,
    initialize_graph_kvcacheinfo,
    set_graph_kvcacheinfo,
    slice_kvcache_info,
)

from .layers import *
from .layers.attention import set_flashmla_cg_meta


class LLM(torch.nn.Module):
    _MAX_GRAPH_BSZ_INTERVAL = 8

    def __init__(self, args):
        from pipeswift.utils import parallel_state

        super().__init__()
        # self.kv_cache:List[torch.Tensor] = [] # 2 [n_layer, num_pages, page_size, n_kv_head, head_dim]
        self.dtype = args.dtype
        # Only differs from self.dtype under --enable_fp8_kv.
        self.kv_cache_dtype = getattr(args, "kv_cache_dtype", args.dtype)
        self.enable_fp8_kv = getattr(args, "enable_fp8_kv", False)
        if self.enable_fp8_kv:
            from pipeswift.utils.fp8_kv import set_enabled, warmup

            set_enabled(True)
            # Allocate the descale constant now, with the per-rank device
            # already set and before any CUDA graph capture.
            warmup(torch.device("cuda", torch.cuda.current_device()))
        self.quant_type = args.quant_type
        self.mem_util_threshold = args.mem_util_threshold  # memory usage
        self.max_batched_tokens = args.max_batched_tokens
        self.max_batchsize = args.max_batchsize
        self.args = args
        self.use_mla = "qk_nope_head_dim" in args.hf_config.keys()
        self.page_size = args.page_size

        tot_n_layer = args.hf_config["num_hidden_layers"]
        draft_vs_layer = args.draft_vs_layer
        if hasattr(args, "num_spec_tokens") and args.num_spec_tokens > 0:
            tot_n_layer += int(args.num_spec_tokens * draft_vs_layer)

        if tot_n_layer % parallel_state.pp_deg != 0:
            _n_layer = tot_n_layer // parallel_state.pp_deg
            _div_layers = [_n_layer + 1] * (tot_n_layer % parallel_state.pp_deg) + [
                _n_layer
            ] * (parallel_state.pp_deg - tot_n_layer % parallel_state.pp_deg)
            if parallel_state.pp_rank == 0 and parallel_state.tp_rank == 0:
                print(
                    f"[\033[31mWarning\033[0m] This model has {tot_n_layer} layers, which can not be evenly divided by {parallel_state.pp_deg} pipeline stages, use division {_div_layers}!"
                )
        else:
            _div_layers = [tot_n_layer // parallel_state.pp_deg] * parallel_state.pp_deg

        self.n_layer_per_device = _div_layers[parallel_state.pp_rank]
        if (
            hasattr(args, "num_spec_tokens")
            and args.num_spec_tokens > 0
            and parallel_state.pp_next_rank is None
        ):
            self.n_layer_per_device -= int(args.num_spec_tokens * draft_vs_layer)
        print(
            f"[rank {parallel_state.global_rank}] has {self.n_layer_per_device} layers"
        )

        self.start_layer_idx = sum(_div_layers[: parallel_state.pp_rank])
        if parallel_state.pp_deg > 1 and parallel_state.global_rank == 0:
            print(f"[INFO] Pipeline Parallelism divides models layers as {_div_layers}")

        self.sampling_params = self.args.sampling_params

        self._MAX_GRAPH_BSZ_DEFAULT = args.cuda_graph_max_batchsize
        assert args.hf_config["num_attention_heads"] % args.tp_deg == 0
        self.q_head_per_device = args.hf_config["num_attention_heads"] // args.tp_deg
        if "qk_nope_head_dim" in args.hf_config.keys():
            self.qkhead_dim = (
                args.hf_config["qk_nope_head_dim"] + args.hf_config["qk_rope_head_dim"]
            )
            self.vhead_dim = args.hf_config["v_head_dim"]
            self.kv_head_per_device = self.q_head_per_device
            self.gqa_group_size = 1
        else:
            self.qkhead_dim = (
                args.hf_config["hidden_size"] // args.hf_config["num_attention_heads"]
                if "head_dim" not in args.hf_config.keys()
                else args.hf_config["head_dim"]
            )
            self.vhead_dim = self.qkhead_dim
            head_per_device = self.q_head_per_device
            self.gqa_group_size = (
                args.hf_config["num_attention_heads"]
                // args.hf_config["num_key_value_heads"]
            )
            import math

            self.kv_head_per_device = math.ceil(head_per_device / self.gqa_group_size)
        self.hidden_size = args.hf_config["hidden_size"]
        self.scaling = 1.0 / (self.qkhead_dim**0.5)
        self.attention_backend = args.attention_backend
        # for cuda graph
        self.static_kvcahceinfo = KVCacheInfo()
        self.static_input_tensor: torch.Tensor = None
        self.static_position_ids: torch.Tensor = None
        self.static_output_tensor: torch.Tensor = None
        self.graph_bs: dict[int, List] = None
        self.graphs = {}
        self.extend_static_kvcacheinfo = KVCacheInfo()
        self.extend_static_input_tensor: torch.Tensor = None
        self.extend_static_position_ids: torch.Tensor = None
        self.extend_static_output_tensor: torch.Tensor = None
        # In-flight PP activation send (buffer + handle), drained before reuse.
        self._pp_send_work = None
        self._pp_send_buf = None
        if args.attention_backend == "flash_mla" and not args.enforce_eager:
            set_flashmla_cg_meta(self._MAX_GRAPH_BSZ_DEFAULT)

    def load_weight(self, weight_path: str):
        import json
        import os

        import torch
        from safetensors import safe_open

        from pipeswift.utils import parallel_state

        class SafeTensorWrapper:
            def __init__(self, path):
                self.handle = safe_open(path, framework="pt", device="cpu")

            def __getitem__(self, key):
                return self.handle.get_tensor(key)

        from tqdm import tqdm

        module_tasks = []
        for _name, _module in self.named_children():
            if isinstance(_module, torch.nn.ModuleList):
                for __module in _module:
                    module_tasks.append((f"{_name}.{__module.layer_id}", __module))
            else:
                module_tasks.append((_name, _module))
        show_progress = parallel_state.global_rank == 0
        iterator = (
            tqdm(module_tasks, desc="Loading model weights", dynamic_ncols=True)
            if show_progress
            else module_tasks
        )

        with torch.no_grad():
            index_file = os.path.join(weight_path, "model.safetensors.index.json")
            if os.path.isfile(index_file):
                weight_map = json.load(open(index_file, "r"))["weight_map"]
            else:
                weight_map = defaultdict(lambda: "model.safetensors")
                weight_map["lm_head.weight"] = "model.safetensors"
            safetensor_dict = {
                safetensor_file: SafeTensorWrapper(
                    os.path.join(weight_path, safetensor_file)
                )
                for safetensor_file in set(weight_map.values())
            }
            for _, module in iterator:
                if isinstance(module, (ParallelVocabEmbedding, ParallelLMHead)):
                    module._load_weight_with_index(safetensor_dict, weight_map)
                elif isinstance(module, RMSNorm):
                    norm_str = "model.norm.weight"
                    norm_tensor = safetensor_dict[weight_map[norm_str]][norm_str]
                    module._load_weight(norm_tensor)
                elif isinstance(module, ParallelMTP):
                    module._load_weight_with_index(safetensor_dict, weight_map)
                elif isinstance(module, ParallelDecoderLayer):
                    module._load_weight_with_index(safetensor_dict, weight_map)
        if show_progress:
            print("[INFO] All weights loaded successfully.")

    def get_model_size_in_bytes(self):
        total_bytes = 0
        for p in self.parameters():
            total_bytes += p.numel() * p.element_size()
        return total_bytes

    def initialize_kv_pages(
        self, n_layer_per_device, head_dim, vocab_size, intermediate_size
    ):
        from pipeswift.utils import parallel_state

        num_cache_layers = n_layer_per_device + int(hasattr(self, "mtp"))
        avaible_mem_bytes4kv_pages = int(
            torch.cuda.get_device_properties("cuda").total_memory
            * self.mem_util_threshold
            - self.get_model_size_in_bytes()
        )

        kv_head_per_device = self.kv_head_per_device
        element_size = torch.Tensor([]).to(self.dtype).element_size()
        # The cache may be narrower than the activations (--enable_fp8_kv), so
        # size the pages off the cache dtype and the activations off args.dtype.
        kv_element_size = torch.Tensor([]).to(self.kv_cache_dtype).element_size()
        peak_act_mem_bytes = max(
            self.max_batched_tokens
            * 2
            * intermediate_size
            * element_size
            // parallel_state.tp_deg,
            self.max_batchsize * vocab_size * element_size,
        )
        avaible_mem_bytes4kv_pages -= peak_act_mem_bytes
        page_size_byte = (
            self.page_size
            * kv_head_per_device
            * head_dim
            * num_cache_layers
            * 2
            * kv_element_size
        )
        num_pages = avaible_mem_bytes4kv_pages // page_size_byte
        if num_pages <= 0:
            model_gb = self.get_model_size_in_bytes() / 2**30
            budget_gb = (
                torch.cuda.get_device_properties("cuda").total_memory
                * self.mem_util_threshold
                / 2**30
            )
            raise RuntimeError(
                f"No GPU memory left for KV cache after model "
                f"({model_gb:.1f} GB) and peak activations "
                f"({peak_act_mem_bytes / 2**30:.1f} GB) within "
                f"the {budget_gb:.1f} GB budget (mem_util_threshold="
                f"{self.mem_util_threshold}). Reduce --max_batchsize, increase "
                f"--mem_util_threshold, or scale up parallelism (tp_deg/pp_deg)."
            )
        all_num_pages = [None] * parallel_state.pp_deg

        torch.distributed.all_gather_object(
            all_num_pages, num_pages, parallel_state.pp_group
        )

        num_pages = min(all_num_pages)
        torch.cuda.empty_cache()

        kv_cache_tensor = torch.empty(
            (
                num_cache_layers,
                num_pages,
                2,
                self.page_size,
                kv_head_per_device,
                head_dim,
            ),
            device="cuda",
            dtype=self.kv_cache_dtype,
        )
        kv_cache = [kv_cache_tensor[:, :, 0], kv_cache_tensor[:, :, 1]]
        if parallel_state.global_rank == 0:
            print(
                f"[INFO] KV cache: dtype={self.kv_cache_dtype}, {num_pages} pages x "
                f"{self.page_size} tokens = {num_pages * self.page_size} tokens/rank, "
                f"{kv_cache_tensor.numel() * kv_element_size / 2**30:.2f} GiB",
                flush=True,
            )
        return num_pages, kv_cache, kv_cache_tensor

    def initialize_mla_kv_pages(self, n_layer_per_device, ckv_dim, kpe_dim, vocab_size):
        from pipeswift.utils import parallel_state

        assert not self.enable_fp8_kv, (
            "FP8 KV cache is not supported for MLA models; --enable_fp8_kv is "
            "GQA/MHA only. This should have been caught in validate_args."
        )

        num_cache_layers = n_layer_per_device + int(hasattr(self, "mtp"))
        avaible_mem_bytes4kv_pages = int(
            torch.cuda.get_device_properties("cuda").total_memory
            * self.mem_util_threshold
            - self.get_model_size_in_bytes()
        )
        self.ckv_dim, self.kpe_dim = ckv_dim, kpe_dim
        element_size = torch.Tensor([]).to(self.dtype).element_size()
        peak_act_mem_bytes = self.max_batchsize * vocab_size * element_size
        avaible_mem_bytes4kv_pages -= peak_act_mem_bytes
        page_size_byte = (
            self.page_size * num_cache_layers * (ckv_dim + kpe_dim) * element_size
        )
        num_pages = avaible_mem_bytes4kv_pages // page_size_byte
        if num_pages <= 0:
            model_gb = self.get_model_size_in_bytes() / 2**30
            budget_gb = (
                torch.cuda.get_device_properties("cuda").total_memory
                * self.mem_util_threshold
                / 2**30
            )
            raise RuntimeError(
                f"No GPU memory left for MLA KV cache after model "
                f"({model_gb:.1f} GB) and peak activations "
                f"({peak_act_mem_bytes / 2**30:.1f} GB) within "
                f"the {budget_gb:.1f} GB budget (mem_util_threshold="
                f"{self.mem_util_threshold}). Reduce --max_batchsize, increase "
                f"--mem_util_threshold, or scale up parallelism (tp_deg/pp_deg)."
            )
        # NOTE (wsj): MLA's kv cache duplicated within a TP group
        all_num_pages = [None] * parallel_state.pp_deg

        torch.distributed.all_gather_object(
            all_num_pages, num_pages, parallel_state.pp_group
        )

        num_pages = min(all_num_pages)
        torch.cuda.empty_cache()
        mla_cache_tensor = torch.empty(
            (num_cache_layers, num_pages, self.page_size, ckv_dim + kpe_dim),
            device="cuda",
            dtype=self.args.dtype,
        )
        kv_cache = []
        kv_cache.append(mla_cache_tensor[..., :ckv_dim])
        kv_cache.append(mla_cache_tensor[..., ckv_dim:])
        return num_pages, kv_cache, mla_cache_tensor

    def capture_cudagraph(self, kv_cache, mla_cache_tensor):
        with torch.no_grad():
            graph_buffer_bsz = self._MAX_GRAPH_BSZ_DEFAULT
            max_bsz = min(self.max_batchsize, self._MAX_GRAPH_BSZ_DEFAULT)
            # NOTE(wsj) max_seqlen equals to max_batched_tokens
            # Decode preemptively reserves tokens AHEAD of the live length in
            # may_allocate_decode (Instance._collect_decode_seqs / MTP draft):
            # up to ``3*num_spec_tokens + 2`` with MTP, 1 otherwise. A sequence
            # at max context therefore yields a page_table one page wider than
            # ``max_context_len // page_size``. The static graph buffer must
            # include that headroom — else set_graph_kvcacheinfo's
            # ``page_table[:bsz, :max_pages].copy_`` shape-mismatches (e.g.
            # 3168 vs 3169) deep in a long run, the offending rank raises, and
            # its PP peers hang on the next collective (NCCL COALESCED/RECV
            # watchdog timeout — the "PP2 deadlock"). Mirrors the reserve in
            # Instance.py:_collect_decode_seqs.
            _n_spec = getattr(self, "num_spec_tokens", 0) or 0
            _decode_reserve = 3 * _n_spec + 2 if _n_spec > 0 else 1
            max_num_pages = (
                self.args.max_context_len + _decode_reserve + self.page_size - 1
            ) // self.page_size
            self.static_kvcahceinfo = initialize_graph_kvcacheinfo(
                graph_buffer_bsz, max_num_pages, max_num_tokens=graph_buffer_bsz
            )
            self.static_kvcahceinfo.mla_cache_tensor = mla_cache_tensor
            self.static_kvcahceinfo.kv_cache = kv_cache

            from pipeswift.utils import parallel_state

            if parallel_state.pp_prev_rank is None:
                self.static_input_tensor = torch.zeros(
                    graph_buffer_bsz, dtype=torch.int32, device="cuda"
                )
            else:
                self.static_input_tensor = torch.zeros(
                    (graph_buffer_bsz, self.hidden_size),
                    dtype=self.args.dtype,
                    device="cuda",
                )

            self.static_output_tensor = torch.zeros(
                (graph_buffer_bsz, self.hidden_size),
                dtype=self.args.dtype,
                device="cuda",
            )

            self.static_position_ids = torch.zeros(
                graph_buffer_bsz, dtype=torch.int32, device="cuda"
            )
            self.extend_static_input_tensor = self.static_input_tensor
            self.extend_static_position_ids = self.static_position_ids
            self.extend_static_output_tensor = self.static_output_tensor
            self.graph_pool = None
            base_graph_bs = [1, 2, 4, 8, 12, 16] + list(
                range(16, max_bsz + 1, LLM._MAX_GRAPH_BSZ_INTERVAL)
            )
            if (
                parallel_state.tp_deg != parallel_state.etp_deg
                and parallel_state.deepep is not None
            ):
                base_graph_bs = [
                    _bsz for _bsz in base_graph_bs if _bsz % parallel_state.tp_deg == 0
                ]
            max_decode_q_len = max(1, getattr(self, "num_spec_tokens", 0) + 1)
            decode_q_lens = [1]
            if max_decode_q_len > 1:
                decode_q_lens.append(max_decode_q_len)
            self.graph_bs = {
                q_len: self.get_decode_graph_bs(base_graph_bs, q_len)
                for q_len in decode_q_lens
            }

            show_progress = parallel_state.global_rank == 0
            maybe_deepep_ctx = (
                parallel_state.deepep.capture()
                if parallel_state.deepep is not None
                else nullcontext()
            )

            decode_graph_items = [
                (q_len, bsz)
                for q_len, graph_bs in self.graph_bs.items()
                for bsz in graph_bs
            ]
            _iter = (
                decode_graph_items
                if not show_progress
                else tqdm(
                    decode_graph_items,
                    desc="Capturing CUDA Graphs",
                    dynamic_ncols=True,
                    mininterval=0,
                )
            )
            with (
                freeze_gc(),
                parallel_state.act_redistributor.capture(),
                parallel_state.adaptive_ar_tp.graph_capture(),
                parallel_state.adaptive_ar_etp.graph_capture(),
                maybe_deepep_ctx,
            ):
                for q_len, bsz in _iter:
                    if show_progress:
                        _iter.set_postfix({"q_len": q_len, "bsz": bsz}, refresh=False)
                    if (
                        parallel_state.deepep is not None
                        and not self.args.disable_deepep_ll
                    ):
                        parallel_state.deepep.set_buffer_low_latency_mode()
                    token_bsz = bsz * q_len
                    self.static_kvcahceinfo.actual_lens_q.fill_(q_len)
                    self.static_kvcahceinfo.actual_lens_q_cpu.fill_(q_len)
                    self.static_kvcahceinfo.cu_seqlens_q.copy_(
                        torch.arange(
                            0,
                            (graph_buffer_bsz + 1) * q_len,
                            q_len,
                            dtype=torch.int32,
                            device="cuda",
                        )
                    )
                    graph = torch.cuda.CUDAGraph()
                    _static_kvcacheinfo = slice_kvcache_info(
                        self.static_kvcahceinfo, bsz, token_bsz
                    )
                    self.plan_attn(_static_kvcacheinfo, is_decode=True, cg_bsz=bsz)
                    self._forward_before_lmhead(
                        self.static_input_tensor[:token_bsz],
                        self.static_position_ids[:token_bsz],
                        _static_kvcacheinfo,
                        is_decode=True,
                    )  # warmup
                    with torch.cuda.graph(graph, self.graph_pool):
                        self.static_output_tensor[:token_bsz] = (
                            self._forward_before_lmhead(
                                self.static_input_tensor[:token_bsz],
                                self.static_position_ids[:token_bsz],
                                _static_kvcacheinfo,
                                is_decode=True,
                            )
                        )  # capture

                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(q_len, bsz)] = graph
                    torch.cuda.synchronize()

    def get_decode_graph_bs(self, graph_bs: List, q_len: int):
        if graph_bs is None:
            return []
        if q_len <= 1:
            return list(graph_bs)
        max_token_bsz = self._MAX_GRAPH_BSZ_DEFAULT
        return [bsz for bsz in graph_bs if bsz * q_len <= max_token_bsz]

    def autotune_flashinfer_cutlass_moe(self, batch_size: int = 512):
        """
        Autotune only unquantized bf16/fp16 FlashInfer CUTLASS fused MoE.
        """
        from pipeswift.utils import parallel_state

        moe_modules = [
            m
            for m in self.modules()
            if hasattr(m, "forward_contiguous") and hasattr(m, "w1")
            # and m.quant_type == QuantType.NO_QUANT
        ]
        if len(moe_modules) == 0:
            return

        # Any MoE layer is enough for kernel autotuning.
        moe = moe_modules[0]
        bsz = min(max(1, batch_size), self.max_batchsize)

        try:
            from flashinfer.autotuner import autotune
        except Exception as e:
            if parallel_state.global_rank == 0:
                print(
                    f"[Warning] flashinfer autotune unavailable, skip moe autotune: {e}"
                )
            return

        top_k = self.args.hf_config["num_experts_per_tok"]
        local_num_experts = getattr(moe, "num_experts_per_device", 1)
        hidden_states = torch.zeros(
            (bsz, self.hidden_size), dtype=self.dtype, device="cuda"
        )
        topk_idx = torch.zeros((bsz, top_k), dtype=torch.int64, device="cuda")
        if local_num_experts > 1:
            topk_idx = torch.arange(top_k, device="cuda").view(1, -1).repeat(bsz, 1)
            topk_idx = topk_idx % local_num_experts
        topk_weights = torch.full(
            (bsz, top_k), 1.0 / float(top_k), dtype=torch.float32, device="cuda"
        )

        if parallel_state.global_rank == 0:
            print(f"[INFO] FlashInfer CUTLASS MoE autotune starts (bsz={bsz})")

        with autotune():
            moe.forward_contiguous(hidden_states, topk_idx, topk_weights)
            torch.cuda.synchronize()

        if parallel_state.global_rank == 0:
            print("[INFO] FlashInfer CUTLASS MoE autotune finished")

    def plan_attn(self, kvcahe_info: KVCacheInfo, is_decode=False, cg_bsz=-1):
        if self.attention_backend == "flash_mla":
            if not is_decode:
                self.plan_flashinfer_wrapper(
                    kvcahe_info,
                    is_decode=is_decode,
                )
            else:
                from pipeswift.utils import parallel_state

                from .layers.attention import set_flash_mla_info

                set_flash_mla_info(
                    kvcahe_info,
                    num_q_head=self.q_head_per_device * parallel_state.dcp_deg,
                    is_decode=True,
                    cg_bsz=cg_bsz,
                )

    def plan_flashinfer_wrapper(self, kv_cache_info: KVCacheInfo, is_decode=False):
        from pipeswift.utils import parallel_state

        from .layers.attention import flashinfer_mla_decode_wrapper

        assert self.use_mla and not is_decode
        flashinfer_mla_decode_wrapper.plan(
            qo_indptr=kv_cache_info.cu_seqlens_q_cpu,
            kv_indptr=kv_cache_info.seqs_kv_page_indptrs_cpu,
            kv_indices=kv_cache_info.seqs_kv_pages,
            kv_len_arr=kv_cache_info.actual_lens_kv_cpu,
            num_heads=self.q_head_per_device * parallel_state.dcp_deg,
            head_dim_ckv=self.ckv_dim,
            head_dim_kpe=self.kpe_dim,
            page_size=self.page_size,
            causal=parallel_state.dcp_deg == 1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            sm_scale=self.scaling,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        kvcache_info: KVCacheInfo,
        is_decode=False,
        have_extend_in_batch=True,
    ):
        import traceback

        from pipeswift.utils import parallel_state

        assert (
            input_tensor.dtype == torch.int32 or parallel_state.pp_rank > 0
        ) and position_ids.dtype == torch.int32
        try:
            # Drain the previous PP activation send before this forward can
            # overwrite/free its buffer (cross-node IB rendezvous reads it late).
            if self._pp_send_work is not None:
                self._pp_send_work.wait()
                self._pp_send_work = None
                self._pp_send_buf = None

            hidden_states = (
                self.forward_decode(input_tensor, position_ids, kvcache_info)
                if is_decode
                else self.forward_prefill(input_tensor, position_ids, kvcache_info)
            )

            # send output to pp_next_rank
            if parallel_state.pp_next_rank is not None:
                self._pp_send_work = dist.isend(
                    hidden_states, dst=parallel_state.pp_next_rank
                )  # async send op
                self._pp_send_buf = hidden_states  # keep alive until next forward
                return None, hidden_states
            elif have_extend_in_batch:
                logits = self.forward_lmhead(hidden_states, kv_cache_info=kvcache_info)
                return self.forward_sampler(logits=logits), hidden_states
            else:
                return None, hidden_states

        except Exception as e:
            print(f"Worker Runtime Crash:\n {traceback.print_exc()}")
            raise

    def forward_prefill(
        self,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo = None,
    ) -> torch.Tensor:
        if input_tensor.shape[0] > 0:
            self.plan_attn(kv_cache_info, is_decode=False, cg_bsz=-1)
        return self._forward_before_lmhead(
            input_tensor, position_ids, kv_cache_info, is_decode=False
        )

    def forward_decode(
        self,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo = None,
    ) -> torch.Tensor:
        from pipeswift.utils import parallel_state

        assert kv_cache_info is not None
        assert kv_cache_info.actual_lens_q_cpu is not None
        q_len = int(kv_cache_info.actual_lens_q_cpu.max().item())
        batch_size = (
            input_tensor.shape[0]
            if q_len == 1
            else kv_cache_info.actual_lens_q.shape[0]
        )
        assert bool(torch.all(kv_cache_info.actual_lens_q_cpu == q_len).item())
        graph_bs = self.graph_bs.get(q_len, []) if self.graph_bs is not None else []
        cg_bsz = (
            min(x for x in graph_bs if x >= batch_size)
            if len(graph_bs) > 0 and graph_bs[-1] >= batch_size
            else -1
        )
        if input_tensor.shape[0] > 0:
            self.plan_attn(kv_cache_info, is_decode=True, cg_bsz=cg_bsz)

        if self.args.enforce_eager or cg_bsz < 0:
            return self._forward_before_lmhead(
                input_tensor, position_ids, kv_cache_info, is_decode=True
            )

        self.static_input_tensor[: input_tensor.shape[0]].copy_(
            input_tensor, non_blocking=True
        )
        self.static_position_ids[: position_ids.shape[0]].copy_(
            position_ids, non_blocking=True
        )
        set_graph_kvcacheinfo(
            self.static_kvcahceinfo,
            kv_cache_info,
            cg_bsz,
            pad_q_len=q_len,
        )
        graph = self.graphs[(q_len, cg_bsz)]
        if parallel_state.deepep is not None and not self.args.disable_deepep_ll:
            parallel_state.deepep.set_buffer_low_latency_mode()
        graph.replay()
        return self.static_output_tensor[: input_tensor.shape[0]]

    def _forward_before_lmhead(
        self,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_info: KVCacheInfo = None,
        is_decode=False,
    ) -> torch.Tensor:
        # forward func for specific models
        raise NotImplementedError("Abstract Class of LLM Model")

    def forward_lmhead(self, hidden_states: torch.Tensor, kv_cache_info: KVCacheInfo):
        from pipeswift.utils import parallel_state

        assert parallel_state.pp_next_rank is None

        if len(hidden_states) > 0:
            actual_lens_q_cpu = kv_cache_info.actual_lens_q_cpu
            hidden_states = hidden_states[
                (torch.cumsum(actual_lens_q_cpu, dim=0) - 1).to(
                    "cuda", non_blocking=True
                )
            ]
        hidden_states = self.lm_head(hidden_states)
        return hidden_states

    def forward_sampler(self, logits: torch.Tensor):
        from pipeswift.utils import parallel_state

        assert parallel_state.pp_next_rank is None

        if logits.shape[0] > 0:
            if (
                self.sampling_params.top_k == 1
                or self.sampling_params.temperature == 0.0
            ):  # greedy
                output_tensor = torch.argmax(logits, dim=-1).to(torch.int32)
            else:
                # Aligned with SGLang sampler.py path (two-step: divide-by-T
                # then torch.softmax, NOT flashinfer.softmax fused) so the
                # resulting probs are numerically identical to SGLang's.
                logits = logits.to(torch.float32)
                logits[:] = flashinfer.softmax(
                    logits, temperature=self.sampling_params.temperature
                )
                # logits.div_(self.sampling_params.temperature)
                # logits[:] = torch.softmax(logits, dim=-1)
                probs = logits
                if (
                    self.sampling_params.top_k == self.args.hf_config["vocab_size"]
                    and self.sampling_params.top_p == 1.0
                ):  # direct sample
                    output_tensor = flashinfer.sampling.sampling_from_probs(
                        probs, deterministic=True
                    )
                elif self.sampling_params.top_k == self.args.hf_config["vocab_size"]:
                    output_tensor = flashinfer.sampling.top_p_sampling_from_probs(
                        probs,
                        self.sampling_params.top_p,
                        deterministic=True,
                    )
                elif self.sampling_params.top_p == 1.0:
                    output_tensor = flashinfer.sampling.top_k_sampling_from_probs(
                        probs,
                        self.sampling_params.top_k,
                        deterministic=True,
                    )
                else:
                    output_tensor = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                        probs,
                        top_k=self.sampling_params.top_k,
                        top_p=self.sampling_params.top_p,
                        filter_apply_order="joint",
                        deterministic=True,
                    )
        else:
            output_tensor = torch.empty(0, dtype=torch.int32, device="cuda")

        return output_tensor
