import os
import random
from contextlib import contextmanager, nullcontext
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap
from flashinfer.comm import (
    create_shared_buffer,
    vllm_all_reduce,
    vllm_get_graph_buffer_ipc_meta,
    vllm_init_custom_ar,
    vllm_meta_size,
    vllm_register_buffer,
    vllm_register_graph_buffers,
)

from .quantization import per_token_group_quant_fp8_blockwise_vllm


def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def min_across_workers(value: int) -> int:
    """Reduce ``value`` to its minimum over every rank.

    The fully distributed scheduler replicates BlockManager state, so page-budget
    decisions must be bit-identical: a per-rank capacity difference makes ranks
    accept different numbers of L2 transfers. The world group spans
    tp x pp x ep x dcp, so this covers every axis; only pp differs today (layer
    split and MTP stage change bytes-per-page).
    """
    if not dist.is_available() or not dist.is_initialized():
        return value
    if dist.get_world_size() == 1:
        return value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reduced = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    return int(reduced.item())


def construct_parallelism_groups(args):
    global parallel_state
    # ATTN: tp, FFN: etp x ep
    tp_deg, pp_deg, etp_deg, ep_deg, dcp_deg = (
        args.tp_deg,
        args.pp_deg,
        args.etp_deg,
        args.ep_deg,
        args.dcp_deg,
    )
    world_size = parallel_state.global_world_size
    rank = parallel_state.global_rank

    tp_groups = [
        list(range(i * tp_deg, (i + 1) * tp_deg)) for i in range(world_size // tp_deg)
    ]
    pp_groups = [
        list(range(i, world_size, world_size // pp_deg))
        for i in range(world_size // pp_deg)
    ]
    ep_groups = [
        list(range(i * ep_deg, (i + 1) * ep_deg)) for i in range(world_size // ep_deg)
    ]
    etp_groups = []
    for pp_rank in range(pp_deg):
        for ep_start_idx in range(
            pp_rank * world_size // pp_deg,
            pp_rank * world_size // pp_deg + ep_deg,
        ):
            etp_groups.append(
                list(
                    range(
                        ep_start_idx,
                        ep_start_idx + ep_deg * etp_deg,
                        ep_deg,
                    )
                )
            )

    parallel_state.pp_deg = pp_deg
    parallel_state.tp_deg = tp_deg
    parallel_state.etp_deg = etp_deg
    parallel_state.ep_deg = ep_deg
    parallel_state.dcp_deg = dcp_deg

    # DCP groups: DCP reuses TP group GPUs
    # For tp_deg=4, dcp_deg=2: TP group [0,1,2,3] -> DCP groups [[0,1], [2,3]]
    dcp_groups = []
    if dcp_deg > 1:
        for tp_gp in tp_groups:
            num_dcp_gps = tp_deg // dcp_deg
            for dcp_gp_idx in range(num_dcp_gps):
                start_idx = dcp_gp_idx * dcp_deg
                end_idx = start_idx + dcp_deg
                dcp_group_ranks = tp_gp[start_idx:end_idx]
                dcp_groups.append(dcp_group_ranks)

    for gp in tp_groups:
        tpg = torch.distributed.new_group(gp)
        if rank in gp:
            parallel_state.tp_group = tpg
            parallel_state.tp_rank = gp.index(rank)
            parallel_state.adaptive_ar_tp = AdaptiveAllReduce(
                tpg,
                disable_vllm_custom_ar=args.disable_vllm_custom_ar,
                _dtype=args.dtype,
            )

    for gp in pp_groups:
        ppg = torch.distributed.new_group(gp)
        if rank in gp:
            parallel_state.pp_group = ppg
            parallel_state.pp_rank = gp.index(rank)
            parallel_state.pp_next_rank = (
                gp[gp.index(rank) + 1] if gp.index(rank) + 1 < pp_deg else None
            )
            parallel_state.pp_prev_rank = (
                gp[gp.index(rank) - 1] if gp.index(rank) >= 1 else None
            )

    if parallel_state.pp_deg > 1:  # warmup batch_isend_irecv p2p
        pp_rank = parallel_state.pp_rank
        pp_deg = parallel_state.pp_deg
        pp_group = parallel_state.pp_group

        ops = []
        send_tensors = []
        recv_tensors = []

        for peer in range(pp_deg):
            if peer == pp_rank:
                continue

            send = torch.zeros((256, 4096), device="cuda")
            recv = torch.zeros((256, 4096), device="cuda")

            send_tensors.append(send)
            recv_tensors.append(recv)

            ops.append(
                dist.P2POp(
                    dist.isend,
                    send,
                    group_peer=peer,
                    group=pp_group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv,
                    group_peer=peer,
                    group=pp_group,
                )
            )

        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

        if parallel_state.pp_next_rank is not None:
            dist.isend(send, parallel_state.pp_next_rank)
        if parallel_state.pp_prev_rank is not None:
            dist.recv(recv, parallel_state.pp_prev_rank)

    # if 'num_experts' in args.hf_config.keys() or 'n_routed_experts' in args.hf_config.keys():
    for gp in etp_groups:
        etpg = torch.distributed.new_group(gp)
        if rank in gp:
            parallel_state.etp_group = etpg
            parallel_state.etp_rank = gp.index(rank)
            parallel_state.adaptive_ar_etp = AdaptiveAllReduce(
                etpg,
                disable_vllm_custom_ar=args.disable_vllm_custom_ar,
                _dtype=args.dtype,
            )

    for gp in ep_groups:
        epg = torch.distributed.new_group(gp)
        if rank in gp:
            parallel_state.ep_group = epg
            parallel_state.ep_rank = gp.index(rank)
            if parallel_state.ep_deg > 1 and args.use_deepep:
                parallel_state.deepep = DeepEP(
                    args, fp8_dispatch=args.enable_fp8_dispatch, ep_group=epg
                )

    # Initialize DCP groups
    for gp in dcp_groups:
        dcpg = torch.distributed.new_group(gp)
        if rank in gp:
            parallel_state.dcp_group = dcpg
            parallel_state.dcp_rank = gp.index(rank)

    if rank == 0:
        print("-" * 20 + "Parallelism Info" + "-" * 20)
        print("[TP]", "Degree:", parallel_state.tp_deg)
        print("groups:", tp_groups)
        print("[PP]", "Degree:", parallel_state.pp_deg)
        print("groups:", pp_groups)
        if dcp_deg > 1:
            print("[DCP]", "Degree:", parallel_state.dcp_deg)
            print("groups:", dcp_groups)
        if (
            "num_experts" in args.hf_config.keys()
            or "n_routed_experts" in args.hf_config.keys()
        ):
            print("[EP]", "Degree:", parallel_state.ep_deg)
            print("groups:", ep_groups)
            print("[ETP]", "Degree:", parallel_state.etp_deg)
            print("groups:", etp_groups)
        print("-" * 70)


def initialize_distributed(args, global_rank):
    global parallel_state
    parallel_state.global_rank = global_rank
    parallel_state.global_world_size = args.tp_deg * args.pp_deg
    _num_gpus_per_node = parallel_state.global_world_size // len(args.ip_addrs)
    parallel_state.num_gpus_per_node = _num_gpus_per_node
    node_rank = global_rank // parallel_state.num_gpus_per_node
    master_addr = args.ip_addrs[0]
    master_port = args.master_port
    _local_rank = global_rank % _num_gpus_per_node

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    world_size = parallel_state.global_world_size
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)
    if global_rank == 0:
        print(f"[NODE {node_rank}] > initializing torch distributed ...", flush=True)

    torch.cuda.set_device(_local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
    )
    # Set the tensor model-parallel, pipeline model-parallel
    # if world_size > 1:
    for _node_rank in range(
        parallel_state.global_world_size // parallel_state.num_gpus_per_node
    ):
        _node_ranks = list(
            range(
                _node_rank * parallel_state.num_gpus_per_node,
                (_node_rank + 1) * parallel_state.num_gpus_per_node,
            )
        )
        _node_gp = torch.distributed.new_group(_node_ranks)
        if global_rank in _node_ranks:
            parallel_state.node_group = _node_gp
    construct_parallelism_groups(args)
    torch.set_float32_matmul_precision("highest")
    set_seed()


class AdaptiveAllReduce:
    _VLLM_CA_MAX_CTAS: int = 36
    _VLLM_CA_MAX_SIZE = 8 * 1024 * 1024
    _VLLM_CA_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _FULL_NVLINK = True  # NOTE(wsj): may be False, but I suppose it to be True
    MiB = 1024 * 1024

    def __init__(
        self,
        group: dist.ProcessGroup,
        disable_vllm_custom_ar=False,
        _dtype=torch.bfloat16,
    ):
        self.group = group
        self.disable_vllm_ca = True
        self.world_size = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(self.group)
        self._IS_CAPTURING = False
        if self.world_size == 1:
            return
        global parallel_state

        if not disable_vllm_custom_ar:
            if (
                parallel_state.num_gpus_per_node
                < dist.get_process_group_ranks(self.group)[-1]
                - dist.get_process_group_ranks(self.group)[0]
                + 1
            ):  # vllm_ca forces intra-node
                if parallel_state.global_rank == 0:
                    print(
                        f"[\033[31mWarning\033[0m] VLLM custom all_reduce doesn't support inter-node allreduce"
                    )
            else:
                if (
                    self.world_size
                    not in AdaptiveAllReduce._VLLM_CA_SUPPORTED_WORLD_SIZES
                ):
                    if parallel_state.global_rank == 0:
                        print(
                            f"[\033[31mWarning\033[0m] VLLM custom all_reduce doesn't support worldsize {self.world_size}"
                        )
                else:
                    self.vllm_ca_meta_ptrs = create_shared_buffer(
                        vllm_meta_size() + AdaptiveAllReduce._VLLM_CA_MAX_SIZE,
                        group=group,
                    )
                    self.vllm_ca_buffer_ptrs = create_shared_buffer(
                        AdaptiveAllReduce._VLLM_CA_MAX_SIZE, group=group
                    )
                    self.vllm_ca_rank_data = torch.empty(
                        8 * 1024 * 1024, dtype=torch.uint8, device="cuda"
                    )
                    self.vllm_ca_ptr = vllm_init_custom_ar(
                        self.vllm_ca_meta_ptrs,
                        self.vllm_ca_rank_data,
                        self.rank,
                        AdaptiveAllReduce._FULL_NVLINK,
                    )
                    vllm_register_buffer(self.vllm_ca_ptr, self.vllm_ca_buffer_ptrs)
                    self.disable_vllm_ca = False

    def forward(self, inp: torch.Tensor):
        if self.world_size == 1:
            return inp
        elif self.should_vllm_custom_ar(inp):
            return self.vllm_custom_all_reduce(inp)
        else:
            torch.distributed.all_reduce(inp, op=dist.ReduceOp.SUM, group=self.group)
            return inp

    def should_vllm_custom_ar(self, inp: torch.Tensor):
        if self.disable_vllm_ca:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False

        def is_weak_contiguous(inp: torch.Tensor):
            return inp.is_contiguous() or (
                inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
                == inp.numel() * inp.element_size()
            )

        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or AdaptiveAllReduce._FULL_NVLINK:
            return inp_size < AdaptiveAllReduce._VLLM_CA_MAX_SIZE
        return False

    def _vllm_custom_all_reduce(
        self, inp: torch.Tensor, out: torch.Tensor = None, registered: bool = False
    ):
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            vllm_all_reduce(
                self.vllm_ca_ptr, inp, out, 0, 0, AdaptiveAllReduce._VLLM_CA_MAX_CTAS
            )
        else:
            vllm_all_reduce(
                self.vllm_ca_ptr,
                inp,
                out,
                self.vllm_ca_buffer_ptrs[self.rank],
                AdaptiveAllReduce._VLLM_CA_MAX_SIZE,
                AdaptiveAllReduce._VLLM_CA_MAX_CTAS,
            )
        return out

    def vllm_custom_all_reduce(self, inp: torch.Tensor):
        if self.disable_vllm_ca or not self.should_vllm_custom_ar(inp):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self._vllm_custom_all_reduce(inp, registered=True)
            else:
                return torch.zeros_like(inp)
        else:
            return self._vllm_custom_all_reduce(inp, registered=False)

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disable_vllm_ca and self.world_size > 1:
                handle, offset = vllm_get_graph_buffer_ipc_meta(self.vllm_ca_ptr)
                all_data = [
                    [None, None] for _ in range(dist.get_world_size(group=self.group))
                ]
                all_data[self.rank] = [handle, offset]
                ranks = sorted(dist.get_process_group_ranks(group=self.group))
                for i, rank in enumerate(ranks):
                    dist.broadcast_object_list(all_data[i], src=rank, group=self.group)
                # Unpack list of tuples to tuple of lists.
                handles = [d[0] for d in all_data]  # type: ignore
                offsets = [d[1] for d in all_data]  # type: ignore
                vllm_register_graph_buffers(self.vllm_ca_ptr, handles, offsets)

    @contextmanager
    def graph_capture(self):
        maybe_ca_context = nullcontext() if self.disable_vllm_ca else self.capture()
        with maybe_ca_context:
            yield


class ActRedistributor:
    def __init__(self):
        self._IS_CAPTURING = False

    def scatter_act_eager(
        self,
        hidden_states: torch.Tensor,
        pg: dist.ProcessGroup,
        scatter_st_idx: int = None,
        scatter_ed_idx: int = None,
    ):
        if scatter_st_idx is not None:
            return hidden_states[scatter_st_idx:scatter_ed_idx], None
        else:
            pg_world_size = dist.get_world_size(pg)
            pg_rank = dist.get_rank(pg)
            hidden_states_list = torch.tensor_split(hidden_states, pg_world_size)
            token_nums = [s.size(0) for s in hidden_states_list]
            return hidden_states_list[pg_rank], token_nums

    def scatter_act_cuda_graph(
        self, hidden_states: torch.Tensor, pg: dist.ProcessGroup
    ):
        pg_world_size = dist.get_world_size(pg)
        pg_rank = dist.get_rank(pg)
        assert hidden_states.shape[0] % pg_world_size == 0
        hidden_states_list = torch.tensor_split(hidden_states, pg_world_size)
        return hidden_states_list[pg_rank], None

    def scatter_act(
        self,
        hidden_states: torch.Tensor,
        pg: dist.ProcessGroup,
        scatter_st_idx: int = None,
        scatter_ed_idx: int = None,
    ):
        if self._IS_CAPTURING:
            return self.scatter_act_cuda_graph(hidden_states, pg)
        else:
            return self.scatter_act_eager(
                hidden_states, pg, scatter_st_idx, scatter_ed_idx
            )

    # NOTE(wsj): for flexibility, this operation is not compatible with torch.compile and cuda graph
    def allgather_act_eager(
        self, hidden_states: torch.Tensor, pg: dist.ProcessGroup, _token_nums=None
    ):
        pg_world_size = dist.get_world_size(pg)
        pg_rank = dist.get_rank(pg)
        if _token_nums is not None:
            token_nums = _token_nums
        else:
            token_nums, token_num = [0] * pg_world_size, hidden_states.shape[0]
            torch.distributed.all_gather_object(token_nums, token_num, pg)
        _hidden_states = torch.empty(
            (sum(token_nums),) + hidden_states.shape[1:],
            dtype=hidden_states.dtype,
            device="cuda",
        )
        offset = 0
        _hidden_states_list = []
        scatter_start_idx, scatter_end_idx = (
            sum(token_nums[:pg_rank]),
            sum(token_nums[: (pg_rank + 1)]),
        )
        for _token_num in token_nums:
            _hidden_states_list.append(_hidden_states[offset : offset + _token_num])
            offset += _token_num  # change this
        torch.distributed.all_gather(_hidden_states_list, hidden_states, pg)
        return _hidden_states, scatter_start_idx, scatter_end_idx

    def allgather_act_cuda_graph(
        self, hidden_states: torch.Tensor, pg: dist.ProcessGroup
    ):
        _hidden_states = torch.empty(
            (dist.get_world_size(pg) * hidden_states.shape[0],)
            + hidden_states.shape[1:],
            dtype=hidden_states.dtype,
            device="cuda",
        )
        dist.all_gather_into_tensor(_hidden_states, hidden_states, group=pg)
        return _hidden_states, None, None

    def allgather_act(
        self, hidden_states: torch.Tensor, pg: dist.ProcessGroup, _token_nums=None
    ):
        if self._IS_CAPTURING:
            return self.allgather_act_cuda_graph(hidden_states, pg)
        else:
            return self.allgather_act_eager(hidden_states, pg, _token_nums)

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


class DeepEP:
    _RETURN_RECV_HOOK = True
    _ASYNC_FINISH = True
    _NUM_MAX_DISPATCH_TOKENS_PER_RANK = 0
    DEEPEP_NORMAL_MODE = 0
    DEEPEP_LL_MODE = 1

    def __init__(self, args, fp8_dispatch: bool, ep_group: dist.ProcessGroup):
        Buffer.set_num_sms(24)
        self.hidden = args.hf_config["hidden_size"]
        self.num_experts = (
            args.hf_config["num_experts"]
            if "num_experts" in args.hf_config.keys()
            else args.hf_config["n_routed_experts"]
        )
        self.handle = None
        self.deepep_buffer = None
        self.use_fp8 = fp8_dispatch
        self.disable_deepep_ll = args.disable_deepep_ll
        self.handle = None
        self._IS_CAPTURING = False
        self._PREVIOUS_NORMAL_MODE = True
        DeepEP._NUM_MAX_DISPATCH_TOKENS_PER_RANK = 128
        self.initialize_deepep_buffer(args, ep_group)

    def initialize_deepep_buffer(self, args, ep_group):
        hidden_size, ele_size = (
            args.hf_config["hidden_size"],
            torch.Tensor([]).to(args.dtype).element_size(),
        )
        num_experts = (
            args.hf_config["num_experts"]
            if "num_experts" in args.hf_config.keys()
            else args.hf_config["n_routed_experts"]
        )
        self.get_buffer(ep_group, hidden_size, ele_size, num_experts)

    def get_buffer(
        self, group: dist.ProcessGroup, hidden: int, ele_size: int, num_experts
    ):
        num_nvl_bytes, num_rdma_bytes, hidden_bytes = 0, 0, hidden * ele_size
        for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                num_rdma_bytes,
            )
        num_qps_per_rank = 24
        if not self.disable_deepep_ll:
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    DeepEP._NUM_MAX_DISPATCH_TOKENS_PER_RANK,
                    hidden,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )
            if (
                self.deepep_buffer is None
                or self.deepep_buffer.group != group
                or not self.deepep_buffer.low_latency_mode
                or self.deepep_buffer.num_rdma_bytes < num_rdma_bytes
            ):
                # NOTES: for the best performance, the QP number **must** be equal to the number of the local experts
                assert num_experts % group.size() == 0
            num_qps_per_rank = max(num_qps_per_rank, num_experts // group.size())
        self.deepep_buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=not self.disable_deepep_ll,
            num_qps_per_rank=num_qps_per_rank,
            allow_mnnvl=True,
        )

    def _deepep_dispatch_normal(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        List,
        Tuple,
        EventOverlap,
    ]:
        # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
        # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
        # refer to the docs of `Buffer.dispatch`
        # Calculate layout before actual dispatch
        global parallel_state
        previous_event = Buffer.capture() if DeepEP._ASYNC_FINISH else None
        self._PREVIOUS_NORMAL_MODE = True
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.deepep_buffer.get_dispatch_layout(
            topk_idx,
            self.num_experts,
            previous_event=previous_event,
            async_finish=DeepEP._ASYNC_FINISH,
            allocate_on_comm_stream=previous_event is not None,
        )
        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
        # Unless you specify `num_worst_tokens`, but this flag is for intranode only
        # For more advanced usages, please refer to the docs of the `dispatch` function
        num_worst_tokens = (
            0 if not self._IS_CAPTURING else x.shape[0] * parallel_state.ep_deg
        )
        if self.use_fp8:
            x = per_token_group_quant_fp8_blockwise_vllm(x)

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            event,
        ) = self.deepep_buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=DeepEP._ASYNC_FINISH,
            allocate_on_comm_stream=(previous_event is not None)
            and DeepEP._ASYNC_FINISH,
            num_worst_tokens=num_worst_tokens,
        )
        event.current_stream_wait() if DeepEP._ASYNC_FINISH else ()
        # For event management, please refer to the docs of the `EventOverlap` class
        return recv_x, recv_topk_idx, recv_topk_weights

    def _deepep_combine_normal(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, EventOverlap]:
        previous_event = Buffer.capture() if DeepEP._ASYNC_FINISH else None
        # Do MoE combine
        # For more advanced usages, please refer to the docs of the `combine` function
        combined_x, _, event = self.deepep_buffer.combine(
            x,
            self.handle,
            async_finish=DeepEP._ASYNC_FINISH,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        self.handle = None
        event.current_stream_wait() if DeepEP._ASYNC_FINISH else ()
        # For event management, please refer to the docs of the `EventOverlap` class
        return combined_x

    def _deepep_dispatch_low_latency(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
    ):
        # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
        expected_m = (
            hidden_states.shape[0] * self.deepep_buffer.group_size * topk_idx.shape[1]
            + self.num_experts
        ) // self.num_experts
        self.set_buffer_low_latency_mode()

        packed_recv_hidden, masked_m, self.handle, event, hook = (
            self.deepep_buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                DeepEP._NUM_MAX_DISPATCH_TOKENS_PER_RANK,
                self.num_experts,
                async_finish=not DeepEP._RETURN_RECV_HOOK,
                return_recv_hook=DeepEP._RETURN_RECV_HOOK,
                use_fp8=self.use_fp8,
            )
        )
        hook() if DeepEP._RETURN_RECV_HOOK else event.current_stream_wait()
        # NOTES: the actual tensor will not be received only if you call `hook()`,
        # it is useful for double-batch overlapping, but **without any SM occupation**
        # If you don't want to overlap, please set `return_recv_hook=False`
        # Later, you can use our GEMM library to do the computation with this specific format
        return packed_recv_hidden, masked_m, expected_m

    def set_buffer_low_latency_mode(self):
        if self._PREVIOUS_NORMAL_MODE:
            self.deepep_buffer.clean_low_latency_buffer(
                DeepEP._NUM_MAX_DISPATCH_TOKENS_PER_RANK, self.hidden, self.num_experts
            )
            self._PREVIOUS_NORMAL_MODE = False

    def _deepep_combine_low_latency(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
        combined_hidden_states, event_overlap, hook = (
            self.deepep_buffer.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                self.handle,
                async_finish=not DeepEP._RETURN_RECV_HOOK,
                return_recv_hook=DeepEP._RETURN_RECV_HOOK,
            )
        )
        self.deepep_buffer.get_next_low_latency_combine_buffer
        self.handle = None
        hook() if DeepEP._RETURN_RECV_HOOK else event_overlap.current_stream_wait()
        # NOTES: the same behavior as described in the dispatch kernel
        return combined_hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        is_decode: bool = False,
    ):
        mode = (
            DeepEP.DEEPEP_LL_MODE
            if (is_decode and not self.disable_deepep_ll)
            else DeepEP.DEEPEP_NORMAL_MODE
        )
        if mode == DeepEP.DEEPEP_NORMAL_MODE:
            return self._deepep_dispatch_normal(hidden_states, topk_idx, topk_weights)
        elif mode == DeepEP.DEEPEP_LL_MODE:
            return self._deepep_dispatch_low_latency(hidden_states, topk_idx)

    def combine(self, hidden_states, topk_idx, topk_weights, is_decode: bool = False):
        mode = (
            DeepEP.DEEPEP_LL_MODE
            if (is_decode and not self.disable_deepep_ll)
            else DeepEP.DEEPEP_NORMAL_MODE
        )
        if mode == DeepEP.DEEPEP_NORMAL_MODE:
            return self._deepep_combine_normal(hidden_states)
        elif mode == DeepEP.DEEPEP_LL_MODE:
            return self._deepep_combine_low_latency(
                hidden_states, topk_idx, topk_weights
            )

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


from dataclasses import dataclass


@dataclass
class ParallelState:
    global_rank: int = 0
    global_world_size: int = 1

    act_redistributor: ActRedistributor = ActRedistributor()

    num_gpus_per_node: int = 1
    node_group: torch.distributed.ProcessGroup = None

    tp_rank: int = 0
    tp_deg: int = 1
    tp_group: torch.distributed.ProcessGroup = None
    adaptive_ar_tp: AdaptiveAllReduce = None
    adaptive_ar_etp: AdaptiveAllReduce = None

    pp_rank: int = 0
    pp_deg: int = 1
    pp_next_rank: int = None
    pp_prev_rank: int = None
    pp_group: torch.distributed.ProcessGroup = None

    ep_rank: int = 0
    ep_deg: int = 1
    ep_group: torch.distributed.ProcessGroup = None
    deepep: DeepEP = None

    etp_rank: int = 0
    etp_deg: int = 1
    etp_group: torch.distributed.ProcessGroup = None

    # Decode Context Parallel
    dcp_rank: int = 0
    dcp_deg: int = 1
    dcp_group: torch.distributed.ProcessGroup = None


parallel_state = ParallelState()
