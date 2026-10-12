import asyncio
import math
import pickle
import traceback
from array import array
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import zmq
from sortedcontainers import SortedList

from pipeswift.llms import LLM, name2model
from pipeswift.utils import (
    KVCacheInfo,
    MsgType,
    Request,
    SamplingParams,
    Sequence,
    WorkerMsg,
    build_socket_conn,
    coalesced_to_cuda,
    mpu,
    prepare_kv_info,
    sequence_offset_desc_key,
)

from .block_manager import BlockManager
from .shm_queue import ShmSPSCQueue


@dataclass
class PendingBatch:
    seqs: List[Sequence]
    token_ids_cpu: torch.Tensor
    token_ids_gpu: torch.Tensor
    num_emitted_draft_tokens: torch.Tensor | None = None
    num_emitted_draft_tokens_gpu: torch.Tensor | None = None
    draft_token_ids_gpu: torch.Tensor | None = None


@dataclass
class PendingBatchRecvPlan:
    seqs: List[Sequence]
    token_ids_gpu: torch.Tensor
    num_emitted_draft_tokens_gpu: torch.Tensor | None = None
    draft_token_ids_gpu: torch.Tensor | None = None


class InstanceProcess:
    throughput_monitor_st = torch.cuda.Event(enable_timing=True)
    throughput_monitor_ed = torch.cuda.Event(enable_timing=True)
    throughput_monitor_steps = 0
    throughput_monitor_tokens = 0
    throughput_monitor_spec_accept_tokens = 0
    throughput_monitor_spec_draft_tokens = 0
    throughput_monitor_spec_drafts = 0

    def __init__(self):
        self.llm: LLM
        self.engine_to_worker_queue: ShmSPSCQueue | None = None
        self.worker_to_engine_queue: ShmSPSCQueue | None = None
        self.sampling_params: SamplingParams
        self.args = None
        self.running: bool = False
        self.pp_irecv_ops_deque = deque()
        # Last-stage result broadcasts accumulated during a prefill burst and
        # flushed once after the burst (see ``_flush_deferred_pp_sends``).
        self._deferred_pp_sends = []

        self.should_process_pending = False
        self.enable_overlap_schedule = False
        self.enable_mtp_spec_decode = False
        self.copy_event_queue = deque()
        self.block_manager: BlockManager = None
        self._FLUSH_OVERLAP_SCHEDULE_STATE = False

        self.received_reqs: List[Request] = []
        self.pending_batch_queue: deque[PendingBatch] = (
            deque()
        )  # used for pp and overlap schedule, where a batches newly generated tokens not merged
        self.waiting_queue: deque[Sequence] = deque()  # FIFO; sequence whose offset = 0
        self.running_queue: SortedList[Sequence] = SortedList(
            key=sequence_offset_desc_key
        )
        self._zigzag_buckets: list[list] = []

        self.max_batchsize = 0
        self.max_batched_tokens = 0
        self.log_interval = 0
        self.control_interval = 1
        self.step_idx = 0
        self.pending_engine_msgs = deque()
        self.finished_outbox = deque()
        self.stopping = False
        self._shutdown_complete = False
        self.profile_initialized = False
        self.pull_cycle_len = 2
        self.prefill_wave_steps = 0
        self._last_prefill_step = 0

    def __call__(
        self,
        local_rank,
        args,
        engine_to_worker_queues,
        worker_to_engine_queues,
    ):
        Sequence.eos_token_ids = args.eos_token_ids
        Sequence.ignore_eos = args.ignore_eos
        Sequence.max_seqlen = args.max_context_len

        self.max_batched_tokens = args.max_batched_tokens
        self.max_batchsize = args.max_batchsize
        self.log_interval = args.log_interval
        self.control_interval = max(1, args.control_interval)
        self.pull_cycle_len = 2 * self.control_interval
        # One prefill wave per engine pull: the cadence the scheduler had before
        # L2 and lost to it (L2 retires a restore, that sequence becomes
        # forwardable, prefill fires again -- measured at a 256-step pull cycle,
        # 27 bursts per cycle at a 3.9-step spacing). Gating prefill to the cycle
        # also gives every restore admitted in it the whole cycle to land.
        self.prefill_wave_steps = self.pull_cycle_len
        self._last_prefill_step = 0

        process_rank = args.node_rank * args.num_gpus_per_node + local_rank
        self.engine_to_worker_queue = (
            engine_to_worker_queues[local_rank] if args.node_rank == 0 else None
        )
        self.worker_to_engine_queue = (
            worker_to_engine_queues[local_rank] if args.node_rank == 0 else None
        )
        self.enable_overlap_schedule = not args.disable_overlap_schedule
        if self.enable_overlap_schedule:
            self._FLUSH_OVERLAP_SCHEDULE_STATE = True

        if args.node_rank != 0:
            del self.engine_to_worker_queue
            del self.worker_to_engine_queue
            self.socket = build_socket_conn(args.ip_addrs[0], process_rank)
        try:
            self.initialize_instance(process_rank, args)
            self.enable_mtp_spec_decode = (
                # hasattr(self.llm, "mtp")
                getattr(args, "num_spec_tokens", 0) > 0
                and args.disable_overlap_schedule
            )

            self.initialize_profile_state()
            self.running = True
            asyncio.run(self.running_loop())
        except Exception as e:
            print(
                f"[Worker Process {process_rank}]: Exception {e}\n {traceback.print_exc()}"
            )
            exit(0)

    def _recv_from_engine(self):
        if self.args.node_rank == 0:
            return self.engine_to_worker_queue.get()
        else:
            self.socket.recv()
            return pickle.loads(self.socket.recv())

    def _send_to_engine(self, msg: WorkerMsg):
        if self.args.node_rank == 0:
            self.worker_to_engine_queue.put(msg)
        else:
            self.socket.send(pickle.dumps(msg))

    def drain_engine_msgs(self):
        while True:
            if self.args.node_rank == 0:
                msg = self.engine_to_worker_queue.get_nowait()
            else:
                if self.socket.poll(timeout=0, flags=zmq.POLLIN) == 0:
                    msg = None
                else:
                    self.socket.recv()
                    msg = pickle.loads(self.socket.recv())
            if msg is None:
                break
            self.pending_engine_msgs.append(msg)

    def initialize_instance(self, process_rank, args):
        mpu.initialize_distributed(args, process_rank)
        self.llm: LLM = name2model[args.model_name](args)
        self.verbose = args.verbose
        self.sampling_params = args.sampling_params
        self.args = args
        self.llm.load_weight(args.weight_path)
        hf_config = args.hf_config
        self.hidden_size = args.hf_config["hidden_size"]
        kv_page_size = args.page_size
        if "kv_lora_rank" not in hf_config.keys():
            num_pages, kv_cache, kv_cache_tensor = self.llm.initialize_kv_pages(
                self.llm.n_layer_per_device,
                hf_config["hidden_size"] // hf_config["num_attention_heads"]
                if "head_dim" not in hf_config.keys()
                else hf_config["head_dim"],
                vocab_size=hf_config["vocab_size"],
                intermediate_size=hf_config["intermediate_size"],
            )
            mla_cache_tensor = None
        else:
            num_pages, kv_cache, mla_cache_tensor = self.llm.initialize_mla_kv_pages(
                self.llm.n_layer_per_device,
                ckv_dim=hf_config["kv_lora_rank"],
                kpe_dim=hf_config["qk_rope_head_dim"],
                vocab_size=hf_config["vocab_size"],
            )
            kv_cache_tensor = None
        self.block_manager = BlockManager(
            num_pages,
            kv_page_size * args.dcp_deg,
            kv_cache,
            mla_cache_tensor,
            kv_cache_tensor,
            l2_cache_gb=args.l2_cache_gb,
            l2_step_ms=args.l2_step_ms,
        )
        if self.block_manager.enable_l2_cache and torch.distributed.get_rank() == 0:
            # Both terms of the cost model as the bandwidth each implies at this
            # page geometry: a transfer whose pages do not coalesce runs at the
            # descriptor-bound rate, and that is the common case. The link rate is
            # also the NUMA gate -- a mirror on the far node reads ~8% low.
            bm = self.block_manager
            for direction, terms in bm.l2_cost_model.items():
                scattered = bm.l2_page_bytes / (
                    bm.l2_views / terms["desc_rate"] + bm.l2_page_bytes / terms["bw"]
                )
                print(
                    f"[L2 COST MODEL] {direction}: link {terms['bw'] / 1e9:.1f} GB/s, "
                    f"{terms['desc_rate']} desc/s -> {scattered / 1e9:.2f} GB/s when "
                    f"nothing coalesces, page_bytes={bm.l2_page_bytes} "
                    f"views={bm.l2_views} "
                    f"l2_pages={bm.l2_num_pages}",
                    flush=True,
                )
                self._warn_if_page_size_starves_link(direction, terms, scattered)

        self.llm.autotune_flashinfer_cutlass_moe(batch_size=512)

        if not args.enforce_eager:
            self.llm.capture_cudagraph(kv_cache, mla_cache_tensor)
        msg_str = f"[PROCESS {torch.distributed.get_rank()} initialized] KV INIT: {num_pages} pages initialized, current memory allocated {torch.cuda.memory_allocated() // (1024) ** 3} GB"
        msg_str += f"\n{self.llm}"
        info_str = "\n" + "-" * 24 + "Initialization Info" + "-" * 24 + "\n"
        if args.pp_deg > 1:
            from pipeswift.utils import parallel_state

            msg_str_list = [None] * parallel_state.pp_deg
            torch.distributed.all_gather_object(
                msg_str_list, msg_str, group=parallel_state.pp_group
            )
            for _msg_str in msg_str_list:
                info_str += _msg_str + "\n"
        else:
            info_str += msg_str + "\n"
        info_str += "-" * 70
        msg = WorkerMsg(
            instr=MsgType.INFO, src_rank=process_rank, dst_rank=-1, content=info_str
        )
        torch.distributed.barrier()
        if process_rank == 0:
            self.worker_to_engine_queue.put(msg)

    # Fraction of the measured link rate a scattered transfer has to reach for
    # the page geometry to count as saturating it. Below this the link idles
    # while the driver walks descriptors.
    L2_SATURATED_FRAC = 0.8

    def _warn_if_page_size_starves_link(self, direction, terms, scattered):
        """Red warning when page_size is too small to saturate the L2 link.

        Descriptor size is the only thing that sets this transport's rate, and
        for a GQA cache it is ``page_size`` x (KV bytes per token per layer) --
        pages are the unit a descriptor covers, so a small page starves the link
        no matter how many of them a transfer batches. Solving the cost model's
        ``scattered / link >= frac`` for bytes-per-descriptor turns the measured
        rates straight into the page_size that would clear it, so the threshold
        is the only number here that is not measured.
        """
        link = terms["bw"]
        if scattered >= self.L2_SATURATED_FRAC * link:
            return
        frac = self.L2_SATURATED_FRAC
        need_bytes = link * frac / (terms["desc_rate"] * (1 - frac))
        bm = self.block_manager
        have_bytes = bm.l2_page_bytes / bm.l2_views
        want = 1 << math.ceil(math.log2(self.args.page_size * need_bytes / have_bytes))
        print(
            f"\033[1;31m[L2 WARNING] {direction}: page_size={self.args.page_size} "
            f"gives {have_bytes / 1024:.0f} KiB per descriptor and only reaches "
            f"{scattered / 1e9:.1f} of {link / 1e9:.1f} GB/s "
            f"({scattered / link:.0%} of link). "
            f"Need >= {need_bytes / 1024:.0f} KiB per descriptor, i.e. "
            f"--page_size {want}.\033[0m",
            flush=True,
        )

    def initialize_profile_state(self):
        InstanceProcess.throughput_monitor_steps = 0
        InstanceProcess.throughput_monitor_tokens = 0
        InstanceProcess.throughput_monitor_spec_accept_tokens = 0
        InstanceProcess.throughput_monitor_spec_draft_tokens = 0
        InstanceProcess.throughput_monitor_spec_drafts = 0
        InstanceProcess.throughput_monitor_st.record()
        self.profile_initialized = True

    def add_requests(self, requests: List[Request]):
        for _req in requests:
            for _ in range(_req.n_seqs):
                from copy import deepcopy

                _seq = Sequence(request=_req, token_ids=deepcopy(_req.prompt_token_ids))
                _req.seqs.append(_seq)
                self.waiting_queue.append(_seq)
            self.received_reqs.append(_req)

    async def running_loop(self):
        tasks = [
            asyncio.create_task(self.generation_loop()),
            asyncio.create_task(self.check_finished_and_clear_page()),
            asyncio.create_task(self.flush_finished_to_engine()),
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        for t in done:
            t.result()

    def generation_done(self):
        return (
            len(self.waiting_queue) == 0
            and len(self.running_queue) == 0
            and len(self.pending_batch_queue) == 0
        )

    async def poll_engine_requests(self):
        from pipeswift.utils import parallel_state

        if len(self.pending_engine_msgs) == 0:
            return False
        msg: WorkerMsg = self.pending_engine_msgs.popleft()
        if self.verbose:
            print(f"[PROCESS {parallel_state.global_rank}] received Msg: {str(msg)}")
        await self.process_engine_msg(msg)
        return True

    async def send_pull_request(self):
        from pipeswift.utils import parallel_state

        if parallel_state.global_rank != 0:
            return
        self._send_to_engine(
            WorkerMsg(
                instr=MsgType.PULL_REQS,
                src_rank=parallel_state.global_rank,
                dst_rank=-1,
                content=self.step_idx,
            ),
        )

    async def flush_finished_to_engine(self):
        while self.running or len(self.finished_outbox) > 0:
            if len(self.finished_outbox) == 0:
                await asyncio.sleep(0)
                continue
            msg = self.finished_outbox.popleft()
            self._send_to_engine(msg)
            self.maybe_finalize_shutdown()

    async def process_engine_msg(self, msg: WorkerMsg):
        if msg.instr == MsgType.REQS:
            self.add_requests(msg.content)
            if len(msg.content) > 0:
                self.should_process_pending = (
                    True if len(self.pending_batch_queue) > 0 else False
                )
                self._FLUSH_OVERLAP_SCHEDULE_STATE = (
                    True
                    if self.enable_overlap_schedule
                    and len(self.pending_batch_queue) > 0
                    else False
                )
        elif msg.instr == MsgType.EXIT:
            self.stopping = True
            self.pending_engine_msgs.clear()
            self.maybe_finalize_shutdown()
            assert self.generation_done()
            if InstanceProcess.throughput_monitor_spec_drafts > 0:
                avg_accept_len_final = 1 + (
                    InstanceProcess.throughput_monitor_spec_accept_tokens
                    / InstanceProcess.throughput_monitor_spec_drafts
                )
                print(f"[FINAL MTP] Avg Accept Len: {avg_accept_len_final:.4f}")
        else:
            raise RuntimeError("Unsupported Message")

    def append_pending_batch(
        self,
        seqs: List[Sequence],
        token_ids_gpu: torch.Tensor,
        to_cpu=True,
        num_emitted_draft_tokens: torch.Tensor | None = None,
        num_emitted_draft_tokens_gpu: torch.Tensor | None = None,
        draft_token_ids_gpu: torch.Tensor | None = None,
    ):
        assert token_ids_gpu.dtype == torch.int32

        if to_cpu:
            if token_ids_gpu.is_cuda:
                self.pending_batch_queue.append(
                    PendingBatch(
                        seqs=seqs,
                        token_ids_cpu=token_ids_gpu.to("cpu", non_blocking=True),
                        token_ids_gpu=token_ids_gpu,
                        num_emitted_draft_tokens=num_emitted_draft_tokens,
                        num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                        draft_token_ids_gpu=draft_token_ids_gpu,
                    )
                )
                _COPY_EVENT = torch.cuda.Event()
                _COPY_EVENT.record()
                self.copy_event_queue.append(_COPY_EVENT)
            else:
                self.pending_batch_queue.append(
                    PendingBatch(
                        seqs=seqs,
                        token_ids_cpu=token_ids_gpu,
                        token_ids_gpu=token_ids_gpu,
                        num_emitted_draft_tokens=num_emitted_draft_tokens,
                        num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                        draft_token_ids_gpu=draft_token_ids_gpu,
                    )
                )
                # no need to synchronize, just for consistency
                _COPY_EVENT = torch.cpu.Event()
                _COPY_EVENT.record()
                self.copy_event_queue.append(_COPY_EVENT)

        else:
            self.pending_batch_queue.append(
                PendingBatch(
                    seqs=seqs,
                    token_ids_cpu=torch.zeros(
                        token_ids_gpu.shape, dtype=torch.int32, device="cpu"
                    ).pin_memory(),  # NOTE(wsj): must use pin_memory for async GPU->CPU tensor copy
                    token_ids_gpu=token_ids_gpu,
                    num_emitted_draft_tokens=num_emitted_draft_tokens,
                    num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                    draft_token_ids_gpu=draft_token_ids_gpu,
                )
            )

    async def check_finished_and_clear_page(self):
        from pipeswift.utils import parallel_state

        while self.running:
            if len(self.pending_batch_queue) == 0:
                await asyncio.sleep(0)
            else:
                # fwd-clear-fwd-clear one-by-one -> self.should_process_pending
                if not self.can_do_step() or self.should_process_pending:
                    if len(self.pending_batch_queue) == len(self.pp_irecv_ops_deque):
                        assert len(self.pp_irecv_ops_deque) > 0
                        self._wait_pending_batch_recv(self.pp_irecv_ops_deque.popleft())
                        pending_batch = self.pending_batch_queue[0]
                        pending_batch.token_ids_cpu.copy_(
                            pending_batch.token_ids_gpu, non_blocking=True
                        )
                        if (
                            pending_batch.num_emitted_draft_tokens_gpu is not None
                            and pending_batch.num_emitted_draft_tokens is None
                        ):
                            pending_batch.num_emitted_draft_tokens = (
                                pending_batch.num_emitted_draft_tokens_gpu.to(
                                    "cpu", non_blocking=True
                                )
                            )
                        if (
                            pending_batch.draft_token_ids_gpu is not None
                            and parallel_state.pp_prev_rank is None
                        ):
                            self.llm._cache_mtp_draft(
                                seqs=pending_batch.seqs,
                                draft_tokens=pending_batch.draft_token_ids_gpu,
                                draft_probs=None,
                            )
                        _COPY_EVENT = torch.cuda.Event()
                        _COPY_EVENT.record()
                        self.copy_event_queue.append(_COPY_EVENT)
                    self.copy_event_queue.popleft().synchronize()
                    InstanceProcess.throughput_monitor_steps += 1
                    _step = InstanceProcess.throughput_monitor_steps
                    if _step % self.log_interval == 1 and _step > 1:
                        InstanceProcess.throughput_monitor_ed.record()
                        InstanceProcess.throughput_monitor_ed.synchronize()
                        delta_time = (
                            InstanceProcess.throughput_monitor_st.elapsed_time(
                                InstanceProcess.throughput_monitor_ed
                            )
                            / 1000
                        )
                        if parallel_state.global_rank == 0:
                            spec_log = ""
                            if InstanceProcess.throughput_monitor_spec_drafts > 0:
                                avg_accept_len = 1 + (
                                    InstanceProcess.throughput_monitor_spec_accept_tokens
                                    / InstanceProcess.throughput_monitor_spec_drafts
                                )
                                spec_log = f", \tAvg Accept Len: {avg_accept_len:.4f}"
                            print(
                                f"[INFO] "
                                f"Step: [{_step - self.log_interval}, {_step}], \t"
                                f"Time: {delta_time:.2f} s, \t"
                                f"Token Throughput: {InstanceProcess.throughput_monitor_tokens / delta_time:.2f} token/s{spec_log}, \t"
                                f"Running Queue Size: {len(self.running_queue)}, \t"
                                f"Waitinig Queue Size: {len(self.waiting_queue)}, \t"
                                f"L1 Hit Pages: {self.block_manager.hit_cnt}, \t"
                                f"L1 Cache Usage: {(self.block_manager.get_usage() * 100):.2f} %, \t"
                                f"L2 Hit/Load/Store Pages: {self.block_manager.l2_hit_cnt}/{self.block_manager.l2_load_cnt}/{self.block_manager.l2_store_cnt}, \t"
                                f"L2 Cache Usage: {(self.block_manager.get_l2_usage() * 100):.2f} %",
                                flush=True,
                            )
                        InstanceProcess.throughput_monitor_tokens = 0
                        InstanceProcess.throughput_monitor_spec_accept_tokens = 0
                        InstanceProcess.throughput_monitor_spec_draft_tokens = 0
                        InstanceProcess.throughput_monitor_spec_drafts = 0
                        InstanceProcess.throughput_monitor_st.record()

                    pending_batch = self.pending_batch_queue.popleft()
                    seqs = pending_batch.seqs
                    token_ids = pending_batch.token_ids_cpu
                    assert len(seqs) > 0

                    is_spec_batch = pending_batch.num_emitted_draft_tokens is not None
                    emitted_draft_tokens = (
                        pending_batch.num_emitted_draft_tokens
                        if is_spec_batch
                        else None
                    )
                    emitted_draft_tokens_arr = (
                        array("i", emitted_draft_tokens.numpy())
                        if is_spec_batch
                        else None
                    )
                    token_ids_arr = (
                        [array("i", row.numpy()) for row in token_ids]
                        if is_spec_batch
                        else array("i", token_ids.numpy())
                    )
                    batch_emitted_tokens = 0
                    batch_accepted_draft_tokens = 0
                    page_size = self.block_manager.page_size
                    finished_seqs = []

                    for i, seq in enumerate(seqs):
                        seq.num_pending_tokens -= 1
                        already_evicted = (
                            not seq.seq_id in self.block_manager.kv_page_table.keys()
                        )
                        if not seq.finished:
                            if is_spec_batch:
                                emitted_draft_token_cnt = int(
                                    emitted_draft_tokens_arr[i]
                                )
                                accepted_tokens = emitted_draft_token_cnt + 1
                                for token_pos in range(accepted_tokens):
                                    if (
                                        len(seq.token_ids) % page_size == 0
                                        and not already_evicted
                                    ):  # NOTE(wsj): may be a sequence is evicted, skip update page table status
                                        self.block_manager.update_page_status_and_merge_possible(
                                            seq
                                        )
                                    _token_id = int(token_ids_arr[i][token_pos])
                                    seq.token_ids.append(_token_id)
                                    batch_emitted_tokens += 1
                                    if token_pos < emitted_draft_token_cnt:
                                        batch_accepted_draft_tokens += 1
                                    if seq.is_finished():
                                        seq.finish_reason = seq.get_finish_reason()
                                        seq.finished = True
                                        finished_seqs.append(seq)
                                        break
                                # Non-spec decode keeps the original offset update timing.
                                if not already_evicted:
                                    seq.offset = len(seq.token_ids) - 1
                            else:
                                batch_emitted_tokens += 1
                                if (
                                    len(seq.token_ids) % page_size == 0
                                    and not already_evicted
                                ):
                                    self.block_manager.update_page_status_and_merge_possible(
                                        seq
                                    )
                                token_id = int(token_ids_arr[i])
                                seq.token_ids.append(token_id)
                                if seq.is_finished():
                                    seq.finish_reason = seq.get_finish_reason()
                                    seq.finished = True
                                    finished_seqs.append(seq)

                    InstanceProcess.throughput_monitor_tokens += batch_emitted_tokens
                    if is_spec_batch:
                        InstanceProcess.throughput_monitor_spec_accept_tokens += (
                            batch_accepted_draft_tokens
                        )
                        InstanceProcess.throughput_monitor_spec_draft_tokens += (
                            len(seqs) * self.llm.num_spec_tokens
                        )
                        InstanceProcess.throughput_monitor_spec_drafts += len(seqs)

                    if finished_seqs:
                        dedup_finished_seq_ids = set()
                        unique_finished_seqs = []
                        for seq in finished_seqs:
                            if seq.seq_id not in dedup_finished_seq_ids:
                                dedup_finished_seq_ids.add(seq.seq_id)
                                unique_finished_seqs.append(seq)
                        finished_seqs = unique_finished_seqs

                    for _seq in finished_seqs:
                        self.block_manager.deallocate(_seq)
                        if hasattr(self.llm, "_clear_mtp_draft"):
                            self.llm._clear_mtp_draft([_seq])

                    if len(finished_seqs) > 0:
                        i = 0
                        while i < len(self.running_queue):
                            if self.running_queue[i].finished:
                                del self.running_queue[i]
                            else:
                                i += 1
                        if parallel_state.global_rank == 0:
                            finished_requests = []
                            seen_req_ids = set()
                            for seq in finished_seqs:
                                req = seq.request
                                if req.req_id in seen_req_ids:
                                    continue
                                if all(_seq.finished for _seq in req.seqs):
                                    seen_req_ids.add(req.req_id)
                                    finished_requests.extend(
                                        self.build_finished_seq_payload(req)
                                    )
                            if finished_requests:
                                self.finished_outbox.append(
                                    WorkerMsg(
                                        instr=MsgType.FINISHED_SEQS,
                                        src_rank=parallel_state.global_rank,
                                        dst_rank=-1,
                                        content=finished_requests,
                                    )
                                )

                    if self.enable_overlap_schedule:
                        # current batch's status changed, i.e., some seqs finished
                        if not self._FLUSH_OVERLAP_SCHEDULE_STATE:
                            self._FLUSH_OVERLAP_SCHEDULE_STATE = len(finished_seqs) > 0
                        self.should_process_pending = (
                            self._FLUSH_OVERLAP_SCHEDULE_STATE
                            and not (len(self.pending_batch_queue) == 0)
                        )
                        if len(self.pending_batch_queue) == 0:
                            self._FLUSH_OVERLAP_SCHEDULE_STATE = False
                    else:
                        # self.should_process_pending = False  # -> generate
                        # FLUSH Pipeline if any sequence is finished
                        self.should_process_pending = len(finished_seqs) > 0 and not (
                            len(self.pending_batch_queue) == 0
                        )
                        if len(finished_seqs) > 0:
                            self._zigzag_buckets.clear()
                else:
                    await asyncio.sleep(0)

    def issue_l2_loads(self):
        """Stage the L2 restore of as many waiting sequences as the window hides.

        Batched, because restores used to go out one sequence per decode window
        (schedule_prefill inspects only the FIFO head, and the head's in-flight
        load stopped it looking further) while the running queue emptied into
        that queue of L2 hits. Runs every step on every rank off replicated
        state, because the page table it commits has to come out identical
        everywhere.
        """
        bm = self.block_manager
        if not bm.enable_l2_cache:
            return
        # Sequences already holding pages, whose decode headroom the prefetch
        # must not eat into. A preempted sequence keeps its pages pinned until
        # its in-flight load retires, so l2_load_pending counts as holding too.
        candidates = []
        committed = len(self.running_queue)
        for seq in self.waiting_queue:
            if seq.l2_load_pending or seq.seq_id in bm.kv_page_table:
                committed += 1
            else:
                candidates.append(seq)

        ps = bm.page_size
        # Prompts of FIFO-earlier candidates that were not prefetched. Prefetching
        # past one of those without holding its pages is what would starve it:
        # schedule_prefill still has to allocate it first.
        ahead_pages = 0
        for seq in candidates:
            # Stop once the backlog would outlast the window. The FIFO head is
            # not exempt: it holds no pages until schedule_prefill allocates it
            # directly, so a bounded prefetch cannot starve it.
            if bm.l2_load_horizon_steps() >= self.steps_until_prefill_wave():
                break
            # Charge the whole prompt: allocate_prefill takes its full page
            # table, and an L1 prefix hit only makes this an overestimate.
            prompt_pages = (len(seq.token_ids) + ps - 1) // ps
            committed += 1  # admitted here or by schedule_prefill; same headroom
            if bm.free_pages() < (
                ahead_pages + prompt_pages + bm.decode_reserve_pages(committed)
            ):
                # FIFO, so stop rather than skip.
                break
            if bm.probe_l2_prefetch(seq):
                bm.allocate_prefill(seq)
            else:
                # Nothing to restore, so nothing to prefetch. It takes no pages
                # here, but it is ahead in the queue, so leave room for it.
                ahead_pages += prompt_pages
        bm.flush_l2_loads()

    def steps_until_prefill_wave(self):
        """Decode steps still owed before the next prefill wave may open.

        The deadline a restore staged now has to land against, and so the only
        depth worth prefetching to: the queue drains serially, so staging past it
        makes nothing land sooner while pinning the pages the transfer reads into
        through the wave that could have used them (unbounded, the pool ran to 94%
        held with nothing decodable). Shrinking within the window is the point --
        late in one, nothing new is stageable, so those pages stay free for the
        wave instead. Replicated inputs, so every rank prefetches equally deep.
        """
        return max(
            0, self.prefill_wave_steps - (self.step_idx - self._last_prefill_step)
        )

    def prefill_wave_open(self):
        """Whether a prefill wave may start on this step.

        What turns a stream of admissions into a P/D wave: a restore needs time
        to land, and prefill firing the moment its sequence is ready leaves the
        copy nowhere to hide, so the scheduler either stalls in front of it or
        re-prefills tokens it was about to be handed. A fixed run of decode steps
        gives every restore admitted in the window the whole window.

        A timer, not a "wait while a restore is in flight" condition: restores are
        staged every step, so that condition never clears and prefill fires on
        whichever step finds the queue momentarily empty (measured far slower).
        The one escape is nothing decodable at all. Replicated state only, so
        every rank opens the same wave on the same step.
        """
        if self.prefill_wave_steps <= 0:
            return True
        if self.step_idx - self._last_prefill_step >= self.prefill_wave_steps:
            return True
        return not self.can_schedule_decode_non_overlap()

    def should_schedule_prefill(self):
        from pipeswift.utils import parallel_state

        # Drains transfers staged by the final decode step of a batch, and
        # restores staged by allocate_prefill inside this same prefill wave,
        # which would otherwise sit unsent until the next forward.
        self.block_manager.flush_l2_stores()
        self.block_manager.flush_l2_loads()
        if len(self.waiting_queue) == 0:
            return False
        # The first sequence that could actually be forwarded. One whose L2 load
        # is in flight cannot be, but it is not a barrier either -- it already
        # holds every page it will read -- so the sequences behind it are what
        # admission should be judged on. See schedule_prefill.
        head_seq = next(
            (seq for seq in self.waiting_queue if not seq.l2_load_pending), None
        )
        # Everything queued is mid-restore: decode instead of spinning here until
        # poll_l2_events retires a transfer.
        if head_seq is None:
            return False
        if len(self.running_queue) == 0:
            return True

        if len(self.running_queue) >= self.max_batchsize * parallel_state.pp_deg:
            return False

        # Pages the head already holds -- a chunked prefill in progress, or an L2
        # load that landed -- are not needed a second time, and a head that needs
        # none of the pool can always run. Same reserve the L2 prefetch applies,
        # so a restored sequence is not admitted into a pool that then has to
        # preempt to keep decoding.
        required_pages = (
            self.block_manager.decode_reserve_pages(len(self.running_queue))
            + len(head_seq.token_ids) // self.block_manager.page_size
            - len(self.block_manager.kv_page_table.get(head_seq.seq_id, ()))
        )
        return self.block_manager.free_pages() >= required_pages

    async def process_a_pending_batch(self):
        if len(self.pending_batch_queue) == 0:
            # check_finished_and_clear_page only clears the flag while its queue
            # is non-empty, so raising it here would park generation_loop for good.
            await asyncio.sleep(0)
            return
        self.should_process_pending = True
        while self.should_process_pending:
            await asyncio.sleep(0)  # clear pending batch

    async def generation_loop(self):
        from pipeswift.utils import parallel_state

        while self.running:
            # Retire due L2 transfers exactly once per step, on every rank: the
            # deadline is priced in steps and a rank that skipped one would fork
            # its page table from the others.
            self.block_manager.poll_l2_events(self.step_idx)
            # Batch the restores of everything waiting, before the prefill wave
            # below asks whether the head is still pending.
            self.issue_l2_loads()

            phase = self.step_idx % self.pull_cycle_len
            # The request goes out half a cycle before the blocking receive, so
            # the engine has that many steps to answer. Both phases must key off
            # the same interval that sizes the cycle, or the send lands on a phase
            # the modulo cannot reach and the receive below blocks forever.
            if not self.stopping and phase == self.control_interval - 1:
                await self.send_pull_request()

            if not self.stopping and phase == self.pull_cycle_len - 1:
                while len(self.pending_engine_msgs) == 0:
                    self.drain_engine_msgs()
                    await asyncio.sleep(0)
                await self.poll_engine_requests()
                if not self.running:
                    break
            if self.should_process_pending:
                await asyncio.sleep(0)
            else:
                if self.can_do_step():
                    # should_schedule_prefill first: it carries the per-step L2
                    # flush, which a gated step still owes.
                    if (
                        self.should_schedule_prefill() and self.prefill_wave_open()
                    ):  # running_queue not ready, do prefill until can not do
                        while len(self.pending_batch_queue) > 0:
                            await self.process_a_pending_batch()
                        self._last_prefill_step = self.step_idx
                        while self.should_schedule_prefill():
                            maybe_prefill_batch, have_extend_in_batch = (
                                self.schedule_prefill()
                            )
                            if not maybe_prefill_batch:
                                break
                            self.carryout_forward(
                                maybe_prefill_batch,
                                is_decode=False,
                                is_overlap_schedule=False,
                                have_extend_in_batch=have_extend_in_batch,
                            )
                        # Prefill burst done: flush deferred result sends so the
                        # recvs drained just below have matching posted sends.
                        self._flush_deferred_pp_sends()
                        # Stage the next wave's restores before awaiting this
                        # wave's forwards, so they run under prefill compute too:
                        # prefill is ~19% of the wall clock and loads were only
                        # ever staged at the top of a loop iteration, while a
                        # burst is many forwards inside one iteration. After the
                        # burst on purpose -- it has finished claiming pages, so
                        # this cannot shorten the wave it prefetches for.
                        self.issue_l2_loads()
                        while len(self.pending_batch_queue) > 0:
                            await self.process_a_pending_batch()

                    elif len(self.pending_batch_queue) == 0:
                        # launch pipeline
                        while self.can_schedule_decode_non_overlap():
                            seqs = self.schedule_decode()
                            self.carryout_forward(
                                seqs, is_decode=True, is_overlap_schedule=False
                            )
                        assert len(self.pending_batch_queue) <= parallel_state.pp_deg, (
                            "Please check the running queue, detect queue size exceeds max_batchsize * pp_deg!"
                        )
                        if not self.enable_overlap_schedule:
                            await self.process_a_pending_batch()

                    else:  # pipeline launched, do decode
                        if (
                            self.can_schedule_decode_non_overlap()
                            and not self.enable_overlap_schedule
                        ):
                            seqs = self.schedule_decode()
                            self.carryout_forward(
                                seqs, is_decode=True, is_overlap_schedule=False
                            )
                            await self.process_a_pending_batch()
                        elif self.enable_overlap_schedule:
                            if self.block_manager.can_allocate_overlap_schedule(
                                self.pending_batch_queue[0].seqs
                            ):
                                self.carryout_forward(
                                    self.pending_batch_queue[0].seqs,
                                    True,
                                    is_overlap_schedule=True,
                                )
                                await self.process_a_pending_batch()
                            else:
                                # stall overlap schedules
                                self.should_process_pending = True  # -> check_finished
                                self._FLUSH_OVERLAP_SCHEDULE_STATE = True
                else:
                    # An idle engine still owes the flush that otherwise rides on
                    # should_schedule_prefill: a server going quiet after a burst
                    # would leave the last pages staged -- unwritten, and pinning
                    # the GPU pages they read -- until the next request arrives.
                    self.block_manager.flush_l2_stores()
                    await asyncio.sleep(0)
            self.maybe_finalize_shutdown()
            self.step_idx += 1

    def maybe_finalize_shutdown(self):
        if not self.stopping or self._shutdown_complete:
            return
        if not self.generation_done():
            return
        if len(self.finished_outbox) > 0:
            return
        self._shutdown_complete = True
        self.running = False
        if hasattr(self.llm, "graphs"):
            del self.llm.graphs
        if hasattr(self.llm, "extend_graphs"):
            del self.llm.extend_graphs
        if hasattr(self.llm, "mtp_decode_graphs"):
            del self.llm.mtp_decode_graphs
        torch.distributed.destroy_process_group()
        if self.verbose:
            from pipeswift.utils import parallel_state

            print(f"[Worker {parallel_state.global_rank}] exiting.")

    def build_finished_seq_payload(self, request: Request):
        payload = []
        for seq in request.seqs:
            payload.append(
                {
                    "request_id": request.req_id,
                    "prompt": request.prompt,
                    "seq_id": seq.seq_id,
                    # Use a compact typed array to reduce pickle and /dev/shm overhead.
                    "response_token_ids": array(
                        "I",
                        seq.token_ids[seq.num_prompt_tokens :],
                    ),
                    "finish_reason": seq.finish_reason,
                }
            )
        return payload

    def can_do_step(self):
        if len(self.waiting_queue) > 0:
            return True
        for seq in self.running_queue:
            if seq.num_pending_tokens == 0 and not seq.finished:
                return True
        if not self.enable_overlap_schedule:
            return False
        else:
            return len(self.pending_batch_queue) > 0

    def can_schedule_decode_non_overlap(self):
        for seq in self.running_queue:
            if seq.num_pending_tokens == 0 and not seq.finished:
                return True
        return False

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        if hasattr(self.llm, "_clear_mtp_draft"):
            self.llm._clear_mtp_draft([seq])
        self.waiting_queue.append(seq)
        seq.offset = 0

    def schedule_prefill(self):
        from pipeswift.utils import parallel_state

        self._zigzag_buckets.clear()
        scheduled_seqs, num_seqs, num_batched_tokens = [], 0, 0
        have_extend_in_batch = False
        bm = self.block_manager
        # prefill. Indexed rather than popped off the front, because a sequence
        # whose restore is in flight has to be stepped over.
        i = 0
        while (
            i < len(self.waiting_queue)
            and num_seqs < self.max_batchsize
            and len(self.running_queue) < self.max_batchsize * parallel_state.pp_deg
        ):
            seq = self.waiting_queue[i]
            # Not forwardable until its restore lands, but allocate_prefill
            # already gave it every page that restore writes into, so stepping
            # over costs the pool nothing. Stopping here instead serialised
            # admission to one sequence per restore: measured with 89 waiting and
            # one load in flight, the running batch drained to 24 of 64.
            if seq.l2_load_pending:
                i += 1
                continue
            if seq.seq_id not in bm.kv_page_table.keys():
                # Over the restore budget: allocating here would stage into a
                # backlog that already cannot land inside the window (measured at
                # 72 steps against a 16-step window, pool 99% held). It holds no
                # pages yet, so stepping over costs nothing and it is picked up
                # when the queue drains. Gated on something being in flight, so
                # the head always makes progress.
                if (
                    bm.pending_l2_loads
                    and bm.l2_load_horizon_steps() >= self.steps_until_prefill_wave()
                    and bm.probe_l2_prefetch(seq)
                ):
                    i += 1
                    continue
                if not bm.can_allocate(seq):
                    break
                else:
                    # for chunked prefill, only allocate all pages at exec of first chunk
                    bm.allocate_prefill(seq)
                    if seq.l2_load_pending:
                        i += 1
                        continue

            kv_page_size = bm.page_size
            assert self.max_batched_tokens >= kv_page_size, (
                f"Please specify a larger --max_batched_tokens no less than {kv_page_size}"
            )
            left_space = self.max_batched_tokens - num_batched_tokens
            remaining = len(seq.token_ids) - seq.offset
            chunk_limit = left_space
            if chunk_limit < remaining:
                if chunk_limit < kv_page_size:
                    break
                chunk_len = chunk_limit // kv_page_size * kv_page_size
                assert seq.offset % (kv_page_size) == 0, (
                    "chunked prefill conducted in page granularity!"
                )
                if have_extend_in_batch:
                    _seq = Sequence(seq.request)._deepcopy_from(seq)
                    _seq.token_ids = _seq.token_ids[: _seq.offset + chunk_len]
                else:
                    _seq = Sequence(seq.request)
                    _seq.seq_id, _seq.offset = seq.seq_id, seq.offset
                    _seq.token_ids = seq.token_ids[: _seq.offset + chunk_len]
                _seq.next_token = seq.token_ids[_seq.offset]
                _seq.is_temporal_for_chunked_prefill = True
                seq.offset += chunk_len
                scheduled_seqs.append(_seq)
                if parallel_state.global_rank == 0:
                    print(
                        f"[INFO] Chunk prefill scheduled, slices [{_seq.offset}, {len(_seq.token_ids)}] of sequence {seq.seq_id}"
                    )
                break  # each batch only contains one chunk not ready for decode
            else:
                num_seqs += 1
                num_batched_tokens += len(seq.token_ids) - seq.offset
                scheduled_seqs.append(seq)
                del self.waiting_queue[i]  # next candidate shifts into i
                self.running_queue.add(seq)
                have_extend_in_batch = True

        if scheduled_seqs:
            return scheduled_seqs, have_extend_in_batch
        return [], False

    def _collect_decode_seqs(self, max_seqs):
        from pipeswift.utils import parallel_state

        scheduled_seqs, num_seqs = [], 0
        reserve_tokens = (
            3 * self.llm.num_spec_tokens + 2 if self.enable_mtp_spec_decode else 1
        )
        i = 0
        while i < len(self.running_queue) and num_seqs < max_seqs:
            seq = self.running_queue[i]
            while not self.block_manager.can_allocate(
                seq, reserve_tokens=reserve_tokens
            ):
                # Pages held by an in-flight L2 store come back by draining it;
                # preempting for them frees nothing.
                if self.block_manager.reclaim_l2_store_pages():
                    continue
                if len(self.running_queue) >= i + 2:
                    _seq = self.running_queue.pop()
                    if parallel_state.global_rank == 0:
                        print(f"[INFO] KV cache poll is full, evict {str(_seq)}")
                    self.preempt(_seq)  # FIFO
                else:
                    if parallel_state.global_rank == 0:
                        print(f"[INFO] KV cache poll is full, evict {str(seq)}")
                    self.preempt(seq)
                    del self.running_queue[i]
                    break
            else:
                num_seqs += 1
                self.block_manager.may_allocate_decode(
                    seq, reserve_tokens=reserve_tokens
                )
                scheduled_seqs.append(seq)
                i += 1
        assert scheduled_seqs, "Not enough pages to support extremely long sequences!"
        return scheduled_seqs

    def schedule_decode(self):
        from pipeswift.utils import parallel_state

        if self._zigzag_buckets:
            return self._zigzag_buckets.pop(0)

        use_zigzag = (
            getattr(self.args, "use_zigzag_schedule_decode", False)
            and parallel_state.pp_deg > 1
        )
        max_seqs = self.max_batchsize * parallel_state.pp_deg
        all_seqs = self._collect_decode_seqs(max_seqs)

        if parallel_state.pp_deg <= 1:
            return all_seqs

        num_buckets = min(parallel_state.pp_deg, len(all_seqs))
        buckets: list[list] = [[] for _ in range(num_buckets)]
        if use_zigzag:
            order = list(range(num_buckets)) + list(range(num_buckets - 1, -1, -1))
            for idx, seq in enumerate(all_seqs):
                buckets[order[idx % len(order)]].append(seq)
        else:
            chunk_size = (len(all_seqs) + num_buckets - 1) // num_buckets
            for idx in range(num_buckets):
                buckets[idx] = all_seqs[idx * chunk_size : (idx + 1) * chunk_size]
        self._zigzag_buckets = [b for b in buckets if b]
        return self._zigzag_buckets.pop(0)

    def _send_pending_batch_to_pp_stages(
        self,
        output_ids: torch.Tensor,
        num_emitted_draft_tokens_gpu: torch.Tensor | None = None,
        draft_token_ids_gpu: torch.Tensor | None = None,
    ):
        from pipeswift.utils import parallel_state

        send_ops = []
        for pp_rank in range(parallel_state.pp_deg - 1):
            assert output_ids.dtype == torch.int32
            send_ops.append(
                dist.P2POp(
                    dist.isend,
                    output_ids,
                    group=parallel_state.pp_group,
                    group_peer=pp_rank,
                )
            )
            if num_emitted_draft_tokens_gpu is not None:
                assert num_emitted_draft_tokens_gpu.dtype == torch.int32
                send_ops.append(
                    dist.P2POp(
                        dist.isend,
                        num_emitted_draft_tokens_gpu,
                        group=parallel_state.pp_group,
                        group_peer=pp_rank,
                    )
                )
            if draft_token_ids_gpu is not None and pp_rank == 0:
                assert draft_token_ids_gpu.dtype == torch.int32
                send_ops.append(
                    dist.P2POp(
                        dist.isend,
                        draft_token_ids_gpu,
                        group=parallel_state.pp_group,
                        group_peer=0,
                    )
                )

        dist.batch_isend_irecv(send_ops)

    def _flush_deferred_pp_sends(self):
        """Last stage: flush the result broadcasts accumulated during a prefill
        burst, in order (no-op on non-last stages)."""
        if not self._deferred_pp_sends:
            return
        for output_ids, num_emitted_gpu, draft_gpu in self._deferred_pp_sends:
            self._send_pending_batch_to_pp_stages(
                output_ids,
                num_emitted_draft_tokens_gpu=num_emitted_gpu,
                draft_token_ids_gpu=draft_gpu,
            )
        self._deferred_pp_sends = []

    def _recv_pending_batch_from_pp_last_stage(
        self,
        seqs: List[Sequence],
        is_decode: bool,
    ):
        from pipeswift.utils import parallel_state

        if self.enable_mtp_spec_decode and is_decode:
            output_ids = torch.empty(
                (len(seqs), self.llm.num_spec_tokens + 1),
                device="cuda",
                dtype=torch.int32,
            )
        else:
            output_ids = torch.empty(len(seqs), device="cuda", dtype=torch.int32)

        num_emitted_draft_tokens_gpu = None
        draft_token_ids_gpu = None
        if self.enable_mtp_spec_decode and is_decode:
            num_emitted_draft_tokens_gpu = torch.empty(
                (len(seqs),), dtype=torch.int32, device="cuda"
            )
        if self.enable_mtp_spec_decode and parallel_state.pp_prev_rank is None:
            draft_token_ids_gpu = torch.empty(
                (len(seqs), self.llm.num_spec_tokens + 1),
                dtype=torch.int32,
                device="cuda",
            )
        self.pp_irecv_ops_deque.append(
            PendingBatchRecvPlan(
                seqs=seqs,
                token_ids_gpu=output_ids,
                num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                draft_token_ids_gpu=draft_token_ids_gpu,
            )
        )
        self.append_pending_batch(
            seqs,
            output_ids,
            to_cpu=False,
            num_emitted_draft_tokens=None,
            num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
            draft_token_ids_gpu=draft_token_ids_gpu,
        )

    def _wait_pending_batch_recv(self, recv_plan: PendingBatchRecvPlan):
        from pipeswift.utils import parallel_state

        recv_ops = [
            dist.P2POp(
                dist.irecv,
                recv_plan.token_ids_gpu,
                group=parallel_state.pp_group,
                group_peer=parallel_state.pp_deg - 1,
            )
        ]
        if recv_plan.num_emitted_draft_tokens_gpu is not None:
            recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_plan.num_emitted_draft_tokens_gpu,
                    group=parallel_state.pp_group,
                    group_peer=parallel_state.pp_deg - 1,
                )
            )
        if recv_plan.draft_token_ids_gpu is not None:
            recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_plan.draft_token_ids_gpu,
                    group=parallel_state.pp_group,
                    group_peer=parallel_state.pp_deg - 1,
                )
            )
        reqs = dist.batch_isend_irecv(recv_ops)
        for req in reqs:
            req.wait()

    def carryout_forward(
        self,
        seqs: List[Sequence],
        is_decode,
        is_overlap_schedule=False,
        have_extend_in_batch=True,
    ):
        from pipeswift.utils import parallel_state

        try:
            with torch.no_grad():
                assert len(seqs) != 0
                for seq in seqs:
                    seq.num_pending_tokens += 1
                num_emitted_draft_tokens = None
                num_emitted_draft_tokens_gpu = None
                draft_token_ids_gpu = None
                if not is_overlap_schedule:
                    if self.enable_mtp_spec_decode:
                        (
                            maybe_output_ids,
                            num_emitted_draft_tokens,
                            num_emitted_draft_tokens_gpu,
                            draft_token_ids_gpu,
                        ) = self._run_model_mtp(seqs, is_decode, have_extend_in_batch)
                        if is_decode:
                            pass
                        else:
                            for seq in seqs:
                                self.block_manager.schedule_l2_stores_for_confirmed_seq(
                                    seq
                                )
                            for seq in seqs:  # update offset
                                seq.offset = len(seq.token_ids)
                    else:
                        maybe_output_ids = self._run_model(
                            seqs, is_decode, have_extend_in_batch
                        )
                        if not is_decode:
                            for seq in seqs:
                                self.block_manager.schedule_l2_stores_for_confirmed_seq(
                                    seq
                                )
                        for seq in seqs:  # update offset
                            seq.offset = len(seq.token_ids)
                else:
                    assert is_decode
                    maybe_output_ids = self._run_model_overlap_schedule(seqs)
                    for seq in seqs:
                        seq.offset += 1

                if have_extend_in_batch:
                    if parallel_state.pp_next_rank is None:
                        assert maybe_output_ids is not None
                        self.append_pending_batch(
                            seqs,
                            maybe_output_ids,
                            num_emitted_draft_tokens=num_emitted_draft_tokens,
                            num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                            draft_token_ids_gpu=draft_token_ids_gpu,
                        )
                        if parallel_state.pp_deg > 1:
                            if is_decode:
                                self._send_pending_batch_to_pp_stages(
                                    maybe_output_ids,
                                    num_emitted_draft_tokens_gpu=num_emitted_draft_tokens_gpu,
                                    draft_token_ids_gpu=draft_token_ids_gpu,
                                )
                            else:
                                # PREFILL: defer to one post-burst flush. Earlier
                                # stages defer their matching recv to after their
                                # burst, so sending per chunk here leaves sends
                                # without a receiver and deadlocks cross-node.
                                self._deferred_pp_sends.append(
                                    (
                                        maybe_output_ids,
                                        num_emitted_draft_tokens_gpu,
                                        draft_token_ids_gpu,
                                    )
                                )
                    else:
                        self._recv_pending_batch_from_pp_last_stage(
                            seqs,
                            is_decode=is_decode,
                        )
                # Decode page completions staged during this step go out as a
                # single batched transfer.
                self.block_manager.flush_l2_stores()

        except Exception as e:
            print(traceback.print_exc())

    def _run_model(self, seqs, is_decode=False, have_extend_in_batch=True):
        from pipeswift.utils import parallel_state

        kv_cache_info: KVCacheInfo = prepare_kv_info(
            seqs,
            self.block_manager.kv_page_table,
            self.block_manager.page_size,
            self.block_manager.kv_cache,
            self.block_manager.mla_cache_tensor,
            dcp_interleave_size=self.args.dcp_interleave_size,
            is_decode=is_decode,
            attention_backend=self.args.attention_backend,
        )
        if len(seqs) > 0:
            token_ids = array("i")
            position_ids = array("i")
            for seq in seqs:
                token_ids.extend(seq.token_ids[seq.offset :])
                position_ids.extend(range(seq.offset, len(seq.token_ids)))
            token_ids, position_ids = coalesced_to_cuda(
                arr_list=[token_ids, position_ids]
            )
        else:
            position_ids = torch.IntTensor([]).cuda()
            token_ids = torch.zeros((0,), device="cuda", dtype=torch.int32)

        if parallel_state.pp_prev_rank != None:
            assert hasattr(self, "hidden_size"), (
                "hidden_size is required to determine the input shape"
            )
            hidden_states = torch.empty(
                (len(token_ids), self.hidden_size), dtype=self.args.dtype, device="cuda"
            )
            dist.recv(hidden_states, src=parallel_state.pp_prev_rank)
            input_tensor = hidden_states
        else:
            input_tensor = token_ids

        maybe_output_ids, _ = self.llm.forward(
            input_tensor,
            position_ids,
            kv_cache_info,
            is_decode=is_decode,
            have_extend_in_batch=have_extend_in_batch,
        )
        return maybe_output_ids

    def _run_model_overlap_schedule(self, seqs):
        from pipeswift.utils import parallel_state

        token_ids_gpu: torch.Tensor = self.pending_batch_queue[0].token_ids_gpu
        assert token_ids_gpu.is_cuda
        position_ids = torch.from_numpy(
            np.array(list(seq.offset for seq in seqs), dtype=np.int32)
        )
        position_ids = position_ids.to("cuda", non_blocking=True)
        for seq in seqs:
            seq.token_ids.append(0)  # append fake token
            self.block_manager.may_allocate_decode(seq)
        kv_cache_info: KVCacheInfo = prepare_kv_info(
            seqs,
            self.block_manager.kv_page_table,
            self.block_manager.page_size,
            self.block_manager.kv_cache,
            self.block_manager.mla_cache_tensor,
            dcp_interleave_size=self.args.dcp_interleave_size,
            is_decode=True,
            attention_backend=self.args.attention_backend,
        )
        for seq in seqs:
            seq.token_ids.pop()

        # input
        if parallel_state.pp_prev_rank is not None:
            assert hasattr(self, "hidden_size"), (
                "hidden_size is required to determine the input shape"
            )
            hidden_states = torch.empty(
                (token_ids_gpu.shape[0], self.hidden_size),
                dtype=self.args.dtype,
                device="cuda",
            )
            dist.recv(hidden_states, src=parallel_state.pp_prev_rank)
            input_tensor = hidden_states
        else:
            input_tensor = token_ids_gpu
        self.llm._current_seqs = seqs

        if parallel_state.pp_next_rank is not None:
            if len(self.pp_irecv_ops_deque) > 0:
                self._wait_pending_batch_recv(self.pp_irecv_ops_deque.popleft())
                pending_batch = self.pending_batch_queue[0]
                pending_batch.token_ids_cpu.copy_(token_ids_gpu, non_blocking=True)
                if (
                    pending_batch.num_emitted_draft_tokens_gpu is not None
                    and pending_batch.num_emitted_draft_tokens is None
                ):
                    pending_batch.num_emitted_draft_tokens = (
                        pending_batch.num_emitted_draft_tokens_gpu.to(
                            "cpu", non_blocking=True
                        )
                    )
                if (
                    pending_batch.draft_token_ids_gpu is not None
                    and parallel_state.pp_prev_rank is None
                ):
                    self.llm._cache_mtp_draft(
                        seqs=pending_batch.seqs,
                        draft_tokens=pending_batch.draft_token_ids_gpu,
                        draft_probs=None,
                    )
                _COPY_EVENT = torch.cuda.Event()
                _COPY_EVENT.record()
                self.copy_event_queue.append(_COPY_EVENT)

        maybe_output_ids, _ = self.llm.forward(
            input_tensor,
            position_ids,
            kv_cache_info,
            is_decode=True,
        )
        if hasattr(self.llm, "_current_seqs"):
            del self.llm._current_seqs
        return maybe_output_ids

    def _run_model_mtp(
        self, seqs: List[Sequence], is_decode=False, have_extend_in_batch=True
    ):
        from pipeswift.utils import parallel_state

        draft_token_ids_gpu = None
        if not is_decode:
            kv_cache_info: KVCacheInfo = prepare_kv_info(
                seqs,
                self.block_manager.kv_page_table,
                self.block_manager.page_size,
                self.block_manager.kv_cache,
                self.block_manager.mla_cache_tensor,
                dcp_interleave_size=self.args.dcp_interleave_size,
                is_decode=False,
                attention_backend=self.args.attention_backend,
            )

            if len(seqs) > 0:
                token_ids = array("i")
                position_ids = array("i")
                for seq in seqs:
                    token_ids.extend(seq.token_ids[seq.offset :])
                    position_ids.extend(range(seq.offset, len(seq.token_ids)))
                token_ids, position_ids = coalesced_to_cuda(
                    arr_list=[token_ids, position_ids]
                )
            else:
                position_ids = torch.IntTensor([]).cuda()
                token_ids = torch.zeros((0,), device="cuda", dtype=torch.int32)

            if parallel_state.pp_prev_rank != None:
                assert hasattr(self, "hidden_size"), (
                    "hidden_size is required to determine the input shape"
                )
                hidden_states = torch.empty(
                    (len(token_ids), self.hidden_size),
                    dtype=self.args.dtype,
                    device="cuda",
                )
                dist.recv(hidden_states, src=parallel_state.pp_prev_rank)
                input_tensor = hidden_states
            else:
                input_tensor = token_ids

            maybe_output_ids, hidden_states = self.llm.forward(
                input_tensor,
                position_ids,
                kv_cache_info,
                is_decode=False,
                have_extend_in_batch=have_extend_in_batch,
            )

            # SPMD: the num_spec_tokens decode-reserve pages the MTP draft writes
            # must be committed to the page table on EVERY pp stage at this same
            # step, not only on the last stage that runs the draft. The page pool
            # (free_pages, per-seq page tables, and the L1/L2 caches keyed on the
            # page indices it hands out) is replicated scheduling state; allocating
            # on one stage only forks it. The free-page count re-converges on the
            # next decode, but during the fork a tight-pool stage preempts /
            # reclaims / admits differently from the others until the PP batch
            # compositions disagree and send/recv deadlocks. Only forward_mtp_draft
            # (the compute) is last-stage work; the allocation is scheduling.
            if have_extend_in_batch:
                for seq in seqs:
                    self.block_manager.may_allocate_decode(
                        seq, reserve_tokens=self.args.num_spec_tokens
                    )
            if parallel_state.pp_next_rank is None:
                cu_seqlens_q_cpu = kv_cache_info.cu_seqlens_q_cpu
                mtp_aux_hiddens = []
                mtp_input_ids = []
                for i, seq in enumerate(seqs):
                    seq_start = cu_seqlens_q_cpu[i]
                    seq_end = cu_seqlens_q_cpu[i + 1]

                    aux = hidden_states[seq_start:seq_end]
                    if seq.next_token is None:
                        assert maybe_output_ids is not None
                        next_token = maybe_output_ids[i : i + 1]
                    else:
                        # chunked prefill
                        next_token = torch.tensor(
                            [seq.next_token], device="cuda", dtype=torch.int32
                        )
                    mtp_aux_hiddens.append(aux)
                    mtp_input_ids.extend(
                        [token_ids[seq_start + 1 : seq_end], next_token]
                    )

                mtp_input_ids = torch.cat(mtp_input_ids, dim=0)
                mtp_aux_hiddens = torch.cat(mtp_aux_hiddens, dim=0)

                hidden_states = self.llm._forward_mtp(
                    mtp_input_ids,
                    mtp_aux_hiddens,
                    position_ids,
                    kv_cache_info,
                    is_decode=False,
                )
                actual_lens_q_cpu = kv_cache_info.actual_lens_q_cpu
                hidden_states = hidden_states[
                    (torch.cumsum(actual_lens_q_cpu, dim=0) - 1).to(
                        "cuda", non_blocking=True
                    )
                ]
                if have_extend_in_batch:
                    draft_token_ids_gpu, _ = self.llm.forward_mtp_draft(
                        seqs,
                        self.block_manager,
                        accepted_cnt_list=[1] * len(seqs),
                        input_token_ids=maybe_output_ids,
                        last_hidden_states=hidden_states,
                    )
            return maybe_output_ids, None, None, draft_token_ids_gpu

        draft_token_ids, draft_probs, draft_position_ids = self.llm._load_mtp_draft(
            seqs
        )
        # assert draft_token_ids.shape[1] == self.llm.num_spec_tokens + 1
        assert draft_position_ids.shape[1] == self.llm.num_spec_tokens + 1

        if parallel_state.pp_prev_rank != None:
            assert hasattr(self, "hidden_size"), (
                "hidden_size is required to determine the input shape"
            )
            hidden_states = torch.empty(
                (len(seqs) * (self.llm.num_spec_tokens + 1), self.hidden_size),
                dtype=self.args.dtype,
                device="cuda",
            )
            dist.recv(hidden_states, src=parallel_state.pp_prev_rank)
            input_tensor = hidden_states
        else:
            input_tensor = draft_token_ids.reshape(-1)

        (
            out_token_ids,
            num_emitted_draft_tokens,
            extend_aux_hidden,
        ) = self.llm.forward_mtp_verify(
            seqs,
            input_tensor,
            draft_position_ids.reshape(-1),
            draft_probs,
            self.block_manager,
            draft_token_ids=draft_token_ids,
        )
        if parallel_state.pp_next_rank is not None:
            return out_token_ids, None, None, None

        num_emitted_draft_tokens_cpu = num_emitted_draft_tokens.to(
            "cpu", non_blocking=True
        )
        verify_d2h_event = torch.cuda.Event()
        verify_d2h_event.record()
        assert out_token_ids.shape[1] == self.llm.num_spec_tokens + 1

        extend_token_ids = out_token_ids
        extend_position_ids = draft_position_ids
        seq_states = self.llm._snapshot_seq_states(seqs)

        for i, seq in enumerate(seqs):
            for _ in range(self.llm.num_spec_tokens):
                seq.token_ids.append(0)
        kv_cache_info: KVCacheInfo = prepare_kv_info(
            seqs,
            self.block_manager.kv_page_table,
            self.block_manager.page_size,
            self.block_manager.kv_cache,
            self.block_manager.mla_cache_tensor,
            dcp_interleave_size=self.args.dcp_interleave_size,
            is_decode=is_decode,
            attention_backend=self.args.attention_backend,
        )
        assert kv_cache_info.actual_lens_q_cpu is not None
        total_q = int(kv_cache_info.actual_lens_q_cpu.sum().item())
        assert total_q == extend_token_ids.flatten().shape[0], (
            f"MTP extend query mismatch: total_q={total_q}, actual_inputs={extend_token_ids.shape[0]}, "
            f"is_decode={is_decode}, num_seqs={len(seqs)}"
        )
        hidden_states = self.llm._forward_mtp(
            extend_token_ids.flatten().contiguous(),
            extend_aux_hidden.reshape(-1, extend_aux_hidden.shape[-1]).contiguous(),
            extend_position_ids.flatten().contiguous(),
            kv_cache_info,
            is_decode=is_decode,
        )

        actual_lens_q_cpu = kv_cache_info.actual_lens_q_cpu
        hidden_states = hidden_states[
            (torch.cumsum(actual_lens_q_cpu, dim=0) - actual_lens_q_cpu).to(
                "cuda", non_blocking=True
            )
            + num_emitted_draft_tokens
        ]
        self.llm._restore_seq_states(seq_states)
        input_token_ids = out_token_ids.take_along_dim(
            num_emitted_draft_tokens.long().unsqueeze(1), dim=1
        ).squeeze(1)
        verify_d2h_event.synchronize()
        draft_token_ids_gpu, _ = self.llm.forward_mtp_draft(
            seqs,
            self.block_manager,
            accepted_cnt_list=(num_emitted_draft_tokens_cpu + 1).tolist(),
            input_token_ids=input_token_ids,
            last_hidden_states=hidden_states,
        )

        return (
            out_token_ids,
            num_emitted_draft_tokens_cpu,
            num_emitted_draft_tokens,
            draft_token_ids_gpu,
        )
