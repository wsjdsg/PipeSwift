from __future__ import annotations

import itertools
from array import array
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


@dataclass
class SamplingParams:
    n_seqs: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    eos_token_ids: List[int] = None
    ignore_eos: bool = False
    max_gen_tokens: int = 32

    def set_from_args(self, args):
        for f in fields(SamplingParams):
            name = f.name
            assert hasattr(args, name)
            setattr(self, name, getattr(args, name))


class Request:
    _id_counter = itertools.count(0)
    _tokenizer: AutoTokenizer = None

    def __init__(
        self,
        prompt: str = None,
        prompt_token_ids: array | List[int] | None = None,
        n_seqs: int = 1,
        max_gen_tokens: int = 32,
    ):
        self.req_id = next(Request._id_counter)
        self.seqs: List[Sequence] = []
        self.prompt = prompt
        self.prompt_token_ids = (
            array("i") if prompt_token_ids is None else array("i", prompt_token_ids)
        )
        self.n_seqs = n_seqs
        self.max_gen_tokens = max_gen_tokens

    def __str__(self):
        import json

        data = {
            "request_id": self.req_id,
            "prompt": self.prompt,
            "responses": {
                str(i + 1): Request._tokenizer.decode(
                    seq.token_ids[seq.num_prompt_tokens :]
                )
                for i, seq in enumerate(self.seqs)
            },
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


class Sequence:
    _id_counter = itertools.count(0)
    ignore_eos: bool = False
    eos_token_ids: List[int]
    max_seqlen: int = 0

    def __init__(self, request, token_ids=None):
        self.seq_id = next(Sequence._id_counter)
        self.request: Request = request
        self.token_ids: array = (
            array("i") if token_ids is None else array("i", token_ids)
        )
        self.offset = 0
        self.num_prompt_tokens = len(self.token_ids)
        self.num_pending_tokens = 0
        self.is_temporal_for_chunked_prefill = False
        self.finished = False
        self.finish_reason: dict | None = None
        self.spec_draft_token_ids = None
        self.spec_draft_probs = None
        self.next_token = None
        self.l2_load_pending = False
        # ``(num_tokens, per-page prefix-chained hashes)``, filled in by the
        # block manager. Cached here because the L2 prefetch probe re-reads it
        # every step; the length invalidates it, since a preempted sequence
        # re-prefills with its generated tokens included.
        self.page_hashes = None
        # Where the L2 prefetch probe's L1-match scan stopped, as
        # ``(num_tokens, page_idx)``: the probe re-runs every step a sequence
        # waits, and on multi-turn traffic that walks a ~700-page all-L1-hit
        # prefix -- tens of ms of Python against a ~19 ms step. Hint only.
        self.l2_probe_resume = None

    def _deepcopy_from(self, o):
        assert isinstance(o, Sequence)
        self.seq_id = o.seq_id
        self.request = o.request
        self.token_ids = deepcopy(o.token_ids)
        self.offset = o.offset
        self.num_prompt_tokens = o.num_prompt_tokens
        self.num_pending_tokens = o.num_pending_tokens
        self.is_temporal_for_chunked_prefill = o.is_temporal_for_chunked_prefill
        self.finished = o.finished
        self.finish_reason = deepcopy(o.finish_reason)
        self.spec_draft_token_ids = o.spec_draft_token_ids  # TODO
        self.spec_draft_probs = o.spec_draft_probs  # TODO
        self.l2_load_pending = o.l2_load_pending
        self.page_hashes = o.page_hashes
        self.l2_probe_resume = o.l2_probe_resume
        return self

    def is_finished(self):
        if self.is_temporal_for_chunked_prefill:
            return False
        max_gen_tokens = self.request.max_gen_tokens
        return (
            self.finished
            or (len(self.token_ids) - self.num_prompt_tokens >= max_gen_tokens)
            or (
                not Sequence.ignore_eos and self.token_ids[-1] in Sequence.eos_token_ids
            )
            or len(self.token_ids) >= Sequence.max_seqlen
        )

    def get_finish_reason(self):
        max_gen_tokens = self.request.max_gen_tokens
        if len(self.token_ids) - self.num_prompt_tokens >= max_gen_tokens:
            return {"type": "length", "length": max_gen_tokens}
        if len(self.token_ids) >= Sequence.max_seqlen:
            return {"type": "length", "length": Sequence.max_seqlen}
        if not Sequence.ignore_eos and self.token_ids[-1] in Sequence.eos_token_ids:
            return {"type": "stop", "matched": self.token_ids[-1]}
        return {"type": "stop", "matched": None}

    def __str__(self):
        return f"Sequence {self.seq_id}: \tRequest Id: {self.request.req_id}, \tprompt_len: {self.num_prompt_tokens}, \tcached seqlen {self.offset}"


def sequence_offset_desc_key(s: Sequence):
    """Sort key for descending offset, with seq_id as deterministic tiebreak so
    every PP rank's running_queue produces an identical ordering."""
    return (-s.offset, s.seq_id)


class MsgType(Enum):
    INFO = 0
    EXIT = 1
    REQS = 2
    PULL_REQS = 3
    FINISHED_SEQS = 4


@dataclass
class WorkerMsg:
    instr: MsgType = MsgType.INFO
    src_rank: int = 0
    dst_rank: int = 0
    content: Any = None

    def __str__(self):
        if isinstance(self.content, List) and self.instr == MsgType.REQS:
            content_str = "[\n"
            flag = False
            for obj in self.content:
                content_str += ",\n" if flag else ""
                content_str += str(obj)
                flag = True
            content_str += "]"
        else:
            content_str = str(self.content)
        return f"[Src Proc {self.src_rank}, Dst Proc {self.dst_rank}] {self.instr}: {content_str}"


class QuantType(Enum):
    NO_QUANT = 0
    FP8_PER_BLOCK = 1
    FP8_PER_CHANNEL = 2


@dataclass
class KVCacheInfo:
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None
    page_table: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    actual_lens_q: torch.Tensor = None
    actual_lens_kv: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None

    seqs_kv_pages: torch.Tensor = None
    seqs_kv_page_indptrs: torch.Tensor = None
    mla_cache_tensor: torch.Tensor = None

    actual_lens_q_cpu: torch.Tensor = None
    seqs_kv_page_indptrs_cpu: torch.Tensor = None
    cu_seqlens_q_cpu: torch.Tensor = None
    actual_lens_kv_cpu: torch.Tensor = None


def prepare_kv_info(
    seqs: List[Sequence],
    kv_page_table,
    kv_page_size,
    kv_cache,
    mla_cache_tensor,
    dcp_interleave_size=1,
    is_decode=False,
    attention_backend: str = "flash_attn",
):
    from .impl_utils import coalesced_to_cuda
    from .mpu import parallel_state

    num_seqs = len(seqs)
    assert num_seqs != 0

    dcp_deg = parallel_state.dcp_deg
    dcp_rank = parallel_state.dcp_rank
    physical_page_size = kv_page_size if dcp_deg == 1 else kv_page_size // dcp_deg

    actual_lens_q = array("i", [0] * num_seqs)
    actual_lens_kv = array("i", [0] * num_seqs)
    logical_offsets = array("i", [0] * num_seqs)
    cu_seqlens_q_arr = array("i", [0])
    max_pages = 0
    total_q = 0
    slot_mapping_ranges = array("i", [0]) * (num_seqs + 1)

    for i, seq in enumerate(seqs):
        num_seq_tokens = len(seq.token_ids)
        logical_num_seq_tokens = num_seq_tokens
        logical_offset = seq.offset
        q_len = logical_num_seq_tokens - logical_offset
        actual_lens_q[i] = q_len
        logical_offsets[i] = logical_offset

        if dcp_deg == 1:
            actual_lens_kv[i] = logical_num_seq_tokens
        else:
            prefix_len = (
                logical_num_seq_tokens if is_decode and q_len == 1 else logical_offset
            )
            actual_lens_kv[i] = (
                prefix_len // (dcp_interleave_size * dcp_deg) * dcp_interleave_size
            )
            remainder = prefix_len % (dcp_interleave_size * dcp_deg)
            if remainder >= (dcp_rank + 1) * dcp_interleave_size:
                actual_lens_kv[i] += dcp_interleave_size
            elif remainder >= dcp_rank * dcp_interleave_size:
                actual_lens_kv[i] += remainder % dcp_interleave_size

        total_q += q_len
        cu_seqlens_q_arr.append(total_q)
        slot_mapping_ranges[i + 1] = total_q
        seq_page_table = kv_page_table[seq.seq_id]
        if len(seq_page_table) > max_pages:
            max_pages = len(seq_page_table)

    seqs_kv_page_table_arr = array("i", [0]) * (len(seqs) * max_pages)
    slot_mapping_arr = array("q", [0]) * total_q
    for i, seq in enumerate(seqs):
        seq_page_table = kv_page_table[seq.seq_id]
        row_start = i * max_pages
        seqs_kv_page_table_arr[row_start : row_start + len(seq_page_table)] = (
            seq_page_table
        )
        logical_offset = logical_offsets[i]
        q_len = actual_lens_q[i]
        logical_num_seq_tokens = logical_offset + q_len
        write_idx = slot_mapping_ranges[i]
        if dcp_deg == 1:
            for pos in range(logical_offset, logical_num_seq_tokens):
                slot_mapping_arr[write_idx] = seq_page_table[
                    pos // kv_page_size
                ] * physical_page_size + (pos % kv_page_size)
                write_idx += 1
        else:
            dcp_stride = dcp_interleave_size * dcp_deg
            for pos in range(logical_offset, logical_num_seq_tokens):
                if ((pos // dcp_interleave_size) % dcp_deg) == dcp_rank:
                    local_pos = (
                        pos % dcp_interleave_size
                        + pos // dcp_stride * dcp_interleave_size
                    )
                    slot_mapping_arr[write_idx] = seq_page_table[
                        pos // kv_page_size
                    ] * physical_page_size + (local_pos % physical_page_size)
                else:
                    slot_mapping_arr[write_idx] = -1
                write_idx += 1

    actual_lens_kv_np = np.frombuffer(actual_lens_kv, dtype=np.int32)
    actual_lens_q_np = np.frombuffer(actual_lens_q, dtype=np.int32)
    cu_seqlens_q_cpu = np.frombuffer(cu_seqlens_q_arr, dtype=np.int32)
    seqs_kv_page_table_np = np.frombuffer(seqs_kv_page_table_arr, dtype=np.int32)
    slot_mapping_np = np.frombuffer(slot_mapping_arr, dtype=np.int64)
    actual_lens_kv_cpu = torch.from_numpy(actual_lens_kv_np)
    actual_lens_q_cpu = torch.from_numpy(actual_lens_q_np)

    seqs_kv_page_table, cu_seqlens_q, actual_lens_q, actual_lens_kv = coalesced_to_cuda(
        [
            seqs_kv_page_table_np.flatten(),
            cu_seqlens_q_cpu,
            actual_lens_q_cpu,
            actual_lens_kv_cpu,
        ]
    )
    seqs_kv_page_table = seqs_kv_page_table.view((len(seqs), max_pages))
    slot_mapping = torch.from_numpy(slot_mapping_np).to("cuda", non_blocking=True)

    kv_cache_info = KVCacheInfo(
        kv_cache=(kv_cache[0], kv_cache[1]),
        mla_cache_tensor=mla_cache_tensor,
        page_table=seqs_kv_page_table,
        slot_mapping=slot_mapping,
        actual_lens_q=actual_lens_q,
        actual_lens_kv=actual_lens_kv,
        cu_seqlens_q=cu_seqlens_q,
        actual_lens_q_cpu=actual_lens_q_cpu,
        cu_seqlens_q_cpu=torch.from_numpy(cu_seqlens_q_cpu),
        actual_lens_kv_cpu=actual_lens_kv_cpu,
    )

    if attention_backend == "flash_mla" and not is_decode:
        seqs_kv_pages = seqs_kv_page_table.view(-1)
        seqs_kv_page_indptrs = torch.arange(
            0,
            seqs_kv_pages.shape[-1] + 1,
            seqs_kv_page_table.shape[-1],
            device="cuda",
            dtype=torch.int32,
        )
        seqs_kv_page_indptrs_cpu = torch.arange(
            0,
            seqs_kv_pages.shape[-1] + 1,
            seqs_kv_page_table.shape[-1],
            dtype=torch.int32,
        )
        kv_cache_info.seqs_kv_pages = seqs_kv_pages
        kv_cache_info.seqs_kv_page_indptrs = seqs_kv_page_indptrs
        kv_cache_info.seqs_kv_page_indptrs_cpu = seqs_kv_page_indptrs_cpu

    return kv_cache_info


def derive_kvcache_info_next_decode_step(
    prev_kv_cache_info: KVCacheInfo,
    seqs: List[Sequence],
    kv_page_table,
    kv_page_size,
    dcp_interleave_size=1,
):  # used for multi-step draft of speculative decoding
    from .mpu import parallel_state

    assert len(seqs) != 0
    dcp_deg = parallel_state.dcp_deg
    dcp_rank = parallel_state.dcp_rank
    physical_page_size = kv_page_size if dcp_deg == 1 else kv_page_size // dcp_deg

    num_seqs = len(seqs)
    slot_mapping_arr = array("q", [0]) * num_seqs
    actual_lens_kv_arr = array(
        "i", prev_kv_cache_info.actual_lens_kv_cpu.numpy().tobytes()
    )

    if dcp_deg > 1:
        dcp_stride = dcp_interleave_size * dcp_deg

    for i, seq in enumerate(seqs):
        abs_seq_pos = seq.offset
        page_id = kv_page_table[seq.seq_id][abs_seq_pos // kv_page_size]

        if dcp_deg == 1:
            slot_mapping_arr[i] = page_id * physical_page_size + (
                abs_seq_pos % kv_page_size
            )
            actual_lens_kv_arr[i] += 1
        else:
            if ((abs_seq_pos // dcp_interleave_size) % dcp_deg) == dcp_rank:
                local_pos = (
                    abs_seq_pos % dcp_interleave_size
                    + abs_seq_pos // dcp_stride * dcp_interleave_size
                )
                slot_mapping_arr[i] = page_id * physical_page_size + (
                    local_pos % physical_page_size
                )
                actual_lens_kv_arr[i] += 1
            else:
                slot_mapping_arr[i] = -1

    slot_mapping_np = np.frombuffer(slot_mapping_arr, dtype=np.int64)
    slot_mapping = torch.from_numpy(slot_mapping_np).to("cuda", non_blocking=True)
    actual_lens_kv_cpu = torch.from_numpy(
        np.frombuffer(actual_lens_kv_arr, dtype=np.int32)
    )
    actual_lens_kv = actual_lens_kv_cpu.to("cuda", non_blocking=True)

    return KVCacheInfo(
        kv_cache=prev_kv_cache_info.kv_cache,
        mla_cache_tensor=prev_kv_cache_info.mla_cache_tensor,
        page_table=prev_kv_cache_info.page_table,
        slot_mapping=slot_mapping,
        actual_lens_q=prev_kv_cache_info.actual_lens_q,
        actual_lens_kv=actual_lens_kv,
        cu_seqlens_q=prev_kv_cache_info.cu_seqlens_q,
        actual_lens_q_cpu=prev_kv_cache_info.actual_lens_q_cpu,
        cu_seqlens_q_cpu=prev_kv_cache_info.cu_seqlens_q_cpu,
        actual_lens_kv_cpu=actual_lens_kv_cpu,
    )
