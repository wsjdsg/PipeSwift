import gc
from contextlib import contextmanager

import torch

from .metadata import KVCacheInfo


def initialize_graph_kvcacheinfo(max_bsz, max_num_pages, max_num_tokens=None):
    if max_num_tokens is None:
        max_num_tokens = max_bsz
    static_kvcacheinfo = KVCacheInfo()
    static_kvcacheinfo.page_table = torch.zeros(
        (max_bsz, max_num_pages), dtype=torch.int32, device="cuda"
    )  #
    static_kvcacheinfo.slot_mapping = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device="cuda"
    )
    static_kvcacheinfo.actual_lens_q = torch.ones(
        (max_bsz,), dtype=torch.int32, device="cuda"
    )
    static_kvcacheinfo.actual_lens_kv = torch.zeros(
        (max_bsz,), dtype=torch.int32, device="cuda"
    )  #
    static_kvcacheinfo.cu_seqlens_q = torch.arange(
        max_bsz + 1, dtype=torch.int32, device="cuda"
    )
    static_kvcacheinfo.seqs_kv_pages = torch.zeros(
        (max_bsz * max_num_pages), dtype=torch.int32, device="cuda"
    )  #
    static_kvcacheinfo.seqs_kv_page_indptrs = torch.zeros(
        (max_bsz + 1,), dtype=torch.int32, device="cuda"
    )
    static_kvcacheinfo.actual_lens_q_cpu = torch.ones((max_bsz,), dtype=torch.int32)
    return static_kvcacheinfo


def slice_kvcache_info(kvcache_info: KVCacheInfo, bsz, num_tokens=None):
    if num_tokens is None:
        num_tokens = bsz
    _kv_cacheinfo = KVCacheInfo()
    _kv_cacheinfo.kv_cache = kvcache_info.kv_cache
    _kv_cacheinfo.mla_cache_tensor = kvcache_info.mla_cache_tensor
    _kv_cacheinfo.page_table = kvcache_info.page_table[:bsz]
    _kv_cacheinfo.slot_mapping = kvcache_info.slot_mapping[:num_tokens]
    _kv_cacheinfo.actual_lens_q = kvcache_info.actual_lens_q[:bsz]
    _kv_cacheinfo.actual_lens_kv = kvcache_info.actual_lens_kv[:bsz]
    _kv_cacheinfo.cu_seqlens_q = kvcache_info.cu_seqlens_q[: (bsz + 1)]
    _kv_cacheinfo.seqs_kv_pages = kvcache_info.seqs_kv_pages
    _kv_cacheinfo.seqs_kv_page_indptrs = kvcache_info.seqs_kv_page_indptrs[: (bsz + 1)]
    _kv_cacheinfo.actual_lens_q_cpu = (
        None
        if kvcache_info.actual_lens_q_cpu is None
        else kvcache_info.actual_lens_q_cpu[:bsz]
    )
    _kv_cacheinfo.actual_lens_kv_cpu = (
        None
        if kvcache_info.actual_lens_kv_cpu is None
        else kvcache_info.actual_lens_kv_cpu[:bsz]
    )
    _kv_cacheinfo.cu_seqlens_q_cpu = (
        None
        if kvcache_info.cu_seqlens_q_cpu is None
        else kvcache_info.cu_seqlens_q_cpu[: (bsz + 1)]
    )
    _kv_cacheinfo.seqs_kv_page_indptrs_cpu = (
        None
        if kvcache_info.seqs_kv_page_indptrs_cpu is None
        else kvcache_info.seqs_kv_page_indptrs_cpu[: (bsz + 1)]
    )
    _kv_cacheinfo.cg_bsz = bsz
    return _kv_cacheinfo


def set_graph_kvcacheinfo(
    static_kvcacheinfo: KVCacheInfo, kvcache_info: KVCacheInfo, cg_bsz, pad_q_len=1
):
    assert kvcache_info is not None
    bsz, max_pages = (
        kvcache_info.actual_lens_q.shape[0],
        kvcache_info.page_table.shape[1],
    )
    static_kvcacheinfo.page_table[:bsz, :max_pages].copy_(
        kvcache_info.page_table, non_blocking=True
    )
    static_kvcacheinfo.slot_mapping[: kvcache_info.slot_mapping.shape[0]].copy_(
        kvcache_info.slot_mapping, non_blocking=True
    )
    static_kvcacheinfo.actual_lens_q[:bsz].copy_(
        kvcache_info.actual_lens_q, non_blocking=True
    )
    static_kvcacheinfo.actual_lens_kv[:bsz].copy_(
        kvcache_info.actual_lens_kv, non_blocking=True
    )
    static_kvcacheinfo.cu_seqlens_q[: (bsz + 1)].copy_(
        kvcache_info.cu_seqlens_q, non_blocking=True
    )

    # pad cuda-graph meta data (necessary)
    if bsz != cg_bsz:
        assert bsz < cg_bsz
        static_kvcacheinfo.actual_lens_q[bsz:].fill_(pad_q_len)
        static_kvcacheinfo.slot_mapping[kvcache_info.slot_mapping.shape[0] :].fill_(-1)
        static_kvcacheinfo.actual_lens_kv[bsz:].zero_()
        static_kvcacheinfo.page_table[bsz:, :max_pages].zero_()
        static_kvcacheinfo.cu_seqlens_q[bsz + 1 :].copy_(
            (
                static_kvcacheinfo.cu_seqlens_q[bsz]
                + torch.arange(
                    pad_q_len,
                    (static_kvcacheinfo.cu_seqlens_q.shape[0] - bsz) * pad_q_len,
                    pad_q_len,
                    dtype=torch.int32,
                    device="cuda",
                )
            ),
            non_blocking=True,
        )


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool = False):
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()
