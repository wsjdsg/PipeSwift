import ctypes
import os
import queue
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3
_batch_memcpy_fn = None

# Descriptor size is the only thing that sets this transport's throughput, and
# it falls off a cliff. Measured H2D on H100/Gen5 x16 at a fixed transfer size,
# so every point moves the same bytes and only granularity varies:
#
#   bytes/descriptor    4K    8K   16K   32K   64K  128K  256K  512K
#   GB/s               3.1   5.6   7.5  40.4  47.9  48.5  51.8  53.4
#
# Below ~32 KB the driver degrades to a synchronous host-staged copy -- at 4 KB
# the submitting thread burns 137 ms of CPU per 512 MiB and the transfer is
# already done when it returns, so batching more pages per call buys nothing.
# Above the cliff it approaches the link ceiling: 64 KB reaches 90% of it, 256 KB
# 97%. D2H is far more tolerant, already at 28 GB/s where H2D is at 5.6.
#
# A descriptor covers one page of one cache tensor for one layer, so its size is
# page_size x (KV bytes per token per layer) -- and for a GQA cache that is only
# 4-16 KB at the default page_size, on the wrong side of the cliff. Two things
# move it, neither of them this file's doing: giving the engine ONE cache tensor
# with the K/V axis inside the page (halves the descriptor count, doubles its
# size) and raising page_size. Instance warns in red when the two together still
# do not clear it. Coalescing runs of adjacent pages would also do it in
# principle, but does not in practice -- pages come off a shared free queue, so
# measured run length is 1.00 at every percentile.


# --------------------------------------------------------------------------- #
# cuMemcpyBatchAsync binding
# --------------------------------------------------------------------------- #


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


def _resolve_batch_memcpy_fn():
    """Bind cuMemcpyBatchAsync once. Needs a 12.8+ driver and cuda-python."""
    global _batch_memcpy_fn
    if _batch_memcpy_fn is None:
        from cuda.bindings import driver as drv

        err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"L2 cache requires cuMemcpyBatchAsync (driver 12.8+): {err}"
            )
        _batch_memcpy_fn = _BATCH_MEMCPY_FUNC_TYPE(ptr)
    return _batch_memcpy_fn


class _BatchMemcpyParams:
    def __init__(self, src_views, dst_views, stream, src_access_order):
        _resolve_batch_memcpy_fn()

        # cuMemcpyBatchAsync rejects both default-stream handles with
        # CUDA_ERROR_INVALID_VALUE, so a dedicated stream is mandatory.
        assert stream.cuda_stream not in (0, 1, 2), (
            "L2 transfers need a dedicated CUDA stream; cuMemcpyBatchAsync "
            f"rejects default-stream handles (got {stream.cuda_stream})."
        )

        assert list(src_views.keys()) == list(dst_views.keys())
        src_bases, dst_bases, bpb = [], [], []
        for src, dst in zip(src_views.values(), dst_views.values()):
            src_bpb = src.stride(0) * src.element_size()
            assert src_bpb == dst.stride(0) * dst.element_size()
            src_bases.append(src.data_ptr())
            dst_bases.append(dst.data_ptr())
            bpb.append(src_bpb)
        self.src_bases = np.array(src_bases, dtype=np.uint64)
        self.dst_bases = np.array(dst_bases, dtype=np.uint64)
        self.bpb = np.array(bpb, dtype=np.uint64)
        self.num_tensors = len(src_bases)
        self.attrs = _CUmemcpyAttributes(srcAccessOrder=src_access_order)
        self.attrs_idx = ctypes.c_size_t(0)
        self.fail_idx = ctypes.c_size_t(0)
        self.stream_handle = stream.cuda_stream
        self.stream = stream


# --------------------------------------------------------------------------- #
# Cache tensor layout
# --------------------------------------------------------------------------- #


def page_bytes(caches):
    """Bytes one page occupies across every layer of every cache tensor."""
    return sum(
        cache[0, 0].numel() * cache.element_size() * cache.shape[0] for cache in caches
    )


def make_dma_views(caches, num_pages):
    """Flatten each layer to an int8 [num_pages, page_bytes] view for descriptors.

    Contiguity is required twice over: ``reshape`` must return a view rather
    than a copy whose storage dies with the local, and ``_coalesce`` relies on
    consecutive pages being consecutive bytes within a layer.
    """
    views = OrderedDict()
    for cache_idx, cache in enumerate(caches):
        assert cache.shape[1] == num_pages
        assert cache.is_contiguous()
        for layer_idx in range(cache.shape[0]):
            layer = cache[layer_idx].reshape(num_pages, -1).view(torch.int8)
            views[f"{cache_idx}.{layer_idx}"] = layer
    return views


# --------------------------------------------------------------------------- #
# Transport engine
# --------------------------------------------------------------------------- #


class CopyHandle:
    def __init__(self, pending=True):
        # None until the copy thread has enqueued the batch; an empty transfer
        # never gets one, hence the flag.
        self.event = None
        self.pending = pending
        self.error = None
        # Set synchronously by _submit, because the retire deadline is priced
        # from them. Runs, not pages: the same bytes take an order of magnitude
        # longer as 4 KiB descriptors (see the table above). Both replicated
        # across ranks; the caller applies the agreed per-rank geometry.
        self.num_runs = 0
        self.num_pages = 0

    def wait(self):
        """Block until the transfer has landed.

        ``event`` is None until the copy thread has enqueued the batch, so the
        spin is over host-side driver calls for whatever is queued ahead, not over
        the copy itself.
        """
        if not self.pending:
            return
        while self.event is None:
            if self.error is not None:
                raise self.error
            os.sched_yield()
        self.event.synchronize()


class CopyEngine:
    """Owns the pinned host mirror of the KV cache and moves pages in and out.

    Transfers are submitted to a background thread so the scheduler never blocks
    on a driver call, and run on dedicated low-priority streams so they overlap
    the forward pass. Each returns a handle the caller waits on when the
    scheduler has contracted the transfer to be done (see poll_l2_events).

    One submitting thread, not several. Splitting a batch across four was worth
    +37-54% while descriptors were below the 32 KB cliff, where the driver stages
    the copy on the submitting thread and throughput is a property of how many
    threads submit rather than of the link. Above the cliff it does not merely
    stop paying: at 64 KB per descriptor it measured 13% *slower* than one thread,
    reproducibly across trials and both cache layouts, and never better than +5%
    anywhere above that. The split was a workaround for a regime the descriptor
    geometry now stays out of.
    """

    def __init__(self, gpu_caches, gpu_num_pages, l2_num_pages):
        # Set before anything fallible so a failed __init__ still unregisters
        # whatever was pinned when __del__ runs.
        self.closed = False
        self._registered = []
        self.thread = None
        self.queue = queue.SimpleQueue()
        self.gpu_num_pages = gpu_num_pages
        self.l2_num_pages = l2_num_pages
        self.host_caches = [
            self._alloc_pinned_mirror(cache, l2_num_pages) for cache in gpu_caches
        ]
        gpu_views = make_dma_views(gpu_caches, gpu_num_pages)
        cpu_views = make_dma_views(self.host_caches, l2_num_pages)

        low_pri, _ = torch.cuda.Stream.priority_range()
        # A _BatchMemcpyParams carries the stream it submits to and the scratch
        # the driver writes its failure index into, so the two directions cannot
        # share one.
        self.load_params = _BatchMemcpyParams(
            cpu_views,
            gpu_views,
            torch.cuda.Stream(priority=low_pri),
            # Not ANY: it lets the driver stage the batch on the submitting
            # thread while the stream is still gated behind the forward pass,
            # which segfaults inside cuMemcpyBatchAsync under load. Correctness
            # does not need it -- nothing writes the mirror while a restore
            # reads it.
            CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self.store_params = _BatchMemcpyParams(
            gpu_views,
            cpu_views,
            torch.cuda.Stream(priority=low_pri),
            CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self.thread = threading.Thread(
            target=self._copy_loop,
            args=(self.queue, torch.cuda.current_device()),
            daemon=True,
        )
        self.thread.start()

    def __del__(self):
        self.close()

    def _alloc_pinned_mirror(self, cache, num_pages):
        """Host tensor shaped like ``cache`` but ``num_pages`` deep, page-locked.

        Pinning is what lets the transfer run async instead of being staged
        through a driver bounce buffer. Allocated by the driver
        (``pin_memory=True`` -> ``cudaHostAlloc``) rather than pinned after the
        fact over a plain ``torch.empty``: same speed, but tens of GiB of movable
        anonymous memory never get pinned for the life of the process and the
        mapping comes out page aligned.
        """
        shape = list(cache.shape)
        shape[1] = num_pages
        tensor = torch.empty(
            tuple(shape), device="cpu", dtype=cache.dtype, pin_memory=True
        )
        self._registered.append(tensor)
        return tensor

    # -- submission ---------------------------------------------------------- #

    def store(self, gpu_page_idxs, l2_page_idxs):
        """D2H. Ordered after the KV writes already queued on the compute stream."""
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream())
        return self._submit(
            gpu_page_idxs, l2_page_idxs, is_store=True, wait_event=compute_done
        )

    def load(self, l2_page_idxs, gpu_page_idxs):
        """H2D. The caller must not read the GPU pages until the handle is waited on.

        Ordered after queued compute for the same reason as ``store``: a page
        handed over by ``allocate_a_page`` may still be read by a launched-ahead
        kernel, and this is the only write to a KV page that does not already go
        through the compute stream.
        """
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream())
        return self._submit(
            l2_page_idxs, gpu_page_idxs, is_store=False, wait_event=compute_done
        )

    def calibrate(self, num_pages=512):
        """Per-direction cost model, solving ``t = descriptors/desc_rate + bytes/bw``.

        Both terms are needed because page count alone does not predict a
        transfer: the same pages consecutive are one descriptor per run at link
        rate, scattered are one per page at a fraction of it (see the table
        above), so calibrating only the consecutive case priced the retire
        deadline ~10x too early. Integer rates, because BlockManager reduces them
        to the cluster minimum. Runs at init on an idle device -- an upper bound
        -- and scribbles over the first ``num_pages`` pages of both tiers, which
        hold nothing this early.
        """
        n = min(num_pages, self.gpu_num_pages, self.l2_num_pages)
        fwd = list(range(n))
        # Reversed: no pair advances together on both sides, so nothing merges
        # and every page becomes its own descriptor.
        rev = list(reversed(fwd))
        model = {}
        for direction, submit in (("store", self.store), ("load", self.load)):
            params = self.store_params if direction == "store" else self.load_params
            samples = {}
            for tag, dst in (("coalesced", fwd), ("scattered", rev)):
                for _ in range(2):  # the first pass pays driver warmup
                    start = time.perf_counter()
                    handle = submit(fwd, dst)
                    handle.wait()
                    elapsed = time.perf_counter() - start
                # Local geometry is the right one here: this is a local timing.
                samples[tag] = (
                    handle.num_runs * params.num_tensors,
                    handle.num_pages * int(params.bpb.sum()),
                    elapsed,
                )
            (d_lo, nbytes, t_lo), (d_hi, _, t_hi) = (
                samples["coalesced"],
                samples["scattered"],
            )
            # Scattering cannot make a transfer faster; if the measurement says
            # otherwise the gap is noise and the per-descriptor term is nil.
            # Floored at 1 ns so the reciprocal stays an int64-safe rate.
            t_desc = min(max((t_hi - t_lo) / max(1, d_hi - d_lo), 1e-9), 1.0)
            # Whatever the coalesced sample did not spend on descriptors.
            link_s = max(t_lo - d_lo * t_desc, 1e-9)
            model[direction] = {
                "desc_rate": int(1 / t_desc),
                "bw": int(nbytes / link_s),
            }
        return model

    def close(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        # __init__ can fail between pinning and starting the thread.
        if getattr(self, "thread", None) is not None:
            self.queue.put(None)
            self.thread.join(timeout=5.0)
        # Only safe once the copy thread is joined: freeing a pinned buffer with
        # a transfer still in flight faults.
        self._registered = []

    def _submit(self, src_ids, dst_ids, is_store, wait_event=None):
        params = self.store_params if is_store else self.load_params
        src_limit, dst_limit = (
            (self.gpu_num_pages, self.l2_num_pages)
            if is_store
            else (self.l2_num_pages, self.gpu_num_pages)
        )
        if not len(src_ids):
            return CopyHandle(pending=False)
        # Coalesced on the calling thread so the caller learns the run count
        # synchronously -- the retire deadline is priced from it.
        src = np.array(src_ids, dtype=np.uint64)
        dst = np.array(dst_ids, dtype=np.uint64)
        # An index no pool handed out becomes an address outside the mapping that
        # the driver dereferences on this thread, i.e. a bare segfault inside
        # cuMemcpyBatchAsync. Two maxima are cheaper than that diagnosis.
        self._check_range(src, dst, src_limit, dst_limit, is_store)
        runs = self._coalesce(src, dst)
        handle = CopyHandle()
        handle.num_runs = len(runs[0])
        handle.num_pages = len(src_ids)
        self.queue.put((runs, wait_event, handle, params))
        return handle

    # -- copy thread --------------------------------------------------------- #

    @staticmethod
    def _check_range(src, dst, src_limit, dst_limit, is_store):
        """Raise before the driver dereferences an index no pool owns."""
        for ids, limit, side in ((src, src_limit, "src"), (dst, dst_limit, "dst")):
            if len(ids) and int(ids.max()) >= limit:
                bad = np.flatnonzero(ids >= limit)
                tier = ("gpu", "l2") if (side == "src") == is_store else ("l2", "gpu")
                raise AssertionError(
                    f"L2 {'store' if is_store else 'load'} {side} page index out of "
                    f"range for the {tier[0]} tier: "
                    f"{[int(ids[i]) for i in bad[:8]]} >= {limit} "
                    f"({len(bad)} of {len(ids)} pages bad)"
                )

    @classmethod
    def _copy_loop(cls, q, device):
        torch.cuda.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            runs, wait_event, handle, params = item
            stream = params.stream
            try:
                with torch.cuda.stream(stream):
                    if wait_event is not None:
                        stream.wait_event(wait_event)
                    cls._copy_blocks(runs, params)
                    event = torch.cuda.Event()
                    event.record(stream)
                handle.event = event
            except Exception as exc:
                handle.error = exc

    @staticmethod
    def _coalesce(src_ids, dst_ids):
        """Merge page pairs that advance together on both tiers into runs.

        A page's bytes for one layer are one contiguous row of that layer's
        ``[num_pages, bpb]`` view, so pages ``p..p+k-1`` mapping to
        ``q..q+k-1`` are one contiguous range on each side and collapse to a
        single ``k * bpb`` descriptor. Sorting first is free: descriptor order
        within a batch carries no meaning.

        Returns the run starts on both sides and the run lengths, in pages.
        """
        order = np.argsort(src_ids, kind="stable")
        src = src_ids[order]
        dst = dst_ids[order]
        breaks = (src[1:] != src[:-1] + 1) | (dst[1:] != dst[:-1] + 1)
        starts = np.flatnonzero(np.concatenate(([True], breaks)))
        # uint64 is mandatory: numpy promotes uint64 * int64 to float64, which
        # silently rounds byte counts above 2**53.
        lens = np.diff(np.append(starts, len(src))).astype(np.uint64)
        return src[starts], dst[starts], lens

    @classmethod
    def _copy_blocks(cls, runs, params):
        """Build and launch the descriptors for pre-coalesced page runs."""
        src_start, dst_start, run_len = runs
        if not len(run_len):
            return
        src_all = (
            params.src_bases[:, None] + src_start[None, :] * params.bpb[:, None]
        ).ravel()
        dst_all = (
            params.dst_bases[:, None] + dst_start[None, :] * params.bpb[:, None]
        ).ravel()
        sz_all = (params.bpb[:, None] * run_len[None, :]).ravel()

        err = _batch_memcpy_fn(
            dst_all.ctypes.data,
            src_all.ctypes.data,
            sz_all.ctypes.data,
            len(src_all),
            ctypes.addressof(params.attrs),
            ctypes.byref(params.attrs_idx),
            1,
            ctypes.byref(params.fail_idx),
            params.stream_handle,
        )
        if err != 0:
            raise RuntimeError(
                f"L2 batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
            )
