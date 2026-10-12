import math
import warnings
from array import array
from collections import OrderedDict, deque
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import xxhash

import pipeswift.utils.copy_engine as copy_engine
from pipeswift.utils import Sequence

DEFAULT_L2_STEP_MS = 10.0
# Decode headroom the scheduler insists on before it spends the page pool on a
# waiting sequence: every sequence already holding pages must be able to run
# this many more tokens without preempting anyone.
DECODE_RESERVE_TOKENS = 512


class L2PagePool:
    """Host page ledger, mirroring the L1 pool's refcount discipline.

    ``ref_counts[p] > 0`` means pinned (in flight, or being read); ``== 0``
    means either free or sitting in ``cached_pages``, an LRU eviction queue.
    """

    def __init__(self, num_pages: int):
        self.num_pages = num_pages
        self.ref_counts = np.zeros(num_pages, dtype=np.int32)
        self.free_page_queue = deque(range(num_pages))
        self.cached_pages = OrderedDict()  # page_idx -> None, LRU order
        self.hash2page_idx_list = {}
        self.page_idx2token_ids = {}
        self.page_idx2hash = [-1] * num_pages
        # (hash, tokens) -> count over staged and in-flight stores, so the
        # duplicate check stays O(1).
        self._pending_store_keys = {}

    # -- lookup -------------------------------------------------------------- #

    def lookup(self, h, token_ids):
        """Return a page_idx whose hash and tokens match, else None."""
        if h == -1 or h not in self.hash2page_idx_list:
            return None
        for pidx in self.hash2page_idx_list[h]:
            if self.page_idx2token_ids[pidx] == token_ids:
                return pidx
        return None

    def has_page_or_pending(self, h, token_ids):
        """True if this prefix is already resident or already being stored."""
        if h == -1:
            return False
        if (h, tuple(token_ids)) in self._pending_store_keys:
            return True
        return self.lookup(h, token_ids) is not None

    def acquire_pending_key(self, key):
        self._pending_store_keys[key] = self._pending_store_keys.get(key, 0) + 1

    def release_pending_key(self, key):
        count = self._pending_store_keys.get(key, 0) - 1
        if count > 0:
            self._pending_store_keys[key] = count
        else:
            self._pending_store_keys.pop(key, None)

    # -- allocation ---------------------------------------------------------- #

    def allocate_for_store(self):
        """Take a free page, else evict the LRU one. None when all are pinned."""
        if self.free_page_queue:
            page_idx = self.free_page_queue.popleft()
        elif self.cached_pages:
            page_idx, _ = self.cached_pages.popitem(last=False)
            self._reset_page(page_idx)
        else:
            return None
        assert self.ref_counts[page_idx] == 0
        self.ref_counts[page_idx] = 1
        return page_idx

    def _reset_page(self, page_idx):
        assert self.ref_counts[page_idx] == 0
        h = self.page_idx2hash[page_idx]
        if h != -1 and h in self.hash2page_idx_list:
            page_idx_list = self.hash2page_idx_list[h]
            if page_idx in page_idx_list:
                page_idx_list.remove(page_idx)
            if len(page_idx_list) == 0:
                del self.hash2page_idx_list[h]
        self.page_idx2hash[page_idx] = -1
        self.page_idx2token_ids[page_idx] = []

    # -- pinning ------------------------------------------------------------- #

    def retain_for_load(self, page_idx):
        """Pin a resident page so eviction cannot recycle it mid-H2D."""
        if self.ref_counts[page_idx] == 0 and page_idx in self.cached_pages:
            self.cached_pages.pop(page_idx)
        self.ref_counts[page_idx] += 1

    def release_from_load(self, page_idx):
        self.ref_counts[page_idx] -= 1
        assert self.ref_counts[page_idx] >= 0
        if self.ref_counts[page_idx] == 0:
            self.cached_pages[page_idx] = None

    def finish_store(self, page_idx, h, token_ids):
        """Drop the store's pin and make the page discoverable."""
        self.ref_counts[page_idx] -= 1
        assert self.ref_counts[page_idx] >= 0
        self.page_idx2hash[page_idx] = h
        self.page_idx2token_ids[page_idx] = token_ids
        self.hash2page_idx_list.setdefault(h, []).append(page_idx)
        if self.ref_counts[page_idx] == 0:
            self.cached_pages[page_idx] = None

    def get_usage(self):
        return 1 - len(self.free_page_queue) / self.num_pages


class BlockManager:
    def __init__(
        self,
        num_pages,
        page_size,
        kv_cache,
        mla_cache_tensor,
        kv_cache_tensor=None,
        l2_cache_gb=0,
        l2_step_ms=DEFAULT_L2_STEP_MS,
    ):
        if l2_step_ms <= 0:
            raise ValueError(f"l2_step_ms must be positive, got {l2_step_ms}")
        self.kv_page_table: Dict[int, array[int]] = {}

        self.kv_cache: List[torch.Tensor] = (
            kv_cache  # 2 [n_layer, num_pages, page_size, n_kv_head, head_dim]
        )
        self.mla_cache_tensor: torch.Tensor = mla_cache_tensor
        # The single [n_layer, num_pages, 2, page_size, n_kv_head, head_dim]
        # allocation ``kv_cache`` is sliced from, when the GQA path is fused.
        self.kv_cache_tensor: torch.Tensor = kv_cache_tensor

        self.num_pages = num_pages
        self.page_size = page_size
        self.ref_counts = np.zeros(self.num_pages, dtype=np.int32)
        self.free_page_queue = deque(range(self.num_pages))
        self.free_page_queue.popleft()  # preserve pages for flashinfer's append_page kernel
        # prefix caching
        self.hash2page_idx_list = {}
        self.page_idx2token_ids = {}
        self.page_idx2hash = [-1] * self.num_pages

        # Cached pages ordered by eviction priority — released seqs insert
        # tail→head so suffix pages evict before their prefixes.
        self.cached_pages = OrderedDict()  # page_idx -> None
        self.hit_cnt = 0

        self.enable_l2_cache = l2_cache_gb > 0
        # Assumed scheduler step, used only to convert a transfer's predicted
        # link-seconds into the step count the retire deadline is priced in. A
        # decode step is ~10 ms on H100.
        self.l2_step_ms = l2_step_ms
        self.l2_hit_cnt = 0
        self.l2_store_cnt = 0
        self.l2_load_cnt = 0
        self.pending_l2_stores = []
        self.pending_l2_loads = []
        # Retire clock: the scheduler step itself. A coarser tick parks a hitting
        # sequence behind a copy that has already landed, and holds the GPU pages
        # its store reads just as long, shrinking the KV pool into preemptions.
        self.l2_step = 0
        # Link-seconds each direction is predicted to still owe the copy engine.
        self._l2_owed_s = {"store": 0.0, "load": 0.0}
        # Decode page completions, flushed as one batched transfer per step.
        self._l2_pending_store_requests = []
        # Restores staged by allocate_prefill, flushed as one batched transfer.
        self._l2_pending_load_requests = []
        if self.enable_l2_cache:
            # Hand the copy engine ONE tensor whenever the layout allows it, so a
            # page's bytes for a layer form a single descriptor instead of one
            # per K/V tensor. Both fused paths (MLA, and GQA since the K/V axis
            # moved inside the page) halve the descriptor count and double its
            # size; the unfused list is the fallback for layouts that have not
            # been converted.
            if self.mla_cache_tensor is not None:
                gpu_caches = [self.mla_cache_tensor]
            elif self.kv_cache_tensor is not None:
                gpu_caches = [self.kv_cache_tensor]
            else:
                gpu_caches = list(self.kv_cache)
            self.l2_num_pages = self._resolve_l2_num_pages(l2_cache_gb, gpu_caches)
            self.l2_copy_engine = copy_engine.CopyEngine(
                gpu_caches, self.num_pages, self.l2_num_pages
            )
            self.l2_pool = L2PagePool(self.l2_num_pages)
            self.l2_cost_model = self._resolve_l2_cost_model()
            # Agreed for the same reason as the cost model they multiply: both
            # scale with num_cache_layers, which differs on an uneven pp split.
            self.l2_page_bytes = max(
                1, self._min_across_workers(copy_engine.page_bytes(gpu_caches))
            )
            self.l2_views = max(
                1, self._min_across_workers(self.l2_copy_engine.load_params.num_tensors)
            )
        else:
            self.l2_num_pages = 0
            self.l2_page_bytes = 0
            self.l2_views = 0
            self.l2_pool = None
            self.l2_copy_engine = None
            self.l2_cost_model = None

    def __del__(self):
        engine = getattr(self, "l2_copy_engine", None)
        if engine is not None:
            engine.close()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # ----------------------------------------------------------------- #
    # L2 capacity
    # ----------------------------------------------------------------- #

    def _resolve_l2_num_pages(self, l2_cache_gb, caches):
        page_bytes = copy_engine.page_bytes(caches)
        max_bytes = int(l2_cache_gb * 1024**3 + 0.5)
        max_pages_per_rank = max_bytes // page_bytes
        if max_pages_per_rank <= 0:
            raise ValueError(
                "L2 cache byte budget is too small: "
                f"per_rank_budget={max_bytes} bytes, "
                f"page_bytes={page_bytes}"
            )
        resolved_pages = max_pages_per_rank
        # num_pages already agrees across ranks, but page_bytes is per-rank
        # (num_cache_layers differs on an uneven pp split or the MTP stage), so
        # this conversion can re-diverge whenever the byte budget binds.
        agreed_pages = self._min_across_workers(resolved_pages)
        if agreed_pages < resolved_pages:
            warnings.warn(
                "Lowering L2 cache pages to the cluster-wide minimum so every "
                f"rank agrees: local={resolved_pages}, agreed={agreed_pages}",
                RuntimeWarning,
            )
            resolved_pages = agreed_pages
        if resolved_pages < self.num_pages:
            warnings.warn(
                "L2 cache is smaller than the GPU pool, so it can only hold a "
                "subset of what L1 evicts; raise --l2_cache_gb. "
                f"num_pages={self.num_pages}, resolved_pages_per_rank={resolved_pages}, "
                f"per_rank_budget={max_bytes} bytes, "
                f"page_bytes={page_bytes}",
                RuntimeWarning,
            )
        return resolved_pages

    @staticmethod
    def _min_across_workers(value):
        if not dist.is_available() or not dist.is_initialized():
            return value
        # Imported lazily: pipeswift.utils.mpu pulls in deep_ep/flashinfer, which
        # the standalone unit tests deliberately do without.
        from pipeswift.utils import min_across_workers

        return min_across_workers(value)

    def _resolve_l2_cost_model(self):
        """Per-direction ``{desc_rate, bw}`` the retire deadline is priced from.

        Measured locally, because no input is uniform: page bytes differ with an
        uneven pp split, and D2H and H2D differ on the same link. Then reduced to
        the cluster minimum, because the deadline it feeds has to come out
        identical on every rank (see poll_l2_events) -- and the minimum is the
        right reducer, being the rank the others would otherwise wait for.
        Integers, so the reduction and the arithmetic after it are exact.
        """
        model = self.l2_copy_engine.calibrate()
        return {
            direction: {
                term: max(1, self._min_across_workers(int(value)))
                for term, value in terms.items()
            }
            for direction, terms in model.items()
        }

    # ----------------------------------------------------------------- #
    # L1 page pool
    # ----------------------------------------------------------------- #

    def can_allocate(self, seq: Sequence, reserve_tokens: int = 0):
        ori_num_pages = len(self.kv_page_table.get(seq.seq_id, ()))
        new_num_pages = (
            len(seq.token_ids) + reserve_tokens + self.page_size - 1
        ) // self.page_size
        return self.free_pages() >= max(0, new_num_pages - ori_num_pages)

    def free_pages(self):
        """Pages the pool can still hand out, counting evictable cached ones."""
        return len(self.free_page_queue) + len(self.cached_pages)

    def decode_reserve_pages(self, num_seqs):
        """Pages ``num_seqs`` need to each decode DECODE_RESERVE_TOKENS further.

        Rounded up, because the last of those tokens lands in a partial page the
        sequence still has to own.
        """
        per_seq = (DECODE_RESERVE_TOKENS + self.page_size - 1) // self.page_size
        return num_seqs * per_seq

    def can_allocate_overlap_schedule(self, seqs: List[Sequence]):
        required_pages = 0
        for seq in seqs:
            ori_num_pages = len(self.kv_page_table.get(seq.seq_id, ()))
            new_num_pages = (len(seq.token_ids) + self.page_size) // self.page_size
            required_pages += max(0, new_num_pages - ori_num_pages)
        return len(self.free_page_queue) + len(self.cached_pages) >= required_pages

    def allocate_a_page(
        self,
    ):
        # Pages pinned by an in-flight store are reclaimable, so drain rather than
        # die on a pool that is only nominally full.
        while (
            not self.free_page_queue
            and not self.cached_pages
            and self.reclaim_l2_store_pages()
        ):
            pass
        if len(self.free_page_queue) > 0:
            page_idx = self.free_page_queue.popleft()
        elif self.cached_pages:
            page_idx, _ = self.cached_pages.popitem(last=False)
            self.reset_page(page_idx)
        else:
            raise RuntimeError("Out of KV Pages!")
        assert self.ref_counts[page_idx] == 0
        self.ref_counts[page_idx] = 1
        return page_idx

    def _lookup_cached_page(self, h, token_ids):
        """Return a cached page_idx whose hash and tokens match, else None."""
        if (
            h == -1
            or h not in self.hash2page_idx_list
            or len(self.hash2page_idx_list[h]) == 0
        ):
            return None
        for pidx in self.hash2page_idx_list[h]:
            if self.page_idx2token_ids[pidx] == token_ids:
                return pidx
        return None

    def _publish_gpu_page(self, page_idx, h, token_ids):
        if self.page_idx2hash[page_idx] == -1:
            self.page_idx2hash[page_idx] = h
            self.page_idx2token_ids[page_idx] = token_ids
            self.hash2page_idx_list.setdefault(h, []).append(page_idx)

    def _retain_gpu_page_for_transfer(self, page_idx):
        if self.ref_counts[page_idx] == 0 and page_idx in self.cached_pages:
            self.cached_pages.pop(page_idx)
        self.ref_counts[page_idx] += 1

    def _release_gpu_pages_from_transfer(self, ordered_page_idxs):
        """Drop the transfer's ref on each page.

        ``ordered_page_idxs`` must be ordered by eviction priority, first
        evicted first: a page reaching ref 0 here is appended to the tail of
        ``cached_pages``, so the caller owns the resulting eviction order the
        same way ``deallocate`` does. A page held by an in-flight transfer
        never reaches ref 0 inside ``deallocate``, so its ordering is skipped
        there and has to be applied here instead.
        """
        for page_idx in ordered_page_idxs:
            self.ref_counts[page_idx] -= 1
            assert self.ref_counts[page_idx] >= 0
            if self.ref_counts[page_idx] == 0:
                if self.page_idx2hash[page_idx] == -1:
                    self.free_page_queue.append(page_idx)
                else:
                    self.cached_pages[page_idx] = None

    # ----------------------------------------------------------------- #
    # L2 stores (GPU -> host)
    # ----------------------------------------------------------------- #

    def _stage_l2_store(self, page_idx, h, token_ids):
        """Queue a store for the end-of-step flush instead of launching now.

        Decode completes pages one at a time; launching per page costs one
        driver call per page, exactly where that overhead dominates.
        """
        if not self.enable_l2_cache or self.l2_pool.has_page_or_pending(h, token_ids):
            return False
        key = (h, tuple(token_ids))
        self._l2_pending_store_requests.append((page_idx, h, token_ids, key))
        self.l2_pool.acquire_pending_key(key)
        return True

    def flush_l2_stores(self):
        if not self._l2_pending_store_requests:
            return 0
        staged = self._l2_pending_store_requests
        self._l2_pending_store_requests = []
        stores = []
        for page_idx, h, token_ids, key in staged:
            self.l2_pool.release_pending_key(key)
            # The page can be merged away or recycled between staging and flush.
            if self.page_idx2hash[page_idx] != h:
                continue
            stores.append((page_idx, h, token_ids))
        return self._schedule_l2_store_batch(stores)

    def schedule_l2_stores_for_confirmed_seq(self, seq: Sequence):
        if not self.enable_l2_cache or seq.seq_id not in self.kv_page_table:
            return
        page_table = self.kv_page_table[seq.seq_id]
        confirmed_pages = min(len(seq.token_ids) // self.page_size, len(page_table))
        stores = []
        for page_idx in page_table[:confirmed_pages]:
            h = self.page_idx2hash[page_idx]
            if h == -1:
                continue
            stores.append((page_idx, h, self.page_idx2token_ids[page_idx]))
        self._schedule_l2_store_batch(stores)

    def _schedule_l2_store_batch(self, stores):
        if not self.enable_l2_cache:
            return 0
        gpu_page_idxs, l2_page_idxs, hashes, token_ids_list = [], [], [], []
        for page_idx, h, token_ids in stores:
            if self.l2_pool.has_page_or_pending(h, token_ids):
                continue
            l2_page_idx = self.l2_pool.allocate_for_store()
            if l2_page_idx is None:
                # Every L2 page is pinned in flight; nothing else fits either.
                break
            # Claim the key now so later duplicates inside this same batch are
            # caught by has_page_or_pending above.
            self.l2_pool.acquire_pending_key((h, tuple(token_ids)))
            self._retain_gpu_page_for_transfer(page_idx)
            gpu_page_idxs.append(page_idx)
            l2_page_idxs.append(l2_page_idx)
            hashes.append(h)
            token_ids_list.append(token_ids)

        if not gpu_page_idxs:
            return 0

        num_pages = len(gpu_page_idxs)
        copy = self.l2_copy_engine.store(gpu_page_idxs, l2_page_idxs)
        self.pending_l2_stores.append(
            {
                "copy": copy,
                "due_step": self._l2_retire_step(copy, "store"),
                "gpu_page_idxs": gpu_page_idxs,
                "l2_page_idxs": l2_page_idxs,
                "hashes": hashes,
                "token_ids_list": token_ids_list,
            }
        )
        self.l2_store_cnt += num_pages
        return num_pages

    def _retire_l2_stores(self, n_ready):
        """Retire the leading ``n_ready`` transfers; completion is FIFO."""
        for pending in self.pending_l2_stores[:n_ready]:
            # Normally already landed: the deadline was priced so it would be.
            pending["copy"].wait()
            for l2_page_idx, h, token_ids in zip(
                pending["l2_page_idxs"],
                pending["hashes"],
                pending["token_ids_list"],
            ):
                self.l2_pool.finish_store(l2_page_idx, h, token_ids)
                # Discoverable via the pool's hash index now.
                self.l2_pool.release_pending_key((h, tuple(token_ids)))
            # Suffix pages first, mirroring deallocate.
            self._release_gpu_pages_from_transfer(reversed(pending["gpu_page_idxs"]))
        self.pending_l2_stores = self.pending_l2_stores[n_ready:]

    # ----------------------------------------------------------------- #
    # L2 loads (host -> GPU)
    # ----------------------------------------------------------------- #

    def _stage_l2_loads(self, seq: Sequence, loads):
        """Queue a restore for the batched flush instead of launching it now.

        One transfer per sequence meant one sequence restored per decode window:
        the scheduler only ever inspects the FIFO head, and the head's pending
        load stops it looking further. Batching does not help the copy engine
        itself -- pages come off a shared free queue, so they never merge into
        larger descriptors -- it only amortises the per-transfer driver call.
        """
        seq.l2_load_pending = True
        self._l2_pending_load_requests.append((seq, loads))

    def flush_l2_loads(self):
        """Send the staged restores, split into transfers a decode step can hide.

        One transfer per wave retired as one unit: the largest measured was 20806
        pages / 8.2 GB / ~1 s of link time, priced to hide inside a 320 ms window,
        and the wave that needed its first sequence waited on its last. Each group
        now gets its own deadline off the same serial water level, so the queue
        drains front to back. Groups split at whole sequences only, because
        ``_retire_l2_loads`` clears ``l2_load_pending`` and advances ``seq.offset``
        per sequence -- a half-retired one would be prefilled over pages still in
        flight -- so a sequence over budget goes alone and overshoots.
        """
        if not self._l2_pending_load_requests:
            return 0
        staged = self._l2_pending_load_requests
        self._l2_pending_load_requests = []
        total = 0
        for group in self._split_l2_loads(staged):
            l2_page_idxs, gpu_page_idxs = [], []
            for _, loads in group:
                for load in loads:
                    self._retain_gpu_page_for_transfer(load["gpu_page_idx"])
                    self.l2_pool.retain_for_load(load["l2_page_idx"])
                    l2_page_idxs.append(load["l2_page_idx"])
                    gpu_page_idxs.append(load["gpu_page_idx"])
            copy = self.l2_copy_engine.load(l2_page_idxs, gpu_page_idxs)
            self.pending_l2_loads.append(
                {
                    "copy": copy,
                    "due_step": self._l2_retire_step(copy, "load"),
                    "seq_loads": group,
                }
            )
            total += len(gpu_page_idxs)
        self.l2_load_cnt += total
        return total

    def _split_l2_loads(self, staged):
        """Group staged restores into transfers of about one scheduler step.

        Priced off the agreed per-page cost, so every rank cuts the batch in the
        same places and commits the same page table.
        """
        budget = max(1, int((self.l2_step_ms / 1000) / self.l2_page_s))
        groups, cur, acc = [], [], 0
        for entry in staged:
            n = len(entry[1])
            if cur and acc + n > budget:
                groups.append(cur)
                cur, acc = [], 0
            cur.append(entry)
            acc += n
        if cur:
            groups.append(cur)
        return groups

    def _retire_l2_loads(self, n_ready):
        """Retire the leading ``n_ready`` transfers; completion is FIFO."""
        for pending in self.pending_l2_loads[:n_ready]:
            pending["copy"].wait()
            for seq, loads in pending["seq_loads"]:
                seq_alive = seq.seq_id in self.kv_page_table
                for load in loads:
                    self._publish_gpu_page(
                        load["gpu_page_idx"],
                        load["hash"],
                        load["token_ids"],
                    )
                    self.l2_pool.release_from_load(load["l2_page_idx"])
                # Only reaches ref 0 when the seq died mid-load; suffix pages
                # first.
                self._release_gpu_pages_from_transfer(
                    [load["gpu_page_idx"] for load in reversed(loads)]
                )
                if seq_alive:
                    seq.offset += len(loads) * self.page_size
                seq.l2_load_pending = False
        self.pending_l2_loads = self.pending_l2_loads[n_ready:]

    def reclaim_l2_store_pages(self):
        """Retire the oldest in-flight store early; True if there was one.

        A store pins the GPU pages it reads, so preempting a sequence whose pages
        are pinned frees nothing. Retiring ahead of the deadline is safe (it is a
        must-by, not an earliest) and the trigger is replicated page state, so
        ranks still drain in step.
        """
        if not self.pending_l2_stores:
            return False
        self._retire_l2_stores(1)
        return True

    def poll_l2_events(self, step_idx):
        if not self.enable_l2_cache:
            return
        self.l2_step = step_idx + 1
        if not self.pending_l2_stores and not self.pending_l2_loads:
            return
        self._retire_l2_stores(self._l2_num_due(self.pending_l2_stores))
        self._retire_l2_loads(self._l2_num_due(self.pending_l2_loads))

    def _l2_retire_step(self, copy, direction):
        """Step by which ``copy`` is contracted to be done.

        The engine drains a direction serially, so a transfer lands only after
        everything queued ahead of it: keep a per-direction water level of
        link-seconds owed and convert it to steps, ``max``'d with the current step
        to write off backlog the engine has already had time to finish. Two-term,
        because page count alone does not predict a transfer (see
        CopyEngine.calibrate). Every input is replicated -- rates and geometry
        min-reduced at init, run and page counts off replicated page tables -- so
        every rank predicts the same step without asking anyone.
        """
        model = self.l2_cost_model[direction]
        cost_s = (
            copy.num_runs * self.l2_views / model["desc_rate"]
            + copy.num_pages * self.l2_page_bytes / model["bw"]
        )
        step_s = self.l2_step_ms / 1000
        owed_s = max(self._l2_owed_s[direction], self.l2_step * step_s) + cost_s
        self._l2_owed_s[direction] = owed_s
        return int(math.ceil(owed_s / step_s))

    def _l2_num_due(self, pendings):
        """Leading transfers whose contracted step has arrived; deadlines
        increase along the list, so the due ones are always a prefix."""
        for i, pending in enumerate(pendings):
            if pending["due_step"] > self.l2_step:
                return i
        return len(pendings)

    @property
    def l2_page_s(self):
        """Link-seconds for one restored page, priced fully scattered.

        One descriptor per page per view is what indices coming off two churned
        free lists cost in practice, and pricing a prefetch budget optimistically
        is what lets the restore queue outgrow the window it hides in.
        """
        model = self.l2_cost_model["load"]
        return self.l2_views / model["desc_rate"] + self.l2_page_bytes / model["bw"]

    def l2_load_horizon_steps(self):
        """Steps until a restore staged now would be due to have landed.

        The load water level the deadline is priced from, less the steps already
        elapsed against it, plus what is staged but not yet flushed -- i.e. would
        one more restore still land inside the decode window it hides in, in the
        units a transfer costs rather than in pages, which do not predict it.
        Replicated inputs, so every rank admits the same sequences.
        """
        if not self.enable_l2_cache:
            return 0
        pages = sum(len(loads) for _, loads in self._l2_pending_load_requests)
        step_s = self.l2_step_ms / 1000
        owed_s = max(self._l2_owed_s["load"] - self.l2_step * step_s, 0.0)
        return int(math.ceil((owed_s + pages * self.l2_page_s) / step_s))

    # ----------------------------------------------------------------- #
    # Sequence-level allocation
    # ----------------------------------------------------------------- #

    def _prompt_page_hashes(self, seq: Sequence):
        """Per-page prefix-chained ``(hash, token_ids)`` for the whole prompt.

        Cached on the sequence because ``probe_l2_prefetch`` re-reads it for
        every waiting sequence on every step. Keyed by length, since a preempted
        sequence comes back to prefill with its generated tokens appended.
        """
        if seq.page_hashes is not None and seq.page_hashes[0] == len(seq.token_ids):
            return seq.page_hashes[1]
        ps = self.page_size
        hashes, h = [], -1
        for i in range((len(seq.token_ids) + ps - 1) // ps):
            token_ids = seq.token_ids[ps * i : ps * (i + 1)]
            h = self.compute_hash(token_ids, h) if len(token_ids) == ps else -1
            hashes.append((h, token_ids))
        seq.page_hashes = (len(seq.token_ids), hashes)
        return hashes

    def probe_l2_prefetch(self, seq: Sequence):
        """True when L2, and only L2, can supply part of ``seq``'s prompt.

        Read-only. Prefetching a sequence commits its entire prompt page table,
        so the scheduler asks first whether there is a restore worth paying for.
        Mirrors ``allocate_prefill``: the last page is never reused, and the L2
        scan begins where the contiguous L1 match stops.

        The scan resumes from where the last probe of this sequence stopped.
        Restarting it from page 0 every step was the dominant host cost of the
        scheduler under multi-turn traffic: the L1 prefix runs almost to the end
        of the prompt, so every waiting sequence walked its whole page table --
        hundreds of dict lookups plus a 16-token list compare each -- on every
        step, for as many steps as it waited. Resuming is safe because this is
        only a hint: ``allocate_prefill`` re-derives the real match from page 0,
        so a resume point stale by an eviction below it costs at most a missed
        prefetch, never a wrong page table. Every input is replicated, so the
        resume point is identical on every rank.
        """
        if not self.enable_l2_cache:
            return False
        hashes = self._prompt_page_hashes(seq)
        start = 0
        if seq.l2_probe_resume is not None and seq.l2_probe_resume[0] == len(
            seq.token_ids
        ):
            start = seq.l2_probe_resume[1]
        for i in range(start, len(hashes) - 1):
            if self._lookup_cached_page(*hashes[i]) is None:
                seq.l2_probe_resume = (len(seq.token_ids), i)
                return self.l2_pool.lookup(*hashes[i]) is not None
        seq.l2_probe_resume = (len(seq.token_ids), max(0, len(hashes) - 1))
        return False

    def allocate_prefill(self, seq: Sequence):
        ps = self.page_size
        page_num = (len(seq.token_ids) + ps - 1) // ps
        hashes = self._prompt_page_hashes(seq)

        # Contiguous leading KV-cache match. The final page is never reused
        # (a seq must run >=1 token to produce a logit), so iterate up to
        # page_num - 1.
        matched_idxs = []
        for i in range(page_num - 1):
            pidx = self._lookup_cached_page(hashes[i][0], hashes[i][1])
            if pidx is None:
                break
            matched_idxs.append(pidx)
        matched_pages = len(matched_idxs)

        l2_loads = []
        l2_matched_pages = 0
        if self.enable_l2_cache:
            for i in range(matched_pages, page_num - 1):
                l2_pidx = self.l2_pool.lookup(hashes[i][0], hashes[i][1])
                if l2_pidx is None:
                    break
                gpu_pidx = self.allocate_a_page()
                l2_loads.append(
                    {
                        "gpu_page_idx": gpu_pidx,
                        "l2_page_idx": l2_pidx,
                        "hash": hashes[i][0],
                        "token_ids": hashes[i][1],
                    }
                )
            l2_matched_pages = len(l2_loads)

        # Build the page table: reuse L1 pages, load L2 pages, allocate fresh
        # pages for the rest. L2-loaded pages are not lookup-visible in L1 until
        # their H2D event completes.
        table = array("i")
        for k in range(matched_pages):
            pidx = matched_idxs[k]
            self.hit_cnt += 1
            self.ref_counts[pidx] += 1
            if self.ref_counts[pidx] == 1:
                self.cached_pages.pop(pidx)
            table.append(pidx)
        seq.offset += matched_pages * ps
        for load in l2_loads:
            self.l2_hit_cnt += 1
            table.append(load["gpu_page_idx"])
        for i in range(matched_pages + l2_matched_pages, page_num):
            hi, token_ids = hashes[i]
            pidx = self.allocate_a_page()
            if hi != -1 and i != (page_num - 1):  # last page do not update
                self.update_page(pidx, hi, token_ids)
            table.append(pidx)
        self.kv_page_table[seq.seq_id] = table
        if l2_loads:
            self._stage_l2_loads(seq, l2_loads)

    def deallocate(self, seq: Sequence):  # TODO
        if seq.seq_id in self.kv_page_table.keys():
            page_ids = self.kv_page_table[seq.seq_id]
            for page_idx in reversed(page_ids):
                self.ref_counts[page_idx] -= 1
                if self.ref_counts[page_idx] == 0:
                    if self.page_idx2hash[page_idx] == -1:
                        self.free_page_queue.append(page_idx)
                    else:
                        self.cached_pages[page_idx] = None
            del self.kv_page_table[seq.seq_id]

    def reset_page(self, page_idx):
        assert self.ref_counts[page_idx] == 0
        if self.page_idx2hash[page_idx] != -1:
            hash = self.page_idx2hash[page_idx]
            if hash in self.hash2page_idx_list.keys():
                i = 0
                while i < len(self.hash2page_idx_list[hash]):
                    if self.hash2page_idx_list[hash][i] == page_idx:
                        del self.hash2page_idx_list[hash][i]
                        break
                    i += 1
                if len(self.hash2page_idx_list[hash]) == 0:
                    del self.hash2page_idx_list[hash]
        self.page_idx2hash[page_idx] = -1
        self.page_idx2token_ids[page_idx] = []

    def update_page(self, page_idx, h, token_ids, from_decode=False):
        assert (
            h != -1
            and self.page_idx2hash[page_idx] == -1
            and self.ref_counts[page_idx] == 1
        )
        if h not in self.hash2page_idx_list.keys():
            self.hash2page_idx_list[h] = [page_idx]
        else:
            self.hash2page_idx_list[h].append(page_idx)
        self.page_idx2token_ids[page_idx] = token_ids
        self.page_idx2hash[page_idx] = h
        # Prompt pages go to L2 via schedule_l2_stores_for_confirmed_seq; only
        # pages completed during decode are staged from here. Decode pages are
        # offloaded too, unlike vLLM: multi-turn agent traffic reuses generated
        # tokens as the next turn's prefix, so they do get hits.
        if from_decode and self.enable_l2_cache:
            self._stage_l2_store(page_idx, h, token_ids)

    def may_allocate_decode(self, seq: Sequence, reserve_tokens: int = 0):
        # NOTE(wsj) The last token of a seq is a fake token, so we don't consider page merging
        assert not seq.finished
        if not seq.finished:
            seq_page_table = self.kv_page_table[seq.seq_id]
            while (
                len(seq_page_table) * self.page_size
                < len(seq.token_ids) + reserve_tokens
            ):
                page_idx = self.allocate_a_page()
                seq_page_table.append(page_idx)

    def update_page_status_and_merge_possible(self, seq: Sequence):
        seq_page_table = self.kv_page_table[seq.seq_id]
        if len(seq.token_ids) % self.page_size == 0:
            check_page_table_idx = len(seq.token_ids) // self.page_size - 1
            check_page_idx = seq_page_table[check_page_table_idx]
            token_ids = seq.token_ids[-self.page_size :]
            if len(seq.token_ids) // self.page_size >= 2:
                prefix = self.page_idx2hash[
                    seq_page_table[len(seq.token_ids) // self.page_size - 2]
                ]
            else:
                prefix = -1
            h = self.compute_hash(token_ids, prefix)
            _may_be_another_page = -1
            if h in self.hash2page_idx_list.keys():
                page_idx_list = self.hash2page_idx_list[h]
                for _page_idx in page_idx_list:
                    if self.page_idx2token_ids[_page_idx] == token_ids:
                        self.hit_cnt += 1
                        _may_be_another_page = _page_idx
                        self.ref_counts[_may_be_another_page] += 1
                        if self.ref_counts[_may_be_another_page] == 1:
                            assert _may_be_another_page in self.cached_pages
                            self.cached_pages.pop(_may_be_another_page)
                        break

            if _may_be_another_page != -1:
                self.ref_counts[check_page_idx] -= 1
                if self.ref_counts[check_page_idx] == 0:
                    # Mirror deallocate: a hashed page must be retired via
                    # reset_page before returning to free_page_queue.
                    if self.page_idx2hash[check_page_idx] != -1:
                        self.reset_page(check_page_idx)
                    self.free_page_queue.append(check_page_idx)
                seq_page_table[check_page_table_idx] = _may_be_another_page
            elif h != -1:
                self.update_page(check_page_idx, h, token_ids, from_decode=True)

    def get_usage(self):
        return 1 - (len(self.free_page_queue) + len(self.cached_pages)) / self.num_pages

    def get_l2_usage(self):
        if not self.enable_l2_cache:
            return 0.0
        return self.l2_pool.get_usage()
