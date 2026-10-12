"""ModuleMTP: shared MTP (multi-token prediction) speculative-decoding mixin.

Provides draft generation, target verification and the MTP draft-decode CUDA
graph capture. Any model that supports MTP inherits this BEFORE its model base
(e.g. ``class DeepSeekV3(ModuleMTP, Llama2)``), so ``capture_cudagraph``'s
``super()`` reaches the model's base capture. The host class must provide
``self.mtp`` (a ParallelMTP-like head), ``self.num_spec_tokens``,
``self.lm_head``, ``self.forward_decode``/``forward_sampler``, and the usual
LLM graph/state attributes.
"""

from contextlib import nullcontext
from typing import List

import flashinfer
import torch
from tqdm import tqdm

from pipeswift.utils import (
    KVCacheInfo,
    coalesced_to_cuda,
    derive_kvcache_info_next_decode_step,
    freeze_gc,
    prepare_kv_info,
    set_graph_kvcacheinfo,
    slice_kvcache_info,
)


class ModuleMTP:
    def _snapshot_seq_states(self, seqs):
        return [(seq, seq.offset, len(seq.token_ids), seq.finished) for seq in seqs]

    def _restore_seq_states(self, seq_states):
        for seq, offset, token_len, finished in seq_states:
            seq.offset = offset
            seq.finished = finished
            if len(seq.token_ids) > token_len:
                del seq.token_ids[token_len:]

    def _cache_mtp_draft(self, seqs, draft_tokens, draft_probs):
        for i, seq in enumerate(seqs):
            seq.spec_draft_token_ids = (
                draft_tokens[i] if draft_tokens is not None else None
            )
            seq.spec_draft_probs = draft_probs[i] if draft_probs is not None else None

    def _load_mtp_draft(self, seqs):
        if seqs[0].spec_draft_token_ids is None:
            draft_tokens = None
        else:
            draft_tokens = torch.stack(
                [seq.spec_draft_token_ids for seq in seqs], dim=0
            )

        if seqs[0].spec_draft_probs is None:
            draft_probs = None
        else:
            draft_probs = torch.stack([seq.spec_draft_probs for seq in seqs], dim=0)
        draft_position_ids = sum(
            (
                list(range(seq.offset, seq.offset + self.num_spec_tokens + 1))
                for seq in seqs
            ),
            start=[],
        )
        draft_position_ids = coalesced_to_cuda(arr_list=[draft_position_ids])[0].view(
            len(seqs), self.num_spec_tokens + 1
        )
        return draft_tokens, draft_probs, draft_position_ids

    def _clear_mtp_draft(self, seqs):
        for seq in seqs:
            seq.spec_draft_token_ids = None
            seq.spec_draft_probs = None

    def draft_sample(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if self.args.spec_rejection_mode == "strict":
            draft_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            draft_probs = None
        else:
            logits = logits.to(torch.float32)
            logits[:] = flashinfer.softmax(logits, self.sampling_params.temperature)
            draft_probs = logits
            draft_tokens = flashinfer.sampling.sampling_from_probs(
                draft_probs, deterministic=True
            )
        return draft_tokens, draft_probs

    def capture_cudagraph(self, kv_cache, mla_cache_tensor):
        # Base (LLM) graph capture, then the MTP draft-decode graphs.
        # ModuleMTP precedes the model base in the MRO, so ``super()`` resolves
        # correctly for DeepSeekV3 (->Llama2->LLM).
        super().capture_cudagraph(kv_cache, mla_cache_tensor)
        from pipeswift.utils import parallel_state

        if self.num_spec_tokens > 0 and parallel_state.pp_next_rank is None:
            with torch.no_grad():
                max_bsz = self._MAX_GRAPH_BSZ_DEFAULT
                # for mtp_decode
                self.mtp_static_input_ids = torch.zeros(
                    max_bsz, dtype=torch.int32, device="cuda"
                )
                self.mtp_static_pos_ids = torch.zeros(
                    max_bsz, dtype=torch.int32, device="cuda"
                )
                self.mtp_static_hidden_states = torch.zeros(
                    (max_bsz, self.hidden_size), dtype=self.args.dtype, device="cuda"
                )
                # self.mtp_decode_graph_pool = None
                self.mtp_decode_graphs = {}

                show_progress = parallel_state.global_rank == 0
                maybe_deepep_ctx = (
                    parallel_state.deepep.capture()
                    if parallel_state.deepep is not None
                    else nullcontext()
                )

                _iter = (
                    self.graph_bs.get(1, [])
                    if not show_progress
                    else tqdm(
                        self.graph_bs.get(1, []),
                        desc="Capturing MTP Decode CUDA Graphs",
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
                    for bsz in _iter:
                        if show_progress:
                            _iter.set_postfix({"q_len": 1, "bsz": bsz}, refresh=False)
                        if (
                            parallel_state.deepep is not None
                            and not self.args.disable_deepep_ll
                        ):
                            parallel_state.deepep.set_buffer_low_latency_mode()
                        self.static_kvcahceinfo.actual_lens_q.fill_(1)
                        self.static_kvcahceinfo.actual_lens_q_cpu.fill_(1)
                        self.static_kvcahceinfo.cu_seqlens_q[: (bsz + 1)].copy_(
                            torch.arange(bsz + 1, dtype=torch.int32, device="cuda")
                        )
                        graph = torch.cuda.CUDAGraph()
                        _static_kvcacheinfo = slice_kvcache_info(
                            self.static_kvcahceinfo, bsz
                        )
                        self.plan_attn(_static_kvcacheinfo, is_decode=True, cg_bsz=bsz)
                        self.mtp.forward(
                            self.mtp_static_hidden_states[:bsz],
                            self.mtp_static_input_ids[:bsz],
                            self.mtp_static_pos_ids[:bsz],
                            _static_kvcacheinfo,
                            is_decode=True,
                        )  # warmup
                        assert self.graph_pool is not None, (
                            "Reuse of Memory Pool is Necessary"
                        )
                        with torch.cuda.graph(graph, self.graph_pool):
                            self.static_output_tensor[:bsz] = self.mtp.forward(
                                self.mtp_static_hidden_states[:bsz],
                                self.mtp_static_input_ids[:bsz],
                                self.mtp_static_pos_ids[:bsz],
                                _static_kvcacheinfo,
                                is_decode=True,
                            )  # capture

                        # if self.mtp_decode_graph_pool is None:
                        #     self.mtp_decode_graph_pool = graph.pool()
                        self.mtp_decode_graphs[bsz] = graph
                        torch.cuda.synchronize()

    def _forward_mtp(
        self,
        input_ids,
        aux_hiddens,
        pos_ids,
        kvcache_info: KVCacheInfo,
        is_decode: bool,
    ):
        from pipeswift.utils import parallel_state

        q_len = 1
        if is_decode:
            assert kvcache_info.actual_lens_q_cpu is not None
            q_len = int(kvcache_info.actual_lens_q_cpu.max().item())
        if input_ids.shape[0] > 0:
            # Only use cuda-graph for mtp's decode with q_len 1
            decode_graph_bs = (
                self.graph_bs.get(1, []) if self.graph_bs is not None else []
            )
            cg_bsz = (
                min(x for x in decode_graph_bs if x >= input_ids.shape[0])
                if is_decode
                and q_len == 1
                and len(decode_graph_bs) > 0
                and decode_graph_bs[-1] >= input_ids.shape[0]
                else -1
            )
            self.plan_attn(kvcache_info, is_decode=is_decode, cg_bsz=cg_bsz)
        else:
            cg_bsz = -1

        if (
            not is_decode
            or q_len > 1
            or self.args.enforce_eager
            or len(self.graph_bs.get(1, [])) == 0
            or input_ids.shape[0] > self.graph_bs[1][-1]
        ):
            hidden_states = self.mtp.forward(
                aux_hiddens, input_ids, pos_ids, kvcache_info, is_decode
            )
        else:
            bsz = input_ids.shape[0]
            self.mtp_static_hidden_states[:bsz].copy_(aux_hiddens, non_blocking=True)
            self.mtp_static_input_ids[:bsz].copy_(input_ids, non_blocking=True)
            self.mtp_static_pos_ids[:bsz].copy_(pos_ids, non_blocking=True)

            set_graph_kvcacheinfo(self.static_kvcahceinfo, kvcache_info, cg_bsz)
            graph = self.mtp_decode_graphs[cg_bsz]
            if parallel_state.deepep is not None and not self.args.disable_deepep_ll:
                parallel_state.deepep.set_buffer_low_latency_mode()
            graph.replay()
            hidden_states = self.static_output_tensor[:bsz]

        return hidden_states

    def forward_mtp_draft(
        self,
        seqs,
        block_manager,
        accepted_cnt_list: List[int],
        input_token_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ):
        seq_states = self._snapshot_seq_states(seqs)

        draft_tokens = [input_token_ids.view(len(seqs), 1)]
        draft_probs = []

        token_ids, _draft_probs = self.draft_sample(last_hidden_states)
        draft_tokens.append(token_ids.unsqueeze(1))
        if _draft_probs is not None:
            draft_probs.append(_draft_probs.unsqueeze(1))

        for seq, accepted_cnt in zip(seqs, accepted_cnt_list):
            # seq.token_ids.extend([0] * (accepted_cnt + 1))
            seq.token_ids.extend([0] * accepted_cnt)
            seq.offset = len(seq.token_ids) - 1

        position_ids = coalesced_to_cuda(arr_list=[[seq.offset for seq in seqs]])[0]

        kv_cache_info: KVCacheInfo = prepare_kv_info(
            seqs,
            block_manager.kv_page_table,
            block_manager.page_size,
            block_manager.kv_cache,
            block_manager.mla_cache_tensor,
            dcp_interleave_size=self.args.dcp_interleave_size,
            is_decode=True,
            attention_backend=self.args.attention_backend,
        )
        assert kv_cache_info.actual_lens_q_cpu is not None
        total_q = int(kv_cache_info.actual_lens_q_cpu.sum().item())
        assert total_q == len(seqs), (
            f"Initial MTP draft query mismatch: total_q={total_q}, batch={len(seqs)}"
        )

        for i in range(self.num_spec_tokens - 1):
            last_hidden_states = self._forward_mtp(
                token_ids,
                last_hidden_states,
                position_ids,
                kv_cache_info,
                is_decode=True,
            )
            token_ids, _draft_probs = self.draft_sample(last_hidden_states)
            position_ids += 1
            draft_tokens.append(token_ids.unsqueeze(1))  # [B, 1]
            if _draft_probs is not None:
                draft_probs.append(_draft_probs.unsqueeze(1))  # [B, 1, D]
            for seq in seqs:
                seq.token_ids.append(0)
                seq.offset += 1

            if i < self.num_spec_tokens - 2:
                kv_cache_info = derive_kvcache_info_next_decode_step(
                    kv_cache_info,
                    seqs,
                    block_manager.kv_page_table,
                    block_manager.page_size,
                    dcp_interleave_size=self.args.dcp_interleave_size,
                )

        draft_tokens = torch.concat(draft_tokens, dim=1)
        if len(draft_probs) > 0:
            draft_probs = torch.concat(draft_probs, dim=1)
        else:
            draft_probs = None
        self._restore_seq_states(seq_states)
        self._cache_mtp_draft(seqs, draft_tokens, draft_probs)
        return draft_tokens, draft_probs

    def forward_mtp_verify(
        self,
        seqs,
        input_tensor,
        position_ids,
        draft_probs,
        block_manager,
        draft_token_ids=None,
    ):
        import torch.distributed as dist

        from pipeswift.utils import parallel_state

        # Drain the previous PP activation send before this verify forward can
        # overwrite/free its buffer (shares the single slot with LLM.forward).
        if self._pp_send_work is not None:
            self._pp_send_work.wait()
            self._pp_send_work = None
            self._pp_send_buf = None
        batch_size = len(seqs)
        # input_tensor = input_tensor.view(batch_size, -1).contiguous()
        position_ids = position_ids.view(batch_size, -1).contiguous()
        flat_position_ids = position_ids.reshape(-1)

        seq_states = self._snapshot_seq_states(seqs)
        for seq in seqs:
            for _ in range(self.num_spec_tokens):
                seq.token_ids.append(0)

        kv_cache_info = prepare_kv_info(
            seqs,
            block_manager.kv_page_table,
            block_manager.page_size,
            block_manager.kv_cache,
            block_manager.mla_cache_tensor,
            dcp_interleave_size=self.args.dcp_interleave_size,
            is_decode=True,
            attention_backend=self.args.attention_backend,
        )
        target_hidden = self.forward_decode(
            input_tensor, flat_position_ids, kv_cache_info
        )
        self._restore_seq_states(seq_states)

        # send output to pp_next_rank
        if parallel_state.pp_next_rank is not None:
            self._pp_send_work = dist.isend(
                target_hidden, dst=parallel_state.pp_next_rank
            )  # async send op
            self._pp_send_buf = target_hidden  # keep alive until next forward
            return None, None, None

        target_logits: torch.Tensor = self.lm_head(target_hidden)

        draft_token_ids = draft_token_ids[:, 1:].contiguous().to(torch.int32)
        if self.args.spec_rejection_mode == "strict":
            out_token_ids = self.forward_sampler(target_logits).view(batch_size, -1)
            prefix_match = out_token_ids[:, : self.num_spec_tokens] == draft_token_ids
            num_emitted_draft_tokens = (
                prefix_match.to(torch.int32)
                .cumprod(dim=1, dtype=torch.int32)
                .sum(dim=1, dtype=torch.int32)
            )
        else:
            target_logits = target_logits.to(torch.float32)
            temperature = float(self.sampling_params.temperature)
            assert temperature > 0, (
                f"Invalid temperature for probabilistic spec mode: {temperature}"
            )
            target_logits[:] = flashinfer.softmax(target_logits, temperature)
            target_probs = target_logits
            del target_logits
            vocab_size = self.args.hf_config["vocab_size"]
            top_k = int(self.sampling_params.top_k)
            top_p = float(self.sampling_params.top_p)
            assert top_k >= 0, f"Invalid top_k: {top_k}"
            assert 0 < top_p <= 1, f"Invalid top_p: {top_p}"
            renorm_topk = 0 < top_k < vocab_size
            renorm_topp = top_p < 1
            if renorm_topk:
                target_probs = flashinfer.top_k_renorm_probs(target_probs, top_k)
            if renorm_topp:
                target_probs = flashinfer.top_p_renorm_probs(target_probs, top_p)
            target_probs = target_probs.view(batch_size, -1, target_probs.shape[-1])
            assert draft_probs is not None

            draft_probs = draft_probs.contiguous()
            draft_token_ids = draft_token_ids.contiguous()
            target_probs = target_probs.contiguous()

            out_token_ids, _, num_emitted_draft_tokens = (
                flashinfer.chain_speculative_sampling(
                    draft_probs,
                    draft_token_ids,
                    target_probs,
                )
            )

        target_hidden = target_hidden.view(batch_size, -1, target_hidden.shape[-1])
        extend_aux_hidden = target_hidden
        return (
            out_token_ids,
            num_emitted_draft_tokens,
            extend_aux_hidden,
        )
