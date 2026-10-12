"""Coalesced parameter loading helper for TP / ETP weight broadcast."""

from typing import List

import torch
import torch.distributed as dist


def coalesced_load_params(
    param_strs,
    safetensor_dict,
    safetensor_index,
    tensors2be_received: List[torch.Tensor] = None,
    is_tp=True,
):
    from .mpu import parallel_state

    if is_tp:
        _bcast_ranks = dist.get_process_group_ranks(parallel_state.tp_group)
    else:  # etp
        _bcast_ranks = dist.get_process_group_ranks(parallel_state.etp_group)

    _node_ranks = list(
        range(
            parallel_state.global_rank
            // parallel_state.num_gpus_per_node
            * parallel_state.num_gpus_per_node,
            (parallel_state.global_rank // parallel_state.num_gpus_per_node + 1)
            * parallel_state.num_gpus_per_node,
        )
    )
    bcast_ranks = [r for r in _bcast_ranks if r in _node_ranks]
    from torch.distributed import P2POp

    if parallel_state.global_rank == bcast_ranks[0]:
        for _param_idx, _param_str in enumerate(param_strs):
            _loaded_tensor = safetensor_dict[safetensor_index[_param_str]][_param_str]
            tensors2be_received[_param_idx].copy_(_loaded_tensor)
            if len(bcast_ranks) > 1:
                _p2p_ops = []
                for _rank in bcast_ranks[1:]:
                    _p2p_ops.append(
                        P2POp(
                            torch.distributed.isend,
                            tensors2be_received[_param_idx],
                            _rank,
                        )
                    )
                torch.distributed.batch_isend_irecv(_p2p_ops)
    else:
        for _recv_tensor in tensors2be_received:
            _reqs = torch.distributed.batch_isend_irecv(
                [P2POp(torch.distributed.irecv, _recv_tensor, bcast_ranks[0])]
            )
            _reqs[0].wait()
