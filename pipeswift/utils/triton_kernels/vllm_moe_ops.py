"""
Wrapper for compiled pipeswift MoE CUDA kernels.
This module keeps only the routing-related wrappers.
"""

from typing import Optional

import torch

# Try to import the compiled CUDA extension
try:
    import pipeswift_kernels

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    pipeswift_kernels = None


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-k softmax for MoE gating.

    Args:
        gating_output: Gating logits, shape [num_tokens, num_experts]
        topk: Number of experts to select
        renormalize: Whether to renormalize the top-k probabilities
        bias: Optional bias to add for expert selection, shape [num_experts]

    Returns:
        topk_weights: Top-k probabilities, shape [num_tokens, topk]
        topk_indices: Top-k expert indices, shape [num_tokens, topk]
        token_expert_indices: Same as topk_indices, shape [num_tokens, topk]
    """
    if not KERNELS_AVAILABLE:
        raise RuntimeError(
            "CUDA kernels not available. Please compile the extension by running: "
            "pip install -e . --no-build-isolation"
        )

    num_tokens, _ = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    if scoring_func == "softmax":
        pipeswift_kernels.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            bias,
            topk,
            renormalize,
        )
    elif scoring_func == "sigmoid":
        pipeswift_kernels.topk_sigmoid(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            bias,
            topk,
            renormalize,
        )
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    return topk_weights, topk_indices


def fused_grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply grouped topk routing to select experts.

    Args:
        gating_output: Gating logits, shape [num_tokens, num_experts]
        topk: Total number of experts to select per token
        renormalize: Whether to renormalize the top-k probabilities
        e_score_correction_bias: Bias for expert selection, shape [num_experts]
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select from
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        routed_scaling_factor: Scaling factor for routed weights

    Returns:
        topk_values: Top-k weights, shape [num_tokens, topk]
        topk_indices: Top-k expert indices, shape [num_tokens, topk]
    """
    if not KERNELS_AVAILABLE:
        raise RuntimeError(
            "CUDA kernels not available. Please compile the extension by running: "
            "pip install -e . --no-build-isolation"
        )

    if scoring_func == "sigmoid":
        topk_values, topk_indices = pipeswift_kernels.grouped_topk(
            gating_output,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            1,
        )
    elif scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
        topk_values, topk_indices = pipeswift_kernels.grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            0,
        )
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    return topk_values, topk_indices
