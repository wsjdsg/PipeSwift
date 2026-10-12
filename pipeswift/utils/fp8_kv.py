"""FP8 (e4m3) KV cache, aligned with vLLM's ``--kv-cache-dtype fp8``.

Scope: e4m3 only, per-tensor scale fixed at 1.0 (uncalibrated, as vLLM defaults),
GQA/MHA only (MLA is rejected in validate_args), SM90+.

The kernel is vLLM's FlashAttention fork, not the upstream ``flash_attn_3`` wheel
PipeSwift uses for bf16: upstream lacks the SM90 two-level FP8 accumulation fix
(vllm-project/flash-attention#104). Hopper's FP8 TensorCore accumulator keeps a
reduced mantissa in practice, so ``Softmax(QK)@V`` drifts low as the context
grows -- measured on an H100 against an exact reference, -1% at 1k, -25% at 16k,
-67% at 128k, versus 0.00% for the fork. Automatic split-k only softens it
(-7.5% at 128k) and prefill does not split at all, so the kernel fix is
load-bearing. ``scripts/check_fp8_kv_backend.py`` is that measurement as a gate.

FA3 runs both GEMMs in FP8, so the query is quantized to e4m3 as well. That is
not optional and it is what vLLM does (see ``Attention.forward``, which builds a
per-tensor ``QuantFP8`` whenever ``kv_cache_dtype`` starts with ``fp8``).
"""

from __future__ import annotations

import torch

FP8_KV_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0

# Set once from args by LLM.__init__, rather than threaded through every
# transformer layer -- matching how PipeSwift already carries parallel_state and
# the flashinfer workspace.
_ENABLED = False
_varlen_func = None
_scale_one: torch.Tensor | None = None


def set_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def is_enabled() -> bool:
    return _ENABLED


def attn_varlen_func():
    """vLLM's FA3 varlen entry point, imported lazily with a pointed error."""
    global _varlen_func
    if _varlen_func is not None:
        return _varlen_func
    try:
        from vllm_flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "--enable_fp8_kv needs vLLM's FlashAttention fork. Install it with:\n"
            "    bash scripts/install_fp8_kv_backend.sh\n"
            "(scripts/build.sh runs this too.)\n\n"
            "The upstream flash_attn_3 wheel cannot stand in for it: it is "
            "missing the SM90 two-level FP8 accumulation fix "
            "(vllm-project/flash-attention#104) and drifts up to 67% low at "
            "128k context."
        ) from exc
    _varlen_func = flash_attn_varlen_func
    return _varlen_func


def check_available() -> None:
    """Fail at startup rather than in the first forward pass."""
    major, _ = torch.cuda.get_device_capability()
    assert major >= 9, (
        f"--enable_fp8_kv requires SM90+ (Hopper or newer), got SM{major}x; "
        "FA3's FP8 path does not exist on earlier architectures."
    )
    attn_varlen_func()


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor e4m3 cast at scale 1.0, saturating like vLLM's scaled_convert."""
    return x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_KV_DTYPE)


def warmup(device) -> None:
    """Allocate the descale constant outside any CUDA graph capture.

    Called from LLM.__init__. Allocating it on first use would be wrong: that
    first use can be inside graph capture, which would place it in the graph's
    private pool where later graph tensors may reuse the same bytes.
    """
    global _scale_one
    assert not torch.cuda.is_current_stream_capturing(), (
        "fp8_kv.warmup must run outside CUDA graph capture"
    )
    _scale_one = torch.ones((), dtype=torch.float32, device=device)


def unit_descale(num_seqs: int, num_kv_heads: int, device) -> torch.Tensor:
    """``{q,k,v}_descale`` for flash-attn, shaped ``(num_seqs, num_kv_heads)``.

    Every scale is 1.0, so one element is expanded: a stride-0 view, no
    allocation per call, CUDA-graph safe. vLLM does the same via
    ``layer._k_scale.expand(descale_shape)``.
    """
    if _scale_one is None or _scale_one.device != torch.device(device):
        warmup(device)
    return _scale_one.expand(num_seqs, num_kv_heads)
