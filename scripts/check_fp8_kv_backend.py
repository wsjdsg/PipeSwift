"""Self-check that the installed FP8 attention backend carries the SM90 fix.

Run after installing vllm_flash_attn. Builds a case whose exact answer is known
analytically -- V is a strictly positive constant, and since the softmax row
sums to 1, the output must equal V exactly -- then measures the drift at a long
context.

This separates a backend that has vllm-project/flash-attention#104 (two-level
FP8 accumulation) from one that does not. Upstream Dao-AILab FA3 drifts ~-67%
here; the vLLM fork is exact.

Exits non-zero if the backend is unusable or inaccurate.
"""

import sys

import torch

KV_LEN = 131072
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64
FP8 = torch.float8_e4m3fn

# The fork measures 0.0 exactly; upstream FA3 measures -0.67. Anything above a
# few 1e-3 means the two-level accumulation is missing or disabled.
TOLERANCE = 5e-3


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        print(f"SKIP: FP8 KV cache needs SM90+, this GPU is SM{major}{minor}")
        return 0

    try:
        from vllm_flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        print(f"FAIL: cannot import vllm_flash_attn ({exc})")
        print("      run scripts/install_fp8_kv_backend.sh")
        return 1

    dev = "cuda"
    torch.manual_seed(0)
    pages = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE

    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16).to(FP8)
    k = torch.randn(
        pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16
    ).to(FP8)
    vals = torch.tensor([1.0, 1.5, 2.0, 3.0], device=dev)  # exact in e4m3
    v_const = vals[torch.arange(NUM_KV_HEADS, device=dev) % len(vals)]
    v = (
        v_const.view(1, 1, NUM_KV_HEADS, 1)
        .expand(pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
        .contiguous()
        .to(torch.bfloat16)
        .to(FP8)
    )

    page_table = torch.arange(pages, device=dev, dtype=torch.int32).unsqueeze(0)
    ones = torch.ones((1, NUM_KV_HEADS), device=dev, dtype=torch.float32)

    out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=1,
        cu_seqlens_q=torch.tensor([0, 1], device=dev, dtype=torch.int32),
        max_seqlen_k=None,
        seqused_k=torch.tensor([KV_LEN], device=dev, dtype=torch.int32),
        block_table=page_table,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        fa_version=3,
        num_splits=1,  # one threadblock accumulates the whole context
        q_descale=ones,
        k_descale=ones,
        v_descale=ones,
    ).float()

    group = NUM_Q_HEADS // NUM_KV_HEADS
    exact = v_const[torch.arange(NUM_Q_HEADS, device=dev) // group]
    exact = exact.view(1, NUM_Q_HEADS, 1).expand_as(out).float()
    drift = ((out - exact) / exact).abs().max().item()

    print(f"context={KV_LEN}  max relative drift={drift:.6f}  (tolerance {TOLERANCE})")
    if drift > TOLERANCE:
        print("FAIL: this FlashAttention build is missing the SM90 two-level FP8")
        print("      accumulation fix (vllm-project/flash-attention#104), or it was")
        print("      compiled with FLASH_ATTENTION_DISABLE_FP8_TWO_LEVEL_ACCUMULATION.")
        print("      FP8 KV cache would silently lose long-context accuracy.")
        return 1
    print("PASS: FP8 attention backend is accurate at long context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
