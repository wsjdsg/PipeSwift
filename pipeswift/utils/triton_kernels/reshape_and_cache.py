import torch
import triton
import triton.language as tl

# e4m3 finite max. vLLM's reshape_and_cache_flash converts with __NV_SATFINITE,
# so out-of-range values clamp here instead of turning into NaN/inf.
# Must be tl.constexpr(...) -- a bare annotation is not readable from a kernel.
FP8_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_stride,
    value_stride,
    block_stride,
    head_stride,
    dim_stride_k,
    dim_stride_v,
    page_stride,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, tile_size)
    tile_pos = tile_i * tile_size + tile_offs

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    cur_head = tile_pos // head_size
    cur_dim = tile_pos % head_size

    tgt_idx_k = (
        block_idx * block_stride
        + block_offset * page_stride
        + cur_head * head_stride
        + cur_dim * dim_stride_k
    )
    tgt_idx_v = (
        block_idx * block_stride
        + block_offset * page_stride
        + cur_head * head_stride
        + cur_dim * dim_stride_v
    )

    mask = tile_pos < (num_heads * head_size)
    key_tile = tl.load(key_ptr + src_key_idx + tile_pos, mask=mask)
    value_tile = tl.load(value_ptr + src_value_idx + tile_pos, mask=mask)

    if IS_FP8:
        # Saturating cast to e4m3 at scale 1.0, matching vLLM's
        # scaled_convert<__NV_SATFINITE>. Clamp in fp32 before the cast so
        # out-of-range values pin to +-448 instead of becoming NaN.
        k32 = tl.minimum(
            tl.maximum(key_tile.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX
        )
        v32 = tl.minimum(
            tl.maximum(value_tile.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX
        )
        tl.store(key_cache_ptr + tgt_idx_k, k32.to(tl.float8e4nv), mask=mask)
        tl.store(value_cache_ptr + tgt_idx_v, v32.to(tl.float8e4nv), mask=mask)
    else:
        tl.store(key_cache_ptr + tgt_idx_k, key_tile, mask=mask)
        tl.store(value_cache_ptr + tgt_idx_v, value_tile, mask=mask)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    assert key.ndim == 3 and value.ndim == 3
    assert key_cache.ndim == 4 and value_cache.ndim == 4
    assert key.is_cuda and value.is_cuda and slot_mapping.is_cuda

    num_heads = key.shape[1]
    head_size = key.shape[2]
    assert value.shape[1] == num_heads
    assert value.shape[2] == head_size
    assert key_cache.shape[2] == num_heads
    assert key_cache.shape[3] == head_size
    assert value_cache.shape[2] == num_heads
    assert value_cache.shape[3] == head_size

    # --enable_fp8_kv: quantize to e4m3 on the way in, per-tensor scale 1.0.
    is_fp8 = key_cache.dtype == torch.float8_e4m3fn
    assert is_fp8 or key_cache.dtype == key.dtype, (
        f"kv cache dtype {key_cache.dtype} does not match key dtype {key.dtype}"
    )
    assert value_cache.dtype == key_cache.dtype

    block_size = key_cache.shape[1]
    n = num_heads * head_size
    tile_size = min(1024, triton.next_power_of_2(n))

    grid = (slot_mapping.shape[0], triton.cdiv(n, tile_size))
    reshape_and_cache_kernel_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(3),
        key_cache.stride(1),
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        tile_size=tile_size,
        IS_FP8=is_fp8,
        num_warps=8,
        num_stages=4,
    )


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,
    value_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    key_stride,
    value_stride,
    block_stride,
    page_stride,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    tgt_base = block_idx * block_stride + block_offset * page_stride

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, tile_size)
    tile_pos = tile_i * tile_size + tile_offs

    key_n = num_heads * head_size_k
    value_n = num_heads * head_size_v
    total_n = key_n + value_n
    mask = tile_pos < total_n

    is_key = tile_pos < key_n
    key_pos = tile_pos
    value_pos = tile_pos - key_n

    key_tile = tl.load(
        key_ptr + token_idx * key_stride + key_pos,
        mask=mask & is_key,
        other=0,
    )
    value_tile = tl.load(
        value_ptr + token_idx * value_stride + value_pos,
        mask=mask & (~is_key),
        other=0,
    )
    out_tile = tl.where(is_key, key_tile, value_tile)

    tl.store(kv_cache_ptr + tgt_base + tile_pos, out_tile, mask=mask)


def triton_reshape_and_cache_flash_diffkv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    assert key.ndim == 3 and value.ndim == 3
    assert kv_cache.ndim == 4
    assert key.is_cuda and value.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda

    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    assert value.shape[1] == num_heads
    assert kv_cache.shape[2] == num_heads
    assert kv_cache.shape[3] == head_size_k + head_size_v

    block_size = kv_cache.shape[1]
    total_n = num_heads * (head_size_k + head_size_v)
    tile_size = min(1024, triton.next_power_of_2(total_n))

    grid = (slot_mapping.shape[0], triton.cdiv(total_n, tile_size))
    reshape_and_cache_kernel_flash_diffkv[grid](
        key,
        value,
        kv_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        num_heads=num_heads,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        block_size=block_size,
        tile_size=tile_size,
        num_warps=8,
        num_stages=4,
    )
