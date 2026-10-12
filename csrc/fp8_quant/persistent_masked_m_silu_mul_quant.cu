#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <type_traits>

#include "dispatch_utils.h"

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

namespace pipeswift {

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + expf(-x)));
}

__device__ __forceinline__ __nv_bfloat162 silu2_v2(float2 x) {
#ifndef USE_ROCM
  return make_bfloat162(__float2bfloat16_rn(silu(x.x)),
                        __float2bfloat16_rn(silu(x.y)));
#else
  return __float22bfloat162_rn(make_float2(silu(x.x), silu(x.y)));
#endif
}

#ifndef USE_ROCM
__device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) {
  static constexpr unsigned FULL_MASK = 0xffffffffu;
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    v = __hmax(v, __shfl_xor_sync(FULL_MASK, v, offset));
  }
  return v;
}
#endif

template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* _smem_ptr, const U* _glob_ptr) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  auto smem_ptr = reinterpret_cast<void*>(_smem_ptr);
  auto glob_ptr = reinterpret_cast<const void*>(_glob_ptr);
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" : : "r"(smem), "l"(glob_ptr), "n"(BYTES));
#else
  _smem_ptr[0] = _glob_ptr[0];
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__device__ __forceinline__ __nv_bfloat16 clip(__nv_bfloat16 v,
                                              __nv_bfloat16 mmin,
                                              __nv_bfloat16 mmax) {
  return __hmin(mmax, __hmax(v, mmin));
}

__device__ __forceinline__ __nv_bfloat162 clip(__nv_bfloat162 v,
                                               __nv_bfloat162 mmin,
                                               __nv_bfloat162 mmax) {
  return __hmin2(mmax, __hmax2(v, mmin));
}

template <class T>
constexpr __nv_bfloat16 get_fp8_max() {
  static_assert(std::is_same_v<T, c10::Float8_e4m3fn> ||
                std::is_same_v<T, c10::Float8_e4m3fnuz>);
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 17376});
  } else {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 17264});
  }
}

template <class T>
constexpr __nv_bfloat16 get_fp8_min() {
  static_assert(std::is_same_v<T, c10::Float8_e4m3fn> ||
                std::is_same_v<T, c10::Float8_e4m3fnuz>);
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 50144});
  } else {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 50032});
  }
}

template <typename Idx_t>
__device__ __forceinline__ int warp_expert_search(
    int idx, int n, const Idx_t* __restrict__ input, Idx_t val) {
  const Idx_t* input_ptr = input + idx;
  int base_offset = 0;
  for (;;) {
    bool move_on = (idx < n && *input_ptr <= val);
    unsigned mask = __ballot_sync(0xffffffff, move_on);
    if (mask != 0xffffffffu) {
      int last_lane = 31 - __clz(mask);
      return base_offset + last_lane;
    }
    input_ptr += 32;
    base_offset += 32;
    idx += 32;
  }
}

template <int num_parallel_tokens>
__device__ __forceinline__ void token_bounds(int32_t n_tokens,
                                             int32_t worker_id,
                                             int32_t& n_tokens_lower,
                                             int32_t& n_tokens_upper) {
  if (n_tokens < num_parallel_tokens && worker_id < n_tokens) {
    if (worker_id >= num_parallel_tokens) return;
    n_tokens_lower = worker_id;
    n_tokens_upper = worker_id + 1;
  } else {
    int32_t chunk_size = n_tokens / num_parallel_tokens;
    int32_t residual = n_tokens - chunk_size * num_parallel_tokens;
    auto calc_id = [&](int32_t id) {
      if (id < residual) {
        return min(n_tokens, id * (chunk_size + 1));
      }
      return min(n_tokens, id * chunk_size + residual);
    };
    n_tokens_lower = calc_id(worker_id);
    n_tokens_upper = calc_id(worker_id + 1);
  }
}

template <int BLOCK_COUNT, int SMEM_SIZE_BYTES_Y, typename fp8_type,
          int THREADS, typename Idx_t, int GROUP_SIZE = 128,
          int NUM_STAGES = 3>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const __nv_bfloat16* __restrict__ _input,
    fp8_type* __restrict__ _y_q,
    float* __restrict__ _y_s,
    const int32_t* __restrict__ tokens_per_expert,
    Idx_t E,
    Idx_t T,
    Idx_t H,
    Idx_t stride_i_e,
    Idx_t stride_i_t,
    Idx_t stride_i_h,
    Idx_t stride_yq_e,
    Idx_t stride_yq_t,
    Idx_t stride_yq_h,
    Idx_t stride_ys_e,
    Idx_t stride_ys_t,
    Idx_t stride_ys_g,
    Idx_t stride_counts_e,
    bool reverse_layout) {
#ifndef USE_ROCM
  static constexpr int NUM_WARPS = THREADS / WARP_SIZE;
  static constexpr int LOAD_STAGE_SIZE = 2 * GROUP_SIZE / 8;
  static constexpr int LOAD_STAGE_MOD = NUM_STAGES * LOAD_STAGE_SIZE;
  static constexpr int COMPUTE_STAGE_SIZE = 2 * GROUP_SIZE / 4;
  static constexpr int COMPUTE_STAGE_MOD = COMPUTE_STAGE_SIZE * NUM_STAGES;

  extern __shared__ __align__(16) __int128_t smem_128[];
  int* s_expert_offsets = reinterpret_cast<int*>(smem_128 + (SMEM_SIZE_BYTES_Y / 16));

  static constexpr __nv_bfloat16 fp8_min = get_fp8_min<fp8_type>();
  static constexpr __nv_bfloat16 fp8_max = get_fp8_max<fp8_type>();
  static constexpr __nv_bfloat16 EPS = (__nv_bfloat16_raw{.x = 11996});

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 0x1f;

  int running_sum{};
  if (!warp_id) {
    for (int i = 0; i < E; i += WARP_SIZE) {
      bool valid = (i + threadIdx.x) < E;
      int value = (valid ? tokens_per_expert[i + threadIdx.x * stride_counts_e] : 0) +
                  (!lane_id ? running_sum : 0);
      for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xFFFFFFFFu, value, offset);
        if (lane_id >= offset) value += n;
      }
      if (valid) {
        s_expert_offsets[i + threadIdx.x + 1] = value;
      }
      running_sum = __shfl_sync(0xFFFFFFFFu, value, WARP_SIZE - 1);
    }
    if (!lane_id) {
      s_expert_offsets[0] = 0;
    }
  }

  __syncthreads();
  int32_t total_tokens = s_expert_offsets[E];

  const int warp_position_yq = warp_id * (H / NUM_WARPS);
  const int warp_position_scales = warp_id * (H / (GROUP_SIZE * NUM_WARPS));

  __int128_t* s_hidden_load = smem_128 + warp_id * ((2 * 128 / 8) * NUM_STAGES);
  __int128_t* smem_load_ptr = s_hidden_load + lane_id;

  const __nv_bfloat16 fp8_inv = __hdiv(__float2bfloat16(1.f), fp8_max);
  int32_t compute_pipeline_offset_64 = 0;
  int32_t load_stage_offset{};
  const __nv_bfloat16 one_bf16 = __float2bfloat16_rn(1.f);

  __int64_t* smem_compute_ptr = reinterpret_cast<__int64_t*>(smem_128) +
                                warp_id * (2 * (GROUP_SIZE / 4) * NUM_STAGES) +
                                lane_id;
  __int64_t* s_gate64_ptr = smem_compute_ptr;
  __int64_t* s_up64_ptr = smem_compute_ptr + GROUP_SIZE / 4;

  int tokens_lower, tokens_upper;
  token_bounds<BLOCK_COUNT>(total_tokens, blockIdx.x, tokens_lower, tokens_upper);

  Idx_t expert_id{}, expert_offset{}, next_expert_offset{};
  int token_id = tokens_lower;
  int32_t t_load{};

  if (token_id < tokens_upper) {
    expert_id = warp_expert_search<int>(lane_id, E, s_expert_offsets, token_id);
    expert_offset = s_expert_offsets[expert_id];
    next_expert_offset = s_expert_offsets[expert_id + 1];
  } else {
    return;
  }

  int t_load_bound = H / (GROUP_SIZE * NUM_WARPS);

  Idx_t base_i = ((expert_id * stride_i_e) / 8) + (token_id - expert_offset) * stride_i_t / 8;
  const Idx_t gate_warp_offset =
      warp_id * ((stride_i_h * H) / (8 * NUM_WARPS)) + (lane_id & 0b1111);

  const __int128_t* input_128_ptr = reinterpret_cast<const __int128_t*>(_input) +
                                    gate_warp_offset +
                                    ((((lane_id < 16) ^ reverse_layout) ? 0 : ((H * stride_i_h) / 8)));
  __int128_t* load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);
  auto token_offset = token_id - expert_offset;

  auto load_and_advance_y_pred = [&] {
    if (t_load < t_load_bound) {
      auto smem_load_ptr_staged = smem_load_ptr + load_stage_offset;
      load_stage_offset += LOAD_STAGE_SIZE;
      load_stage_offset %= LOAD_STAGE_MOD;
      cp_async4(smem_load_ptr_staged, load_ptr);
      load_ptr += GROUP_SIZE / 8;
      ++t_load;
    } else if (token_id + 1 < tokens_upper) {
      ++token_id;
      t_load = 0;
      if (token_id >= next_expert_offset) {
        do {
          ++expert_id;
          expert_offset = next_expert_offset;
          next_expert_offset = s_expert_offsets[expert_id + 1];
        } while (next_expert_offset == expert_offset);

        base_i = expert_id * (stride_i_e / 8);
        token_offset = 0;
        load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);
      } else {
        base_i += stride_yq_t / 4;
        token_offset++;
      }

      load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);
      auto smem_load_ptr_staged = smem_load_ptr + load_stage_offset;
      load_stage_offset += LOAD_STAGE_SIZE;
      load_stage_offset %= LOAD_STAGE_MOD;
      cp_async4(smem_load_ptr_staged, load_ptr);
      load_ptr += GROUP_SIZE / 8;
      ++t_load;
    }
    cp_async_fence();
  };

#pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; i++) {
    load_and_advance_y_pred();
  }

  __nv_fp8x4_e4m3* y_q_base_ptr = reinterpret_cast<__nv_fp8x4_e4m3*>(_y_q) + lane_id;
  Idx_t scale_group_offset = warp_position_scales * stride_ys_g;
  float* const y_scale_base_ptr = _y_s + scale_group_offset;

  for (auto j = tokens_lower; j < tokens_upper; j++) {
    int current_group_id = warp_position_scales;
    const Idx_t base_ys = expert_id * stride_ys_e;
    auto y_s_ptr = y_scale_base_ptr + base_ys + token_offset * stride_ys_t;
    __nv_fp8x4_e4m3* y_q_ptr =
        y_q_base_ptr + (expert_id * stride_yq_e + token_offset * stride_yq_t +
                        warp_position_yq * stride_yq_h) /
                           4;
    const int COMPUTE_LIMIT = H / (GROUP_SIZE * NUM_WARPS);

    for (int i = 0; i < COMPUTE_LIMIT; i++) {
      cp_async_wait<NUM_STAGES - 2>();
      __syncthreads();
      load_and_advance_y_pred();

      __int64_t* gate64_ptr = s_gate64_ptr + compute_pipeline_offset_64;
      __int64_t* up64_ptr = s_up64_ptr + compute_pipeline_offset_64;
      compute_pipeline_offset_64 += COMPUTE_STAGE_SIZE;
      compute_pipeline_offset_64 %= COMPUTE_STAGE_MOD;

      __int64_t gate64 = *gate64_ptr;
      __int64_t up64 = *up64_ptr;

      __nv_bfloat162 res[2];
      __nv_bfloat162* s_up_comp = reinterpret_cast<__nv_bfloat162*>(&up64);
      __nv_bfloat162* s_gate_comp = reinterpret_cast<__nv_bfloat162*>(&gate64);

#pragma unroll
      for (int32_t k = 0; k < 2; ++k) {
        __nv_bfloat162 gate = silu2_v2(__bfloat1622float2(s_gate_comp[k]));
        res[k] = __hmul2(gate, s_up_comp[k]);
      }

      auto _y_max2 = __hmax2(__habs2(res[0]), __habs2(res[1]));
      _y_max2.x = __hmax(__hmax(_y_max2.x, _y_max2.y), EPS);
      __nv_bfloat16 y_s = __hmul(warp_max(_y_max2.x), fp8_inv);
      __nv_bfloat16 inv_y = __hdiv(one_bf16, y_s);
      auto y_s2 = make_bfloat162(inv_y, inv_y);

#pragma unroll
      for (int32_t k = 0; k < 2; ++k) {
        res[k] = clip(__hmul2(res[k], y_s2), __bfloat162bfloat162(fp8_min),
                      __bfloat162bfloat162(fp8_max));
      }

      *y_q_ptr = __nv_fp8x4_e4m3(res[0], res[1]);
      y_q_ptr += WARP_SIZE * stride_yq_h;

      if (!lane_id) {
        *y_s_ptr = y_s;
        y_s_ptr += stride_ys_g;
        current_group_id += 1;
      }
    }
  }
#endif
}

}  // namespace pipeswift

void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,
    const at::Tensor& tokens_per_expert,
    at::Tensor& y_q,
    at::Tensor& y_s,
    bool cast_scale_ue8m0,
    bool reverse_layout) {
#ifndef USE_ROCM
  static constexpr int GROUP_SIZE = 128;

  TORCH_CHECK(input.dtype() == torch::kBFloat16,
              "input must be bf16");
  TORCH_CHECK(y_q.dtype() == torch::kFloat8_e4m3fn ||
              y_q.dtype() == torch::kFloat8_e4m3fnuz,
              "y_q must be fp8 e4m3");
  TORCH_CHECK(input.size(-1) % (GROUP_SIZE * 2) == 0,
              "input hidden dim must be divisible by 256");
  TORCH_CHECK(tokens_per_expert.dtype() == torch::kInt32,
              "tokens_per_expert must be int32");

  TORCH_CHECK(!cast_scale_ue8m0,
              "ue8m0 scale path is disabled in pipeswift kernel copy");
  TORCH_CHECK(y_s.dtype() == torch::kFloat32,
              "y_s must be float32 when ue8m0 is disabled");

  using Idx_t = int64_t;

  Idx_t E = input.size(0);
  Idx_t T = input.size(1);
  Idx_t H = input.size(2) / 2;
  Idx_t stride_i_e = input.stride(0);
  Idx_t stride_i_t = input.stride(1);
  Idx_t stride_i_h = input.stride(2);
  Idx_t stride_yq_e = y_q.stride(0);
  Idx_t stride_yq_t = y_q.stride(1);
  Idx_t stride_yq_h = y_q.stride(2);
  Idx_t stride_counts_e = tokens_per_expert.stride(0);

  int const NUM_GROUPS = H / GROUP_SIZE;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static constexpr int SILU_V2_BLOCK_COUNT = 132 * 32;

  Idx_t stride_ys_e = y_s.stride(0);
  Idx_t stride_ys_t = y_s.stride(1);
  Idx_t stride_ys_g = y_s.stride(2);

#define KERNEL(BLOCK_COUNT, THREAD_COUNT, STAGES)                                \
  do {                                                                            \
    static constexpr int NUM_WARPS = THREAD_COUNT / 32;                   \
    static constexpr int max_shared_mem_bytes =                                  \
        GROUP_SIZE * 2 * STAGES * NUM_WARPS * 2;                                 \
    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);                                 \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
    VLLM_DISPATCH_FP8_TYPES(y_q.scalar_type(),                                   \
                            "silu_mul_fp8_quant_deep_gemm_kernel", [&] {        \
      pipeswift::silu_mul_fp8_quant_deep_gemm_kernel<                            \
          BLOCK_COUNT, max_shared_mem_bytes, fp8_t, THREAD_COUNT, Idx_t,         \
          GROUP_SIZE, STAGES><<<grid, block, max_shared_mem_bytes + (E + 1) * 16,\
                                 stream>>>(                                       \
          reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),                    \
          (fp8_t*)y_q.data_ptr(),                                                 \
          reinterpret_cast<float*>(y_s.data_ptr()),                               \
          reinterpret_cast<int32_t*>(tokens_per_expert.data_ptr()), E, T, H,     \
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,          \
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,   \
          reverse_layout);                                                        \
    });                                                                           \
  } while (0)

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    KERNEL(SILU_V2_BLOCK_COUNT, 256, 4);
  } else {
    KERNEL(SILU_V2_BLOCK_COUNT, 32, 2);
  }

#undef KERNEL
#endif
}
