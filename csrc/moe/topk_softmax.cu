#include "moe_ops.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

// Type conversion helpers
template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        return __bfloat162float(__nv_bfloat16(value));
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        return __half2float(__half(value));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
    return static_cast<float>(value);
}

// Warp-level shuffle for different types
#define WARP_SIZE 32
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(0xffffffff, var, lane_mask, WARP_SIZE)

// Scoring function enums
enum ScoringFunc {
  SCORING_SOFTMAX = 0, // apply softmax
  SCORING_SIGMOID = 1  // apply sigmoid
};

// Forward declarations for fallback implementation
template <typename InputType>
void launchSoftmaxKernel(
    const InputType* input,
    float* output,
    int num_tokens,
    int num_experts,
    cudaStream_t stream);

void launchTopKKernel(
    const float* softmax_output,
    const float* bias,
    float* topk_weights,
    int* topk_indices,
    int* token_expert_indices,
    int num_tokens,
    int num_experts,
    int topk,
    bool renormalize,
    cudaStream_t stream);

// Optimized topk gating kernel for power-of-2 expert counts
// SF: Scoring Function (SCORING_SOFTMAX or SCORING_SIGMOID)
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, typename InputType, int TOPK, int SF>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE) __global__
void topkGatingKernel(
    const InputType* __restrict__ input,
    const InputType* __restrict__ bias,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_indices,
    int* __restrict__ token_expert_indices,
    const int num_tokens,
    const int topk,
    const bool renormalize)
{
    // Compile-time constants
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // Compute row indices
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_tokens) return;

    // Setup read pointers
    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Load data with vectorization
    float row_chunk[VPT];

    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else {
        // For half precision types, load and convert using the SAME strided
        // layout as the float path above (each LDG group is COLS_PER_GROUP_LDG
        // apart). The previous consecutive read `thread_read_ptr[ii]` was only
        // correct when LDG_PER_THREAD == 1 (num_experts <= 256); for 512 experts
        // (LDG_PER_THREAD == 2) it loaded the wrong experts into the upper half
        // of row_chunk, while argmax/clear index them with the strided layout.
#pragma unroll
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                row_chunk[ldg * ELTS_PER_LDG + ii] =
                    toFloat(thread_read_ptr[ldg * COLS_PER_GROUP_LDG + ii]);
            }
        }
    }

    // Apply scoring function
    if constexpr (SF == SCORING_SOFTMAX) {
        // Softmax: find max
        float thread_max = row_chunk[0];
#pragma unroll
        for (int ii = 1; ii < VPT; ++ii) {
            thread_max = fmaxf(thread_max, row_chunk[ii]);
        }

        // Warp-level max reduce
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            thread_max = fmaxf(thread_max, VLLM_SHFL_XOR_SYNC(thread_max, mask));
        }

        // Softmax: compute exp and sum
        float row_sum = 0.0f;
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii) {
            row_chunk[ii] = expf(row_chunk[ii] - thread_max);
            row_sum += row_chunk[ii];
        }

        // Warp-level sum reduce
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            row_sum += VLLM_SHFL_XOR_SYNC(row_sum, mask);
        }

        // Normalize
        const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii) {
            row_chunk[ii] *= reciprocal_row_sum;
        }
    } else if constexpr (SF == SCORING_SIGMOID) {
        // Sigmoid: apply sigmoid function
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii) {
            row_chunk[ii] = 1.0f / (1.0f + expf(-row_chunk[ii]));
        }
    }

    // Apply bias for selection if provided
    float row_chunk_for_choice[VPT];
    if (bias != nullptr) {
#pragma unroll
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                const int expert = first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
                float bias_val = expert < NUM_EXPERTS ? toFloat(bias[expert]) : 0.0f;
                row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] = row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
            }
        }
    } else {
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii) {
            row_chunk_for_choice[ii] = row_chunk[ii];
        }
    }

    // TopK selection
    int start_col = first_elt_read_by_thread;
    float selected_sum = 0.0f;

    for (int k_idx = 0; k_idx < topk; ++k_idx) {
        // Local argmax
        float max_val_for_choice = row_chunk_for_choice[0];
        float max_val = row_chunk[0];
        int expert = start_col;

#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                if (val_for_choice > max_val_for_choice) {
                    max_val_for_choice = val_for_choice;
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // Warp-level argmax reduce
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max_for_choice = VLLM_SHFL_XOR_SYNC(max_val_for_choice, mask);
            float other_max = VLLM_SHFL_XOR_SYNC(max_val, mask);
            int other_expert = VLLM_SHFL_XOR_SYNC(expert, mask);

            if (other_max_for_choice > max_val_for_choice ||
                (other_max_for_choice == max_val_for_choice && other_expert < expert)) {
                max_val_for_choice = other_max_for_choice;
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write results (only first thread in group)
        if (thread_group_idx == 0) {
            const int idx = topk * thread_row + k_idx;
            topk_weights[idx] = max_val;
            topk_indices[idx] = expert;
            token_expert_indices[idx] = expert;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        // Clear the selected expert for next iteration
        if (k_idx + 1 < topk) {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.0f;
            }
        }
    }

    // Renormalize
    if (renormalize && thread_group_idx == 0) {
        const float denom = selected_sum > 0.0f ? selected_sum : 1.0f;
        for (int k_idx = 0; k_idx < topk; ++k_idx) {
            const int idx = topk * thread_row + k_idx;
            topk_weights[idx] /= denom;
        }
    }
}

// Helper to compute compile-time constants
template <int EXPERTS, int BYTES_PER_LDG, typename InputType>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

// Launch helper
template <int EXPERTS, int WARPS_PER_TB, int MAX_BYTES_PER_LDG, typename InputType, int TOPK, int SF>
void launchTopkGating(
    const InputType* input,
    const InputType* bias,
    float* topk_weights,
    int* topk_indices,
    int* token_expert_indices,
    const int num_tokens,
    const int topk,
    const bool renormalize,
    cudaStream_t stream)
{
    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, (int)(sizeof(InputType) * EXPERTS));
    using Constants = TopkConstants<EXPERTS, BYTES_PER_LDG, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;

    const int num_warps = (num_tokens + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingKernel<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, InputType, TOPK, SF>
        <<<num_blocks, block_dim, 0, stream>>>(
            input, bias, topk_weights, topk_indices, token_expert_indices,
            num_tokens, topk, renormalize);
}

#define LAUNCH_TOPK_KERNEL(NUM_EXPERTS, TOPK, SF) \
    launchTopkGating<NUM_EXPERTS, 4, 16, scalar_t, TOPK, SF>( \
        gating_output_ptr, bias_ptr, topk_weights_ptr, topk_indices_ptr, \
        token_expert_indices_ptr, num_tokens, topk, renormalize, stream);

void topk_softmax(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output,
    torch::optional<torch::Tensor> bias,
    int topk,
    bool renormalize) {

    const int num_tokens = gating_output.size(0);
    const int num_experts = gating_output.size(1);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        gating_output.scalar_type(), "topk_softmax_kernel", [&] {
            const scalar_t* gating_output_ptr = gating_output.data_ptr<scalar_t>();
            const scalar_t* bias_ptr = bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr;
            float* topk_weights_ptr = topk_weights.data_ptr<float>();
            int* topk_indices_ptr = topk_indices.data_ptr<int>();
            int* token_expert_indices_ptr = token_expert_indices.data_ptr<int>();

            // Dispatch based on num_experts (power of 2 only for now)
            if (num_experts == 8) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(8, 8, SCORING_SOFTMAX);
            } else if (num_experts == 16) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(16, 8, SCORING_SOFTMAX);
            } else if (num_experts == 64) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(64, 8, SCORING_SOFTMAX);
            } else if (num_experts == 128) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(128, 8, SCORING_SOFTMAX);
            } else if (num_experts == 256) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(256, 8, SCORING_SOFTMAX);
            } else if (num_experts == 512) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(512, 8, SCORING_SOFTMAX);
            } else if (num_experts == 32) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(32, 8, SCORING_SOFTMAX);
            } else if (num_experts == 4) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(4, 8, SCORING_SOFTMAX);
            } else if (num_experts == 2) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(2, 8, SCORING_SOFTMAX);
            } else if (num_experts == 1) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(1, 8, SCORING_SOFTMAX);
            } else {
                // Fallback path for non-power-of-2 or unsupported expert counts
                // Use a generic implementation with separate softmax and topk
                // This path supports any number of experts (e.g., 160, 192, 320, etc.)
                // Note: vLLM has optimized paths for 192, 320, 384, 448, 576 (multiples of 64)
                // but these require special kernel parameters and are not implemented here yet.
                // Fallback path for non-power-of-2 or unsupported expert counts
                // Use a generic implementation with separate softmax and topk
                // This path supports any number of experts (e.g., 160, 192, 320, etc.)

                // Allocate workspace for softmax output
                auto workspace = torch::empty({num_tokens, num_experts},
                                             torch::dtype(torch::kFloat32).device(gating_output.device()));
                float* workspace_ptr = workspace.data_ptr<float>();

                // Step 1: Compute softmax
                launchSoftmaxKernel<scalar_t>(gating_output_ptr, workspace_ptr,
                                             num_tokens, num_experts, stream);

                // Step 2: Convert bias to float if needed
                const float* bias_float_ptr = nullptr;
                torch::Tensor bias_float;
                if (bias_ptr != nullptr) {
                    if constexpr (std::is_same_v<scalar_t, float>) {
                        bias_float_ptr = reinterpret_cast<const float*>(bias_ptr);
                    } else {
                        bias_float = bias.value().to(torch::kFloat32);
                        bias_float_ptr = bias_float.data_ptr<float>();
                    }
                }

                // Step 3: Compute topK on softmax output (bias applied to softmax for selection)
                launchTopKKernel(workspace_ptr, bias_float_ptr,
                               topk_weights_ptr, topk_indices_ptr, token_expert_indices_ptr,
                               num_tokens, num_experts, topk, renormalize, stream);
            }
        });
}

// ====================== Fallback Implementation ===============================

// Generic softmax kernel for any number of experts
template <typename InputType>
__global__ void genericSoftmaxKernel(
    const InputType* __restrict__ input,
    float* __restrict__ output,
    const int num_experts)
{
    const int token_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int offset = token_id * num_experts;
    const InputType* input_row = input + offset;
    float* output_row = output + offset;

    // Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_experts; i += num_threads) {
        float val = toFloat(input_row[i]);
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction for max
    __shared__ float shared_max[32];  // Assuming max 1024 threads = 32 warps
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Reduce within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        thread_max = (lane_id < (num_threads + 31) / 32) ? shared_max[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_id == 0) {
            shared_max[0] = thread_max;
        }
    }
    __syncthreads();

    const float row_max = shared_max[0];

    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_experts; i += num_threads) {
        float val = toFloat(input_row[i]);
        float exp_val = expf(val - row_max);
        output_row[i] = exp_val;
        thread_sum += exp_val;
    }

    // Warp-level reduction for sum
    __shared__ float shared_sum[32];

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        thread_sum = (lane_id < (num_threads + 31) / 32) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = thread_sum;
        }
    }
    __syncthreads();

    const float row_sum = shared_sum[0];
    const float inv_sum = 1.0f / (row_sum + 1e-9f);

    // Normalize
    for (int i = tid; i < num_experts; i += num_threads) {
        output_row[i] *= inv_sum;
    }
}

// Generic topK kernel for any number of experts
// Uses CUB library for efficient block-level reduction (matching vLLM implementation)
// NOTE: Bias is applied to softmax output for selection (matching vLLM semantics)
template <int TPB>
__global__ void genericTopKKernel(
    const float* __restrict__ softmax_output,  // Softmax values
    const float* __restrict__ bias,            // Bias applied to softmax for selection
    float* __restrict__ topk_weights,
    int* __restrict__ topk_indices,
    int* __restrict__ token_expert_indices,
    const int num_experts,
    const int topk,
    const bool renormalize)
{
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int token_id = blockIdx.x;
    const int offset = token_id * num_experts;
    const float* softmax_row = softmax_output + offset;

    float selected_sum = 0.0f;

    for (int k_idx = 0; k_idx < topk; ++k_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -FLT_MAX;

        cub_kvp inp_kvp;
        // Each thread scans through experts
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            inp_kvp.key = expert;
            inp_kvp.value = softmax_row[expert];

            // Apply bias for selection
            if (bias != nullptr) {
                inp_kvp.value += bias[expert];
            }

            // Check if this expert was already selected in previous iterations
            bool already_selected = false;
            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                if (topk_indices[topk * token_id + prior_k] == expert) {
                    already_selected = true;
                    break;
                }
            }

            // Update thread-local maximum if not already selected
            if (!already_selected) {
                thread_kvp = arg_max(inp_kvp, thread_kvp);
            }
        }

        // Block-level reduction using CUB
        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);

        if (threadIdx.x == 0) {
            const int expert = result_kvp.key;
            const int idx = topk * token_id + k_idx;

            // Store unbiased weight from softmax output
            topk_weights[idx] = softmax_row[expert];
            topk_indices[idx] = expert;
            token_expert_indices[idx] = expert;

            if (renormalize) {
                selected_sum += softmax_row[expert];
            }
        }
        __syncthreads();
    }

    // Renormalize the k weights for this row to sum to 1, if requested
    if (renormalize) {
        if (threadIdx.x == 0) {
            const float denom = selected_sum > 0.0f ? selected_sum : 1.0f;
            for (int k_idx = 0; k_idx < topk; ++k_idx) {
                const int idx = topk * token_id + k_idx;
                topk_weights[idx] = topk_weights[idx] / denom;
            }
        }
    }
}

// Launcher functions for fallback path
template <typename InputType>
void launchSoftmaxKernel(
    const InputType* input,
    float* output,
    int num_tokens,
    int num_experts,
    cudaStream_t stream)
{
    const int TPB = 256;
    genericSoftmaxKernel<InputType><<<num_tokens, TPB, 0, stream>>>(
        input, output, num_experts);
}

void launchTopKKernel(
    const float* softmax_output,
    const float* bias,
    float* topk_weights,
    int* topk_indices,
    int* token_expert_indices,
    int num_tokens,
    int num_experts,
    int topk,
    bool renormalize,
    cudaStream_t stream)
{
    const int TPB = 256;
    // No external shared memory needed - CUB manages its own temp storage
    genericTopKKernel<TPB><<<num_tokens, TPB, 0, stream>>>(
        softmax_output, bias, topk_weights, topk_indices,
        token_expert_indices, num_experts, topk, renormalize);
}

void topk_sigmoid(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output,
    torch::optional<torch::Tensor> bias,
    int topk,
    bool renormalize) {

    const int num_tokens = gating_output.size(0);
    const int num_experts = gating_output.size(1);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        gating_output.scalar_type(), "topk_sigmoid_kernel", [&] {
            const scalar_t* gating_output_ptr = gating_output.data_ptr<scalar_t>();
            const scalar_t* bias_ptr = bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr;
            float* topk_weights_ptr = topk_weights.data_ptr<float>();
            int* topk_indices_ptr = topk_indices.data_ptr<int>();
            int* token_expert_indices_ptr = token_expert_indices.data_ptr<int>();

            // Dispatch based on num_experts (power of 2 only for now)
            if (num_experts == 8) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(8, 8, SCORING_SIGMOID);
            } else if (num_experts == 16) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(16, 8, SCORING_SIGMOID);
            } else if (num_experts == 64) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(64, 8, SCORING_SIGMOID);
            } else if (num_experts == 128) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(128, 8, SCORING_SIGMOID);
            } else if (num_experts == 256) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(256, 8, SCORING_SIGMOID);
            } else if (num_experts == 512) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(512, 8, SCORING_SIGMOID);
            } else if (num_experts == 32) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(32, 8, SCORING_SIGMOID);
            } else if (num_experts == 4) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(4, 8, SCORING_SIGMOID);
            } else if (num_experts == 2) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(2, 8, SCORING_SIGMOID);
            } else if (num_experts == 1) {
                // topk is a runtime kernel arg; the TOPK template param is unused, so a
                // single instantiation handles any topk (matches vLLM).
                LAUNCH_TOPK_KERNEL(1, 8, SCORING_SIGMOID);
            } else {
                TORCH_CHECK(false, "topk_sigmoid fallback for non-power-of-2 experts not implemented yet. num_experts: ", num_experts);
            }
        });
}
