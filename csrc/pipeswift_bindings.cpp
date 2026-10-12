#include <torch/extension.h>
#include "moe/moe_ops.h"
#include "fp8_quant/per_token_group_quant_8bit.h"


// Fused masked SiLU*Mul + FP8 block quantization
void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,
    const at::Tensor& tokens_per_expert,
    at::Tensor& y_q,
    at::Tensor& y_s,
    bool use_ue8m0,
    bool reverse);

// Forward declarations for CUTLASS scaled_mm
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);
void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);
void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::Tensor& expert_first_token_offset,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    const int64_t n, const int64_t k, const bool swap_ab);
void moe_permute(
    const torch::Tensor& input, const torch::Tensor& topk_ids,
    const torch::Tensor& token_expert_indices,
    const std::optional<torch::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk,
    const std::optional<int64_t>& align_block_size, torch::Tensor& permuted_input,
    torch::Tensor& expert_first_token_offset, torch::Tensor& inv_permuted_idx,
    torch::Tensor& permuted_idx, torch::Tensor& m_indices);
void moe_unpermute(
    const torch::Tensor& permuted_hidden_states,
    const torch::Tensor& topk_weights, const torch::Tensor& inv_permuted_idx,
    const std::optional<torch::Tensor>& expert_first_token_offset, int64_t topk,
    torch::Tensor& hidden_states);
bool moe_permute_unpermute_supported();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // MoE operations
  m.def("moe_align_block_size", &moe_align_block_size,
        "Align tokens to block size for efficient MoE processing");

  m.def("moe_sum", &moe_sum,
        "Sum MoE results from all selected experts");

  m.def("topk_softmax", &topk_softmax,
        "Apply topk softmax to gating outputs");

  m.def("topk_sigmoid", &topk_sigmoid,
        "Apply topk sigmoid to gating outputs");

#ifndef USE_ROCM
  m.def("grouped_topk", &grouped_topk,
        "Apply grouped topk routing to select experts");
#endif

  // FP8 quantization operations
  m.def("per_token_group_fp8_quant",
        &per_token_group_quant_fp8,
        "Per-token-group FP8 quantization kernel from vLLM",
        py::arg("input"),
        py::arg("output_q"),
        py::arg("output_s"),
        py::arg("group_size"),
        py::arg("eps"),
        py::arg("fp8_min"),
        py::arg("fp8_max"),
        py::arg("scale_ue8m0"));

  m.def("persistent_masked_m_silu_mul_quant",
        &persistent_masked_m_silu_mul_quant,
        "Fused masked SiLU*Mul + FP8 block quantization",
        py::arg("input"),
        py::arg("tokens_per_expert"),
        py::arg("y_q"),
        py::arg("y_s"),
        py::arg("use_ue8m0"),
        py::arg("reverse") = false);

  m.def("rms_norm",
        &rms_norm,
        "vLLM RMSNorm kernel",
        py::arg("out"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("epsilon"));

  m.def("fused_add_rms_norm",
        &fused_add_rms_norm,
        "vLLM fused add RMSNorm kernel",
        py::arg("input"),
        py::arg("residual"),
        py::arg("weight"),
        py::arg("epsilon"));

  // CUTLASS FP8 scaled_mm operations
  m.def("cutlass_scaled_mm",
        &cutlass_scaled_mm,
        "CUTLASS FP8 scaled matrix multiplication",
        py::arg("out"),
        py::arg("a"),
        py::arg("b"),
        py::arg("scale_a"),
        py::arg("scale_b"),
        py::arg("bias") = py::none());

  m.def("cutlass_scaled_mm_supports_fp8",
        &cutlass_scaled_mm_supports_fp8,
        "Check if CUTLASS FP8 scaled_mm is supported on this GPU",
        py::arg("cuda_device_capability"));

  m.def("cutlass_scaled_mm_supports_block_fp8",
        &cutlass_scaled_mm_supports_block_fp8,
        "Check if CUTLASS block FP8 scaled_mm is supported on this GPU",
        py::arg("cuda_device_capability"));

  m.def("cutlass_moe_mm",
        &cutlass_moe_mm,
        "CUTLASS FP8 grouped GEMM for MoE",
        py::arg("out_tensors"),
        py::arg("a_tensors"),
        py::arg("b_tensors"),
        py::arg("a_scales"),
        py::arg("b_scales"),
        py::arg("expert_offsets"),
        py::arg("problem_sizes"),
        py::arg("a_strides"),
        py::arg("b_strides"),
        py::arg("c_strides"),
        py::arg("per_act_token"),
        py::arg("per_out_ch"));

  m.def("get_cutlass_moe_mm_problem_sizes_from_expert_offsets",
        &get_cutlass_moe_mm_problem_sizes_from_expert_offsets,
        "Compute per-expert problem sizes for CUTLASS MoE grouped GEMM",
        py::arg("expert_first_token_offset"),
        py::arg("problem_sizes1"),
        py::arg("problem_sizes2"),
        py::arg("n"),
        py::arg("k"),
        py::arg("swap_ab"));

  m.def("moe_permute",
        &moe_permute,
        "Permute MoE inputs by expert assignment",
        py::arg("input"),
        py::arg("topk_ids"),
        py::arg("token_expert_indices"),
        py::arg("expert_map"),
        py::arg("n_expert"),
        py::arg("n_local_expert"),
        py::arg("topk"),
        py::arg("align_block_size"),
        py::arg("permuted_input"),
        py::arg("expert_first_token_offset"),
        py::arg("inv_permuted_idx"),
        py::arg("permuted_idx"),
        py::arg("m_indices"));

  m.def("moe_unpermute",
        &moe_unpermute,
        "Unpermute and reduce MoE outputs",
        py::arg("permuted_hidden_states"),
        py::arg("topk_weights"),
        py::arg("inv_permuted_idx"),
        py::arg("expert_first_token_offset"),
        py::arg("topk"),
        py::arg("hidden_states"));

  m.def("moe_permute_unpermute_supported",
        &moe_permute_unpermute_supported,
        "Check if moe_permute/moe_unpermute kernels are supported");
}
