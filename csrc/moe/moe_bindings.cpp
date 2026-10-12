#include <torch/extension.h>
#include "moe_ops.h"

// Simplified bindings - only essential MoE operations
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
