#include <torch/extension.h>
#include "per_token_group_quant_8bit.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
