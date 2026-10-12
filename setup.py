import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
root_dir = os.path.dirname(os.path.abspath(__file__))

# Set TORCH_CUDA_ARCH_LIST to specify GPU architectures
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'

# Common NVCC flags
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--generate-code=arch=compute_90a,code=[sm_90a]",
    "-DENABLE_SCALED_MM_SM90=1",
    "-DENABLE_CUTLASS_MOE_SM90=1",
    "-DNDEBUG",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
]

# CUDA extensions
ext_modules = [
    CUDAExtension(
        name="pipeswift_kernels",
        sources=[
            # Unified bindings
            os.path.join(root_dir, "csrc/pipeswift_bindings.cpp"),
            # MoE kernels (from vLLM)
            os.path.join(root_dir, "csrc/moe/moe_align_sum_kernels.cu"),
            os.path.join(root_dir, "csrc/moe/topk_softmax.cu"),
            os.path.join(root_dir, "csrc/moe/grouped_topk_kernels.cu"),
            os.path.join(root_dir, "csrc/moe/moe_permute_unpermute_op.cu"),
            os.path.join(root_dir, "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"),
            # FP8 quantization kernels
            os.path.join(root_dir, "csrc/fp8_quant/per_token_group_quant.cu"),  # Per-block quantization
            os.path.join(root_dir, "csrc/fp8_quant/persistent_masked_m_silu_mul_quant.cu"),
            os.path.join(root_dir, "csrc/layernorm_kernels.cu"),
            # CUTLASS FP8 kernels
            os.path.join(root_dir, "csrc/cutlass_extensions/common.cpp"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm90.cu"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8.cu"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/moe/moe_data.cu"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x_sm90.cu"),
        ],
        include_dirs=[
            os.path.join(root_dir, "csrc"),
            os.path.join(root_dir, "csrc/moe"),
            os.path.join(root_dir, "csrc/fp8_quant"),
            os.path.join(root_dir, "csrc/cutlass_extensions"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/c3x"),
            os.path.join(root_dir, "csrc/quantization/cutlass_w8a8/moe"),
            os.path.join(root_dir, "_deps/cutlass/include"),
            os.path.join(root_dir, "_deps/cutlass/tools/util/include"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-fPIC"],
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    name="pipeswift",
    version="0.1.0",
    description=(
        "A high-performance pipeline-parallel runtime for large-scale "
        "agentic LLM serving on NVIDIA Hopper GPUs"
    ),
    packages=find_packages(),
    python_requires=">=3.12,<3.13",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    entry_points={"console_scripts": ["pipeswift=pipeswift.server.app:run_server"]},
)
