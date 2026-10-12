# Third-Party Notices

PipeSwift incorporates or adapts code from open-source projects. The applicable
copyright and license notices in the source files remain in effect.

## vLLM

Selected MoE, FP8 quantization, and CUTLASS integration code under `csrc/` is
derived from the [vLLM project](https://github.com/vllm-project/vllm), licensed
under the Apache License 2.0.

## NVIDIA CUTLASS

Selected headers under `csrc/cutlass_extensions/` are derived from
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) and retain their
BSD-3-Clause notices. The build also downloads a pinned CUTLASS revision into
the ignored `_deps/` directory.

## Runtime Dependencies

The build script installs FlashAttention, FlashInfer, FlashMLA, DeepEP, and
DeepGEMM from their upstream projects. Those packages are not vendored in this
repository and remain governed by their respective licenses.

This file is informational and does not replace the license text associated
with PipeSwift or any third-party component.
