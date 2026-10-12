from typing import Tuple, Union

import flashinfer
import torch

from pipeswift.utils import QuantType
from pipeswift.utils.quantization import *


class QuantLinear(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        with_bias,
        dtype,
        quant_type,
        fp8_gemm_backend: str = "vllm_cutlass",
    ):
        super().__init__()
        self.quant_type = quant_type
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.with_bias = with_bias
        self.bias = None
        self.dtype = dtype
        self.fp8_gemm_backend = fp8_gemm_backend

        if self.with_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_dim, device="cuda", dtype=dtype)
            )
        assert dtype in [torch.bfloat16, torch.float16]
        if self.quant_type == QuantType.NO_QUANT:
            self.weight = torch.nn.Parameter(
                torch.empty((out_dim, inp_dim), device="cuda", dtype=dtype)
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(
                    (out_dim, inp_dim), device="cuda", dtype=torch.float8_e4m3fn
                )
            )
            if self.quant_type == QuantType.FP8_PER_BLOCK:
                assert inp_dim % BLOCK_QUANT_SIZE == 0
                # NOTE(wsj): allow out_dim % 128 != 0
                self.weight_scale = torch.nn.Parameter(
                    torch.empty(
                        (
                            (out_dim + BLOCK_QUANT_SIZE - 1) // BLOCK_QUANT_SIZE,
                            inp_dim // BLOCK_QUANT_SIZE,
                        ),
                        device="cuda",
                        dtype=torch.float32,
                    )
                )
            elif self.quant_type == QuantType.FP8_PER_CHANNEL:
                self.weight_scale = torch.nn.Parameter(
                    torch.empty((out_dim, 1), device="cuda", dtype=torch.float32)
                )

    def forward(self, inp: torch.Tensor):
        return quant_matmul(
            inp,
            self.weight
            if self.quant_type == QuantType.NO_QUANT
            else (self.weight, self.weight_scale),
            dtype=self.dtype,
            quant_type=self.quant_type,
            bias=self.bias,
            fp8_gemm_backend=self.fp8_gemm_backend,
        )


def quant_matmul(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    y: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    dtype,
    quant_type: QuantType,
    bias: torch.Tensor = None,
    fp8_gemm_backend: str = "vllm_cutlass",
):  # x @ y.t() + bias
    if quant_type == QuantType.NO_QUANT:
        return torch.nn.functional.linear(x, y, bias)
    else:
        weight, weight_scale = y[0], y[1]
        if quant_type == QuantType.FP8_PER_BLOCK:
            if not isinstance(x, tuple):
                inp_fp8, inp_scale = per_token_group_quant_fp8_blockwise_vllm(
                    x,
                    group_size=BLOCK_QUANT_SIZE,
                    use_ue8m0=False,
                    column_major_scales=True,
                )
            else:
                inp_fp8, inp_scale = x

            out = torch.empty(
                inp_fp8.shape[0], y[0].shape[0], device="cuda", dtype=dtype
            )
            if fp8_gemm_backend == "vllm_cutlass" and inp_fp8.shape[0] % 4 == 0:
                import pipeswift_kernels

                pipeswift_kernels.cutlass_scaled_mm(
                    out, inp_fp8, weight.t(), inp_scale, weight_scale.t(), bias=bias
                )
            else:
                assert dtype == torch.bfloat16, "DeepGEMM forces bfloat16"
                from deep_gemm import fp8_gemm_nt

                fp8_gemm_nt(
                    a=(inp_fp8, inp_scale),
                    b=(weight, weight_scale),
                    d=out,
                    c=None,
                    disable_ue8m0_cast=True,
                )
                if bias is not None:
                    out += bias
        elif quant_type == QuantType.FP8_PER_CHANNEL:
            import pipeswift_kernels

            # Quantize input with per-token scale
            if not isinstance(x, tuple):
                inp_fp8, inp_scale = per_token_group_quant_fp8_channelwise_torchnative(
                    x
                )
            else:
                inp_fp8, inp_scale = x
            out = torch.empty(
                inp_fp8.shape[0], y[0].shape[0], device="cuda", dtype=dtype
            )

            if fp8_gemm_backend == "vllm_cutlass":
                assert weight.is_contiguous()
                pipeswift_kernels.cutlass_scaled_mm(
                    out, inp_fp8, weight.t(), inp_scale, weight_scale, bias=bias
                )
            elif fp8_gemm_backend == "torch_native":
                torch._scaled_mm(
                    inp_fp8,
                    weight.t(),
                    inp_scale,
                    weight_scale.t(),
                    bias=bias,
                    out_dtype=dtype,
                    out=out,
                )
                return out  # bias can be fused here
            else:
                raise NotImplementedError(
                    f"Not supported backend {fp8_gemm_backend} for quant_type per_channel"
                )

        return out


class QuantExperts(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        moe_intermediate_size,
        dtype,
        quant_type: QuantType,
    ):
        super().__init__()
        assert dtype in [torch.bfloat16, torch.float16]
        from pipeswift.utils import parallel_state

        self.dtype = dtype
        self.quant_type = quant_type
        self.num_experts_per_device = num_experts
        self.tot_num_experts = num_experts * parallel_state.ep_deg
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        proj_dtype = (
            dtype if self.quant_type == QuantType.NO_QUANT else torch.float8_e4m3fn
        )
        self.w1 = torch.nn.Parameter(
            torch.empty(
                (
                    self.num_experts_per_device,
                    self.moe_intermediate_size * 2 // parallel_state.etp_deg,
                    self.hidden_size,
                ),
                device="cuda",
                dtype=proj_dtype,
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                (
                    self.num_experts_per_device,
                    self.hidden_size,
                    self.moe_intermediate_size // parallel_state.etp_deg,
                ),
                device="cuda",
                dtype=proj_dtype,
            )
        )
        if self.quant_type == QuantType.FP8_PER_BLOCK:
            assert (
                hidden_size % BLOCK_QUANT_SIZE == 0
                and moe_intermediate_size % BLOCK_QUANT_SIZE == 0
            )
            self.w1_scale = torch.nn.Parameter(
                torch.empty(
                    (
                        self.num_experts_per_device,
                        self.moe_intermediate_size
                        // parallel_state.etp_deg
                        // BLOCK_QUANT_SIZE
                        * 2,
                        self.hidden_size // BLOCK_QUANT_SIZE,
                    ),
                    device="cuda",
                    dtype=torch.float32,
                )
            )
            self.w2_scale = torch.nn.Parameter(
                torch.empty(
                    (
                        self.num_experts_per_device,
                        self.hidden_size // BLOCK_QUANT_SIZE,
                        self.moe_intermediate_size
                        // parallel_state.etp_deg
                        // BLOCK_QUANT_SIZE,
                    ),
                    device="cuda",
                    dtype=torch.float32,
                )
            )
        elif self.quant_type == QuantType.FP8_PER_CHANNEL:
            self.w1_scale = torch.nn.Parameter(
                torch.empty(
                    (
                        self.num_experts_per_device,
                        self.moe_intermediate_size * 2 // parallel_state.etp_deg,
                        1,
                    ),
                    device="cuda",
                    dtype=torch.float32,
                )
            )
            self.w2_scale = torch.nn.Parameter(
                torch.empty(
                    (
                        self.num_experts_per_device,
                        self.hidden_size,
                        1,
                    ),
                    device="cuda",
                    dtype=torch.float32,
                )
            )
            # required for vllm_cutlass kernel
            e = self.num_experts_per_device
            n = self.moe_intermediate_size // parallel_state.etp_deg
            ab_strides1_c_strides2 = torch.full(
                (e,), hidden_size, device="cuda", dtype=torch.int64
            )
            ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
            c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
            self.ab_strides1 = ab_strides1_c_strides2
            self.ab_strides2 = ab_strides2
            self.c_strides1 = c_strides1
            self.c_strides2 = ab_strides1_c_strides2
            if parallel_state.ep_deg > 1:
                self.expert_map = torch.full(
                    (self.tot_num_experts,),
                    -1,
                    device="cuda",
                    dtype=torch.int32,
                )
                expert_st_idx = self.num_experts_per_device * parallel_state.ep_rank
                expert_ed_idx = expert_st_idx + self.num_experts_per_device
                self.expert_map[expert_st_idx:expert_ed_idx] = torch.arange(
                    self.num_experts_per_device, device="cuda", dtype=torch.int32
                )
            else:
                self.expert_map = None
        else:
            self.expert_map = None

    def forward_contiguous(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        from pipeswift.utils import parallel_state

        # skip if doesn't have any tokens
        if isinstance(hidden_states, tuple):
            assert self.quant_type == QuantType.FP8_PER_BLOCK
            if hidden_states[0].shape[0] == 0:
                return torch.empty(
                    hidden_states[0].shape, dtype=self.dtype, device="cuda"
                )
        else:
            if hidden_states.shape[0] == 0:
                return hidden_states

        if self.quant_type == QuantType.NO_QUANT:
            flashinfer.fused_moe.cutlass_fused_moe(
                input=hidden_states,
                token_selected_experts=topk_idx.to(torch.int32),
                token_final_scales=topk_weights,
                fc1_expert_weights=self.w1,
                fc2_expert_weights=self.w2,
                output_dtype=hidden_states.dtype,
                quant_scales=None,
                output=hidden_states,
                ep_rank=parallel_state.ep_rank,
                ep_size=parallel_state.ep_deg,
            )
        elif self.quant_type == QuantType.FP8_PER_BLOCK:
            if isinstance(hidden_states, tuple):  # requires bf16 act
                hidden_states = fp8_block_dequant(hidden_states[0], hidden_states[1])

            flashinfer.fused_moe.cutlass_fused_moe(
                input=hidden_states,
                token_selected_experts=topk_idx.to(torch.int32),
                token_final_scales=topk_weights,
                fc1_expert_weights=self.w1,
                fc2_expert_weights=self.w2,
                output_dtype=hidden_states.dtype,
                quant_scales=[self.w1_scale, self.w2_scale],
                use_deepseek_fp8_block_scale=self.quant_type == QuantType.FP8_PER_BLOCK,
                output=hidden_states,
                ep_rank=parallel_state.ep_rank,
                ep_size=parallel_state.ep_deg,
            )
        elif self.quant_type == QuantType.FP8_PER_CHANNEL:
            a1q, a1q_scale = per_token_group_quant_fp8_channelwise_torchnative(
                hidden_states
            )
            vllm_cutlass_fp8_moe(
                a1q=a1q,
                a1q_scale=a1q_scale,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                w1=self.w1,
                w1_scale=self.w1_scale,
                w2=self.w2,
                w2_scale=self.w2_scale,
                out=hidden_states,
                ab_strides1=self.ab_strides1,
                ab_strides2=self.ab_strides2,
                c_strides1=self.c_strides1,
                c_strides2=self.c_strides2,
                global_num_experts=self.tot_num_experts,
                expert_map=self.expert_map,
            )

        return hidden_states

    def forward_masked(
        self,
        hidden_states: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: torch.Tensor,
    ):
        from pipeswift.utils import mul_and_silu_masked

        if self.quant_type == QuantType.NO_QUANT:
            assert hidden_states.dim() == 3
            from deep_gemm import m_grouped_bf16_gemm_nt_masked

            _gemm_out = torch.empty(
                hidden_states.shape[:-1] + (self.w1.shape[1],),
                device="cuda",
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                hidden_states, self.w1, _gemm_out, masked_m, expected_m
            )
            _gemm_out = mul_and_silu_masked(_gemm_out, masked_m)
            m_grouped_bf16_gemm_nt_masked(
                _gemm_out, self.w2, hidden_states, masked_m, expected_m
            )
            return hidden_states
        elif self.quant_type == QuantType.FP8_PER_BLOCK:
            if not isinstance(hidden_states, tuple):
                inp_fp8, inp_scale = masked_fp8_block_quant(hidden_states, masked_m)
            else:
                inp_fp8, inp_scale = hidden_states
            assert inp_fp8.dim() == 3
            from deep_gemm import m_grouped_fp8_gemm_nt_masked

            _gemm_out = torch.empty(
                inp_fp8.shape[:-1] + (self.w1.shape[1],),
                device="cuda",
                dtype=torch.bfloat16,
            )
            m_grouped_fp8_gemm_nt_masked(
                (inp_fp8, inp_scale),
                (self.w1, self.w1_scale),
                _gemm_out,
                masked_m,
                expected_m,
                disable_ue8m0_cast=True,
            )
            _gemm_out_fp8, _gemm_out_scale = fused_masked_m_silu_mul_fp8_block_quant(
                _gemm_out, masked_m, reverse=True
            )
            hidden_states = torch.empty(
                inp_fp8.shape,
                device="cuda",
                dtype=torch.bfloat16,
            )
            m_grouped_fp8_gemm_nt_masked(
                (_gemm_out_fp8, _gemm_out_scale),
                (self.w2, self.w2_scale),
                hidden_states,
                masked_m,
                expected_m,
                disable_ue8m0_cast=True,
            )
            return hidden_states
