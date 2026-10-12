from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List, Optional

from .openai_types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorDetail,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    UsageInfo,
)

UNSUPPORTED_FIELDS = (
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


def canonical_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def served_model_aliases(served_model: str) -> set[str]:
    aliases = {served_model}
    aliases.add(f"openai/{served_model}")
    return aliases


def validate_requested_model(model_name: str | None, served_model: str) -> None:
    if canonical_model_name(model_name) != canonical_model_name(served_model):
        raise ValueError(
            f"model '{model_name}' not served by this PipeSwift instance; expected one of {sorted(served_model_aliases(served_model))}"
        )


def normalize_reasoning_request(request: ChatCompletionRequest) -> None:
    if request.chat_template_kwargs is None:
        request.chat_template_kwargs = {}
    else:
        request.chat_template_kwargs = dict(request.chat_template_kwargs)

    chat_template_reasoning_effort = request.chat_template_kwargs.pop(
        "reasoning_effort", None
    )
    if (
        chat_template_reasoning_effort in {"none", "low", "medium", "high"}
        and request.reasoning_effort is None
    ):
        request.reasoning_effort = chat_template_reasoning_effort

    if request.reasoning is not None:
        effort = request.reasoning.effort or request.reasoning.reasoning_effort
        if effort in {"none", "low", "medium", "high"}:
            request.reasoning_effort = effort

        enabled = (
            request.reasoning.enabled
            if request.reasoning.enabled is not None
            else request.reasoning.enable
        )
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if enabled:
            request.chat_template_kwargs.setdefault("thinking", True)

    if request.reasoning_effort == "none":
        request.chat_template_kwargs.setdefault("thinking", False)
        request.chat_template_kwargs.setdefault("enable_thinking", False)

    if not request.chat_template_kwargs:
        request.chat_template_kwargs = None


def _reasoning_model_family(served_model: str) -> str:
    model = served_model.lower()
    if model in {"qwen3", "dpsk-distill-qwen", "qwen3coder"}:
        return "qwen3"
    if model == "dpskv3":
        return "deepseek-v3"
    if model == "glm4":
        return "glm45"
    return "none"


def align_reasoning_request_with_model(
    request: ChatCompletionRequest, served_model: str
) -> None:
    normalize_reasoning_request(request)

    family = _reasoning_model_family(served_model)
    if family == "none":
        return

    if request.chat_template_kwargs is None:
        request.chat_template_kwargs = {}
    else:
        request.chat_template_kwargs = dict(request.chat_template_kwargs)

    enabled = None
    if request.reasoning is not None:
        enabled = (
            request.reasoning.enabled
            if request.reasoning.enabled is not None
            else request.reasoning.enable
        )
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}

    if family == "deepseek-v3":
        if enabled is True:
            request.chat_template_kwargs["thinking"] = True
        elif enabled is False:
            request.chat_template_kwargs["thinking"] = False
        elif request.reasoning_effort == "none":
            request.chat_template_kwargs["thinking"] = False
    elif family in {"qwen3", "glm45"}:
        if enabled is False or request.reasoning_effort == "none":
            request.chat_template_kwargs["enable_thinking"] = False
        else:
            request.chat_template_kwargs.setdefault("enable_thinking", True)

    if not request.chat_template_kwargs:
        request.chat_template_kwargs = None


def validate_chat_request(
    request: ChatCompletionRequest, served_model: str | None = None
) -> None:
    if served_model is not None:
        align_reasoning_request_with_model(request, served_model)
    else:
        normalize_reasoning_request(request)
    if not request.messages:
        raise ValueError("messages cannot be empty")
    if request.stream:
        raise ValueError("stream=true is not supported")
    if request.n < 1:
        raise ValueError("n must be at least 1")
    if request.functions is not None or request.function_call is not None:
        raise ValueError(
            "legacy functions/function_call is not supported; use tools/tool_choice"
        )
    if request.response_format is not None:
        raise ValueError(
            "response_format is not supported; PipeSwift does not support constrained decoding"
        )

    if request.tools == []:
        raise ValueError(
            "`tools` must not be an empty array. Either provide at least one tool or omit the field entirely."
        )

    if request.tools is not None and request.tool_choice is None:
        request.tool_choice = "auto"

    if request.tool_choice == "none":
        pass
    elif request.tool_choice is not None:
        if request.tools is None:
            raise ValueError("When using `tool_choice`, `tools` must be set.")
        is_named_tool_choice = hasattr(request.tool_choice, "function")
        if not is_named_tool_choice and request.tool_choice not in {"auto", "required"}:
            raise ValueError(
                f"Invalid value for `tool_choice`: {request.tool_choice}! "
                'Only named tools, "none", "auto" or "required" are supported.'
            )
        if is_named_tool_choice:
            function_name = request.tool_choice.function.name
            if not any(tool.function.name == function_name for tool in request.tools):
                raise ValueError(
                    "The tool specified in `tool_choice` does not match any of the specified `tools`"
                )

    for field_name in UNSUPPORTED_FIELDS:
        if getattr(request, field_name) is not None:
            raise ValueError(f"{field_name} is not supported")


def validate_completion_request(
    request: CompletionRequest, served_model: str | None = None
) -> None:
    if served_model is not None:
        validate_requested_model(request.model, served_model)
    if not request.prompt:
        raise ValueError("prompt cannot be empty")
    if request.stream:
        raise ValueError("stream=true is not supported")
    if request.n < 1:
        raise ValueError("n must be at least 1")
    for field_name in UNSUPPORTED_FIELDS:
        if getattr(request, field_name) is not None:
            raise ValueError(f"{field_name} is not supported")


def validate_generate_request(
    request: GenerateRequest, served_model: str | None = None
) -> None:
    if served_model is not None and request.model is not None:
        validate_requested_model(request.model, served_model)
    if not request.text:
        raise ValueError("text cannot be empty")
    if request.stream:
        raise ValueError("stream=true is not supported")


def normalize_stop(stop: Optional[str | List[str]]) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [item for item in stop if item]


def clamp_completion_text(text: str, stop_strings: Iterable[str]) -> tuple[str, str]:
    finish_reason = "length"
    final_text = text
    for stop in stop_strings:
        idx = final_text.find(stop)
        if idx >= 0:
            final_text = final_text[:idx]
            finish_reason = "stop"
            break
    return final_text, finish_reason


def normalize_prompt_style_output(text: str) -> str:
    final_text = text.replace("<|im_end|>", "").strip()

    if "<tool_call>" in final_text or "</tool_call>" not in final_text:
        return final_text

    candidate, _, _ = final_text.partition("</tool_call>")
    candidate = candidate.strip()
    if not candidate:
        return final_text

    try:
        parsed = json.loads(candidate)
    except Exception:
        return final_text

    if not isinstance(parsed, dict):
        return final_text
    if "name" not in parsed or "arguments" not in parsed:
        return final_text

    normalized_json = json.dumps(parsed, ensure_ascii=False)
    return f"<tool_call>\n{normalized_json}\n</tool_call>"


def compute_usage(tokenizer, prompt: str, completion: str) -> UsageInfo:
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(completion))
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def compute_usage_multi(tokenizer, prompt: str, completions: List[str]) -> UsageInfo:
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = sum(
        len(tokenizer.encode(completion)) for completion in completions
    )
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def build_chat_response(
    request: ChatCompletionRequest,
    choices: List[ChatCompletionChoice],
    usage: UsageInfo,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=usage,
    )


def build_completion_response(
    request: CompletionRequest,
    choices: List[CompletionChoice],
    usage: UsageInfo,
) -> CompletionResponse:
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=usage,
    )


def build_generate_response(text: str) -> GenerateResponse:
    return GenerateResponse(text=text)


def build_error_response(
    message: str, err_type: str = "invalid_request_error"
) -> ErrorResponse:
    return ErrorResponse(error=ErrorDetail(message=message, type=err_type))
