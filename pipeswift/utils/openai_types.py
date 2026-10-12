from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool | str] = None
    enable: Optional[bool | str] = None
    effort: Optional[Literal["none", "low", "medium", "high"]] = None
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None


class ChatMessageTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[ChatMessageTextPart]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    function_call: Optional["FunctionCall"] = None
    tool_calls: Optional[List["ToolCall"]] = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[ChatMessage]
    model: str
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    n: int = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    user: Optional[str] = None
    skip_special_tokens: Optional[bool] = None

    # NOTE(wsj): sampling args are fixed at engine startup; per-request overrides
    # are intentionally unsupported by PipeSwift's chat/completions server.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    reasoning: Optional[ReasoningConfig] = None
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[ChatCompletionToolParam]] = None
    tool_choice: Optional[
        Union[
            Literal["none", "auto", "required"],
            ChatCompletionNamedToolChoiceParam,
        ]
    ] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None


class ChatCompletionResponseMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    max_tokens: Optional[int] = Field(default=None)
    n: int = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    user: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    skip_special_tokens: Optional[bool] = None
    reasoning: Optional[ReasoningConfig] = None
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None
    thinking: Optional[Dict[str, Any]] = None
    provider: Optional[Dict[str, Any]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    finish_reason: Literal["stop", "length"]


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class GenerateSamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    skip_special_tokens: Optional[bool] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    text: str
    sampling_params: Optional[GenerateSamplingParams] = None
    stream: bool = False
    skip_special_tokens: Optional[bool] = None
    main_key: Optional[str] = None


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
