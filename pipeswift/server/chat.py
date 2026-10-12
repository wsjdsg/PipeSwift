from __future__ import annotations

from pipeswift.server.inference import run_prompt_request
from pipeswift.utils.chat_template import render_messages_to_prompt
from pipeswift.utils.chat_utils import (
    build_chat_response,
    compute_usage_multi,
    validate_chat_request,
    validate_requested_model,
)
from pipeswift.utils.openai_types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponseMessage,
    ToolCall,
)
from pipeswift.utils.tool_parsers import get_tool_parser_for_model, make_tool_call_id


def _parse_tool_response(
    completion: str,
    request: ChatCompletionRequest,
    served_model: str,
) -> tuple[str | None, list[ToolCall], bool]:
    if request.tools is None or request.tool_choice == "none":
        return completion, [], False

    parser = get_tool_parser_for_model(served_model)
    if parser is None:
        return completion, [], False

    extracted = parser.extract_tool_calls(completion, request)
    if not extracted.tools_called or not extracted.tool_calls:
        return completion, [], False

    response_tool_calls: list[ToolCall] = []
    for tool_call in extracted.tool_calls:
        response_tool_calls.append(
            ToolCall(
                id=tool_call.id or make_tool_call_id(),
                type="function",
                function=tool_call.function,
            )
        )
    return extracted.content, response_tool_calls, True


def create_chat_completion(
    request_dict: dict, tokenizer, engine_bridge, served_model: str
):
    request = ChatCompletionRequest.model_validate(request_dict)
    validate_chat_request(request, served_model=served_model)
    validate_requested_model(request.model, served_model)
    prompt = render_messages_to_prompt(
        request.messages,
        tokenizer,
        reasoning_effort=request.reasoning_effort,
        chat_template_kwargs=request.chat_template_kwargs,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    max_tokens = request.max_completion_tokens or request.max_tokens
    completions, finish_reasons = run_prompt_request(
        prompt=prompt,
        n=request.n,
        max_tokens=max_tokens,
        stop=request.stop,
        tokenizer=tokenizer,
        engine_bridge=engine_bridge,
    )

    choices = []
    rendered_completions = []
    for choice_idx, completion in enumerate(completions):
        finish_reason = finish_reasons[choice_idx]

        response_content, response_tool_calls, tools_called = _parse_tool_response(
            completion,
            request,
            served_model,
        )

        if tools_called and request.tool_choice != "none":
            finish_reason = "tool_calls"
        rendered_completions.append(completion or "")
        choices.append(
            ChatCompletionChoice(
                index=choice_idx,
                message=ChatCompletionResponseMessage(
                    content=response_content,
                    tool_calls=response_tool_calls or None,
                ),
                finish_reason=finish_reason,
            )
        )
    usage = compute_usage_multi(tokenizer, prompt, rendered_completions)
    return build_chat_response(
        request,
        choices,
        usage,
    )
