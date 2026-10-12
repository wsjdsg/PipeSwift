from __future__ import annotations

import json
from typing import Dict, List, Optional

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils

from .openai_types import ChatCompletionToolParam, ChatMessage


def _normalize_content(
    message: ChatMessage,
    *,
    preserve_structured_content: bool,
) -> Optional[str | List[dict[str, str]]]:
    content = message.content
    if content is None or isinstance(content, str):
        return content

    if preserve_structured_content:
        return [part.model_dump(exclude_none=True) for part in content]

    if message.role == "user":
        return "".join(part.text for part in content)

    if len(content) != 1:
        raise ValueError(f"{message.role} message should be a single text part")
    return content[0].text


def _normalize_tool_call_arguments(arguments: object) -> object:
    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _normalize_tool_call(tool_call) -> dict:
    item = tool_call.model_dump(exclude_none=True)
    function = item.get("function")
    if isinstance(function, dict) and "arguments" in function:
        function["arguments"] = _normalize_tool_call_arguments(function["arguments"])
    return item


def _to_hf_message(
    message: ChatMessage,
    *,
    preserve_structured_content: bool,
) -> dict:
    item = {
        "role": message.role,
        "content": _normalize_content(
            message,
            preserve_structured_content=preserve_structured_content,
        ),
    }
    if message.name is not None:
        item["name"] = message.name
    if message.tool_call_id is not None:
        item["tool_call_id"] = message.tool_call_id
    if message.function_call is not None:
        item["function_call"] = message.function_call.model_dump(exclude_none=True)
    if message.tool_calls is not None:
        item["tool_calls"] = [
            _normalize_tool_call(tool_call) for tool_call in message.tool_calls
        ]
    return item


def _build_chat_template_kwargs(
    chat_template_kwargs: Optional[Dict[str, object]],
    reasoning_effort: Optional[str],
) -> Dict[str, object]:
    kwargs = dict(chat_template_kwargs or {})
    if reasoning_effort is not None:
        kwargs.setdefault("reasoning_effort", reasoning_effort)
    return kwargs


def _to_hf_tools(
    tools: Optional[List[ChatCompletionToolParam]],
    tool_choice,
) -> Optional[List[dict]]:
    if tools is None or tool_choice == "none":
        return None
    return [tool.model_dump(exclude_none=True) for tool in tools]


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str | None = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _try_extract_ast(chat_template: str):
    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception:
        return None


def _detect_jinja_template_content_format(chat_template: str | None) -> str:
    if not chat_template:
        return "string"

    if any(
        keyword in chat_template for keyword in ["image", "audio", "video", "vision"]
    ):
        return "openai"

    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return "string"

    try:
        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):
            loop_iter = loop_ast.iter
            if _is_var_or_elems_access(loop_iter, "message", "content"):
                return "openai"
            if _is_var_or_elems_access(
                loop_iter, "msg", "content"
            ) or _is_var_or_elems_access(loop_iter, "m", "content"):
                return "openai"
        return "string"
    except Exception:
        return "string"


def _should_preserve_structured_content(tokenizer) -> bool:
    return (
        _detect_jinja_template_content_format(getattr(tokenizer, "chat_template", None))
        == "openai"
    )


def render_messages_to_prompt(
    messages: List[ChatMessage],
    tokenizer,
    reasoning_effort: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, object]] = None,
    tools: Optional[List[ChatCompletionToolParam]] = None,
    tool_choice=None,
) -> str:
    if not messages:
        raise ValueError("messages cannot be empty")

    preserve_structured_content = _should_preserve_structured_content(tokenizer)
    hf_messages = [
        _to_hf_message(
            message,
            preserve_structured_content=preserve_structured_content,
        )
        for message in messages
    ]
    template_kwargs = _build_chat_template_kwargs(
        chat_template_kwargs, reasoning_effort
    )
    hf_tools = _to_hf_tools(tools, tool_choice)
    try:
        return tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=hf_tools,
            **template_kwargs,
        )
    except Exception as e:
        raise ValueError(f"failed to apply chat template: {e}") from e
