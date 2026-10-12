from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from .openai_types import ChatCompletionRequest, FunctionCall, ToolCall


@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: list[ToolCall]
    content: Optional[str] = None


def get_json_schema_constraint(
    tools: list | None,
    tool_choice,
) -> Optional[dict[str, Any]]:
    if not tools:
        return None

    if hasattr(tool_choice, "function"):
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name != fn_name:
                continue
            return {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"const": fn_name},
                        "parameters": tool.function.parameters or {"type": "object"},
                    },
                    "required": ["name", "parameters"],
                    "additionalProperties": False,
                },
            }
        return None

    if tool_choice == "required":
        any_of = []
        for tool in tools:
            any_of.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": tool.function.name},
                        "parameters": tool.function.parameters or {"type": "object"},
                    },
                    "required": ["name", "parameters"],
                    "additionalProperties": False,
                }
            )
        return {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": any_of,
            },
        }

    return None


def _requires_constrained_tool_choice(tool_choice) -> bool:
    return tool_choice == "required" or hasattr(tool_choice, "function")


def parse_constrained_tool_calls(
    model_output: str,
) -> list[ToolCall]:
    tool_call_data = json.loads(model_output)
    tool_calls: list[ToolCall] = []
    for item in tool_call_data:
        if not isinstance(item, dict):
            raise ValueError("tool call item must be an object")
        name = item["name"]
        parameters = item["parameters"]
        tool_calls.append(
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(parameters, ensure_ascii=False),
                ),
            )
        )
    return tool_calls


def make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _deserialize_argument(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    return value


def _tool_argument_type(
    tools: Optional[list],
    tool_name: str,
    arg_name: str,
) -> Optional[str]:
    if not tools:
        return None
    for tool in tools:
        if tool.function.name != tool_name:
            continue
        parameters = tool.function.parameters or {}
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return None
        spec = properties.get(arg_name, {})
        if not isinstance(spec, dict):
            return None
        arg_type = spec.get("type")
        return arg_type if isinstance(arg_type, str) else None
    return None


class BaseNonStreamingToolParser:
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError


class LlamaToolParser(BaseNonStreamingToolParser):
    def __init__(self) -> None:
        self.bot_token = "<|python_tag|>"
        self.tool_call_separator = ";"
        self.json_decoder = json.JSONDecoder()

    def _convert_python_dict_to_json(self, text: str) -> str:
        try:
            parsed = ast.literal_eval(text.strip())
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.bot_token not in model_output and not model_output.startswith("{"):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if self.bot_token in model_output:
            normal_text, action_text = model_output.split(self.bot_token, maxsplit=1)
        else:
            normal_text, action_text = "", model_output

        idx = 0
        safe_idx = 0
        tool_calls: list[ToolCall] = []
        try:
            while idx < len(action_text):
                try:
                    obj, end = self.json_decoder.raw_decode(action_text[idx:])
                    idx += end + len(self.tool_call_separator)
                    safe_idx = idx
                except json.JSONDecodeError:
                    dict_end = idx
                    brace_count = 0
                    for i in range(idx, len(action_text)):
                        if action_text[i] == "{":
                            brace_count += 1
                        elif action_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                dict_end = i + 1
                                break
                    if dict_end > idx:
                        potential_dict = action_text[idx:dict_end]
                        json_version = self._convert_python_dict_to_json(potential_dict)
                        if json_version != potential_dict:
                            obj, _ = self.json_decoder.raw_decode(json_version)
                            idx = dict_end + len(self.tool_call_separator)
                            safe_idx = idx
                        else:
                            next_obj_start = action_text.find('{"name":', idx + 1)
                            if next_obj_start == -1:
                                break
                            idx = next_obj_start
                            continue
                    else:
                        next_obj_start = action_text.find('{"name":', idx + 1)
                        if next_obj_start == -1:
                            break
                        idx = next_obj_start
                        continue

                name = obj["name"]
                arguments_or_params = (
                    obj["arguments"] if "arguments" in obj else obj["parameters"]
                )
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(
                                arguments_or_params,
                                ensure_ascii=False,
                            ),
                        ),
                    )
                )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if tool_calls:
            trailing_text = (
                action_text[safe_idx:].strip() if safe_idx < len(action_text) else ""
            )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=(normal_text + trailing_text) or None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )


class DeepSeekV3ToolParser(BaseNonStreamingToolParser):
    def __init__(self) -> None:
        self.tool_calls_start_token = "<｜tool▁calls▁begin｜>"
        self.func_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>", re.DOTALL
        )
        self.func_detail_regex = re.compile(
            r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>",
            re.DOTALL,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )
        try:
            matches = self.func_call_regex.findall(model_output)
            tool_calls = []
            for match in matches:
                func_detail = self.func_detail_regex.search(match)
                if not func_detail:
                    continue
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_detail.group(2),
                            arguments=json.dumps(
                                json.loads(func_detail.group(3)),
                                ensure_ascii=False,
                            ),
                        ),
                    )
                )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        content = model_output[: model_output.find(self.tool_calls_start_token)]
        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content if content else None,
        )


class Glm47ToolParser(BaseNonStreamingToolParser):
    def __init__(self) -> None:
        self.bot_token = "<tool_call>"
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>",
            re.DOTALL,
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        matched_tool_calls = self.func_call_regex.findall(model_output)
        try:
            normal_text_parts = []
            last_end = 0
            for match in self.func_call_regex.finditer(model_output):
                if match.start() > last_end:
                    normal_text_parts.append(model_output[last_end : match.start()])
                last_end = match.end()
            if last_end < len(model_output):
                normal_text_parts.append(model_output[last_end:])
            normal_text = "".join(normal_text_parts).strip()

            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                details = self.func_detail_regex.search(match)
                if not details:
                    continue
                tool_name = (details.group(1) or "").strip()
                tool_args = details.group(2) or ""
                pairs = self.func_arg_regex.findall(tool_args)
                arg_dict: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    arg_type = _tool_argument_type(request.tools, tool_name, arg_key)
                    if arg_type != "string":
                        arg_val = _deserialize_argument(arg_val)
                    arg_dict[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(arg_dict, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=normal_text or None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )


class Qwen25ToolParser(BaseNonStreamingToolParser):
    def __init__(self) -> None:
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if _requires_constrained_tool_choice(request.tool_choice):
            try:
                tool_calls = parse_constrained_tool_calls(model_output.strip())
            except Exception:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content="",
            )

        idx = model_output.find(self.bot_token)
        normal_text = model_output[:idx].strip() if idx != -1 else model_output
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=normal_text,
            )

        try:
            pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
            match_result_list = re.findall(pattern, model_output, re.DOTALL)
            tool_calls = []
            for match_result in match_result_list:
                parsed_call = json.loads(match_result.strip())
                name = parsed_call["name"]
                arguments_or_params = (
                    parsed_call["arguments"]
                    if "arguments" in parsed_call
                    else parsed_call["parameters"]
                )
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(
                                arguments_or_params, ensure_ascii=False
                            ),
                        ),
                    )
                )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=normal_text or None,
        )


class Qwen3CoderToolParser(BaseNonStreamingToolParser):
    def __init__(self) -> None:
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_prefix = "<function="
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

    def _get_arguments_config(
        self,
        func_name: str,
        tools: Optional[list],
    ) -> dict:
        if tools is None:
            return {}
        for tool in tools:
            if tool.type != "function" or tool.function.name != func_name:
                continue
            params = tool.function.parameters
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            if isinstance(params, dict):
                return params
            return {}
        return {}

    def _convert_param_value(
        self,
        param_value: str,
        param_name: str,
        param_config: dict,
    ) -> Any:
        if param_value.lower() == "null":
            return None
        if param_name not in param_config:
            return param_value

        spec = param_config.get(param_name)
        if not isinstance(spec, dict):
            return param_value
        param_type = str(spec.get("type", "string")).strip().lower()
        if param_type in {"string", "str", "text", "varchar", "char", "enum"}:
            return param_value
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except Exception:
                return param_value
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    "." not in param_value and "e" not in param_value.lower()
                )
                converted = float(param_value)
                if maybe_convert and converted.is_integer():
                    return int(converted)
                return converted
            except Exception:
                return param_value
        if param_type in {"boolean", "bool", "binary"}:
            return param_value.lower() == "true"
        if (
            param_type in {"object", "array", "arr"}
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value)
            except Exception:
                pass
        try:
            return ast.literal_eval(param_value)
        except Exception:
            return param_value

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            raw_tool_calls = self.tool_call_regex.findall(model_output)
            if not raw_tool_calls and self.tool_call_prefix in model_output:
                raw_tool_calls = [model_output]

            tool_calls: list[ToolCall] = []
            for tool_content in raw_tool_calls:
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue
                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]
                    param_config = self._get_arguments_config(func_name, request.tools)
                    parsed_params: dict[str, Any] = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]
                        parsed_params[p_name] = self._convert_param_value(
                            p_val,
                            p_name,
                            param_config,
                        )

                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=json.dumps(parsed_params, ensure_ascii=False),
                            ),
                        )
                    )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if tool_calls:
            start_idx = model_output.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = model_output.find(self.tool_call_prefix)
            normal_text = model_output[:start_idx] if start_idx > 0 else ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=normal_text or None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )


def get_tool_parser_for_model(model_name: str) -> Optional[BaseNonStreamingToolParser]:
    model = model_name.lower()
    if model == "qwen3coder":
        return Qwen3CoderToolParser()
    if model.startswith("qwen3"):
        return Qwen25ToolParser()
    if model == "dpsk-distill-qwen":
        return Qwen25ToolParser()
    if model == "glm4":
        return Glm47ToolParser()
    if model in {"dpskv3"}:
        return DeepSeekV3ToolParser()
    if model in {"llama2", "llama3"}:
        return LlamaToolParser()
    return None
