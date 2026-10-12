from __future__ import annotations

from array import array

from pipeswift.utils.chat_utils import (
    clamp_completion_text,
    normalize_prompt_style_output,
    normalize_stop,
)


def clip_completion_tokens(tokenizer, completion: str, max_tokens: int | None):
    if max_tokens is None:
        return completion, False
    token_ids = tokenizer.encode(completion)
    if len(token_ids) <= max_tokens:
        return completion, False
    return tokenizer.decode(token_ids[:max_tokens]), True


def map_finish_reason(raw_reason):
    if not raw_reason:
        return "stop"
    reason_type = raw_reason.get("type")
    if reason_type == "length":
        return "length"
    return "stop"


def encode_prompt_tokens(
    *,
    prompt: str,
    tokenizer,
    max_tokens: int | None,
    max_context_len: int,
) -> array:
    prompt_token_ids = array("i", tokenizer.encode(prompt))
    prompt_len = len(prompt_token_ids)
    if prompt_len == 0:
        raise ValueError("Input length (0) is invalid; prompt cannot be empty.")
    if prompt_len >= max_context_len:
        raise ValueError(
            f"Input length ({prompt_len}) exceeds model's maximum context length ({max_context_len})."
        )
    return prompt_token_ids


def run_prompt_request(
    *,
    prompt: str,
    n: int,
    max_tokens: int | None,
    stop,
    tokenizer,
    engine_bridge,
):
    prompt_token_ids = encode_prompt_tokens(
        prompt=prompt,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_context_len=engine_bridge.engine.args.max_context_len,
    )
    request_id = engine_bridge.submit_prompt(
        prompt,
        prompt_token_ids=prompt_token_ids,
        n_seqs=n,
        max_gen_tokens=max_tokens,
    )
    result = engine_bridge.wait_for_result(request_id)

    completions: list[str] = []
    finish_reasons: list[str] = []
    for choice_idx in range(n):
        completion = result["responses"][str(choice_idx + 1)]
        raw_finish_reason = result.get("finish_reasons", {}).get(str(choice_idx + 1))
        completion, truncated = clip_completion_tokens(
            tokenizer, completion, max_tokens
        )
        completion, stop_finish_reason = clamp_completion_text(
            completion,
            normalize_stop(stop),
        )
        completion = normalize_prompt_style_output(completion)
        finish_reason = map_finish_reason(raw_finish_reason)
        if stop_finish_reason == "stop":
            finish_reason = "stop"
        if truncated and finish_reason != "stop":
            finish_reason = "length"
        completions.append(completion or "")
        finish_reasons.append(finish_reason)

    return completions, finish_reasons
