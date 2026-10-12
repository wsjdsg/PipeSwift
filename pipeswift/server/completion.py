from __future__ import annotations

from pipeswift.server.inference import run_prompt_request
from pipeswift.utils.chat_utils import (
    build_completion_response,
    compute_usage_multi,
    validate_completion_request,
)
from pipeswift.utils.openai_types import CompletionChoice, CompletionRequest


def create_completion(request_dict: dict, tokenizer, engine_bridge, served_model: str):
    request = CompletionRequest.model_validate(request_dict)
    validate_completion_request(request, served_model=served_model)

    completions, finish_reasons = run_prompt_request(
        prompt=request.prompt,
        n=request.n,
        max_tokens=request.max_tokens,
        stop=request.stop,
        tokenizer=tokenizer,
        engine_bridge=engine_bridge,
    )

    choices = []
    for choice_idx, completion in enumerate(completions):
        choices.append(
            CompletionChoice(
                index=choice_idx,
                text=completion,
                finish_reason=finish_reasons[choice_idx],
            )
        )

    usage = compute_usage_multi(tokenizer, request.prompt, completions)
    return build_completion_response(request, choices, usage)
