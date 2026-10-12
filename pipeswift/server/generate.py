from __future__ import annotations

from pipeswift.server.inference import run_prompt_request
from pipeswift.utils.chat_utils import (
    build_generate_response,
    validate_generate_request,
)
from pipeswift.utils.openai_types import GenerateRequest


def create_generate(request_dict: dict, tokenizer, engine_bridge, served_model: str):
    request = GenerateRequest.model_validate(request_dict)
    validate_generate_request(request, served_model=served_model)

    sampling_params = request.sampling_params
    max_tokens = None if sampling_params is None else sampling_params.max_new_tokens
    stop = None if sampling_params is None else sampling_params.stop

    completions, _ = run_prompt_request(
        prompt=request.text,
        n=1,
        max_tokens=max_tokens,
        stop=stop,
        tokenizer=tokenizer,
        engine_bridge=engine_bridge,
    )
    return build_generate_response(completions[0])
