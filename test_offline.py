import argparse
import json
import os
import time

import pandas as pd
from transformers import AutoTokenizer

from pipeswift import LLMEngine
from pipeswift.utils import Request, inference_args, parse_args
from pipeswift.utils.chat_utils import _reasoning_model_family

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "aime26.parquet",
)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--n_sample", type=int, default=4)
    script_args, remaining = parser.parse_known_args()
    engine_args = parse_args(inference_args().parse_args(remaining))
    return script_args, engine_args


def load_messages_list(dataset_path: str, n_sample: int):
    df = pd.read_parquet(dataset_path)
    rows = df.head(n_sample).to_dict(orient="records")
    return [[{"role": "user", "content": row["problem"]}] for row in rows]


def build_chat_template_kwargs(model_name: str) -> dict:
    family = _reasoning_model_family(model_name)
    if family == "deepseek-v3":
        return {"thinking": True}
    if family in {"qwen3", "glm45"}:
        return {"enable_thinking": True}
    return {}

if __name__ == "__main__":
    total_start = time.perf_counter()
    script_args, args = parse_script_args()

    assert os.path.isfile(os.path.join(args.weight_path, "tokenizer.json")), (
        f"--weight_path={args.weight_path} must contain tokenizer.json"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.weight_path, trust_remote_code=True
    )
    messages_list = load_messages_list(DEFAULT_DATASET_PATH, script_args.n_sample)

    # Apply chat template to convert messages to model-specific format
    prompts = []
    chat_template_kwargs = build_chat_template_kwargs(args.model_name)
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        prompts.append(prompt)

    engine_init_start = time.perf_counter()
    llm_engine = LLMEngine(args)
    engine_init_end = time.perf_counter()

    inference_start = time.perf_counter()
    llm_engine.offline_batch_inference(prompts)
    llm_engine.exit()
    inference_end = time.perf_counter()
    total_end = time.perf_counter()

    with open(args.o, "+a") as f:
        f.write(
            json.dumps(
                {
                    "summary": {
                        "dataset_path": DEFAULT_DATASET_PATH,
                        "n_sample": len(prompts),
                        "generate_gather_time_s": round(
                            inference_end - inference_start, 3
                        )
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    print(f"[Timing] Engine init: {engine_init_end - engine_init_start:.3f} s")
    print(f"[Timing] Generate + gather: {inference_end - inference_start:.3f} s")
    print(f"[Timing] Total: {total_end - total_start:.3f} s")
