import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from random import Random

import pandas as pd

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "aime26.parquet",
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--server_url", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--min_n", type=int, default=1)
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument(
        "--reasoning_effort_choices",
        type=str,
        default="none,low,medium,high",
        help="Comma-separated reasoning_effort choices sampled per request.",
    )
    parser.add_argument("--n_sample", type=int, default=4)
    parser.add_argument("--max_inject_delay_s", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server_start_timeout_s", type=float, default=600.0)
    parser.add_argument("--keep_server_alive", action="store_true")
    parser.add_argument(
        "--request_log_path",
        "-o",
        type=str,
        default=None,
    )
    return parser.parse_known_args()


def resolve_output_path(path: str):
    if path is None:
        path = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    if os.path.isabs(path):
        return path
    return os.path.join(DEFAULT_OUTPUT_DIR, path)


def resolve_pipeswift_output_path(path: str, model_name: str):
    if path is None:
        path = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}-{model_name}"
    return os.path.abspath(os.path.join(DEFAULT_OUTPUT_DIR, f"{path}.jsonl"))


def load_problems(n_sample: int):
    df = pd.read_parquet(DEFAULT_DATASET_PATH)
    return df.head(n_sample).to_dict(orient="records")


def health_url_from_server_url(server_url: str):
    suffix = "/v1/chat/completions"
    if server_url.endswith(suffix):
        return server_url[: -len(suffix)] + "/health"
    return server_url.rstrip("/") + "/health"


def wait_for_server_ready(server_url: str, timeout_s: float):
    health_url = health_url_from_server_url(server_url)
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"timed out waiting for server readiness at {health_url}: {last_error}"
    )


def launch_local_server(args, remaining):
    server_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    cmd = [
        sys.executable,
        "-m",
        "pipeswift.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model_name",
        args.model_name,
        "-o",
        args.server_output_arg,
    ]
    cmd.extend(remaining)

    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__),
    )
    try:
        wait_for_server_ready(server_url, args.server_start_timeout_s)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        raise
    return proc, None, server_url


def post_chat_completion(
    server_url: str,
    model_name: str,
    problem: str,
    n: int,
    reasoning_effort: str | None,
):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": problem}],
        "stream": False,
        "n": n,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8"), response.status


def main():
    args, remaining = parse_args()
    if args.min_n < 1:
        raise ValueError("min_n must be at least 1")
    if args.max_n < args.min_n:
        raise ValueError("max_n must be greater than or equal to min_n")
    reasoning_effort_choices = [
        item.strip()
        for item in args.reasoning_effort_choices.split(",")
        if item.strip()
    ]
    allowed_reasoning_efforts = {"none", "low", "medium", "high"}
    if any(item not in allowed_reasoning_efforts for item in reasoning_effort_choices):
        raise ValueError(
            "reasoning_effort_choices must be a comma-separated subset of "
            "{none,low,medium,high}"
        )
    if not reasoning_effort_choices:
        reasoning_effort_choices = [None]
    args.server_output_arg = args.request_log_path
    if args.server_output_arg is None:
        args.server_output_arg = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}-{args.model_name}"
    if args.server_url is None:
        args.request_log_path = resolve_pipeswift_output_path(
            args.server_output_arg,
            args.model_name,
        )
    else:
        args.request_log_path = resolve_output_path(args.request_log_path)
    os.makedirs(os.path.dirname(args.request_log_path), exist_ok=True)
    server_proc = None
    server_log_file = None
    server_url = args.server_url
    if server_url is None:
        server_proc, server_log_file, server_url = launch_local_server(
            args,
            remaining,
        )

    rows = load_problems(args.n_sample)
    rng = Random(args.seed)
    inject_delays = [rng.uniform(0.0, args.max_inject_delay_s) for _ in rows]
    request_ns = [rng.randint(args.min_n, args.max_n) for _ in rows]
    request_reasoning_efforts = [rng.choice(reasoning_effort_choices) for _ in rows]
    # NOTE(wsj): Measure request injection/serving time from the moment the
    # server is ready, so model init / graph capture time is excluded.
    start_zero = time.perf_counter()
    log_lock = threading.Lock()

    def run_one(sample_idx, row, inject_delay_s, request_n, request_reasoning_effort):
        time.sleep(inject_delay_s)
        st_time = time.perf_counter()
        try:
            response_text, status_code = post_chat_completion(
                server_url,
                args.model_name,
                row["problem"],
                request_n,
                request_reasoning_effort,
            )
            ed_time = time.perf_counter()
            record = {
                "sample_idx": sample_idx,
                "problem_idx": row.get("problem_idx"),
                "request_n": request_n,
                "request_reasoning_effort": request_reasoning_effort,
                "inject_delay_s": round(inject_delay_s, 6),
                "st_time_s": round(st_time - start_zero, 6),
                "ed_time_s": round(ed_time - start_zero, 6),
                "latency_s": round(ed_time - st_time, 6),
                "status_code": status_code,
                "response": json.loads(response_text),
            }
        except urllib.error.HTTPError as exc:
            ed_time = time.perf_counter()
            record = {
                "sample_idx": sample_idx,
                "problem_idx": row.get("problem_idx"),
                "request_n": request_n,
                "request_reasoning_effort": request_reasoning_effort,
                "inject_delay_s": round(inject_delay_s, 6),
                "st_time_s": round(st_time - start_zero, 6),
                "ed_time_s": round(ed_time - start_zero, 6),
                "latency_s": round(ed_time - st_time, 6),
                "status_code": exc.code,
                "error": exc.read().decode("utf-8", errors="replace"),
            }
        except Exception as exc:
            ed_time = time.perf_counter()
            record = {
                "sample_idx": sample_idx,
                "problem_idx": row.get("problem_idx"),
                "request_n": request_n,
                "request_reasoning_effort": request_reasoning_effort,
                "inject_delay_s": round(inject_delay_s, 6),
                "st_time_s": round(st_time - start_zero, 6),
                "ed_time_s": round(ed_time - start_zero, 6),
                "latency_s": round(ed_time - st_time, 6),
                "status_code": None,
                "error": str(exc),
            }

        with log_lock:
            with open(args.request_log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return record

    with open(args.request_log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "summary": {
                        "dataset_path": DEFAULT_DATASET_PATH,
                        "model_name": args.model_name,
                        "server_url": server_url,
                        "launched_local_server": args.server_url is None,
                        "time_origin": "post_server_ready",
                        "request_n_range": [args.min_n, args.max_n],
                        "reasoning_effort_choices": reasoning_effort_choices,
                        "n_sample": len(rows),
                        "max_inject_delay_s": args.max_inject_delay_s,
                        "seed": args.seed,
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(rows))) as executor:
            futures = [
                executor.submit(
                    run_one,
                    idx,
                    row,
                    inject_delays[idx],
                    request_ns[idx],
                    request_reasoning_efforts[idx],
                )
                for idx, row in enumerate(rows)
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        if server_proc is not None and not args.keep_server_alive:
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=10)
        if server_log_file is not None:
            server_log_file.close()


if __name__ == "__main__":
    main()
