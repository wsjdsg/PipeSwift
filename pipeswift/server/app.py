from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from transformers import AutoTokenizer

from pipeswift.server.bridge import EngineBridge
from pipeswift.server.chat import create_chat_completion
from pipeswift.server.completion import create_completion
from pipeswift.server.generate import create_generate
from pipeswift.utils.arguments import inference_args, parse_args
from pipeswift.utils.chat_utils import build_error_response, served_model_aliases
from pipeswift.utils.metadata import Request as EngineRequest


def _load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.weight_path, trust_remote_code=True)


def create_app(
    engine,
    tokenizer,
    *,
    request_timeout_seconds: float = 300.0,
    max_concurrency: int = 128,
) -> FastAPI:
    bridge = EngineBridge(engine)

    # No server-side concurrency semaphore — let the engine's own
    # running_queue / waiting_queue be the cap. Whatever the bench sends,
    # the server accepts; engine running_queue size is bounded by
    # max_batchsize * pp_deg, and the rest queues at the engine.
    #
    # We still need a sufficiently large thread pool so asyncio.to_thread()
    # can dispatch many in-flight create_completion() calls (each blocks on
    # bridge.wait_for_result). Default ThreadPoolExecutor is 32 — too small.
    request_executor = ThreadPoolExecutor(
        max_workers=max(max_concurrency * 4, 1024),
        thread_name_prefix="pipeswift-request",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(request_executor)
        try:
            yield
        finally:
            engine.exit()
            request_executor.shutdown(wait=False)

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    app.state.bridge = bridge
    app.state.request_timeout_seconds = request_timeout_seconds
    app.state.max_concurrency = max_concurrency

    def error_json(status_code: int, message: str, err_type: str):
        return JSONResponse(
            status_code=status_code,
            content=build_error_response(message, err_type).model_dump(),
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "ready": True,
            "inflight_requests": bridge.inflight_count(),
        }

    @app.get("/v1/models")
    async def list_models():
        model_name = engine.args.model_name
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "pipeswift",
                }
                for alias in sorted(served_model_aliases(model_name))
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            return error_json(400, f"invalid json body: {exc}", "invalid_request_error")

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    create_chat_completion,
                    body,
                    tokenizer,
                    bridge,
                    engine.args.model_name,
                ),
                timeout=request_timeout_seconds,
            )
            return JSONResponse(status_code=200, content=response.model_dump())
        except (ValueError, ValidationError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        except TimeoutError as exc:
            return error_json(408, str(exc), "timeout_error")
        except asyncio.TimeoutError:
            return error_json(408, "request timed out", "timeout_error")
        except Exception as exc:
            return error_json(500, str(exc), "internal_server_error")

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            return error_json(400, f"invalid json body: {exc}", "invalid_request_error")

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    create_completion,
                    body,
                    tokenizer,
                    bridge,
                    engine.args.model_name,
                ),
                timeout=request_timeout_seconds,
            )
            return JSONResponse(status_code=200, content=response.model_dump())
        except (ValueError, ValidationError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        except TimeoutError as exc:
            return error_json(408, str(exc), "timeout_error")
        except asyncio.TimeoutError:
            return error_json(408, "request timed out", "timeout_error")
        except Exception as exc:
            return error_json(500, str(exc), "internal_server_error")

    @app.post("/generate")
    async def generate(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            return error_json(400, f"invalid json body: {exc}", "invalid_request_error")

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    create_generate,
                    body,
                    tokenizer,
                    bridge,
                    engine.args.model_name,
                ),
                timeout=request_timeout_seconds,
            )
            return JSONResponse(status_code=200, content=response.model_dump())
        except (ValueError, ValidationError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        except TimeoutError as exc:
            return error_json(408, str(exc), "timeout_error")
        except asyncio.TimeoutError:
            return error_json(408, "request timed out", "timeout_error")
        except Exception as exc:
            return error_json(500, str(exc), "internal_server_error")

    return app


def run_server():
    cli_parser = argparse.ArgumentParser(
        description="PipeSwift OpenAI-compatible inference server"
    )
    server_group = cli_parser.add_argument_group("Server Arguments")
    server_group.add_argument("--host", type=str, default="0.0.0.0")
    server_group.add_argument("--port", type=int, default=8000)
    server_group.add_argument("--server-max-concurrency", type=int, default=128)
    server_group.add_argument(
        "--server-request-timeout-seconds", type=float, default=300.0
    )
    inference_args(cli_parser)
    parsed_args = cli_parser.parse_args()

    host = parsed_args.host
    port = parsed_args.port
    server_max_concurrency = parsed_args.server_max_concurrency
    server_request_timeout_seconds = parsed_args.server_request_timeout_seconds
    for server_arg in (
        "host",
        "port",
        "server_max_concurrency",
        "server_request_timeout_seconds",
    ):
        delattr(parsed_args, server_arg)

    args = parse_args(parsed_args)
    from pipeswift.engine.Engine import LLMEngine

    tokenizer = _load_tokenizer(args)
    EngineRequest._tokenizer = tokenizer
    engine = LLMEngine(args)
    app = create_app(
        engine,
        tokenizer,
        request_timeout_seconds=server_request_timeout_seconds,
        max_concurrency=server_max_concurrency,
    )
    print(f"[PipeSwift Server] Listening on http://{host}:{port}/v1/chat/completions")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
