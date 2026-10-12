from __future__ import annotations

import threading
import time
from array import array
from dataclasses import dataclass
from typing import Dict, Optional

from pipeswift.utils.metadata import Request


@dataclass
class InflightRequest:
    submitted_at: float


class EngineBridge:
    def __init__(
        self,
        engine,
        result_poll_interval: float = 0.005,
        result_ttl_seconds: float = 300.0,
    ):
        self.engine = engine
        self.result_poll_interval = result_poll_interval
        self.result_ttl_seconds = result_ttl_seconds
        self._finished_by_request: Dict[int, dict] = {}
        self._finished_at: Dict[int, float] = {}
        self._inflight: Dict[int, InflightRequest] = {}
        self._cv = threading.Condition()
        self._collector_thread = threading.Thread(
            target=self._collect_finished_results,
            name="pipeswift-server-results",
            daemon=True,
        )
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_finished_results,
            name="pipeswift-server-cleanup",
            daemon=True,
        )
        self._collector_thread.start()
        self._cleanup_thread.start()

    def submit_prompt(
        self,
        prompt: str,
        prompt_token_ids: array,
        n_seqs: int = 1,
        max_gen_tokens: int | None = None,
    ) -> int:
        if max_gen_tokens is None:
            max_gen_tokens = self.engine.args.max_gen_tokens
        req = Request(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            n_seqs=n_seqs,
            max_gen_tokens=max_gen_tokens,
        )
        with self.engine.request_lock:
            self.engine.pending_request_queue.append(req)
            self.engine.total_submitted_requests += 1
        with self._cv:
            self._inflight[req.req_id] = InflightRequest(submitted_at=time.time())
        return req.req_id

    def wait_for_result(self, request_id: int, timeout: Optional[float] = None) -> dict:
        with self._cv:
            end_time = None if timeout is None else time.time() + timeout
            while request_id not in self._finished_by_request:
                remaining = None if end_time is None else end_time - time.time()
                if remaining is not None and remaining <= 0:
                    self._inflight.pop(request_id, None)
                    raise TimeoutError(f"timed out waiting for request {request_id}")
                self._cv.wait(timeout=remaining)
            self._inflight.pop(request_id, None)
            self._finished_at.pop(request_id, None)
            return self._finished_by_request.pop(request_id)

    def inflight_count(self) -> int:
        with self._cv:
            return len(self._inflight)

    def _collect_finished_results(self) -> None:
        # NOTE(wsj): Online serving path; drain finished results continuously and
        # wake the matching HTTP request as soon as that request finishes.
        while True:
            drained = []
            with self.engine.result_lock:
                while self.engine.finished_results:
                    drained.extend(self.engine.finished_results.popleft())
            if drained:
                with self._cv:
                    now = time.time()
                    for result in drained:
                        request_id = result["request_id"]
                        self._finished_by_request[request_id] = result
                        self._finished_at[request_id] = now
                    self._cv.notify_all()
            else:
                time.sleep(self.result_poll_interval)

    def _cleanup_finished_results(self) -> None:
        while True:
            time.sleep(max(1.0, self.result_ttl_seconds / 10.0))
            cutoff = time.time() - self.result_ttl_seconds
            with self._cv:
                stale_ids = [
                    request_id
                    for request_id, finished_at in self._finished_at.items()
                    if finished_at < cutoff
                ]
                for request_id in stale_ids:
                    self._finished_at.pop(request_id, None)
                    self._finished_by_request.pop(request_id, None)
