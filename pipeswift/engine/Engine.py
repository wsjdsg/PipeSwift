import json
import os
import pickle
import shlex
import subprocess
import sys
import threading
import time
from array import array
from collections import deque
from typing import List

import torch.multiprocessing as torch_mp
import zmq
from transformers import AutoTokenizer

from pipeswift.llms import name2model
from pipeswift.utils.metadata import MsgType, Request, WorkerMsg

from .Instance import InstanceProcess
from .shm_queue import ShmSPSCQueue


class LLMEngine:
    def __init__(self, args):
        self.sampling_params = args.sampling_params
        self.args = args
        self.pending_request_queue = deque()
        self.finished_results = deque()
        self.result_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.total_submitted_requests = 0
        self.running = True
        self.exiting = False
        model_name = args.model_name
        assert args.model_name in name2model.keys(), (
            f"Model {model_name} Not Implemented"
        )
        if args.node_rank == 0:
            Request._tokenizer = self._load_tokenizer(args)

        self.world_size = args.tp_deg * args.pp_deg
        if len(args.ip_addrs) == 1:
            assert args.ip_addrs[0] == "localhost", (
                "The first machine is always localhost!"
            )
        elif args.node_rank == 0:
            assert args.ip_addrs[0] != "localhost", (
                "Please specify concrete ip address!"
            )
            from copy import deepcopy

            self.router = launch_fake_engine(deepcopy(args))
        self.initialize_workers()
        if args.node_rank != 0:
            exit(0)
        self._router_poller = None
        if hasattr(self, "router"):
            self._router_poller = zmq.Poller()
            self._router_poller.register(self.router, zmq.POLLIN)
        self.engine_thread = threading.Thread(
            target=self._engine_loop,
            name="pipeswift-engine-control",
            daemon=True,
        )
        self.engine_thread.start()

    @staticmethod
    def _load_tokenizer(args):
        return AutoTokenizer.from_pretrained(args.weight_path, trust_remote_code=True)

    def initialize_workers(self):
        worker_to_engine_queues = []
        engine_to_worker_queues = []
        for _ in range(self.args.num_gpus_per_node):
            worker_to_engine_queues.append(ShmSPSCQueue())
            engine_to_worker_queues.append(ShmSPSCQueue())
        self.worker_to_engine_queues = tuple(worker_to_engine_queues)
        self.engine_to_worker_queues = tuple(engine_to_worker_queues)
        self.worker_ctx = torch_mp.spawn(
            InstanceProcess(),
            args=(
                self.args,
                self.engine_to_worker_queues,
                self.worker_to_engine_queues,
            ),
            nprocs=self.args.num_gpus_per_node,
            join=False,
        )
        if self.args.node_rank == 0:
            self._recv_from_worker(0)
            print("[Engine] All workers launched.")

    def _send2worker(self, rank: int, msg: WorkerMsg):
        if rank < self.args.num_gpus_per_node:
            self.engine_to_worker_queues[rank].put(msg)
        else:
            identity = f"worker-{rank}".encode()
            payload = pickle.dumps(msg)
            self.router.send_multipart([identity, b"", payload])

    def _recv_from_worker(self, rank: int):
        if rank < self.args.num_gpus_per_node:
            msg = self.worker_to_engine_queues[rank].get()
        else:
            self.router.recv()
            payload = self.router.recv()
            msg = pickle.loads(payload)
        if self.args.verbose:
            print(msg)
        return msg  # zmq can not guarantee the sender

    def _try_recv_from_local_worker(self, rank: int):
        msg = self.worker_to_engine_queues[rank].get_nowait()
        if msg is None:
            return None
        if self.args.verbose:
            print(msg)
        return msg

    def _drain_remote_workers(self):
        if self._router_poller is None:
            return []
        msgs = []
        while self._router_poller.poll(0):
            identity, payload = self.router.recv_multipart()
            msg = pickle.loads(payload)
            if self.args.verbose:
                print(
                    f"[Engine] recv remote msg from {identity.decode(errors='ignore')}: {msg}"
                )
            msgs.append(msg)
        return msgs

    def broadcast_to_workers(self, msg: WorkerMsg):
        for rank in range(self.world_size):
            self._send2worker(rank, msg)

    def offline_batch_inference(
        self, prompts: List[str]
    ):  # NOTE(wsj): offline on-policy path
        reqs = [
            Request(
                prompt=_prompt,
                prompt_token_ids=array("i", Request._tokenizer.encode(_prompt)),
                n_seqs=self.sampling_params.n_seqs,
                max_gen_tokens=self.sampling_params.max_gen_tokens,
            )
            for _prompt in prompts
        ]
        with self.request_lock:
            self.pending_request_queue.extend(reqs)
            self.total_submitted_requests += len(reqs)
        finished_requests = []
        while True:
            with self.request_lock:
                pending = len(self.pending_request_queue)
            with self.result_lock:
                while self.finished_results:
                    finished_requests.extend(self.finished_results.popleft())
            if pending == 0 and len(finished_requests) == self.total_submitted_requests:
                break

        self.total_submitted_requests = 0
        with open(self.args.o, "+a") as f:
            for req_payload in finished_requests:
                f.write(
                    json.dumps(
                        req_payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
        print(f"Results saved in {self.args.o}!")
        if self.exiting and hasattr(self, "worker_ctx"):
            self.worker_ctx.join()
            self.running = False

    def exit(self):
        self.exiting = True

    def _engine_loop(self):
        if self.args.node_rank != 0:
            return
        while self.running:
            for rank in range(min(self.args.num_gpus_per_node, self.world_size)):
                msg = self._try_recv_from_local_worker(rank)
                if msg is None:
                    continue
                self._handle_worker_msg(rank, msg)
            for msg in self._drain_remote_workers():
                self._handle_worker_msg(msg.src_rank, msg)

    def _handle_worker_msg(self, rank: int, msg: WorkerMsg):
        if msg.instr == MsgType.PULL_REQS:
            with self.request_lock:
                reqs = list(self.pending_request_queue)
                self.pending_request_queue.clear()
            if self.exiting and len(reqs) == 0:
                engine_msg = WorkerMsg(
                    instr=MsgType.EXIT,
                    src_rank=-1,
                    dst_rank=-1,
                    content="Exit and Release Resources",
                )
            else:
                engine_msg = WorkerMsg(
                    instr=MsgType.REQS,
                    src_rank=-1,
                    dst_rank=-1,
                    content=reqs,
                )
            self.broadcast_to_workers(engine_msg)
        elif msg.instr == MsgType.FINISHED_SEQS:
            req_results = self._group_finished_seqs_by_request(msg.content)
            with self.result_lock:
                self.finished_results.append(req_results)

    def _group_finished_seqs_by_request(self, finished_seqs):
        grouped = {}
        order = []
        for seq in finished_seqs:
            req_id = seq["request_id"]
            if req_id not in grouped:
                grouped[req_id] = {
                    "request_id": req_id,
                    "prompt": seq["prompt"],
                    "responses": {},
                    "finish_reasons": {},
                }
                order.append(req_id)
            response_idx = str(len(grouped[req_id]["responses"]) + 1)
            response_token_ids = seq["response_token_ids"]
            if hasattr(response_token_ids, "tolist"):
                response_token_ids = response_token_ids.tolist()
            elif not isinstance(response_token_ids, list):
                response_token_ids = list(response_token_ids)
            grouped[req_id]["responses"][response_idx] = (
                Request._tokenizer.decode(response_token_ids)
                if Request._tokenizer is not None
                else str(response_token_ids)
            )
            grouped[req_id]["finish_reasons"][response_idx] = seq.get("finish_reason")
        return [grouped[req_id] for req_id in order]


_LAUNCH_PORT = 3678
_COMM_PORT = 8888


def launch_fake_engine(args):
    _file_path = os.path.abspath(__file__)
    _root_dir = os.path.abspath(f"{os.path.dirname(_file_path)}/../../")
    _log_dir = os.path.join(_root_dir, "output", "launch_logs")
    os.makedirs(_log_dir, exist_ok=True)
    _python_bin = sys.executable
    _ps = []
    _remote_logs = {}
    for node_rank, ip_addr in enumerate(args.ip_addrs):
        if node_rank >= 1:
            cmd_prefix_str = ""
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                cmd_prefix_str = (
                    "CUDA_VISIBLE_DEVICES="
                    f"{shlex.quote(os.environ['CUDA_VISIBLE_DEVICES'])} "
                )
            if "NCCL_IB_HCA" in os.environ.keys():
                cmd_prefix_str += (
                    f"NCCL_IB_HCA={shlex.quote(os.environ['NCCL_IB_HCA'])} "
                )
            conda_env = os.environ.get(
                "PIPESWIFT_CONDA_ENV", os.environ.get("CONDA_DEFAULT_ENV", "pipeswift")
            )
            remote_cmd = (
                f"cd {shlex.quote(_root_dir)}; "
                "if command -v conda >/dev/null 2>&1; then "
                'eval "$(conda shell.bash hook)" && '
                f"conda activate {shlex.quote(conda_env)}; "
                "fi; "
                f"PYTHONPATH={shlex.quote(_root_dir)}:${{PYTHONPATH:-}} "
                f"{cmd_prefix_str}{shlex.quote(_python_bin)} "
                "-m pipeswift._launch_fake_engine"
            )
            cmd = ["pdsh", "-w", f"ssh:{ip_addr}", "bash", "-lc", remote_cmd]
            log_path = os.path.join(_log_dir, f"launch-{ip_addr}.log")
            log_f = open(log_path, "ab")
            _remote_logs[ip_addr] = log_path
            _ps.append(
                subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
            )

    for node_rank, ip_addr in enumerate(args.ip_addrs):
        if node_rank >= 1:
            args.node_rank = node_rank
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PUSH)
            sock.connect(f"tcp://{ip_addr}:{_LAUNCH_PORT}")
            sock.send(pickle.dumps(args))
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind(f"tcp://*:{_COMM_PORT}")
    expected_workers = args.num_gpus_per_node * (len(args.ip_addrs) - 1)
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    launched = 0
    # Remote environment activation, Python imports, and kernel autotuning can
    # take several minutes after a cold start. Override this timeout as needed.
    deadline = time.time() + int(os.environ.get("PIPESWIFT_REMOTE_LAUNCH_TIMEOUT", 600))
    while launched < expected_workers:
        timeout_ms = max(0, int((deadline - time.time()) * 1000))
        if timeout_ms == 0:
            raise RuntimeError(
                "Timed out waiting for remote workers to connect. "
                f"Expected {expected_workers}, got {launched}. "
                f"Check remote launch logs: {_remote_logs}"
            )
        events = dict(poller.poll(timeout_ms))
        if router not in events:
            raise RuntimeError(
                "Timed out waiting for remote workers to connect. "
                f"Expected {expected_workers}, got {launched}. "
                f"Check remote launch logs: {_remote_logs}"
            )
        identity, _ = router.recv_multipart()
        router.send_multipart([identity, b"ACK"])
        launched += 1
    print("[INFO] All remote workers launched!")
    return router
