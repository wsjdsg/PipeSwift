import pickle
import traceback

import zmq

from pipeswift.engine import LLMEngine

_LAUNCH_PORT = 3678
if __name__ == "__main__":
    # recv args and launch engine
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(f"tcp://*:{_LAUNCH_PORT}")
    args = pickle.loads(sock.recv())
    try:
        engine = LLMEngine(args)
    except Exception as e:
        print(f"{traceback.print_exc()}")
