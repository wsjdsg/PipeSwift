_COMM_PORT = 8888
import zmq


def build_socket_conn(master_ip, process_rank):
    ctx = zmq.Context()
    identity = f"worker-{process_rank}".encode()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.connect(f"tcp://{master_ip}:{_COMM_PORT}")
    sock.send(b"HELLO")
    sock.recv()
    return sock
