import ctypes
import pickle
import struct
import time
from multiprocessing import Value
from multiprocessing.shared_memory import SharedMemory


class ShmSPSCQueue:
    def __init__(self, num_slots=128, slot_size=64 * 1024 * 1024):
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.meta_size = 4
        self.slot_stride = self.meta_size + self.slot_size
        self._shm = SharedMemory(create=True, size=self.num_slots * self.slot_stride)
        self.read_idx = Value(ctypes.c_ulonglong, 0, lock=False)
        self.write_idx = Value(ctypes.c_ulonglong, 0, lock=False)

    def put(self, obj):
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        payload_len = len(payload)
        if payload_len > self.slot_size:
            raise ValueError(
                f"payload size {payload_len} exceeds shared-memory slot size {self.slot_size}"
            )
        while self.write_idx.value - self.read_idx.value >= self.num_slots:
            time.sleep(0)
        slot = self.write_idx.value % self.num_slots
        base = slot * self.slot_stride
        buf = self._shm.buf
        struct.pack_into("<I", buf, base, payload_len)
        buf[base + self.meta_size : base + self.meta_size + payload_len] = payload
        self.write_idx.value += 1

    def get_nowait(self):
        if self.read_idx.value >= self.write_idx.value:
            return None
        slot = self.read_idx.value % self.num_slots
        base = slot * self.slot_stride
        buf = self._shm.buf
        payload_len = struct.unpack_from("<I", buf, base)[0]
        payload = bytes(
            buf[base + self.meta_size : base + self.meta_size + payload_len]
        )
        self.read_idx.value += 1
        return pickle.loads(payload)

    def get(self):
        while True:
            msg = self.get_nowait()
            if msg is not None:
                return msg
            time.sleep(0)
