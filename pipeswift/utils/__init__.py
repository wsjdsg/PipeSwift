from __future__ import annotations

import importlib

_SUBMODULES = (
    "arguments",
    "cudagraph",
    "fp8_kv",
    "impl_utils",
    "internode",
    "load_weight",
    "metadata",
    "mpu",
    "quantization",
)


def __getattr__(name: str):
    for module_name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
