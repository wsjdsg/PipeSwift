__all__ = [
    "LLMEngine",
]


def __getattr__(name: str):
    if name == "LLMEngine":
        from .engine.Engine import LLMEngine

        return LLMEngine
    raise AttributeError(f"module 'pipeswift' has no attribute {name}")
