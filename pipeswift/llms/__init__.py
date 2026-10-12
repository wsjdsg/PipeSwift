from .dpskv3 import DeepSeekV3
from .glm4 import GLM4
from .llama2 import Llama2
from .llm import LLM as LLM
from .qwen3 import Qwen3

name2model = {
    "llama2": Llama2,
    "llama3": Llama2,
    "qwen3": Qwen3,
    "qwen3coder": Qwen3,
    "glm4": GLM4,
    "dpskv3": DeepSeekV3,
    "dpsk-distill-qwen": Llama2,
}
