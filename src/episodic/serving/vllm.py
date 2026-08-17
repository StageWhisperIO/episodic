from . import register
from .base import HTTPBackend


class VLLMBackend(HTTPBackend):
    name = "vllm"
    default_base_url = "http://localhost:8000"


register(VLLMBackend)
