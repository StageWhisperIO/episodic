from . import register
from .base import HTTPBackend


class OpenAIBackend(HTTPBackend):
    name = "openai"
    default_base_url = "https://api.openai.com"


register(OpenAIBackend)
