SCHEMA_VERSION = "0.1.0"


class BackendUnavailable(RuntimeError):
    def __init__(self, backend, hint):
        self.backend = backend
        self.hint = hint
        super().__init__(hint)


_REGISTRY = {}
_DISCOVERED = False


def register(backend_cls):
    _REGISTRY[backend_cls.name] = backend_cls
    return backend_cls


def _discover():
    global _DISCOVERED
    if _DISCOVERED:
        return
    from . import openai, ollama, vllm, tinker  # noqa: F401  built-ins self-register on import
    _DISCOVERED = True


def get(name):
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}'; available: {', '.join(available())}")
    return _REGISTRY[name]


def available():
    _discover()
    return sorted(_REGISTRY)


def build(name, config=None):
    backend_cls = get(name)
    return backend_cls.from_config(config or {})
