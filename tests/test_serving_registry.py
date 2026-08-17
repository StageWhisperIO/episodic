import pytest

from episodic import serving
from episodic.serving.base import HTTPBackend


def test_builtin_backends_registered():
    names = serving.available()
    assert {"openai", "ollama", "vllm", "tinker"}.issubset(set(names))


def test_get_returns_backend_class():
    backend_cls = serving.get("openai")
    assert issubclass(backend_cls, HTTPBackend)
    assert backend_cls.name == "openai"


def test_unknown_backend_raises_key_error():
    with pytest.raises(KeyError):
        serving.get("does-not-exist")


def test_build_instantiates_backend_from_config():
    backend = serving.build("openai", {"base_url": "http://upstream.local", "model": "distilled-1"})
    assert backend.base_url == "http://upstream.local"
    assert backend.model == "distilled-1"


def test_backend_unavailable_carries_hint():
    exc = serving.BackendUnavailable("tinker", "install tinker")
    assert exc.backend == "tinker"
    assert exc.hint == "install tinker"
    assert str(exc) == "install tinker"
