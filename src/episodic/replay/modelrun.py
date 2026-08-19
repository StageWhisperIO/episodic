import subprocess
import threading

_FENCE = "```"

_DIFF_INSTRUCTION = "\n\nRespond with a unified diff of your changes inside a ```diff code block."

_BACKENDS = ("stub", "mlx", "serving", "tinker")


def extract_diff(text):
    if not text:
        return ""
    if _FENCE in text:
        blocks = text.split(_FENCE)
        for i in range(1, len(blocks), 2):
            body = blocks[i]
            newline = body.find("\n")
            lang = body[:newline].strip().lower() if newline != -1 else ""
            content = body[newline + 1:] if newline != -1 else body
            if lang in ("diff", "patch") or content.lstrip().startswith("diff --git"):
                return content
    return text


def apply_diff(diff_text, workspace):
    if not diff_text or not diff_text.strip():
        return False, "empty diff"
    if not diff_text.endswith("\n"):
        diff_text += "\n"
    log = ""
    for extra in (["--recount"], ["--recount", "--3way"]):
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", *extra, "-"],
                input=diff_text,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        if result.returncode == 0:
            return True, result.stdout + result.stderr
        log = result.stdout + result.stderr
    return False, log


def _stub_generate(config):
    stub = config.get("eval_stub")

    def generate(model, messages):
        if stub is None:
            return ""
        if callable(stub):
            return stub(model, messages)
        return stub.get(model, "")

    return generate


def _mlx_generate(config, start=None):
    from ..trainers.mlx import load_predictor, _require_mlx

    _require_mlx()
    real_base = config.get("eval_model_dir")
    if not real_base:
        raise ValueError("eval_backend 'mlx' needs eval_model_dir")
    label_base = config.get("base_model", "base")
    predictors = {}
    lock = threading.Lock()

    def generate(model, messages):
        with lock:
            predictor = predictors.get(model)
            if predictor is None:
                adapter_path = None if model == label_base else model
                predictor = load_predictor(real_base, adapter_path=adapter_path)
                predictors[model] = predictor
            return predictor(messages)

    return generate


def _tinker_generate(config, start=None):
    from ..trainers.tinker import open_sampler, sample_text

    sampler_path = config.get("eval_sampler_path")
    if not sampler_path:
        raise ValueError("eval_backend 'tinker' needs eval_sampler_path")
    sampler = open_sampler(sampler_path, base_model=config.get("eval_model_dir"))
    max_tokens = config.get("eval_max_tokens", 400)

    def generate(model, messages):
        return sample_text(sampler, messages, max_tokens=max_tokens)

    return generate


def _serving_generate(config):
    from .. import serving

    backend_config = dict(config.get("eval_backend_config") or {})
    name = backend_config.pop("backend", "openai")
    if name == "openai" and not backend_config.get("base_url") and not backend_config.get("api_key"):
        raise serving.BackendUnavailable(
            "openai",
            "eval_backend 'serving' is not configured; refusing to proxy to the public OpenAI API by "
            "default. Set eval_backend_config.base_url (self-hosted / vLLM / Ollama) or an api_key.",
        )
    backend = serving.build(name, backend_config)

    def generate(model, messages):
        response = backend.chat_completions({"model": model, "messages": messages}, stream=False)
        choices = response.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "")

    return generate


def resolve_generate(config, start=None):
    backend = config.get("eval_backend", "stub")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown eval_backend {backend!r}; choose from: {_BACKENDS}")
    if backend == "stub":
        return _stub_generate(config)
    if backend == "mlx":
        return _mlx_generate(config, start)
    if backend == "serving":
        return _serving_generate(config)
    return _tinker_generate(config, start)


def build_runner(generate):
    def runner(model, workspace, prompt_text):
        messages = [{"role": "user", "content": (prompt_text or "") + _DIFF_INSTRUCTION}]
        text = generate(model, messages)
        diff = extract_diff(text)
        ok, log = apply_diff(diff, workspace)
        return log, 0 if ok else 1

    return runner
