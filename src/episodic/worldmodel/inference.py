from . import WM_SYSTEM

DEFAULT_MAX_TOKENS = 200


def _wm_messages(sample, system):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": sample["history"]},
    ]


def mlx_predictor(model, adapter_path=None, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.0, system=None):
    from ..trainers.mlx import load_predictor

    predict_text = load_predictor(model, adapter_path=adapter_path, max_tokens=max_tokens, temperature=temperature)
    system = system or WM_SYSTEM

    def predict(sample):
        return predict_text(_wm_messages(sample, system))

    return predict


def tinker_predictor(sampler_path, base_model=None, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.0, system=None):
    from ..trainers.tinker import open_sampler, sample_text

    sampler = open_sampler(sampler_path, base_model=base_model)
    system = system or WM_SYSTEM

    def predict(sample):
        return sample_text(sampler, _wm_messages(sample, system), max_tokens=max_tokens, temperature=temperature)

    return predict
