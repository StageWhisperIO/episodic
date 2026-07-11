from . import TrainerUnavailable

HINT = "local critic needs torch + transformers: pip install 'episodic[trl]'"
DEFAULT_CRITIC_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_CRITIC_TOKENS = 1024


def _require_torch(name):
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable(name, HINT) from exc


def attention_parameter(name):
    lowered = name.lower()
    return "attn" in lowered or "attention" in lowered


def _pick_device(torch):
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalCritic:
    def __init__(self, model_name=DEFAULT_CRITIC_MODEL, learning_rate=1e-5, freeze_attention=True, device=None):
        _require_torch("local-critic")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or _pick_device(torch)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device).float()
        self.head = torch.nn.Linear(self.backbone.config.hidden_size, 1).to(self.device)
        if freeze_attention:
            for name, parameter in self.backbone.named_parameters():
                if attention_parameter(name):
                    parameter.requires_grad_(False)
        trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable + list(self.head.parameters()), lr=learning_rate)

    def _forward(self, texts):
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CRITIC_TOKENS,
        ).to(self.device)
        hidden = self.backbone(**encoded).last_hidden_state
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        rows = self.torch.arange(hidden.shape[0], device=self.device)
        return self.head(hidden[rows, lengths]).squeeze(-1)

    def value(self, texts):
        self.backbone.eval()
        with self.torch.no_grad():
            return [float(v) for v in self._forward(texts)]

    def update(self, texts, targets):
        self.backbone.train()
        predicted = self._forward(texts)
        target_tensor = self.torch.tensor(targets, dtype=predicted.dtype, device=self.device)
        loss = self.torch.nn.functional.mse_loss(predicted, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def pretrain(self, pairs, epochs=1, batch_size=8):
        losses = []
        for _ in range(epochs):
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                losses.append(self.update([text for text, _ in batch], [target for _, target in batch]))
        return losses


def pretrain_pairs_from_reward_rows(rows):
    pairs = []
    for row in rows:
        prompt = row.get("prompt")
        target = row.get("scalar_reward")
        if prompt and target is not None:
            pairs.append((prompt, float(target)))
    return pairs


def build_critic(config, name):
    model_name = config.get("critic_model")
    if not model_name:
        return None
    _require_torch(name)
    critic = LocalCritic(
        model_name=model_name,
        learning_rate=config.get("critic_lr", 1e-5),
        freeze_attention=config.get("critic_freeze_attention", True),
        device=config.get("critic_device"),
    )
    pretrain = config.get("critic_pretrain")
    if pretrain and pretrain.get("dataset"):
        from .trl import _read_rows
        pairs = pretrain_pairs_from_reward_rows(_read_rows(pretrain["dataset"]))
        if pairs:
            critic.pretrain(pairs, epochs=pretrain.get("epochs", 1), batch_size=pretrain.get("batch_size", 8))
    return critic
