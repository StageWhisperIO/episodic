from . import register, TrainerUnavailable
from .trl import _read_rows, _training_kwargs

HINT = "Unsloth backend needs extras: pip install 'episodic[unsloth]' (unsloth, trl, torch — CUDA GPU required)"
DEFAULT_MODEL = "unsloth/SmolLM2-135M-Instruct"
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _require_unsloth():
    try:
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable("unsloth", HINT) from exc


def _load_peft_model(config):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.get("model", DEFAULT_MODEL),
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=config.get("load_in_4bit", True),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0.0,
        target_modules=config.get("target_modules", DEFAULT_TARGETS),
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _save(model, tokenizer, out_dir, config):
    if config.get("save_merged"):
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)


class UnslothSFTTrainer:
    name = "unsloth-sft"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_unsloth()
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])
        model, tokenizer = _load_peft_model(config)

        args = SFTConfig(output_dir=out_dir, **_training_kwargs(config))
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        _save(model, tokenizer, out_dir, config)
        return {
            "model_dir": out_dir,
            "base_model": config.get("model", DEFAULT_MODEL),
            "method": "lora-merged" if config.get("save_merged") else "lora",
            "examples": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


class UnslothDPOTrainer:
    name = "unsloth-dpo"
    consumes = ("dpo",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_unsloth()
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer

        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ])
        model, tokenizer = _load_peft_model(config)

        args = DPOConfig(output_dir=out_dir, beta=config.get("beta", 0.1), **_training_kwargs(config))
        trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        _save(model, tokenizer, out_dir, config)
        return {
            "model_dir": out_dir,
            "base_model": config.get("model", DEFAULT_MODEL),
            "method": "lora-merged" if config.get("save_merged") else "lora",
            "pairs": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


register(UnslothSFTTrainer())
register(UnslothDPOTrainer())
