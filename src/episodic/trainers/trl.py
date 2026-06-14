import json

from . import register, TrainerUnavailable

HINT = "TRL backend needs extras: pip install 'episodic[trl]' (torch, transformers, trl, datasets, accelerate)"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _require_trl():
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable("trl", HINT) from exc


def _read_rows(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _training_kwargs(config):
    return {
        "num_train_epochs": config.get("epochs", 1),
        "per_device_train_batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("grad_accum", 1),
        "learning_rate": config.get("learning_rate", 2e-5),
        "max_steps": config.get("max_steps", -1),
        "logging_steps": config.get("logging_steps", 1),
        "report_to": [],
    }


class TRLSFTTrainer:
    name = "trl-sft"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = SFTConfig(output_dir=out_dir, **_training_kwargs(config))
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "examples": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


class TRLDPOTrainer:
    name = "trl-dpo"
    consumes = ("dpo",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = DPOConfig(output_dir=out_dir, beta=config.get("beta", 0.1), **_training_kwargs(config))
        trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "pairs": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


register(TRLSFTTrainer())
register(TRLDPOTrainer())
