import json
import subprocess
import sys
from pathlib import Path

from . import register, TrainerUnavailable
from .trl import _read_rows

HINT = (
    "MLX backend needs extras: pip install 'episodic[mlx]' (mlx-lm). "
    "MLX runs on Apple Silicon (arm64 Mac) only."
)
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TAIL = 2000


def _require_mlx():
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable("mlx-sft", HINT) from exc


def split_rows(rows, valid_frac):
    chat = [{"messages": row["messages"]} for row in rows if row.get("messages")]
    if not chat:
        raise ValueError("mlx-sft needs SFT rows with 'messages'; dataset is empty")
    if len(chat) == 1:
        return chat, chat
    valid_count = min(len(chat) - 1, max(1, round(len(chat) * valid_frac)))
    return chat[valid_count:], chat[:valid_count]


def _write_split(data_dir, train, valid):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class MLXSFTTrainer:
    name = "mlx-sft"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_mlx()
        out = Path(out_dir)
        data_dir = out / "data"
        adapters = out / "adapters"
        train, valid = split_rows(_read_rows(dataset_path), config.get("valid_frac", 0.2))
        _write_split(data_dir, train, valid)

        model_name = config.get("model", DEFAULT_MODEL)
        iters = config.get("iters", 100)
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model_name,
            "--train",
            "--data", str(data_dir),
            "--adapter-path", str(adapters),
            "--iters", str(iters),
            "--batch-size", str(config.get("batch_size", 1)),
            "--num-layers", str(config.get("num_layers", 4)),
            "--learning-rate", str(config.get("learning_rate", 1e-5)),
            "--max-seq-length", str(config.get("max_seq_length", 2048)),
            "--val-batches", str(config.get("val_batches", 1)),
            "--steps-per-eval", str(config.get("steps_per_eval", iters)),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        result = {
            "model_dir": str(adapters),
            "base_model": model_name,
            "method": "lora",
            "train_examples": len(train),
            "valid_examples": len(valid),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-TAIL:],
            "stderr_tail": proc.stderr[-TAIL:],
        }
        if proc.returncode != 0:
            raise RuntimeError(f"mlx_lm.lora exited {proc.returncode}: {proc.stderr[-500:]}")
        return result


register(MLXSFTTrainer())
