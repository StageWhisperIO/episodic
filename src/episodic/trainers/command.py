import json
import subprocess
from pathlib import Path

from . import register

TAIL = 2000


class CommandTrainer:
    name = "command"
    consumes = ("sft", "dpo", "reward", "rlds", "jsonl")
    version = "1"

    def train(self, dataset_path, out_dir, config):
        template = config.get("command")
        if not template:
            raise ValueError(
                "command trainer needs config.command, e.g. "
                "'my-trainer --data {dataset} --out {out}'"
            )
        cmd = template.format(dataset=dataset_path, out=out_dir)
        proc = subprocess.run(
            cmd,
            shell=True,
            input=json.dumps(config),
            text=True,
            capture_output=True,
        )
        result = {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-TAIL:],
            "stderr_tail": proc.stderr[-TAIL:],
        }
        metrics_path = Path(out_dir) / "metrics.json"
        if metrics_path.exists():
            try:
                result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if proc.returncode != 0:
            raise RuntimeError(f"command trainer exited {proc.returncode}: {proc.stderr[-500:]}")
        return result


register(CommandTrainer())
