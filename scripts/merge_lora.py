"""Folds a LoRA adapter into its base model, so a warm start can serve as the base of the next run.

    python merge_lora.py <base model directory> <adapter directory> <output directory>
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def merge(base: Path, adapter: Path, out: Path) -> int:
    config = json.loads((adapter / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    lora = load_file(adapter / "adapter_model.safetensors")
    out.mkdir(parents=True, exist_ok=True)
    merged = 0
    for path in sorted(base.iterdir()):
        if path.suffix != ".safetensors":
            shutil.copy(path, out / path.name)
            continue
        weights = load_file(path)
        for name, weight in weights.items():
            stem = name.removesuffix(".weight")
            if f"{stem}.lora_A.weight" in lora:
                delta = lora[f"{stem}.lora_B.weight"].float() @ lora[f"{stem}.lora_A.weight"].float()
                weights[name] = (weight.float() + scale * delta).to(weight.dtype)
                merged += 1
        save_file(weights, out / path.name, metadata={"format": "pt"})
    if merged * 2 != len(lora):
        sys.exit(f"MERGE_FAILED: {merged} modules merged for {len(lora)} adapter tensors")
    return merged


if __name__ == "__main__":
    print(f"MERGED {merge(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))} modules")
