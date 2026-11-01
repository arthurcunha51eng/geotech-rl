import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

spec = importlib.util.spec_from_file_location("merge_lora", Path(__file__).parents[1] / "scripts" / "merge_lora.py")
merge_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_lora)

WEIGHT = "model.layers.0.mlp.up_proj.weight"


def write(base, adapter, config, extra=None):
    base.mkdir()
    adapter.mkdir()
    safetensors.save_file({WEIGHT: torch.ones(2, 3), "model.norm.weight": torch.ones(3)}, base / "model.safetensors")
    (base / "config.json").write_text("{}")
    lora = {"model.layers.0.mlp.up_proj.lora_A.weight": torch.tensor([[1.0, 0.0, 2.0]]),
            "model.layers.0.mlp.up_proj.lora_B.weight": torch.tensor([[1.0], [3.0]])}
    safetensors.save_file(lora | (extra or {}), adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps(config))


def test_adds_the_scaled_low_rank_product_and_keeps_the_rest(tmp_path):
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    write(base, adapter, dict(r=1, lora_alpha=2.0))
    assert merge_lora.merge(base, adapter, out) == 1
    merged = safetensors.load_file(out / "model.safetensors")
    expected = torch.tensor([[3.0, 1.0, 5.0], [7.0, 1.0, 13.0]])
    assert torch.equal(merged[WEIGHT], expected) and torch.equal(merged["model.norm.weight"], torch.ones(3))
    assert (out / "config.json").exists()


def test_refuses_an_adapter_that_does_not_fit_the_model(tmp_path):
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    extra = {"model.layers.9.mlp.up_proj.lora_A.weight": torch.ones(1, 3), "model.layers.9.mlp.up_proj.lora_B.weight": torch.ones(2, 1)}
    write(base, adapter, dict(r=1, lora_alpha=1.0), extra)
    with pytest.raises(SystemExit):
        merge_lora.merge(base, adapter, out)
