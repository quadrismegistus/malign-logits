"""The cost model and the shard partitioner.

These two decide what a cloud run costs and which models share a card. Both
replace a proxy that agreed with its target on every case anyone had seen:
`len(prompts)` for cost, and a worker COUNT for a memory constraint.
"""

import ast
import json
import os

import pytest

from malign_logits import model_cost as MC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "data", "grid_spec.json")


def test_module_selftest_passes():
    assert MC.selftest(False)


@pytest.mark.parametrize("model,cls", [
    ("tiiuae/Falcon-H1-7B-Base", "hybrid"),
    ("allenai/Olmo-Hybrid-7B", "hybrid"),
    ("tiiuae/falcon-mamba-7b", "ssm"),
    ("RWKV/rwkv-4-7b-pile", "ssm"),
    ("allenai/OLMoE-1B-7B-0125", "moe"),
    ("tiiuae/Falcon3-7B-Base", "transformer"),
    ("meta-llama/Llama-3.1-8B", "transformer"),
])
def test_arch_class(model, cls):
    assert MC.arch_class(model) == cls


def test_equal_prompt_counts_do_not_mean_equal_cost():
    """The blindness this module exists for.

    `spec.sort(key=lambda e: len(e["prompts"]))` ranks these two EQUAL. They
    are not: one runs at ~2.9 p/s and the other at ~0.65.
    """
    n = 2583
    t = MC.cost_hours("tiiuae/Falcon3-7B-Base", n, {})
    h = MC.cost_hours("tiiuae/Falcon-H1-7B-Base", n, {})
    assert h > 3.5 * t


def test_a_rate_from_too_few_prompts_is_not_a_rate():
    """A model that OOM'd after five prompts still emitted a `p/s` line."""
    warmup = {"m": {"p_per_s": 0.98, "p_per_s_prompts": 5}}
    real = {"m": {"p_per_s": 0.98, "p_per_s_prompts": 2583}}
    assert MC.rate_for("m", warmup) == MC.CLASS_RATE["transformer"]
    assert MC.rate_for("m", real) == 0.98
    assert "discarded" in MC.rate_source("m", warmup)
    assert MC.rate_source("m", real) == "measured"


def _shard_spec():
    """Load shard_spec without importing twp_cloud (which imports torch)."""
    src = open(os.path.join(ROOT, "scripts", "twp_cloud.py")).read()
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "shard_spec"][0]
    g = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "twp_cloud", "exec"), g)
    return g["shard_spec"]


@pytest.mark.skipif(not os.path.exists(SPEC), reason="no grid spec on disk")
def test_sharding_partitions_exactly_and_isolates_heavy_models():
    shard_spec = _shard_spec()
    raw = json.load(open(SPEC))
    spec = raw["spec"] if isinstance(raw, dict) else raw
    n = 3
    parts = [shard_spec(spec, n, i, 80.0, quiet=True) for i in range(n + 1)]

    seen = [e["model"] for p in parts for e in p]
    assert len(seen) == len(spec), "a partition loses or duplicates nothing"
    assert len(set(seen)) == len(spec)

    costs = MC.load_costs()
    per_worker = 80.0 / n
    leaked = [e["model"] for p in parts[:n] for e in p
              if MC.gpu_gb(e["model"], costs) > per_worker]
    assert leaked == [], (
        "a model too large to share the card reached a PARALLEL shard; "
        "this is the condition that produced the 67-of-80 GB thrash")

    hours = [sum(MC.cost_hours(e["model"], len(e["prompts"]), costs) for e in p)
             for p in parts[:n]]
    assert max(hours) <= 1.35 * min(hours), (
        "parallel shards are balanced on COST, so they finish together")


@pytest.mark.skipif(not os.path.exists(SPEC), reason="no grid spec on disk")
def test_shard_index_out_of_range_is_refused():
    shard_spec = _shard_spec()
    raw = json.load(open(SPEC))
    spec = raw["spec"] if isinstance(raw, dict) else raw
    with pytest.raises(SystemExit):
        shard_spec(spec, 3, 4, 80.0, quiet=True)
    with pytest.raises(SystemExit):
        shard_spec(spec, 3, -1, 80.0, quiet=True)


def test_profiles_declare_the_torch_floor():
    """The floor that cost the July grid 13 models must be an artifact."""
    path = os.path.join(ROOT, "data", "cloud_profiles.json")
    profiles = json.load(open(path))
    real = {k: v for k, v in profiles.items() if not k.startswith("_")}
    assert real, "at least one profile"
    for name, p in real.items():
        assert any(pin.startswith("torch>=") for pin in p.get("pins", [])), (
            f"profile {name} declares no torch floor")
    assert "mamba-ssm" in profiles["ssm"]["pins"], (
        "the ssm profile exists to carry the kernels; without them a bigger "
        "card does not help")
