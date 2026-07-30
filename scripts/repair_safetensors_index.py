"""Rebuild a missing `model.safetensors.index.json` from the shards' own headers.

    uv run .venv/bin/python scripts/repair_safetensors_index.py --probe MODEL_ID
    uv run .venv/bin/python scripts/repair_safetensors_index.py --repair MODEL_ID --out DIR

WHY THIS EXISTS. `HuggingFaceH4/mistral-7b-sft-beta` ships BOTH weight formats and
only one index:

    model-00001-of-00002.safetensors      shards PRESENT
    model-00002-of-00002.safetensors
    pytorch_model.bin.index.json          .bin index present
    (no model.safetensors.index.json)     SAFETENSORS INDEX ABSENT

A sharded checkpoint needs its index to know which tensor lives in which shard.
Without it transformers falls back to the `.bin` index and then REFUSES the load
("a serious vulnerability issue in torch.load"), so the safetensors sitting right
there are unreachable. `use_safetensors=True` does not help: the flag selects a
format, it does not synthesise the map. That model is zephyr's SFT arm, so its
loss takes zephyr's three-arm decomposition with it -- the only one of the v3
grid's three exclusions that costs a MEASUREMENT rather than a comparison.

THE INDEX IS RECOVERABLE WITHOUT DOWNLOADING THE WEIGHTS. A safetensors file
begins with a u64 little-endian header length followed by that many bytes of
JSON naming every tensor and its byte range. An HTTP range request over the
first few MB of each shard therefore yields the whole map: 291 tensors across
two shards, verified 2026-07-30, at no meaningful bandwidth cost.

WHAT THIS DOES NOT DO, and the distinction is the point: it does not convert,
re-serialise, or unpickle anything. It reads metadata the repository already
publishes and writes down what the repository failed to write down. The RWKV
arms are NOT fixable this way -- they publish no safetensors at all, so
recovering them means unpickling 29.6 GB of `.bin` per model, which is a trust
decision rather than a technical one and is not made here.
"""
import argparse
import json
import os
import struct
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HDR_BYTES = 20_000_000      # headers are far smaller; this is slack, not a guess
INDEX = "model.safetensors.index.json"


def shard_names(model_id):
    from huggingface_hub import list_repo_files
    fs = list_repo_files(model_id)
    return sorted(f for f in fs if f.endswith(".safetensors"))


def read_header(model_id, fn, revision="main"):
    """The shard's tensor map, over a range request. No weights are fetched."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{fn}"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HDR_BYTES}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = r.read()
    n = struct.unpack("<Q", buf[:8])[0]
    if n > len(buf) - 8:
        raise RuntimeError(
            f"{fn}: header is {n:,} bytes, larger than the {HDR_BYTES:,} fetched. "
            f"Raise HDR_BYTES -- do NOT parse a truncated header.")
    return json.loads(buf[8:8 + n])


def build_index(model_id, revision="main"):
    """(index_dict, per_shard_counts). total_size sums the tensors' own extents."""
    shards = shard_names(model_id)
    if not shards:
        raise RuntimeError(f"{model_id} publishes no .safetensors at all -- this "
                           f"script cannot help it; see the module docstring.")
    weight_map, total, counts = {}, 0, {}
    for fn in shards:
        h = read_header(model_id, fn, revision)
        keys = [k for k in h if k != "__metadata__"]
        for k in keys:
            weight_map[k] = fn
            a, b = h[k]["data_offsets"]
            total += b - a
        counts[fn] = len(keys)
    return {"metadata": {"total_size": total}, "weight_map": weight_map}, counts


def main(a):
    mid = a.probe or a.repair
    idx, counts = build_index(mid, a.revision)
    for fn, n in counts.items():
        print(f"  {fn}: {n} tensors")
    print(f"{mid}: {len(idx['weight_map'])} tensors, "
          f"total_size {idx['metadata']['total_size'] / 1e9:.2f} GB")
    if a.probe:
        print("\nPROBE ONLY -- nothing written. Re-run with --repair to materialise.")
        return 0

    from huggingface_hub import snapshot_download
    # EXCLUDE THE .bin DELIBERATELY. Pulling both formats doubles the download and
    # leaves the refused format on disk next to the usable one, which is how the
    # repo got into this state from the loader's point of view.
    local = snapshot_download(
        mid, revision=a.revision, local_dir=a.out,
        ignore_patterns=["*.bin", "*.bin.index.json", "*.pth", "*.msgpack", "*.h5"])
    path = os.path.join(local, INDEX)
    with open(path, "w") as fh:
        json.dump(idx, fh, indent=1)
    print(f"\nwrote {path}")

    missing = [t for t, fn in idx["weight_map"].items()
               if not os.path.exists(os.path.join(local, fn))]
    if missing:
        print(f"!! {len(missing)} tensors map to shards not present locally. "
              f"The index would be a lie; NOT usable.")
        return 1

    # LOAD IT, rather than declare it fixed. An index that parses is not an index
    # that loads, and the whole failure this repairs was a load-time refusal.
    if a.verify:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            local, torch_dtype=torch.float16, use_safetensors=True)
        print(f"VERIFIED: loads from safetensors, "
              f"{sum(p.numel() for p in m.parameters()) / 1e9:.2f}B params")
    else:
        print("index written; pass --verify to load it before trusting it.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", metavar="MODEL_ID",
                   help="read the headers and report; write nothing")
    g.add_argument("--repair", metavar="MODEL_ID",
                   help="download safetensors + config to --out and write the index")
    ap.add_argument("--out", default=None, help="local dir for --repair")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--verify", action="store_true",
                    help="load the result before declaring it fixed")
    a = ap.parse_args()
    if a.repair and not a.out:
        ap.error("--repair needs --out")
    raise SystemExit(main(a))
