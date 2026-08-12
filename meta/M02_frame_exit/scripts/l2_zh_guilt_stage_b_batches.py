#!/usr/bin/env python
"""Stage B batches: fresh 800 zh continuations for the ruled guilt rubric.

    uv run python meta/M02_frame_exit/scripts/l2_zh_guilt_stage_b_batches.py

[5569] design, Stage B: drawn from the ~112k unread zh passages, same
arm/role balance as round 1 / Stage A (50% both, 25% each control, split
across arms), same 100-char window, same blind-batch format. Poolable
with Stage A (same instrument); never with round 1.

Deltas from the round-1 builder, each declared:
  seed       20260813 (round 1 used 20260812; a fresh draw needs a fresh
             seed, declared here)
  exclusion  the Stage-A 800 are excluded by exact (model, prompt,
             sample_idx): model and sample_idx from the round-1
             UNBLINDING_KEY, prompt recovered from the batch files by id
             (the key does not store prompts; the batch files do)
  ids        30000+ (round 1 used 20000+, controls 29001-29008)
  output     data/opus_readers_zh_stage_b/ with its own UNBLINDING_KEY;
             the two controls mini-batches (batch_09, batch_10) are
             copied verbatim from Stage A so every stage rides the same
             probes.
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

SRC = "meta/M02_frame_exit/data/opus_readers_zh"
OUTDIR = "meta/M02_frame_exit/data/opus_readers_zh_stage_b"
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
N = 800
SEED = 20260813
WINDOW = 100


def unescape(s):
    for e, c in (("\\t", "\t"), ("\\n", "\n"), ("\\'", "'"), ("\\\\", "\\")):
        s = s.replace(e, c)
    return s


def stage_a_exclusions():
    key = json.load(open(f"{SRC}/UNBLINDING_KEY.json"))["key"]
    prompt_of = {}
    for b in range(1, 9):
        txt = open(f"{SRC}/batch_{b:02d}.txt").read()
        for m in re.finditer(r"### (\d+)\nPROMPT: (.*)\n", txt):
            prompt_of[m.group(1)] = m.group(2)
    excl = set()
    for pid, k in key.items():
        if pid in prompt_of:
            excl.add((k["model"], prompt_of[pid], k["sample_idx"]))
    assert len(excl) == 800, f"expected 800 exclusions, got {len(excl)}"
    return excl


def main():
    excl = stage_a_exclusions()
    bap = json.load(open("data/base_aligned_pairs.json"))
    bap = bap if isinstance(bap, list) else bap.get("pairs", bap)
    BASES, ALIGNEDS = {}, {}
    for p in bap:
        pid = f"{p['base']}>{p['aligned']}"
        BASES[p["base"]] = pid
        ALIGNEDS[p["aligned"]] = pid

    cat = json.load(open("data/prompt_categorisation.json"))["prompts"]
    role_of, group_of = {}, {}
    for r in cat:
        g = str(r.get("group_id") or "")
        if g.startswith("f11") and g.endswith("_zh"):
            role_of[r["prompt"]] = r.get("group_role")
            group_of[r["prompt"]] = g

    q = ("SELECT model, pair, prompt, sample_idx, text FROM "
         "malign_logits.gen_sequences WHERE corpus='f11_l2' AND "
         "match(prompt, '[\\\\x{4e00}-\\\\x{9fff}]') FORMAT TSV")
    r = subprocess.run([CH, "client", "--query", q], capture_output=True,
                       text=True, check=True)
    rows, excluded = [], 0
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        model, _, prompt, sidx = parts[0], parts[1], parts[2], parts[3]
        text = unescape("\t".join(parts[4:]))
        prompt = unescape(prompt)
        role = role_of.get(prompt)
        if role not in ("BOTH", "CONTROL_A", "CONTROL_B"):
            continue
        if (model, prompt, int(sidx)) in excl:
            excluded += 1
            continue
        role = role.lower()
        if model in BASES:
            arm, pairid = "base", BASES[model]
        elif model in ALIGNEDS:
            arm, pairid = "aligned", ALIGNEDS[model]
        else:
            continue
        rows.append(dict(model=model, pair=pairid, prompt=prompt,
                         sample_idx=int(sidx), text=text, role=role,
                         group=group_of[prompt], arm=arm))
    print(f"zh rows with wanted roles: {len(rows)} "
          f"(stage-A exclusions hit: {excluded}/800)")
    assert excluded == 800, "stage-A exclusion did not match all 800"

    by = defaultdict(list)
    for x in rows:
        by[(x["role"], x["arm"])].append(x)
    rng = random.Random(SEED)
    TARGET = {"both": N // 4, "control_a": N // 8, "control_b": N // 8}
    sample = []
    for (role, arm), v in sorted(by.items()):
        sample += rng.sample(v, min(TARGET[role], len(v)))
    rng.shuffle(sample)
    sample = sample[:N]
    print(f"sampled {len(sample)} (targets: {TARGET}, per arm)")

    os.makedirs(OUTDIR, exist_ok=True)
    key = {}
    header = (f"100 story continuations in Chinese. Each is the first "
              f"{WINDOW} characters after its prompt.\n"
              f"The prompt is context only.\n")
    for b in range(len(sample) // 100):
        lines = [header]
        for i, x in enumerate(sample[b * 100:(b + 1) * 100]):
            pid = 30000 + b * 100 + i
            key[pid] = dict(model=x["model"], pair=x["pair"], arm=x["arm"],
                            role=x["role"], group=x["group"],
                            sample_idx=x["sample_idx"])
            cont = x["text"][:WINDOW].replace("\n", " ")
            lines.append(f"### {pid}\nPROMPT: {x['prompt']}\n"
                         f"CONTINUATION: {cont}\n")
        open(f"{OUTDIR}/batch_{b + 1:02d}.txt", "w").write("\n".join(lines))
    for c in ("batch_09.txt", "batch_10.txt"):
        shutil.copy(f"{SRC}/{c}", f"{OUTDIR}/{c}")
    json.dump(dict(seed=SEED, window_chars=WINDOW, n=len(sample),
                   stage="B", excludes="stage-A 800 by (model,prompt,"
                   "sample_idx)", key=key),
              open(f"{OUTDIR}/UNBLINDING_KEY.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"wrote {len(sample) // 100} batches + controls copies + key "
          f"to {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
