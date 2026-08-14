#!/usr/bin/env python
"""zh reader batches for the persisted Opus second-order workflow.

    uv run python meta/M02_frame_exit/scripts/l2_zh_reader_batches.py

Mirrors the EN design exactly (lacan's m02-opus-second-order, session
cdbe9c9e, script persisted with the run): blind batches of 100
continuations, prompt as context only, ids unique, arm/model/role never
shown. Differences, each declared:

  window   first 100 CHARACTERS (EN used 50 words; zh has no word
           boundary — 100 chars is the same declared-and-dumb cut at
           roughly matching information volume)
  source   ClickHouse gen_sequences, corpus=f11_l2, zh prompts (the
           receipt does not mention zh; the store holds 112,520 rows)
  roles    joined from data/prompt_categorisation.json group_role
           (the store's role column is empty for zh)

Population: roles both / control_a / control_b, en-code groups f11_*_zh,
balanced base/aligned within role, seed 20260812. 800 sampled -> 8
batches of 100 under data/raw/l2_zh_batches/, with the UNBLINDING KEY
(id -> model, arm, role, group) written beside them; the key never
enters a batch file.
"""
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

OUTDIR = "meta/M02_frame_exit/data/opus_readers_zh"
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
N = 800
SEED = 20260812
WINDOW = 100  # characters


def unescape(s):
    for e, c in (("\\t", "\t"), ("\\n", "\n"), ("\\'", "'"), ("\\\\", "\\")):
        s = s.replace(e, c)
    return s


def main():
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
    rows = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        model, pair, prompt, sidx = parts[0], parts[1], parts[2], parts[3]
        text = unescape("\t".join(parts[4:]))
        prompt = unescape(prompt)
        role = role_of.get(prompt)
        if role not in ("BOTH", "CONTROL_A", "CONTROL_B"):
            continue
        role = role.lower()
        #: `pair` is EMPTY in gen_sequences for f11_l2 — arm and pair id
        #: come from the registry (data/base_aligned_pairs.json membership).
        if model in BASES:
            arm, pairid = "base", BASES[model]
        elif model in ALIGNEDS:
            arm, pairid = "aligned", ALIGNEDS[model]
        else:
            continue
        pair = pairid
        rows.append(dict(model=model, pair=pair, prompt=prompt,
                         sample_idx=int(sidx), text=text, role=role,
                         group=group_of[prompt], arm=arm))
    print(f"zh rows with wanted roles: {len(rows)}")
    by = defaultdict(list)
    for x in rows:
        by[(x["role"], x["arm"])].append(x)
    for k, v in sorted(by.items()):
        print(f"  {k}: {len(v)}")

    rng = random.Random(SEED)
    #: mirror the EN round-1 composition (400 BOTH / ~200 per control):
    #: 50% both, 25% each control, split evenly across arms.
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
            pid = 20000 + b * 100 + i
            key[pid] = dict(model=x["model"], pair=x["pair"], arm=x["arm"],
                            role=x["role"], group=x["group"],
                            sample_idx=x["sample_idx"])
            cont = x["text"][:WINDOW].replace("\n", " ")
            lines.append(f"### {pid}\nPROMPT: {x['prompt']}\n"
                         f"CONTINUATION: {cont}\n")
        open(f"{OUTDIR}/batch_{b + 1:02d}.txt", "w").write("\n".join(lines))
    json.dump(dict(seed=SEED, window_chars=WINDOW, n=len(sample), key=key),
              open(f"{OUTDIR}/UNBLINDING_KEY.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"wrote {len(sample) // 100} batches + UNBLINDING_KEY.json "
          f"to {OUTDIR}/ (key never enters a batch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
