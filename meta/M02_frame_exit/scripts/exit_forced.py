#!/usr/bin/env python
"""The forced-arm exit pass: does uttering the demoted word expel the scene?

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/exit_forced.py

The repair-work discriminator Findings W left unrun, asked with M02's marker
battery: force the checkpoint to begin with the FALLER (the word alignment
demoted) vs a RISER (a word it promoted) at the SAME site, and count exit
markers in the 10 tokens that follow. Same checkpoint, same prompt, same
decoder on both sides — the within-checkpoint protection class that survived
the [4994]-[5000] audits. RH's constraint (2026-08-08): risers must be the
WAVE-3 LEXICAL selection, not the earlier waves' function words — the
builder's own docstring records that wave as "a lexical effect diluted by a
function-word half that carries nothing." Filter: value `design` starts with
"wave3-lexical" (design lives in the VALUE, not the key — [4996]: key-level
reads see design=None on all 41,342 records).

Qualifiers carried: forced sites are the k>=2 mover population ([4817]);
10-token window, so E-QA-class markers are reachable and E-ASSIST mostly is
not (X.3g); REFUSAL rows at beam grain measure dialogue apology, not refusal
([4995] sampling check) — printed but struck through in reading.

Writes results/exit_forced_bysite.csv; prints faller-vs-riser per checkpoint
(all + 12-sampler-clean columns) and the forced-vs-undisturbed context read.
First look, not a measurement to quote.
"""
import csv
import os
import re
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from malign_logits.cache import CacheManager  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "exit_forced_bysite.csv")

TYPES = [
    ("E-QUIZ",    re.compile(r"_{3,}|\([A-D]\)\s|\b[A-D]\.\s+\w|possible answers|"
                             r"multiple choice|fill in the blank|choose the correct", re.I)),
    ("E-QA",      re.compile(r"\bQ\s*:|\bA\s*:|\bquestion\s*:|\banswer\s*:|答案", re.I)),
    ("E-TASK",    re.compile(r"\btranslate\b|\brewrite\b|\bsummari[sz]e\b|your task is|"
                             r"\bexercise\b|\bcomplete the\b|\bconvert the\b", re.I)),
    ("E-ASSIST",  re.compile(r"you are an? (?:ai|helpful|language)|as an ai|"
                             r"i(?:'m| am) an ai|i cannot (?:assist|help|provide)|"
                             r"it(?:'s| is) important to (?:approach|note|remember)|"
                             r"raises (?:some )?(?:significant )?concerns", re.I)),
    ("E-MENTION", re.compile(r"words like|the (?:word|term|phrase) [\"'“]|"
                             r"\bmeans\b.{0,20}\bdefinition|refer(?:s|ring) to in the", re.I)),
    ("E-META",    re.compile(r"content warning|trigger warning|this (?:excerpt|passage|text) is|"
                             r"\bgenre\b|editor(?:'s|ial) note|\bexcerpt from\b", re.I)),
    ("REFUSAL",   re.compile(r"i (?:cannot|can't|won't|will not) (?:write|continue|create|"
                             r"generate|produce|engage|describe)|i(?:'m| am) sorry,? but|"
                             r"i apologi[sz]e,? but|cannot fulfill|against my (?:guidelines|"
                             r"principles|programming)", re.I)),
]
TNAMES = [t for t, _ in TYPES]

SAMPLERS = {"HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-8B", "deepseek-ai/deepseek-llm-7b-chat", "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct", "openbmb/MiniCPM5-1B",
            "openbmb/MiniCPM5-1B-SFT", "allenai/Olmo-3.1-32B-Instruct",
            "microsoft/phi-4-reasoning", "allenai/Llama-3.1-Tulu-3-8B-DPO"}


def sweep():
    cm = CacheManager()
    rows = []
    fallers, risers = defaultdict(int), defaultdict(int)
    lex_sites = set()          # (pair, role, prompt) with a wave3-lexical forced record
    und = {}                   # undisturbed cells, matched later
    for key, val in cm._stash("beam_fc").items():
        arm = key.get("arm")
        beams = (val.get("beams") or []) if isinstance(val, dict) else []
        if not beams:
            continue
        k3 = (key["pair"], key["role"], key["prompt"])
        if arm == "undisturbed":
            n = len(beams)
            cts = [0] * len(TYPES)
            for b in beams:
                t = b.get("text") or ""
                for i, (_, rx) in enumerate(TYPES):
                    if rx.search(t):
                        cts[i] += 1
            und[k3] = (n, cts)
            continue
        design = (val.get("design") or "") if isinstance(val, dict) else ""
        if not design.startswith("wave3-lexical"):
            continue
        lex_sites.add(k3)
        n = len(beams)
        cts = [0] * len(TYPES)
        for b in beams:
            t = b.get("text") or ""
            for i, (_, rx) in enumerate(TYPES):
                if rx.search(t):
                    cts[i] += 1
        rows.append([key["pair"], key["role"], key["prompt"], arm, key.get("word"), n, *cts])
        (fallers if arm == "force_faller" else risers)[key.get("word")] += 1
    # undisturbed context rows only at lexical sites
    for k3 in lex_sites:
        if k3 in und:
            n, cts = und[k3]
            rows.append([k3[0], k3[1], k3[2], "undisturbed", "", n, *cts])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "role", "prompt", "arm", "word", "n_beams", *TNAMES])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    print("top faller words:", sorted(fallers.items(), key=lambda x: -x[1])[:10])
    print("top riser words: ", sorted(risers.items(), key=lambda x: -x[1])[:10])


if not os.path.exists(OUT):
    sweep()


def wilcox(ds):
    from scipy.stats import wilcoxon
    nz = [d for d in ds if d != 0]
    if len(nz) < 5:
        return float("nan"), len(nz)
    return wilcoxon(nz, alternative="two-sided").pvalue, len(nz)


# aggregate: (checkpoint, prompt, arm) -> pooled counts (risers pooled over words)
cell = defaultdict(lambda: [0] + [0] * len(TYPES))
for r in csv.DictReader(open(OUT)):
    m = r["pair"].split(">")[0 if r["role"] == "base" else 1]
    c = cell[(m, r["prompt"], r["arm"])]
    c[0] += int(r["n_beams"])
    for i, t in enumerate(TNAMES):
        c[1 + i] += int(r[t])

ckpts = sorted({m for m, _, _ in cell})
prompts_of = defaultdict(set)
for m, p, a in cell:
    prompts_of[(m, a)].add(p)

print("\n=== FALLER vs RISER, per checkpoint on matched sites ===")
print("delta = %marker(force_faller) - %marker(force_riser); positive = the demoted word expels more")
print(f"{'type':10s} | {'ALL mean':>8s} {'p':>7s} {'nz':>3s} {'n':>3s} | {'CLEAN mean':>10s} {'p':>7s} {'nz':>3s} {'n':>3s}")
for i, t in enumerate(TNAMES):
    cols = []
    for drop in (False, True):
        ds = []
        for m in ckpts:
            if drop and m in SAMPLERS:
                continue
            sites = prompts_of.get((m, "force_faller"), set()) & prompts_of.get((m, "force_riser"), set())
            nf = sum(cell[(m, p, "force_faller")][0] for p in sites)
            cf = sum(cell[(m, p, "force_faller")][1 + i] for p in sites)
            nr = sum(cell[(m, p, "force_riser")][0] for p in sites)
            cr = sum(cell[(m, p, "force_riser")][1 + i] for p in sites)
            if nf >= 300 and nr >= 300:
                ds.append(100 * cf / nf - 100 * cr / nr)
        p, nz = wilcox(ds)
        cols.append((statistics.mean(ds) if ds else float("nan"), p, nz, len(ds)))
    a, c = cols
    print(f"{t:10s} | {a[0]:+8.4f} {a[1]:7.4f} {a[2]:3d} {a[3]:3d} | {c[0]:+10.4f} {c[1]:7.4f} {c[2]:3d} {c[3]:3d}")

print("\n=== FORCED (either word) vs UNDISTURBED at the same sites — context read ===")
print(f"{'type':10s} | {'ALL mean':>8s} {'p':>7s} {'nz':>3s} {'n':>3s}")
for i, t in enumerate(TNAMES):
    ds = []
    for m in ckpts:
        sites = ((prompts_of.get((m, "force_faller"), set()) | prompts_of.get((m, "force_riser"), set()))
                 & prompts_of.get((m, "undisturbed"), set()))
        nf = cf = nu = cu = 0
        for p in sites:
            for a in ("force_faller", "force_riser"):
                if (m, p, a) in cell:
                    nf += cell[(m, p, a)][0]
                    cf += cell[(m, p, a)][1 + i]
            nu += cell[(m, p, "undisturbed")][0]
            cu += cell[(m, p, "undisturbed")][1 + i]
        if nf >= 300 and nu >= 300:
            ds.append(100 * cf / nf - 100 * cu / nu)
    p, nz = wilcox(ds)
    print(f"{t:10s} | {statistics.mean(ds) if ds else float('nan'):+8.4f} {p:7.4f} {nz:3d} {len(ds):3d}")

# ── the member split that resolved RH's question (2026-08-08) ────────
# Scene effect vs word effect, separated: the MARKED-UNMARKED twin gap
# (E-QA) is present in ALL THREE arms (undisturbed +0.36 p 0.0009;
# faller +0.22 p 0.063; riser +0.23 p 0.010) while faller-vs-riser is
# null WITHIN both members. The frame apparatus reads the scene, not
# the signifier.
sample_m = {}
with open(os.path.join(REPO, "data", "beam_sample_105.csv")) as f:
    for r in csv.DictReader(f):
        sample_m[r["prompt"].strip()] = r["member"]
mcell = defaultdict(lambda: [0] + [0] * len(TYPES))
for r in csv.DictReader(open(OUT)):
    pr = r["prompt"].strip()
    if pr not in sample_m:
        continue
    m = r["pair"].split(">")[0 if r["role"] == "base" else 1]
    c = mcell[(m, sample_m[pr], r["arm"])]
    c[0] += int(r["n_beams"])
    for i, t in enumerate(TNAMES):
        c[1 + i] += int(r[t])
mck = sorted({m for m, _, _ in mcell})
print("\n=== twin effect (MARKED - UNMARKED) inside each arm ===")
print(f"{'type':10s} | {'undisturbed':>11s} {'p':>7s} | {'faller':>8s} {'p':>7s} | {'riser':>8s} {'p':>7s}")
for i, t in enumerate(TNAMES):
    line = f"{t:10s}"
    for arm in ("undisturbed", "force_faller", "force_riser"):
        ds = []
        for m in mck:
            mk = mcell.get((m, "MARKED", arm))
            un = mcell.get((m, "UNMARKED", arm))
            if mk and un and mk[0] >= 200 and un[0] >= 200:
                ds.append(100 * mk[1 + i] / mk[0] - 100 * un[1 + i] / un[0])
        p, nz = wilcox(ds)
        line += f" | {statistics.mean(ds) if ds else float('nan'):+8.4f} {p:7.4f}"
    print(line)
