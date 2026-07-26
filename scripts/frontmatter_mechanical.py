#!/usr/bin/env python3
"""Mechanical half of the findings frontmatter: date, data, scripts, families.

The judgment fields -- status, grade, role, instruments, chapters,
superseded_by -- are NOT derivable from the repo and are supplied by the
TheoryMachines seat from the claims ledger and audit record. This script
emits only what the repo can prove, and every path it emits is verified to
exist on disk rather than merely named in prose.

Sources, in order of reliability:
  1. data/README.md   -- the explicit finding -> output provenance map (36 rows)
  2. the finding's own prose -- data/ and scripts/ paths it names
  3. git             -- first-commit date of the finding document
"""
import glob, json, os, re, subprocess, sys

from malign_logits import MODEL_FAMILIES

FILE_RE = re.compile(r"([\w./-]+\.(?:csv|parquet|json|jsonl))")

# Files named in a finding ONLY inside a note explaining their fate. Recording
# the resolution here rather than letting a heuristic guess, and rather than
# reporting them as gaps a human must adjudicate -- both were traced on
# 2026-07-26 and the answers are known.
# Findings whose source of record is a stash or an external document rather than
# a file on disk. Desktop ruling 2026-07-26: the data field admits three
# visibly-typed shapes -- file paths, "stash:<name> (<pattern>)", and external
# citations. For a DOCUMENTARY finding the external ID is a PRECISE pin, not a
# coarse one; a stash reference there would be wrong rather than weaker.
NON_FILE_SOURCES = {
    "F01": ["stash:logits (battery prompts, all families)"],
    "F04": ["stash:logits (OLMo Think-SFT step checkpoints)"],
    "F07": ["arXiv:2512.13961"],                       # documentary; type specimen
    "F23": ["stash:reasoning_logits (source of record)"],
    "F31": ["stash:word_probs (4,098 entries, 37 models x ~120 prompts)"],
    "F33": ["stash:logits (32B/70B, 10 prompts each)", "logits_32b/", "logits_70b/"],
    "F34": ["stash:logits (CHINESE_PROMPTS, 73 prompts, 6 families) (source of record)"],
}

# Files associated with a finding by SEARCH rather than by citation -- the
# finding does not name them and the provenance map does not list them; they
# were matched on content. Kept separate from cited paths so the weaker basis
# of the association is visible rather than absorbed into the same list.
SCRIPTS_BY_SEARCH = {
    "F34": ["qwen_chinese_displacement.py", "f25_chinese_displacement_figure.py"],
}

IDENTIFIED_BY_SEARCH = {
    "F34": ["qwen_chinese_logits.csv (qwen only; matched by content, not cited)",
            "qwen_chinese_generations.csv (qwen only; matched by content, not cited)"],
}

# Per-finding annotations kept IN the frontmatter so a reader following the file
# list is told at the point of reading what it does and does not cover.
DATA_NOTES = {
    "F34": {"qwen_chinese_logits.csv": "(qwen only)",
            "qwen_chinese_generations.csv": "(qwen only)"},
}

RESOLVED = {
    "displacement_taxonomy.csv":
        "renamed to taxonomy_olmo.csv at 39a3886 (2026-05-03)",
    "reasoning_thinking_chains.csv":
        "never committed; source of record is the reasoning_logits stash",
}
SCRIPT_RE = re.compile(r"([\w./-]+\.py)")


def provenance_map():
    """finding number -> (commands, output files) from data/README.md."""
    out = {}
    for line in open("data/README.md"):
        m = re.match(r"\|\s*(F\d+)\b([^|]*)\|([^|]*)\|([^|]*)\|", line)
        if not m:
            continue
        num, cmds, outs = m.group(1), m.group(3), m.group(4)
        prev = out.get(num, (set(), set()))
        out[num] = (prev[0] | set(SCRIPT_RE.findall(cmds)),
                    prev[1] | set(FILE_RE.findall(outs)))
    return out


def first_commit(path):
    d = [x for x in subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", path],
        capture_output=True, text=True).stdout.split() if x]
    return d[-1] if d else None


def exists_data(name):
    for cand in (name, f"data/{name}", os.path.join("data", os.path.basename(name))):
        if os.path.exists(cand):
            return os.path.relpath(cand, "data") if cand.startswith("data/") else os.path.basename(cand)
    return None


def exists_script(name):
    base = os.path.basename(name)
    for cand in (name, f"scripts/{base}"):
        if os.path.exists(cand):
            return base
    return None


def main():
    prov = provenance_map()
    fams = sorted(MODEL_FAMILIES, key=len, reverse=True)
    out = {}
    for f in sorted(glob.glob("findings/F*.md")):
        if open(f).read(4).startswith("---"):
            continue                      # already has frontmatter
        num = re.match(r"findings/(F\d+)", f).group(1)
        txt = open(f).read()

        data_named = set(FILE_RE.findall(txt)) | prov.get(num, (set(), set()))[1]
        scr_named = set(SCRIPT_RE.findall(txt)) | prov.get(num, (set(), set()))[0]

        data = sorted({v for v in (exists_data(x) for x in data_named) if v})
        notes = DATA_NOTES.get(num, {})
        data = [f"{d} {notes[d]}" if d in notes else d for d in data]
        data = NON_FILE_SOURCES.get(num, []) + data + IDENTIFIED_BY_SEARCH.get(num, [])
        scripts = sorted({v for v in (exists_script(x) for x in scr_named) if v})
        scripts += [x for x in SCRIPTS_BY_SEARCH.get(num, []) if x not in scripts]
        missing = sorted({x for x in data_named if not exists_data(x)
                          and os.path.basename(x) not in RESOLVED})
        resolved = sorted({os.path.basename(x) for x in data_named
                           if os.path.basename(x) in RESOLVED})

        famhits = sorted({k for k in fams if re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", txt)})

        out[num] = dict(file=os.path.basename(f), date=first_commit(f),
                        data=data, scripts=scripts, families=famhits,
                        data_named_but_missing=missing,
                        historical_refs={k: RESOLVED[k] for k in resolved},
                        provenance_map_row=num in prov)
    json.dump(out, open("data/frontmatter_mechanical.json", "w"), indent=1)

    print(f"{'F':6s}{'date':>12s}{'data':>6s}{'scr':>5s}{'fam':>5s}{'gone':>6s}  map")
    for num in sorted(out, key=lambda x: int(x[1:])):
        r = out[num]
        print(f"{num:6s}{str(r['date']):>12s}{len(r['data']):>6d}{len(r['scripts']):>5d}"
              f"{len(r['families']):>5d}{len(r['data_named_but_missing']):>6d}  "
              f"{'yes' if r['provenance_map_row'] else 'NO'}")
    tot = sum(len(r["data"]) for r in out.values())
    gone = sum(len(r["data_named_but_missing"]) for r in out.values())
    print(f"\n{len(out)} findings | {tot} verified data files | {gone} named-but-missing "
          f"| {sum(len(r['scripts']) for r in out.values())} scripts")
    if gone:
        print("\nNAMED BUT NOT ON DISK (needs a human decision, not a guess):")
        for num in sorted(out, key=lambda x: int(x[1:])):
            for m in out[num]["data_named_but_missing"]:
                print(f"   {num}  {m}")
    hist = {(n, k): v for n, r in out.items() for k, v in r["historical_refs"].items()}
    if hist:
        print("\nHISTORICAL REFERENCES (named only inside a correction note, resolved):")
        for (n, k), v in sorted(hist.items()):
            print(f"   {n}  {k} -> {v}")


if __name__ == "__main__":
    main()
