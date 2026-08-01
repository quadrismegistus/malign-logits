"""Mechanical template-audit of the Sonnet pair drafts against
agents/lacan/pair_authoring_template.md §5 THE BIRTH CHECKLIST.

Checks ONLY what is mechanical. The semantic gates -- (b) unmarked member
genuinely innocuous, (c) prompt states rather than elicits, (d) an agent
transgresses -- are RH's construct read and are NOT adjudicated here.
Every count prints beside the population it was taken over.
"""
import glob, os, re, sys, collections
import yaml

REQ = ["pair_id", "contrast_type", "domain", "language", "writer",
       "MARKED", "UNMARKED"]
files = sorted(glob.glob(os.path.expanduser(
    "~/Dropbox/Prof/Articles/TheoryMachines/pair_drafts/*.yaml")))

rows, per_file = [], collections.Counter()
for f in files:
    d = yaml.safe_load(open(f))
    for r in d:
        r["_file"] = os.path.basename(f)
        rows.append(r); per_file[os.path.basename(f)] += 1

print(f"POPULATION: {len(rows)} pairs over {len(files)} files")
for f, n in sorted(per_file.items()):
    print(f"   {f:26s} {n:>4}")

def toks(s):
    return s.split()

fail = collections.defaultdict(list)

# --- field presence, and contrast_type must MEAN it -------------------------
for r in rows:
    for k in REQ:
        if not r.get(k):
            fail[f"missing field: {k}"].append(r.get("pair_id", "?"))
    if r.get("contrast_type") != "transgressive_swap":
        fail["contrast_type not transgressive_swap"].append(r.get("pair_id"))
    if "group_role" in r and r.get("group_role") != r.get("pair_role"):
        fail["group_role disagrees with pair_role"].append(r.get("pair_id"))

# --- pair_id uniqueness ACROSS all files -----------------------------------
ids = collections.Counter(r.get("pair_id") for r in rows)
for k, n in ids.items():
    if n > 1:
        fail["pair_id collides across drafts"].append(f"{k} x{n}")

# --- string duplication across the whole draft pool ------------------------
seen = collections.defaultdict(list)
for r in rows:
    for role in ("MARKED", "UNMARKED"):
        if r.get(role):
            seen[r[role].strip()].append(f"{r['pair_id']}:{role}")
for s, where in seen.items():
    if len(where) > 1:
        fail["duplicate string in pool"].append(" == ".join(where))

# --- one substitution, and it must not be final ----------------------------
for r in rows:
    m, u = r.get("MARKED", ""), r.get("UNMARKED", "")
    if not m or not u:
        continue
    tm, tu = toks(m), toks(u)
    # longest common prefix / suffix -> the differing span
    p = 0
    while p < min(len(tm), len(tu)) and tm[p] == tu[p]:
        p += 1
    s = 0
    while (s < min(len(tm), len(tu)) - p and tm[-1 - s] == tu[-1 - s]):
        s += 1
    span_m, span_u = tm[p:len(tm) - s], tu[p:len(tu) - s]
    if not span_m and not span_u:
        fail["members identical"].append(r["pair_id"]); continue
    # tokens that FOLLOW the substitution, excluding the scored slot ___
    tail = [t for t in tm[len(tm) - s:] if t != "___"]
    if len(tail) == 0:
        fail["ANTI-PATTERN (a): swap is final, nothing between it and ___"].append(
            f"{r['pair_id']}  {m}")
    r["_span"] = (" ".join(span_m), " ".join(span_u))
    r["_tail"] = len(tail)
    if abs(len(tm) - len(tu)) > 1:
        fail["length differs by >1 token"].append(
            f"{r['pair_id']}  {len(tm)}v{len(tu)}")
    # declared swap must match the measured one
    dec = (r.get("swap") or "").lower()
    if dec:
        a = " ".join(span_m).lower().strip(".,")
        b = " ".join(span_u).lower().strip(".,")
        if a and b and (a not in dec or b not in dec):
            fail["declared swap != measured swap"].append(
                f"{r['pair_id']}  declared '{r['swap']}'  measured '{a} -> {b}'")

print("\nMECHANICAL CHECKLIST (template §5). Each count is over all "
      f"{len(rows)} pairs unless noted.\n")
clean = True
for k in sorted(fail):
    clean = False
    v = fail[k]
    print(f"  {len(v):>4}  {k}")
    for x in v[:6]:
        print(f"          {x}")
    if len(v) > 6:
        print(f"          ... and {len(v) - 6} more")
if clean:
    print("  0     every mechanical gate clears")

print(f"\nTAIL LENGTH (words between the swap and the scored slot), "
      f"over {sum(1 for r in rows if '_tail' in r)} pairs:")
td = collections.Counter(r["_tail"] for r in rows if "_tail" in r)
for k in sorted(td):
    print(f"   {k} word(s) after the swap: {td[k]:>4} pairs"
          + ("   <- ZERO IS ANTI-PATTERN (a)" if k == 0 else ""))

print("\nNOT ADJUDICATED HERE (semantic, RH's construct read):")
print("  (b) is the UNMARKED member genuinely innocuous read on its own")
print("  (c) does the prompt STATE the transgression or ELICIT it")
print("  (d) is there an agent who transgresses")
print("  catalogue collision against existing rows -- malign's field audit")
