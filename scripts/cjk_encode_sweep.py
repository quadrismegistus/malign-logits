"""Roster precondition 5: what happens to CJK INPUT, per model.

    uv run .venv/bin/python scripts/cjk_encode_sweep.py

The coverage build (cjk_coverage.csv) measured which tokens EXIST. This measures
what a tokenizer DOES with CJK it is given. They are different questions and
only the second predicts a failure.

deepseek-llm-7b-base is the case that forced it: vocab 100,000, English encodes
normally, and `encode("她非常生气")` returns []. Not an UNK, not a replacement,
not an error -- the characters vanish. An empty id list then crashes the
expansion with a dtype error naming nothing relevant.

THE WORSE CASE IS THE ONE THAT DOES NOT CRASH. A MIXED prompt would be silently
truncated to its English remainder, score normally, and look fine -- a plausible
number computed on the wrong input, which is this project's recurring enemy.
So the sweep records BOTH: does the pure-CJK probe survive, and does the mixed
probe keep its CJK.
"""
import csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA
import importlib.util as _ilu
_sp=_ilu.spec_from_file_location("tc", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "twp_cloud.py"))
_tc=_ilu.module_from_spec(_sp); _sp.loader.exec_module(_tc)

CJK = re.compile(r'[一-鿿㐀-䶿]')
PURE = "她非常生气，想要"
MIXED = "The word 她 means she"


def probe(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    back = tok.decode(ids)
    return len(ids), sum(1 for c in CJK.findall(back))


def main():
    seen, rows = set(), []
    for fam, F in sorted(MODEL_FAMILIES.items()):
        for pos in ("base", "ego", "superego", "reinforced_superego"):
            mid = getattr(F, pos, None)
            if not mid or mid in seen:
                continue
            seen.add(mid)
            try:
                tok, _loader = _tc.load_tokenizer(mid)   # THROUGH THE TABLE
            except Exception as e:
                rows.append(dict(model=mid, family=fam, pure_ids=-1, pure_cjk=-1,
                                 mixed_ids=-1, mixed_cjk=-1, drops_cjk="",
                                 note=type(e).__name__))
                continue
            want = len(CJK.findall(PURE))
            pi, pc = probe(tok, PURE)
            mi, mc = probe(tok, MIXED)
            drops = (pi == 0) or (pc == 0 and want > 0)
            partial = (not drops) and pc < want
            rows.append(dict(model=mid, family=fam, pure_ids=pi, pure_cjk=pc,
                             mixed_ids=mi, mixed_cjk=mc,
                             drops_cjk=str(bool(drops)).lower(),
                             note="partial_loss" if partial else ""))
            flag = "  *** DROPS CJK" if drops else ("  partial" if partial else "")
            print(f"  {mid[:50]:<52}pure {pi:>3}ids/{pc}cjk  mixed {mi:>3}/{mc}{flag}",
                  flush=True)

    out = os.path.join(PATH_DATA, "cjk_encode_sweep.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = [r for r in rows if r["drops_cjk"] == "true"]
    p = [r for r in rows if r["note"] == "partial_loss"]
    print(f"\n{len(rows)} models. DROPS CJK: {len(d)}   partial loss: {len(p)}")
    for r in d:
        print(f"   DROPS  {r['model']}")
    for r in p[:8]:
        print(f"   partial {r['model']}  ({r['pure_cjk']}/{len(CJK.findall(PURE))} chars survive)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
