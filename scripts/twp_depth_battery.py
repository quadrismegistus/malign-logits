#!/usr/bin/env python
"""twp_depth_battery.py — every depth measure, one pair loaded once, N prompts.

    scripts/twp_depth_battery.py --pairs 0,1 --n-prompts 10
    scripts/twp_depth_battery.py --base LLM360/Amber --aligned LLM360/AmberChat

The individual scripts each load both checkpoints, which is 20 s of the 30 s
they take. This loads a pair ONCE and sweeps prompts against it, so a roster
sweep costs load-per-pair rather than load-per-measurement.

PER PAIR (once):
    weight delta per block          where the optimizer put its updates
    head delta, and per-row         the frozen-head precondition

PER (PAIR, PROMPT):
    ||dh_L|| / ||h_L||              representational divergence by depth
    repr-patch recovery(L)          aligned RESIDUAL at depth L, base weights
    weight-patch top-K / bottom-K   aligned BLOCKS from the top / from the bottom

**READ THE PATCH PAIR TOGETHER, NEVER SEPARATELY.** On Llama/angry the
representation patch said alignment's work is not recoverable until L26 of 32 --
the downstream picture -- while the weight patch said the last two aligned
blocks give 12% and the first thirty give 91%. Both are true: the early changes
are not independently sufficient AND not inert. Quoting either alone inverts the
conclusion, which is why this driver computes both or neither.

WORD SETS COME FROM THE STORE via Step/Cell/movement(rule), so a prompt without
a cell for this pair is SKIPPED AND NAMED rather than silently recomputed.

CAVEAT carried from the single scripts: a hybrid model is not a model anyone
trained; blocks compose only because one checkpoint is a fine-tune of the other.
Ordinal readings only -- the top-K curve is non-monotone at ~0.05.
"""
import argparse, glob, json, math, os, re, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def weight_deltas(base, aligned):
    """||dW_L||/||W_L|| per block, read from safetensors without loading models."""
    from head_frozen_survey import snapshot_dir
    from safetensors import safe_open
    sb, sa = snapshot_dir(base), snapshot_dir(aligned)
    if not sb or not sa: return None
    def index(snap):
        idx = os.path.join(snap, "model.safetensors.index.json")
        if os.path.exists(idx): return json.load(open(idx))["weight_map"]
        m = {}
        for x in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(x, framework="pt") as fh:
                for k in fh.keys(): m[k] = os.path.basename(x)
        return m
    mb, ma = index(sb), index(sa)
    per, handles = defaultdict(lambda: [0.0, 0.0]), {}
    def get(snap, mp, k):
        p = os.path.join(snap, mp[k])
        if p not in handles: handles[p] = safe_open(p, framework="pt")
        return handles[p].get_tensor(k).float()
    for k in [k for k in mb if k in ma]:
        mm = re.search(r'layers?\.(\d+)\.', k)
        if not mm: continue
        tb, ta = get(sb, mb, k), get(sa, ma, k)
        if tb.shape != ta.shape: continue
        L = int(mm.group(1))
        per[L][0] += float((ta - tb).pow(2).sum()); per[L][1] += float(tb.pow(2).sum())
    return {L: (v[0] ** .5) / (v[1] ** .5) for L, v in sorted(per.items()) if v[1] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--aligned")
    ap.add_argument("--pairs", help="registry pair indices, comma separated")
    ap.add_argument("--prompts", nargs="*")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rule", default="canonical", choices=("canonical", "lens"))
    ap.add_argument("--kstep", type=int, default=4)
    ap.add_argument("--out", default="data/twp_depth_battery.jsonl")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.registry import Registry
    from malign_logits import movement as MV

    rule = MV.CANONICAL if a.rule == "canonical" else MV.LENS
    if a.base and a.aligned:
        pairs = [{"base": a.base, "aligned": a.aligned}]
    else:
        allp = Registry().base_aligned_pairs()
        pairs = ([allp[int(i)] for i in a.pairs.split(",")] if a.pairs else allp[:1])

    if a.prompts:
        prompts = a.prompts
    else:
        P = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
        items = list(P.values()) if isinstance(P, dict) else P
        want = ("sexual_liminal", "sexual_explicit", "violence_liminal", "violence_explicit")
        prompts = [v["prompt"] for v in items
                   if any((v.get("prompt_id") or "").startswith(w) for w in want)]
        prompts += [v["prompt"] for v in items
                    if (v.get("prompt_id") or "").startswith("e7_")]
        prompts = prompts[:a.n_prompts] if a.n_prompts else prompts

    fh = open(os.path.join(ROOT, a.out), "a")
    for pr in pairs:
        b, al = pr["base"], pr["aligned"]
        t0 = time.time()
        wd = weight_deltas(b, al)
        wsum = None
        if wd:
            v = [wd[L] for L in sorted(wd)]; n = len(v)
            wsum = {"mean": sum(v)/n, "first_third": sum(v[:n//3])/(n//3),
                    "last_third": sum(v[-n//3:])/(n//3), "n_blocks": n}
        print("\n######## %s -> %s" % (b, al.split('/')[-1]))
        if wsum:
            print("  weights: mean %.4f | first third %.4f | last third %.4f | ratio %.2f"
                  % (wsum["mean"], wsum["first_third"], wsum["last_third"],
                     wsum["last_third"]/wsum["first_third"]))

        step = Step(Checkpoint(b), Checkpoint(al))
        usable = []; c_pre = {}
        for p in prompts:
            try:
                c = step.cell(p); m = c.movement(rule)
                if len(m.fallers) + len(m.risers) >= 4:
                    usable.append((p, m)); c_pre[p] = dict(c.pre.probs)
            except Exception as e:
                print("  skip %-38r %s" % (p[:38], type(e).__name__))
        print("  prompts with a cell and >=4 movers: %d of %d" % (len(usable), len(prompts)))
        if not usable: continue

        tok, _ = twp.load_tokenizer(b); dev = twp.pick_device()
        base = AM.from_pretrained(b, dtype=getattr(torch, a.dtype)).to(dev).eval()
        alm = AM.from_pretrained(al, dtype=getattr(torch, a.dtype)).to(dev).eval()
        N = base.config.num_hidden_layers
        bb, ab = list(base.model.layers), list(alm.model.layers)
        print("  %-30s %5s %6s %6s %6s %5s %7s" %
              ("prompt", "mov", "top2", "bot-2", "ceil", "L50", "lensOOD"))
        for p, m in usable:
            words = list(m.fallers) + list(m.risers)
            #: **THE FLATS ARE THE LENS'S NOISE MODEL AND WITHOUT THEM ITS
            #: NUMBERS CANNOT BE READ.** head(norm(h_L)) mid-stack is out of
            #: distribution, so a mover's gap at L4 has no scale: measured raw
            #: it produced +0.250 at L4 against +0.022 at the output, a
            #: mid-stack magnitude LARGER than the final one, with sign flips
            #: between adjacent layers. CANONICAL's own third class -- words
            #: alignment did NOT move -- gives the per-layer spread that says
            #: whether a mover's gap is signal or the lens breathing.
            pre = c_pre[p]
            flats = [w for w in pre if w not in set(words)
                     and pre[w] >= max(rule.min_prob, 1e-9)]
            tid = lambda w: (tok.encode(" " + w, add_special_tokens=False) or [None])[0]
            ids = sorted({tid(w) for w in words if tid(w) is not None})
            fids = sorted({tid(w) for w in flats if tid(w) is not None} - set(ids))
            pids, _s, _r = twp._prompt_ids(tok, p, "inherited")
            x = torch.tensor([pids], device=dev)
            with torch.no_grad():
                ob = base(x, output_hidden_states=True)
                oa = alm(x, output_hidden_states=True)
            ob_hs = [h.detach().clone() for h in ob.hidden_states]
            lb = torch.log_softmax(ob.logits[0, -1, :].float(), -1).cpu()
            la = torch.log_softmax(oa.logits[0, -1, :].float(), -1).cpu()
            HA = [h.detach().clone() for h in oa.hidden_states]
            dh = {L: float((oa.hidden_states[L].float() - ob.hidden_states[L].float()).norm()
                           / ob.hidden_states[L].float().norm())
                  for L in range(len(ob.hidden_states))}
            del ob, oa
            use = [i for i in ids if abs(float(la[i] - lb[i])) > 1e-3]
            if len(use) < 3:
                print("  %-38s %6d  (too few separating tokens)" % (p[:38], len(words)))
                continue
            den = {i: float(la[i] - lb[i]) for i in use}
            def rec(lp): return statistics.median(float(lp[i]-lb[i])/den[i] for i in use)

            # representation patch
            st = {"L": None}
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                r = HA[st["L"]+1].to(h.dtype)
                return (r.clone(),) + tuple(out[1:]) if isinstance(out, tuple) else r.clone()
            repr_rec = {}
            for L in range(N):
                st["L"] = L
                hd = base.model.layers[L].register_forward_hook(hook)
                with torch.no_grad():
                    repr_rec[L] = rec(torch.log_softmax(base(x).logits[0,-1,:].float(),-1).cpu())
                hd.remove()
            l50 = next((L for L in range(N) if repr_rec[L] >= 0.5), None)

            # weight patch
            def runsel(sel):
                base.model.layers = torch.nn.ModuleList(
                    [ab[i] if sel[i] else bb[i] for i in range(N)])
                with torch.no_grad():
                    return rec(torch.log_softmax(base(x).logits[0,-1,:].float(),-1).cpu())
            ks = sorted(set(list(range(0, N+1, a.kstep)) + [2, N-2, N]))
            top = {k: runsel([i >= N-k for i in range(N)]) for k in ks}
            bot = {k: runsel([i < k for i in range(N)]) for k in ks}
            base.model.layers = torch.nn.ModuleList(bb)
            ceil = top[N] if top.get(N) else float('nan')
            nrm = lambda v: v/ceil if ceil and abs(ceil) > 1e-6 else float('nan')
            # ---- LAYER-BY-LAYER LENS CONTRAST, WITH ITS PERMISSIBILITY ----
            #: The naive form reads each arm through ITS OWN head, which mixes
            #: stack change with head change -- and no pair in the roster froze
            #: its head, so the naive form is never permissible. THE RESOLUTION
            #: IS NOT TO SKIP IT: read BOTH arms through the BASE head, which
            #: makes the lens defect common-mode by construction.
            #:
            #: That leaves ONE gate: is the aligned state through the base head
            #: still in distribution? Measured as the ratio of its top-1 mass to
            #: the true reading's. Amber fails this at 5x; Llama passes.
            lens = {"permitted": None, "reason": None, "gap": {}, "ood": {}}
            Wb = base.get_output_embeddings().weight.detach()
            nb = twp._final_norm(base)
            gain = getattr(nb, "weight", None)
            eps = float(getattr(nb, "variance_epsilon", 1e-5))
            worst = 0.0
            with torch.no_grad():
                for L in range(N + 1):
                    hb = ob_hs[L][0, -1, :].to(Wb.dtype)
                    ha = HA[L][0, -1, :].to(Wb.dtype)
                    if L != N and gain is not None:
                        f = lambda h: (h * torch.rsqrt(h.float().pow(2).mean(-1,
                                       keepdim=True) + eps).to(h.dtype) * gain)
                        hb, ha = f(hb), f(ha)
                    pb = torch.softmax((hb @ Wb.T).float(), -1)
                    pa = torch.softmax((ha @ Wb.T).float(), -1)   # BASE head, both
                    gp = lambda idx: [float(torch.log10(pb[i].clamp_min(1e-30) /
                                       pa[i].clamp_min(1e-30))) for i in idx]
                    mv = gp(use)
                    fl = gp(fids) if len(fids) >= 4 else []
                    lens["gap"][L] = statistics.median(mv)
                    if fl:
                        fl_s = sorted(fl)
                        lo = fl_s[int(.025 * len(fl_s))]
                        hi = fl_s[min(len(fl_s) - 1, int(.975 * len(fl_s)))]
                        lens.setdefault("flat_lo", {})[L] = lo
                        lens.setdefault("flat_hi", {})[L] = hi
                        lens.setdefault("outside", {})[L] = bool(
                            lens["gap"][L] > hi or lens["gap"][L] < lo)
                    r = float(pa.max() / pb.max().clamp_min(1e-30))
                    lens["ood"][L] = r
                    worst = max(worst, r, 1.0 / max(r, 1e-9))
            lens["worst_ratio"] = worst
            lens["n_flats"] = len(fids)
            run, lens["first_outside"] = 0, None
            for L in range(N + 1):
                if (lens.get("outside") or {}).get(L):
                    run += 1
                    if run >= 3 and lens["first_outside"] is None:
                        lens["first_outside"] = L - 2
                else:
                    run = 0
            lens["permitted"] = bool(worst < 3.0)
            lens["reason"] = ("fixed base head; cross-read within 3x"
                              if lens["permitted"] else
                              "cross-read %.1fx out of distribution -- the "
                              "aligned state is not readable by the base head "
                              "at some depth" % worst)

            rowd = {"base": b, "aligned": al, "prompt": p, "rule": rule.name,
                    "lens": lens,
                    "n_fallers": len(m.fallers), "n_risers": len(m.risers),
                    "n_tokens": len(use), "N": N, "weights": wsum,
                    "dh": dh, "repr_recovery": repr_rec, "repr_L50": l50,
                    "top": top, "bottom": bot, "ceiling": ceil}
            fh.write(json.dumps(rowd) + "\n"); fh.flush()
            print("  %-30s %5d %6.2f %6.2f %6s %5s %6.1fx"
                  % (p[:30], len(words), top.get(2, float('nan')),
                     bot.get(N-2, float('nan')), "%.2f"%ceil, l50,
                     lens["worst_ratio"]) + "  lensL%s/%df" % (
                         lens.get("first_outside"), lens.get("n_flats", 0)))
        del base, alm
        try: torch.mps.empty_cache()
        except Exception: pass
        print("  pair done in %.1f min" % ((time.time()-t0)/60))
    fh.close()
    print("\nappended to %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
