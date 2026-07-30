"""true_word_probs on a cloud box. NO PROJECT DEPENDENCIES, JSONL output.

    python twp_cloud.py --models models.txt --out /workspace/twp

WHY JSONL AND NOT THE STASH. HashStash is lmdb: one large file that changes on
every write, so rsync re-transfers the whole thing each sync and a kill mid-write
risks the store. Here each line is a COMPLETE record appended and flushed, so a
kill loses at most the line in flight, a finished model's file never changes
again, and repeated rsync pulls only the model currently in progress.

The local machine keeps HashStash as the canonical store; these files merge into
it through CacheManager, where the pinned open is enforced. Same round trip as
F37, which worked.

MODELS ARE PROCESSED SMALLEST FIRST and the HF cache entry is deleted after each,
because the binding constraint is DOWNLOAD (~1.3 TB for the roster), not compute.
Ascending order also means anything too large for the card sorts to the end,
where cancelling costs nothing.
"""
import argparse, gc, json, os, re, shutil, subprocess, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THETA, MAX_DEPTH = 0.001, 6
PUNCT = set(".,;:!?\"'()[]{}—-–…/\\*#") | {"\n", "\r", "\t"}
# FULLWIDTH CJK PUNCTUATION WAS MISSING AND IT IS NOT COSMETIC. The set above is
# ASCII-only, so `。` `，` `？` `！` were never boundaries -- in Chinese, where
# sentence punctuation is the ONLY boundary this mask can see, that meant no
# boundary at all. Measured on CT-LLM: adding these takes resolved mass from
# 0.061 to 0.472 with no other change. (Dictionary word boundaries take it to
# 0.860; that is a separate extension. These two are independent bugs and this
# is the one-line half.)
CJK_PUNCT = set("。，、；：！？「」『』（）《》〈〉【】…—～·　")
PUNCT |= CJK_PUNCT


CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "dict", "jieba_dict_big.txt")


def is_cjk(s):
    return bool(s) and bool(CJK.search(s))


def clean_surface(s):
    """A word cannot CONTAIN punctuation or whitespace, so truncate at the first.

    The boundary mask asks whether a token STARTS a word, which is a test on its
    FIRST character. Byte-level BPE merges punctuation into the middle of tokens
    -- Qwen has one token for the string 'X?' -- so a token beginning with a CJK
    glyph can carry a clause end inside it that the first-character test cannot
    see. Truncating is exact rather than cosmetic: the mass on 'W?' IS the mass
    on word W terminated by punctuation. Surfaces differing only in trailing
    punctuation merge, which is correct -- they are the same word.
    """
    for i, c in enumerate(s):
        if c in PUNCT or c.isspace():
            return s[:i]
    return s


def load_prefix_trie(path=DICT):
    """Dictionary words plus every PROPER PREFIX of one, or None if absent.

    Proper prefixes must be present because the test during expansion is "could
    this surface still grow into a word". Without them every multi-character
    word is cut at its first character, which is character-split by another
    route -- and character-split reproduces at character level exactly the
    uninterpretability that motivated word-level probabilities in the first
    place.
    """
    if not os.path.exists(path):
        return None
    pref = set()
    with open(path) as f:
        for line in f:
            w = line.split(" ")[0].strip()
            if not w or not CJK.search(w):
                continue
            pref.add(w)
            for i in range(1, len(w)):
                pref.add(w[:i])
    return pref


def cjk_vocab(tok, n):
    """(ids, strings) for tokens that DECODE to bare CJK.

    DECODE, never convert_ids_to_tokens: byte-level BPE returns the byte
    mangling for CJK, so a script test on the token string finds ZERO CJK
    tokens in a Chinese model's vocabulary.
    """
    ids, strs = [], []
    for i in range(n):
        try:
            t = tok.decode([i])
        except Exception:
            continue
        if t and is_cjk(t) and t == t.strip():
            ids.append(i); strs.append(t)
    return np.array(ids, dtype=int), strs


def boundary_mask(tok, n):
    """MODEL VOCAB SIZE AND TOKENIZER PIECE COUNT ARE NOT THE SAME NUMBER.

    `config.vocab_size` is the unembedding width, which is padded up to a
    hardware-friendly multiple; the tokenizer holds fewer real pieces. CT-LLM's
    sentencepiece raises `IndexError: piece id is out of range` on the padding
    ids rather than returning None, which killed the roster from inside a
    function nobody thought could fail.

    An id with no piece is treated as a BOUNDARY. It can never be produced as
    real text, so the only question is whether it terminates a word or extends
    one, and terminating is the safe answer: an unknown id extending a prefix
    would silently glue garbage onto a real word, while terminating merely ends
    it. The mass involved is negligible either way -- these ids are untrained.
    """
    m = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            s = tok.convert_ids_to_tokens(i)
        except Exception:
            m[i] = True; continue      # padding id past the tokenizer's pieces
        if s is None:
            m[i] = True; continue
        if s.startswith("Ġ") or s.startswith("▁") or s.startswith(" "):
            m[i] = True
        elif s and (s[0] in PUNCT or s.strip() == ""):
            m[i] = True
        elif s.startswith("<") and s.endswith(">"):
            m[i] = True
    return m


def free(*objs):
    """del -> gc.collect() -> empty_cache(), IN THAT ORDER.

    `del model` drops one reference; HF modules hold cycles (child -> parent,
    config, hooks), so the object survives until the cycle collector runs, and
    `empty_cache()` only returns blocks the allocator has ALREADY reclaimed --
    with a live cycle it is a no-op. On the exception path it is worse: the
    traceback holds the frame that holds the activations, which is how a 1.5B
    model came to OOM against 65 GiB in use.
    """
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def next_dist(model, tok, pids, prefixes, dev, batch=64):
    """Batch is ADAPTIVE because it is architecture-blind.

    A dense transformer's peak scales with batch x seq x vocab. An SSM's does
    not: Falcon-H1's `torch_forward` materialises B_decay[...,None,:] *
    hidden_states[...,None], which is batch x seq x heads x state x dim and hit
    24 GiB at batch=64 on a 1.5B model -- it OOM'd where a 7B transformer was
    fine. Rather than maintain a per-architecture table that is wrong the moment
    a new family is registered, halve on OOM and carry on.
    """
    out, i = [], 0
    while i < len(prefixes):
        ch = prefixes[i:i + batch]
        seqs = [pids + list(p) for p in ch]
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([[pad]*(L-len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0]*(L-len(s)) + [1]*len(s) for s in seqs], device=dev)
        try:
            lg = model(ids, attention_mask=att).logits[:, -1, :].float()
        except torch.OutOfMemoryError:
            del ids, att
            gc.collect(); torch.cuda.empty_cache()
            if batch == 1:
                raise                      # genuinely cannot fit; let it surface
            batch = max(1, batch // 2)
            print(f"    [oom] batch -> {batch}", flush=True)
            continue                       # retry the SAME slice, smaller
        out.append(torch.softmax(lg, -1).cpu().numpy())
        i += len(ch)
    return np.concatenate(out, 0)


@torch.no_grad()
def expand(model, tok, prompt, dev, bmask, theta=THETA, cjk=None):
    """cjk=(prefix_trie, ids, strings) enables dictionary word boundaries.

    ONE INSTRUMENT, DISPATCHING ON SCRIPT -- not two runs side by side. A non-CJK
    surface keeps the whitespace/punctuation rule; a CJK surface takes the
    dictionary rule. Mixed-script text is therefore handled correctly WITHIN a
    cell, which two parallel datasets could not do since they would force a
    per-cell choice mixed text cannot make.
    """

    pids = tok.encode(prompt)
    lg = model(torch.tensor([pids], device=dev)).logits[0, -1, :].float()
    P0 = torch.softmax(lg, -1).cpu().numpy()
    sel = np.flatnonzero(P0 >= theta)
    live = [((int(t),), float(P0[t]), int(t)) for t in sel]
    words, calls, bcache = {}, 0, {}
    res_tail, res_drop = float(1.0 - P0[sel].sum()), 0.0
    for _ in range(MAX_DEPTH):
        if not live:
            break
        dist = next_dist(model, tok, pids, [p for p, _, _ in live], dev); calls += 1
        nxt = []
        for (pref, mass, t1), row in zip(live, dist):
            surf = clean_surface(tok.decode(list(pref)).strip())
            b = bmask
            if cjk is not None and is_cjk(surf):
                trie, cids, cstrs = cjk
                b = bcache.get(surf)
                if b is None:
                    # EVERY CJK token is judged, not just the probable ones.
                    # Termination sums over ALL boundary tokens, so improbable
                    # CJK tokens that should end the word must be marked or the
                    # word runs to the clause end and its mass drains via `drop`.
                    b = bmask.copy()
                    inside = np.fromiter(((surf + t) in trie for t in cstrs),
                                         dtype=bool, count=len(cstrs))
                    b[cids] = ~inside
                    bcache[surf] = b
            term = float(row[b].sum())
            if surf:
                words[(surf, t1)] = words.get((surf, t1), 0.0) + mass * term
            else:
                res_drop += mass * term
            cont = np.flatnonzero(~b)
            m2 = mass * row[cont]
            keep = m2 >= theta
            for t, mm in zip(cont[keep], m2[keep]):
                nxt.append(((*pref, int(t)), float(mm), t1))
            res_drop += float(m2[~keep].sum())
        live = nxt
    res_open = float(sum(m for _, m, _ in live))
    return words, dict(tail=res_tail, drop=res_drop, open=res_open,
                       total=res_tail + res_drop + res_open), calls


def done_prompts(path):
    """Resume by reading back what was written. Tolerates a truncated last line."""
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    seen.add(json.loads(ln)["prompt"])
                except Exception:
                    pass          # partial final line from a kill: ignore, redo it
    return seen


def main(a):
    spec = json.load(open(a.models))          # [{model, prompts:[...]}, ...]
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trie = None if a.no_dict else load_prefix_trie(a.dict)
    if trie is None:
        print("NO CJK DICTIONARY -- Chinese resolves ~3-16% of mass against "
              "80-90% for English. Chinese cells produced without it are not "
              "usable at word level.", flush=True)
    else:
        print(f"cjk dictionary: {len(trie):,} words+prefixes", flush=True)
    for mi, entry in enumerate(spec, 1):
        mid, prompts = entry["model"], entry["prompts"]
        safe = mid.replace("/", "__")
        path = os.path.join(a.out, f"{safe}.jsonl")
        todo = [p for p in prompts if p not in done_prompts(path)]
        print(f"\n[{mi}/{len(spec)}] {mid}  {len(todo)}/{len(prompts)} to do", flush=True)
        if not todo:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        except Exception as e:
            print(f"  LOAD FAILED: {str(e)[:120]}", flush=True)
            free()                 # the traceback held the partial load
            continue
        # INSIDE THE GUARD. This sat BETWEEN the guarded load and the guarded
        # run, so a tokenizer that cannot decode every id in range(vocab_size)
        # killed the whole roster from the one unguarded line -- CT-LLM's
        # sentencepiece raises "piece id is out of range" because the model's
        # config vocab_size exceeds the tokenizer's actual piece count. Guarding
        # two of three phases is guarding none of them.
        try:
            bmask = boundary_mask(tok, model.config.vocab_size)
            cjk = None
            if trie is not None:
                cids, cstrs = cjk_vocab(tok, model.config.vocab_size)
                if len(cids):
                    cjk = (trie, cids, cstrs)
                    print(f"  cjk: {len(cids):,} tokens", flush=True)
        except Exception as e:
            print(f"  MASK FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            free(model, tok)
            continue
        t0, i = time.time(), 0
        # ONE MODEL MUST NOT END THE ROSTER. The first version guarded only the
        # load, so a mid-run OOM on model 17 of 103 took the other 87 with it.
        # Per-prompt writes are already flushed, so a model that dies partway
        # keeps what it finished and resumes there on the next pass.
        try:
            with open(path, "a") as f:
                for i, p in enumerate(todo, 1):
                    w, res, calls = expand(model, tok, p, dev, bmask, cjk=cjk)
                    tot = sum(w.values()) + res["total"]
                    f.write(json.dumps({
                        "model": mid, "prompt": p, "theta": THETA,
                        "rows": [{"word": s_, "t1": t_, "p": m_} for (s_, t_), m_ in w.items()],
                        "residual": res, "batches": calls, "conservation": tot}) + "\n")
                    f.flush()                  # crash-safe: complete line on disk
                    if i % 50 == 0:
                        print(f"    {i}/{len(todo)}  {i/(time.time()-t0):.2f} p/s", flush=True)
            print(f"  done {len(todo)} in {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print(f"  RUN FAILED after {i-1}/{len(todo)}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        free(model, tok)
        if a.purge:                            # download is the binding constraint
            for d in ("~/.cache/huggingface/hub",):
                for sub in os.listdir(os.path.expanduser(d)):
                    if sub.startswith("models--") and mid.replace("/", "--") in sub:
                        shutil.rmtree(os.path.join(os.path.expanduser(d), sub),
                                      ignore_errors=True)
                        print(f"  purged {sub}", flush=True)
    print("\nALL MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="/workspace/twp")
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--dict", default=DICT,
                    help="CJK prefix dictionary; on a cloud box the repo-relative "
                         "default will not resolve, so pass the uploaded path")
    ap.add_argument("--no-dict", action="store_true",
                    help="disable CJK dictionary boundaries (reproduces the "
                         "pre-fix rule; Chinese cells will be unusable)")
    main(ap.parse_args())
