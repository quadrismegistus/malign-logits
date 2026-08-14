#!/usr/bin/env python3
"""bge-m3 sentence embeddings over the exported passage set, shardable for a fleet.

    python scripts/bge_cloud.py --input blt_passages.jsonl.gz --out /workspace/bge \
        --shard 0 --of 4 --mixed-policy dominant [--limit N]

lacan's commission [5896], run on the SAME 483,085 passages as the BLT pass and
keyed the same way, so the two join on (prompt, text_sha) with no re-derivation.

THE THREE TRAPS ARE STORAGE-SHAPED, and each is answered structurally here
rather than by remembering to be careful:

 1. NAMESPACE THE KEY BY SPLITTER. `BAAI/bge-m3` alone is not the identity of
    this result: the same passage under nltk-en and under stanza-zh yields
    DIFFERENT sentences, hence different vectors and a different count of them.
    Every row carries `ref` = "BAAI/bge-m3|<splitter>", written from the
    splitter actually used on THAT passage, never from a run-level default. A
    shared key would let the second splitter's run silently read the first's.
 2. NDARRAY, NEVER `.tolist()`. Vectors go to a .f32 sidecar via `.tobytes()`.
    `.tolist()` would cost ~4x the bytes, lose the exact float32 bit pattern to
    repr rounding, and make the jsonl unreadable line-by-line.
 3. KEY ON RAW TEXT, NEVER THE JOINED SENTENCES. `text_sha` is sha256 of the
    passage exactly as it arrived. Splitting then rejoining normalises
    whitespace, so a joined-text key would (a) differ between splitters for one
    passage and (b) fail to join back to the BLT pass, which hashed raw text.

THE `mixed` STRATUM IS A SPECIFICATION GAP AND THIS SCRIPT REFUSES TO CLOSE IT
SILENTLY. The corpus is en 372,103 / zh 78,879 / **mixed 32,103 (6.6%)**, and
the commission named two splitters for three strata. The mixed passages are
genuinely mixed -- median CJK share of letter characters 0.25, only 1.5% above
0.9 -- so a dominant-script rule sends ~84% of them to nltk, which does not
split on the ideographic full stop at all and will return CJK spans as one
sentence. Sentence count and length ARE the unit of this job, so that is a
methodological choice, not a default. `--mixed-policy` is therefore REQUIRED and
its value is recorded on every row it touched.
"""
import argparse, gzip, hashlib, json, os, time
import numpy as np

BGE = "BAAI/bge-m3"
DIM = 1024


def done_keys(*paths):
    """(prompt, text_sha) already HANDLED in this shard's own output.

    HANDLED, not embedded -- the refused file counts too. A refused passage is
    a settled outcome, not pending work: re-reading it on resume re-appends an
    identical refusal row, so a shard restarted N times accumulates N copies of
    every refusal. The embeddings stay correct throughout, which is what makes
    it easy to miss; what drifts is the COUNT. `bge_fleet_sweep` reports
    refusals as `wc -l` over this file, and the manifest carries the total, so
    both would inflate with restarts while the data underneath was fine.
    """
    got = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                got.add((r.get("prompt"), r.get("text_sha")))
    return got


def cjk_share(text):
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return cjk / max(cjk + lat, 1)


class Splitters:
    """Lazy, so a shard that never meets zh does not pay for stanza."""

    def __init__(self):
        self._nltk = None
        self._stanza = None

    def en(self, text):
        if self._nltk is None:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            self._nltk = nltk.sent_tokenize
        return [s for s in self._nltk(text) if s.strip()]

    def zh(self, text):
        if self._stanza is None:
            import stanza
            #: No `download_method=None`. It reads as "do not download" and does
            #: not prevent anything: stanza >=1.6 resolves models through the
            #: HuggingFace hub (stanfordnlp/stanza-zh-hans), not the legacy
            #: ~/stanza_resources tree, so the flag described a path this
            #: version no longer uses.
            self._stanza = stanza.Pipeline(
                lang="zh", processors="tokenize", verbose=False)
        return [s.text for s in self._stanza(text).sentences if s.text.strip()]

    def warm(self):
        """Build BOTH splitters before the loop, and prove each one splits.

        THE FAILURE THIS PREVENTS LOOKS LIKE FULL THROUGHPUT. Built lazily, a
        stanza failure on the first zh passage leaves `self._stanza` None, the
        caller's except turns that passage into a refusal, and every subsequent
        zh passage repeats it -- 78,879 passages, 16.3% of the corpus, quietly
        becoming refusals while the shard reports its normal rate and exits 0.
        A missing model must kill the run in the first second instead.
        """
        #: PROBES MUST BE UNAMBIGUOUS FOR THE TOOL, not merely for me. The first
        #: zh probe here was `我今天很好。你呢？`, which stanza returns as ONE
        #: sentence -- correctly, since `你呢？` is an interrogative fragment
        #: continuing the statement -- and the assert then failed on a splitter
        #: that was working, having encoded my segmentation intuition rather
        #: than tested the instrument. A probe whose FAIL is ambiguous cannot
        #: gate a run. Three full independent sentences leave no such room.
        n_en = len(self.en("A sentence. And a second one. Then a third one."))
        n_zh = len(self.zh("他既是美丽的又是丑陋的。她想要更多。这是第三句。"))
        assert n_en >= 3, "nltk-en split a three-sentence probe into %d" % n_en
        assert n_zh >= 3, "stanza-zh split a three-sentence probe into %d" % n_zh
        print("  splitters warm: nltk-en -> %d, stanza-zh -> %d on the probes"
              % (n_en, n_zh), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="/workspace/bge")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    #: REQUIRED, no default. See the module docstring: 6.6% of the corpus has no
    #: declared splitter and the producer must not invent one by running.
    ap.add_argument("--mixed-policy", required=True,
                    choices=["dominant", "zh", "en", "refuse"],
                    help="dominant = route by CJK share of letter chars (>=0.5 -> zh)")
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s | shard %d/%d | mixed-policy %s"
          % (dev, a.shard, a.of, a.mixed_policy), flush=True)

    model = SentenceTransformer(BGE, device=dev)
    sp = Splitters()
    #: BEFORE the loop, always -- both splitters are used under every
    #: policy, including `refuse`, which drops only the mixed stratum.
    sp.warm()

    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, "bge_shard%02d.jsonl" % a.shard)
    fb = os.path.join(a.out, "bge_shard%02d.f32" % a.shard)
    rp = os.path.join(a.out, "bge_shard%02d.refused.jsonl" % a.shard)
    done = done_keys(jl, rp)
    #: ROW COUNTER FROM THE FILE'S OWN SIZE, never a remembered count -- the
    #: defect twp_cloud.py fixed for its .f16, and blt_cloud.py for its .f32.
    row = os.path.getsize(fb) // 4 if os.path.exists(fb) else 0
    assert row % DIM == 0, (
        "sidecar holds %d floats, not a multiple of dim %d -- a previous run was "
        "killed mid-write and the file is torn; truncate to %d before resuming"
        % (row, DIM, (row // DIM) * DIM))
    #: "handled", not "embedded" -- `done` now includes refusals, so the word
    #: has to match the quantity or the next reader subtracts the wrong number.
    print("resuming: %d already handled (embedded + refused), %d vectors in "
          "the sidecar" % (len(done), row // DIM), flush=True)

    n_seen = n_new = n_sent = 0
    refused = []
    t0 = time.time()
    with gzip.open(a.input, "rt") as src, open(jl, "a") as out, open(fb, "ab") as sb:
        for i, line in enumerate(src):
            if i % a.of != a.shard:
                continue
            r = json.loads(line)
            n_seen += 1
            if a.limit and n_seen > a.limit:
                break
            text, prompt = r["text"], r["prompt"]
            #: RAW text, matching the BLT pass so the two tables join.
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            if (prompt, sha) in done:
                continue

            script = r.get("script")
            if script == "mixed":
                if a.mixed_policy == "refuse":
                    refused.append({"prompt": prompt, "text_sha": sha,
                                    "script": script, "why": "mixed-policy=refuse"})
                    continue
                if a.mixed_policy == "dominant":
                    use = "zh" if cjk_share(text) >= 0.5 else "en"
                else:
                    use = a.mixed_policy
            else:
                use = script if script in ("en", "zh") else "en"

            splitter = {"en": "nltk-en", "zh": "stanza-zh"}[use]
            try:
                sents = sp.zh(text) if use == "zh" else sp.en(text)
            except Exception as e:
                refused.append({"prompt": prompt, "text_sha": sha, "script": script,
                                "splitter": splitter, "why": "split failed: %s" % e})
                continue
            if not sents:
                refused.append({"prompt": prompt, "text_sha": sha, "script": script,
                                "splitter": splitter, "why": "no sentences"})
                continue

            #: ndarray throughout. `.tolist()` is never CALLED here -- grep it
            #: and you get three hits, all prose (this line and the docstring),
            #: which is said explicitly because a bare count of 3 reads as a
            #: violation to whoever checks.
            vecs = model.encode(sents, batch_size=a.batch, convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float32)
            assert vecs.shape == (len(sents), DIM), vecs.shape
            sb.write(vecs.tobytes())

            out.write(json.dumps({
                "prompt": prompt, "text_sha": sha, "script": script,
                "splitter": splitter,
                #: THE KEY, namespaced by the splitter ACTUALLY used on this row.
                "ref": "%s|%s" % (BGE, splitter),
                "mixed_policy": a.mixed_policy if script == "mixed" else None,
                "corpora": r.get("corpora"), "n_chars": r.get("n_chars"),
                "n_sentences": len(sents),
                "sent_chars": [len(s) for s in sents],
                "row": row, "n": int(vecs.size), "dim": DIM,
                "normalized": True}) + "\n")
            row += int(vecs.size)
            n_sent += len(sents)
            n_new += 1
            if n_new % 200 == 0:
                out.flush(); sb.flush()
                el = (time.time() - t0) / 60
                print("  %d embedded  %d sentences  %.1f min  %.1f/s"
                      % (n_new, n_sent, el, n_new / max(el * 60, 1)), flush=True)

    #: RUN MANIFEST CARRYING `_about`, per registrar's rider [5956]: a refused
    #: stratum is an ABSENCE in the output, and an absence has to be visible
    #: where the reader meets the data, not only in the docket. Written every
    #: run, including partial ones, so a shard killed mid-way still says what it
    #: was excluding and under which policy.
    mf = os.path.join(a.out, "bge_shard%02d.manifest.json" % a.shard)
    json.dump({"_about": {
        "ref": "%s|<splitter>, per row; nltk-en and stanza-zh both appear" % BGE,
        "commission": "lacan [5896]; mixed-policy decided at [5955]",
        "mixed_policy": a.mixed_policy,
        #: `.format`, not `%` -- these sentences carry literal percent signs
        #: (6.6%, 25%) and %-formatting reads them as conversion specifiers.
        #: It raised at the MANIFEST WRITE, i.e. after every passage had been
        #: embedded: the run would have done all its work, exited non-zero, and
        #: left no manifest, which the fleet sweep reads as a crashed shard.
        "mixed_stratum": (
            "The corpus is en 372,103 / zh 78,879 / mixed 32,103 (6.6%, median "
            "CJK share of letter characters 0.25). Under policy '{}' the mixed "
            "stratum is {}.".format(
                a.mixed_policy,
                "REFUSED and absent from this table"
                if a.mixed_policy == "refuse"
                else "segmented, see per-row mixed_policy")),
        "why_refused": (
            "No monolingual splitter is defensible on text that is median 25% "
            "CJK, and the mis-segmentation scales with CJK share -- a confound "
            "correlated with the stratum-defining variable, which is worse than "
            "a smaller n because it is unrecoverable from the output."
            if a.mixed_policy == "refuse" else None),
        "vectors": "float32, L2-normalized, dim %d, in the .f32 sidecar" % DIM,
        "join": "(prompt, text_sha) with the BLT pass; text_sha is sha256 of RAW text",
        "shard": a.shard, "of": a.of,
        "passages_embedded": n_new, "sentences": n_sent, "refused": len(refused),
    }}, open(mf, "w"), indent=1)
    print("  manifest -> %s" % os.path.basename(mf), flush=True)

    if refused:
        with open(rp, "a") as fh:
            for r in refused:
                fh.write(json.dumps(r) + "\n")
        print("  REFUSED %d passage(s); recorded in %s"
              % (len(refused), os.path.basename(rp)), flush=True)
    print("shard %d done: %d seen, %d newly embedded, %d sentences, %d refused, %.1f min"
          % (a.shard, n_seen, n_new, n_sent, len(refused),
             (time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
