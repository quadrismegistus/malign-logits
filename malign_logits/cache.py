"""Unified cache manager for all stash types.

Each data type gets its own HashStash with dict keys:

    cache/
    ├── logits/            {'model', 'prompt'}
    ├── reasoning_logits/  {'model', 'prompt'} → {'thinking', 'post_logits', 'raw_logits'}
    ├── generations/       {'model', 'prompt', 'temp', 'idx'}
    ├── mega_generations/  {'model', 'prompt', 'temp', 'idx'} → [position dicts]
    ├── gen_logprobs/      {'model', 'prompt', 'temp', 'idx'}
    ├── gen_annotations/   {'tagger', 'model', 'prompt', 'temp', 'idx'}
    ├── sent_embeddings/   {'embedder', 'prompt', 'text'}
    ├── ref_surprisal/     {'ref', 'prompt', 'text'}
    ├── self_surprisal/    {'model', 'prompt', 'text'}
    ├── word_embeddings/   {'model', 'prompt', 'word', 'k'}
    ├── top_words_v2/      {'type', 'model', 'prompt', 'k'} — discover_top_words results
    ├── score_vocab_v2/    {'type', 'model', 'prompt', 'words'} — word-level probabilities
    ├── beams/             {'type', 'model', 'prompt', ...} — beam storylines + annotations
    ├── trees/             {'type', 'model', 'prompt', ...} — tree exploration results
    ├── logit_lens/        {'model', 'prompt', 'k'} — per-layer top-k projections
    ├── logit_lens_raw/    {'model', 'prompt'} — per-layer raw projection data
    └── perplexity/        {'model', 'prompt'}

    (psyche_derived, the legacy junk drawer, was retired 2026-07-05: every entry
    was shadowed by the typed stashes above.)

Text values in keys are hashed (SHA256[:16]) to avoid matching issues.
"""

import os

from . import PATH_DATA_RAW

CACHE_ROOT = os.path.join(PATH_DATA_RAW, "cache")

# hashstash encodes serializer/compress/b64 into the on-disk path
# (e.g. lmdb.hashstash.lz4+b64/data.db), so any open that relies on
# library defaults silently resolves to a different, empty store when
# those defaults change (b64 flips to False in hashstash 1.0). Every
# stash open in this project must pin the full format explicitly.
STASH_OPTIONS = dict(
    engine="lmdb",
    serializer="hashstash",
    compress="lz4",
    b64=True,
    map_size=200 * 1024**3,  # 200GB limit
)


def open_stash(root_dir, **overrides):
    """Open a HashStash with every format option pinned (see STASH_OPTIONS)."""
    from hashstash import HashStash
    return HashStash(root_dir=root_dir, **{**STASH_OPTIONS, **overrides})


def canonical_key(key):
    """Sort dict keys so the on-disk key never depends on insertion order.

    hashstash 0.4.0 serializes dicts in INSERTION ORDER, so
    {"model": m, "prompt": p} and {"prompt": p, "model": m} are different
    keys, and a reader using one order is silently blind to entries written
    with the other. hashstash 1.0.1 canonicalizes and the two agree.

    That difference bit us on 2026-07-26: the confirmation census was written
    under 1.0.1 (.venv) and a 0.4.0 reader (venv) got None for every entry.
    Sorting here makes the key identical under both versions, so the code
    depends on no version's canonicalization behaviour.

    MODE-KEY CONVENTION IS SPLIT ACROSS STASHES. READ THIS BEFORE WRITING A READER.

        true_word_probs   ALWAYS emits mode        {model, prompt, theta, mode}
        logits            conditional              mode present only if != "raw"
        word_probs        conditional
        beam_words        conditional
        (the other 23)    no mode field at all

    A reader that learns the convention from `true_word_probs` and applies it to
    `logits` builds a four-field key against three-field data and gets None --
    silently, because a miss is indistinguishable from an absent entry. That is
    the hazard this note exists to catch.

    Why they differ: true_word_probs was migrated (13,815 keys converted, one
    shape, verified at two seats); the other three were not. Migrating them is
    RH-GATED -- his word was "hold off on rekeying any other stash, it deserves
    special care" -- and governs until he says otherwise.

    A mode-less key IS raw, by declaration, for every stash that omits it.

    """
    return {k: key[k] for k in sorted(key)} if isinstance(key, dict) else key


class _CanonicalStash:
    """Proxy that canonicalizes dict keys on every access.

    Wrapping at the stash boundary fixes all ~45 key-construction sites in
    this module at once, and catches any added later.
    """

    __slots__ = ("_s",)

    def __init__(self, stash):
        object.__setattr__(self, "_s", stash)

    def __getitem__(self, key):
        return self._s[canonical_key(key)]

    def __setitem__(self, key, value):
        self._s[canonical_key(key)] = value

    def __contains__(self, key):
        return canonical_key(key) in self._s

    def __delitem__(self, key):
        del self._s[canonical_key(key)]

    def get(self, key, default=None):
        k = canonical_key(key)
        return self._s[k] if k in self._s else default

    def __len__(self):
        return len(self._s)

    def __iter__(self):
        return iter(self._s)

    def __getattr__(self, name):
        return getattr(self._s, name)


def normalize_text(text: str) -> str:
    """Canonical text normalization for cache keys.

    Always rstrip to avoid trailing whitespace mismatches.
    HashStash hashes the key internally, so storing full text
    doesn't affect path length — it just keeps keys readable.
    """
    return text.rstrip()


class CacheManager:
    def __init__(self, root=None):
        self.root = root or CACHE_ROOT
        self._stashes = {}

    def _stash(self, name):
        if name not in self._stashes:
            self._stashes[name] = _CanonicalStash(
                open_stash(os.path.join(self.root, name)))
        return self._stashes[name]

    # ── logits ──────────────────────────────────────────────────

    # THE ARCHIVED STASH'S TWO DEFECTS, AND HOW THESE AVOID THEM.
    #
    # The store retired on 2026-08-02 held 52,800 entries in TWO key shapes --
    # 31,402 {model, prompt} and 21,398 {mode, model, prompt} -- because the
    # three methods below each carried `if mode != "raw": key["mode"] = mode`.
    # Raw was therefore IMPLICIT and a pre-mode entry was indistinguishable
    # from a raw one. `mode` is now always present, via the declared schema.
    #
    # It also mixed float16 and float32 with NOTHING recording which, in key or
    # value. `dtype` is now KEYED: a dtype difference is a logit difference,
    # and a next-token probability is this campaign's quantity.
    #
    # `dtype` is REQUIRED and deliberately has no default. A default would
    # invent provenance for a caller who did not know it -- the same refusal
    # `set_true_word_probs` makes about `rule_version`.

    def _logits_resolve_dtype(self, model, prompt, mode, dtype):
        """RESOLVE, OR REFUSE -- the same third way as the twp rule dimension.

        A read that names no dtype is answerable exactly while this
        (model, prompt, mode) holds ONE, and ambiguous the instant it holds
        two. Requiring the caller to name it would break every `has_logits`
        used as "should I compute this?"; DEFAULTING it would invent
        provenance. So: one present -> fill it; two -> raise, naming both;
        none -> _NO_RULE, which makes has False and get None rather than an
        error. [2970].1's bootstrap lesson, applied before it could bite twice.
        """
        if dtype is not None:
            return dtype
        found = {d.get("dtype") for d in self.iter_keys(
            "logits", model=model, prompt=prompt, mode=mode)}
        if len(found) == 1:
            return next(iter(found))
        if not found:
            return CacheManager._NO_RULE
        raise KeyError(
            f"logits for {model} / {str(prompt)[:32]!r} exist at {len(found)} "
            f"dtypes {sorted(found)} -- a read that names none is AMBIGUOUS. "
            f"Pass dtype=; a dtype difference is a logit difference.")

    # ── THE LOGITS VALUE CONTRACT: AN INDEX, NEVER AN ARRAY ───────────
    #
    # The value is ALWAYS {"file": basename, "row": int, "dim": int} and the
    # vectors stay in their .f16 files, memmapped on read. Measured on the real
    # data 2026-08-02: lz4 compresses float16 logits to 100.0% of original and
    # hashstash b64 adds 33%, so an lmdb copy would cost 66.6 GB to hold 50 GB
    # -- and the .f16 files are not transient, they are the archive of a
    # 30-hour run, so ingesting them means holding the same bytes TWICE.
    #
    # ONE VALUE SHAPE, ENFORCED. The archived store held two KEY shapes and
    # that is what retired it; founding two VALUE shapes in the successor would
    # be the same defect one field over. `set_logits` refuses an array.
    #
    # `file` is a BASENAME resolved against LOGIT_ROOT at read time, so moving
    # the payloads is a config change and not a re-index.

    _LOGIT_MMAP = {}          #: basename -> np.memmap, one handle per file

    def _logit_root(self):
        """Where .f16 payloads live, DERIVED FROM THIS INSTANCE'S CACHE ROOT.

        THE FIRST VERSION READ A MODULE CONSTANT AND IGNORED THE INSTANCE, so a
        CacheManager pointed at a temp directory still wrote payloads to the
        production one: running the test suite deposited `model-a.float32.f16`
        and `model-a.float64.f16` into data/raw/cloud_run_20260801, beside 50 GB
        of real run output. An isolated cache that is not isolated is worse than
        no isolation, because the caller believes it.

        Same shape as [2970].3 -- a class attribute answering for every
        instance -- one layer down and in the direction that WRITES.
        """
        import os as _os
        env = _os.environ.get("MALIGN_LOGIT_ROOT")
        if env:
            return env
        #: self.root is this instance's cache dir; the payloads sit beside it
        #: under the run directory, so a temp-rooted manager gets a temp payload
        #: dir and cannot touch the real one.
        base = getattr(self, "root", None) or CACHE_ROOT
        if _os.path.abspath(base) == _os.path.abspath(CACHE_ROOT):
            from . import PATH_DATA_RAW
            return _os.path.join(PATH_DATA_RAW, "cloud_run_20260801")
        return _os.path.join(base, "logit_payloads")

    def _logit_array(self, entry, dtype):
        """The vector for one index entry, via a cached memmap.

        THE DTYPE COMES FROM THE KEY, NEVER A CONSTANT. This hardcoded
        float16 and was harmless only while every payload happened to be
        float16 -- a float32 file read at 2 bytes per value returns garbage
        that is finite, plausibly ranged, and wrong, which is the exact failure
        class lacan named for a wrong stride. I saw it, called it latent, and
        left it; `set_logits` accepting arrays made it live within the hour,
        and tests/test_cache.py::test_logits_roundtrip caught it immediately.
        A LATENT BUG IS ONE WHOSE PRECONDITION HAS NOT ARRIVED YET.
        """
        import os as _os
        import numpy as _np
        f, row, dim = entry["file"], entry["row"], entry["dim"]
        dt = _np.dtype(str(dtype))
        ck = (f, dt.str)
        mm = CacheManager._LOGIT_MMAP.get(ck)
        if mm is None:
            path = _os.path.join(self._logit_root(), f)
            mm = _np.memmap(path, dtype=dt, mode="r").reshape(-1, dim)
            CacheManager._LOGIT_MMAP[ck] = mm
        return _np.array(mm[row])

    def get_logits(self, model, prompt, mode="raw", dtype=None):
        dt = self._logits_resolve_dtype(model, prompt, mode, dtype)
        if dt is CacheManager._NO_RULE:
            return None
        entry = self.get("logits", model=model, prompt=prompt,
                         mode=mode, dtype=dt)
        if entry is None:
            return None
        if not isinstance(entry, dict) or "file" not in entry:
            raise TypeError(
                f"logits value for {model} / {str(prompt)[:32]!r} is a "
                f"{type(entry).__name__}, not an index entry. The store holds "
                f"{{file, row, dim}} and nothing else; a raw array here is a "
                f"second dialect and the reason the previous store was retired.")
        return self._logit_array(entry, dt)

    def get_logits_entry(self, model, prompt, mode="raw", dtype=None):
        """The INDEX entry itself -- {file, row, dim} -- without touching the
        payload. The cheap question stays cheap."""
        dt = self._logits_resolve_dtype(model, prompt, mode, dtype)
        if dt is CacheManager._NO_RULE:
            return None
        return self.get("logits", model=model, prompt=prompt, mode=mode, dtype=dt)

    #: where newly COMPUTED vectors are appended. The 2026-08-01 run's payloads
    #: are read-only history; anything computed since needs its own file.
    LOGIT_WRITE_DIR = "computed"

    def _append_logit_payload(self, model, vec):
        """Append one vector to this model's writable .f16 and return its row.

        THE STORE IS AN INDEX, SO A WRITER SUPPLYING AN ARRAY NEEDS SOMEWHERE
        TO PUT IT. Twenty-one call sites across psyche, circuit, step_analysis
        and the f36/f11/r1 scripts pass `logits.cpu().numpy()`; refusing them
        outright would have made the contract correct and the codebase broken.
        So an array is accepted, WRITTEN TO THE PAYLOAD, and indexed -- one
        value dialect, and every caller unchanged.
        """
        import os as _os
        import numpy as _np
        vec = _np.ascontiguousarray(vec)
        d = _os.path.join(self._logit_root(), CacheManager.LOGIT_WRITE_DIR)
        _os.makedirs(d, exist_ok=True)
        rel = _os.path.join(CacheManager.LOGIT_WRITE_DIR,
                            model.replace("/", "__") + f".{vec.dtype}.f16")
        path = _os.path.join(self._logit_root(), rel)
        isz = vec.dtype.itemsize
        dim = int(vec.shape[-1])
        n = _os.path.getsize(path) if _os.path.exists(path) else 0
        #: DIM MUST BE CONSTANT WITHIN A FILE -- lacan [3012]. A file whose
        #: stride changes mid-way returns real floats at wrong offsets forever
        #: after, and no value check can see it.
        if n % (dim * isz):
            raise ValueError(
                f"{rel}: size {n} is not a multiple of dim {dim} x {isz}. "
                f"A vector of a different width was appended to this file; "
                f"every row after it reads at the wrong offset.")
        with open(path, "ab") as fh:
            fh.write(vec.tobytes())
        for k in [k for k in CacheManager._LOGIT_MMAP if k[0] == rel]:
            CacheManager._LOGIT_MMAP.pop(k, None)        # stale handle
        return rel, n // (dim * isz), dim

    def set_logits(self, model, prompt, value, mode="raw", dtype=None):
        """Write logits. Accepts an index entry OR a vector.

        An index entry `{file, row, dim}` is stored as-is (the indexer's path).
        A VECTOR is appended to this model's writable payload and the resulting
        entry stored -- so the store keeps ONE value dialect while callers that
        naturally hold an array keep working.
        """
        if isinstance(value, dict) and {"file", "row", "dim"} <= set(value):
            dt = dtype or value.get("dtype")
            if dt is None:
                raise KeyError(
                    f"refusing to write logits for {model} / "
                    f"{str(prompt)[:40]!r}: no dtype. A dtype difference IS a "
                    f"logit difference; it is keyed and never defaulted.")
            self.set("logits", {k: value[k] for k in ("file", "row", "dim")},
                     model=model, prompt=prompt, mode=mode, dtype=str(dt))
            return
        dt = dtype or getattr(value, "dtype", None)
        if dt is None:
            raise KeyError(
                f"refusing to write logits for {model} / {str(prompt)[:40]!r}: "
                f"value is neither an index entry nor an array with a dtype. "
                f"A dtype difference IS a logit difference; never defaulted.")
        rel, row, dim = self._append_logit_payload(model, value)
        self.set("logits", {"file": rel, "row": row, "dim": dim},
                 model=model, prompt=prompt, mode=mode, dtype=str(dt))

    def has_logits(self, model, prompt, mode="raw", dtype=None):
        dt = self._logits_resolve_dtype(model, prompt, mode, dtype)
        if dt is CacheManager._NO_RULE:
            return False
        return self.has("logits", model=model, prompt=prompt,
                        mode=mode, dtype=dt)

    # ── generations ─────────────────────────────────────────────

    def get_generation(self, model, prompt, temp=1.0, idx=0):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        s = self._stash("generations")
        return s[key] if key in s else None

    def set_generation(self, model, prompt, text, temp=1.0, idx=0):
        self._stash("generations")[{
            "model": model, "prompt": prompt, "temp": temp, "idx": idx
        }] = text

    def count_generations(self, model, prompt, temp=1.0):
        """Count how many generations exist for this (model, prompt, temp).

        Uses binary search for O(log n) instead of O(n).
        """
        s = self._stash("generations")
        if {"model": model, "prompt": prompt, "temp": temp, "idx": 0} not in s:
            return 0
        # Binary search for upper bound
        lo, hi = 0, 1
        while {"model": model, "prompt": prompt, "temp": temp, "idx": hi} in s:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if {"model": model, "prompt": prompt, "temp": temp, "idx": mid} in s:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def iter_generations(self, model, prompt, temp=1.0):
        """Yield (idx, text) for all generations matching (model, prompt, temp)."""
        n = self.count_generations(model, prompt, temp)
        for idx in range(n):
            text = self.get_generation(model, prompt, temp=temp, idx=idx)
            if text is not None:
                yield idx, text

    # ── generation logprobs (API models) ─────────────────────────

    def get_gen_logprobs(self, model, prompt, temp=1.0, idx=0):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        s = self._stash("gen_logprobs")
        return s[key] if key in s else None

    def set_gen_logprobs(self, model, prompt, logprobs, temp=1.0, idx=0):
        self._stash("gen_logprobs")[{
            "model": model, "prompt": prompt, "temp": temp, "idx": idx
        }] = logprobs

    def has_gen_logprobs(self, model, prompt, temp=1.0, idx=0):
        return {"model": model, "prompt": prompt, "temp": temp,
                "idx": idx} in self._stash("gen_logprobs")

    # ── generation annotations (LLM tagger scores) ──────────────

    def get_gen_annotation(self, tagger, model, prompt, temp=1.0, idx=0):
        key = {"tagger": tagger, "model": model, "prompt": prompt,
               "temp": temp, "idx": idx}
        s = self._stash("gen_annotations")
        return s[key] if key in s else None

    def set_gen_annotation(self, tagger, model, prompt, annotation,
                           temp=1.0, idx=0):
        self._stash("gen_annotations")[{
            "tagger": tagger, "model": model, "prompt": prompt,
            "temp": temp, "idx": idx,
        }] = annotation

    def has_gen_annotation(self, tagger, model, prompt, temp=1.0, idx=0):
        return {"tagger": tagger, "model": model, "prompt": prompt,
                "temp": temp, "idx": idx} in self._stash("gen_annotations")

    # ── sentence embeddings ─────────────────────────────────────

    def get_sent_embeddings(self, embedder, prompt, text):
        key = {"embedder": embedder, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("sent_embeddings")
        return s[key] if key in s else None

    def set_sent_embeddings(self, embedder, prompt, text, vectors):
        self._stash("sent_embeddings")[{
            "embedder": embedder, "prompt": prompt, "text": normalize_text(text)
        }] = vectors

    def has_sent_embeddings(self, embedder, prompt, text):
        return {"embedder": embedder, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("sent_embeddings")

    # ── reference surprisal ─────────────────────────────────────

    def get_ref_surprisal(self, ref_model, prompt, text):
        key = {"ref": ref_model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("ref_surprisal")
        return s[key] if key in s else None

    def set_ref_surprisal(self, ref_model, prompt, text, tok_surps):
        self._stash("ref_surprisal")[{
            "ref": ref_model, "prompt": prompt, "text": normalize_text(text)
        }] = tok_surps

    def has_ref_surprisal(self, ref_model, prompt, text):
        return {"ref": ref_model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("ref_surprisal")

    # ── token metrics (drift from hidden states) ─────────────────

    def get_token_metrics(self, ref_model, prompt, text):
        key = {"ref": ref_model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("token_metrics")
        return s[key] if key in s else None

    def set_token_metrics(self, ref_model, prompt, text, metrics):
        self._stash("token_metrics")[{
            "ref": ref_model, "prompt": prompt, "text": normalize_text(text)
        }] = metrics

    def has_token_metrics(self, ref_model, prompt, text):
        return {"ref": ref_model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("token_metrics")

    # ── self-surprisal ──────────────────────────────────────────

    def get_self_surprisal(self, model, prompt, text):
        key = {"model": model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("self_surprisal")
        return s[key] if key in s else None

    def set_self_surprisal(self, model, prompt, text, tok_surps):
        self._stash("self_surprisal")[{
            "model": model, "prompt": prompt, "text": normalize_text(text)
        }] = tok_surps

    def has_self_surprisal(self, model, prompt, text):
        return {"model": model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("self_surprisal")

    # ── word embeddings ─────────────────────────────────────────

    def get_word_embedding(self, model, prompt, word, k):
        key = {"model": model, "prompt": prompt, "word": word, "k": k}
        s = self._stash("word_embeddings")
        return s[key] if key in s else None

    def set_word_embedding(self, model, prompt, word, k, embedding):
        self._stash("word_embeddings")[{
            "model": model, "prompt": prompt, "word": word, "k": k
        }] = embedding

    def has_word_embedding(self, model, prompt, word, k):
        return {"model": model, "prompt": prompt, "word": word,
                "k": k} in self._stash("word_embeddings")

    # ── reasoning logits (post-thinking distributions) ──────────

    def get_reasoning(self, model, prompt):
        """Get cached reasoning result: thinking text + post-thinking logits.

        Returns dict with keys: 'thinking', 'post_logits', 'raw_logits'
        or None if not cached.
        """
        key = {"model": model, "prompt": prompt}
        s = self._stash("reasoning_logits")
        return s[key] if key in s else None

    def set_reasoning(self, model, prompt, thinking, post_logits, raw_logits):
        """Cache reasoning result: thinking text + post-thinking logits."""
        self._stash("reasoning_logits")[{"model": model, "prompt": prompt}] = {
            "thinking": thinking,
            "post_logits": post_logits,
            "raw_logits": raw_logits,
        }

    def has_reasoning(self, model, prompt):
        return {"model": model, "prompt": prompt} in self._stash("reasoning_logits")

    # ── mega-generations (F25 position-level trajectories) ──────

    def _mega_key(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        if mode != "raw":
            key["mode"] = mode
        return key

    def get_mega_generation(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        """Get cached position-level trajectory for a single generation."""
        s = self._stash("mega_generations")
        key = self._mega_key(model, prompt, temp, idx, mode)
        return s[key] if key in s else None

    def set_mega_generation(self, model, prompt, positions, temp=1.0, idx=0, mode="raw"):
        """Cache position-level trajectory (list of dicts with step/entropy/top5)."""
        self._stash("mega_generations")[
            self._mega_key(model, prompt, temp, idx, mode)
        ] = positions

    def has_mega_generation(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        return self._mega_key(model, prompt, temp, idx, mode) in self._stash("mega_generations")

    def count_mega_generations(self, model, prompt, temp=1.0, mode="raw"):
        """Count cached mega-generations (binary search on idx)."""
        s = self._stash("mega_generations")
        if self._mega_key(model, prompt, temp, 0, mode) not in s:
            return 0
        lo, hi = 0, 1
        while self._mega_key(model, prompt, temp, hi, mode) in s:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self._mega_key(model, prompt, temp, mid, mode) in s:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ── probe: per-position logits/hidden, per-gen meta, per-model embeddings ──

    def _probe_pos_key(self, model, prompt, gen, pos, max_tokens):
        return {"model": model, "prompt": prompt, "gen": gen,
                "pos": pos, "T": max_tokens}

    def get_probe_logits(self, model, prompt, gen=0, pos=0, max_tokens=20):
        s = self._stash("probe_logits")
        key = self._probe_pos_key(model, prompt, gen, pos, max_tokens)
        return s[key] if key in s else None

    def set_probe_logits(self, model, prompt, logits, gen=0, pos=0, max_tokens=20):
        self._stash("probe_logits")[
            self._probe_pos_key(model, prompt, gen, pos, max_tokens)] = logits

    def get_probe_hidden(self, model, prompt, gen=0, pos=0, max_tokens=20):
        s = self._stash("probe_hidden")
        key = self._probe_pos_key(model, prompt, gen, pos, max_tokens)
        return s[key] if key in s else None

    def set_probe_hidden(self, model, prompt, hidden, gen=0, pos=0, max_tokens=20):
        self._stash("probe_hidden")[
            self._probe_pos_key(model, prompt, gen, pos, max_tokens)] = hidden

    def get_probe_meta(self, model, prompt, gen=0, max_tokens=20):
        s = self._stash("probe_meta")
        key = {"model": model, "prompt": prompt, "gen": gen, "T": max_tokens}
        return s[key] if key in s else None

    def set_probe_meta(self, model, prompt, meta, gen=0, max_tokens=20):
        self._stash("probe_meta")[
            {"model": model, "prompt": prompt, "gen": gen, "T": max_tokens}] = meta

    def has_probe(self, model, prompt, gen=0, pos=0, max_tokens=20):
        return self._probe_pos_key(model, prompt, gen, pos, max_tokens) in self._stash("probe_logits")

    def count_probe_gens(self, model, prompt, max_tokens=20):
        if not self.has_probe(model, prompt, gen=0, pos=0, max_tokens=max_tokens):
            return 0
        lo, hi = 0, 1
        while self.has_probe(model, prompt, gen=hi, pos=0, max_tokens=max_tokens):
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self.has_probe(model, prompt, gen=mid, pos=0, max_tokens=max_tokens):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get_probe_embeddings(self, model):
        """Load embedding matrix from numpy file (too large for lmdb)."""
        import numpy as np
        path = os.path.join(self.root, "probe_embeddings",
                            model.replace("/", "--") + ".npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def set_probe_embeddings(self, model, embeddings):
        """Save embedding matrix as numpy file."""
        import numpy as np
        d = os.path.join(self.root, "probe_embeddings")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, model.replace("/", "--") + ".npy")
        np.save(path, embeddings)

    # ── top words (discover_top_words results) ──────────────────

    def get_top_words(self, model, prompt, k=200):
        key = {"type": "top_words", "model": model, "prompt": prompt, "k": k}
        s = self._stash("top_words_v2")
        return s[key] if key in s else None

    def set_top_words(self, model, prompt, words, k=200):
        self._stash("top_words_v2")[{
            "type": "top_words", "model": model, "prompt": prompt, "k": k
        }] = words

    def has_top_words(self, model, prompt, k=200):
        return {"type": "top_words", "model": model, "prompt": prompt, "k": k} in self._stash("top_words_v2")

    # ── score vocab (word-level probabilities) ─────────────────

    @staticmethod
    def _vocab_hash(words):
        """Stable id for a vocabulary (order-insensitive)."""
        import hashlib
        return hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest()[:16]

    def get_score_vocab(self, model, prompt, words=None):
        """Word-probability scores for one (model, prompt, vocabulary).

        Keys include a hash of the vocabulary: two families sharing a base
        model (llama/tulu) have different focused vocabularies, and the old
        {model, prompt} key let the second family silently read the first's
        scores (missing words read as probability 0 in formation_df).
        Ambiguous legacy {model, prompt} entries are deliberately NOT read.
        words=None scans for any vocabulary's entry (display use only).
        """
        s = self._stash("score_vocab_v2")
        if words is not None:
            key = {"model": model, "prompt": prompt,
                   "vocab": self._vocab_hash(words)}
            if key in s:
                return s[key]
            # Oldest format was already vocabulary-exact — safe to read
            old_key = {"type": "score_vocab", "model": model, "prompt": prompt,
                       "words": tuple(words)}
            if old_key in s:
                return s[old_key]
            return None
        for k in s.keys():
            if isinstance(k, dict) and k.get("model") == model \
                    and k.get("prompt") == prompt and "vocab" in k:
                return s[k]
        return None

    def set_score_vocab(self, model, prompt, scores, words=None):
        if words is None:
            raise ValueError("set_score_vocab requires the vocabulary (words=) — "
                             "unkeyed score_vocab entries collide across vocabularies")
        key = {"model": model, "prompt": prompt, "vocab": self._vocab_hash(words)}
        self._stash("score_vocab_v2")[key] = scores

    def has_score_vocab(self, model, prompt, words=None):
        return self.get_score_vocab(model, prompt, words) is not None

    # ── word probs (hybrid: exact logit + beam for multi-token) ──

    def get_word_probs(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        s = self._stash("word_probs")
        return s[key] if key in s else None

    def set_word_probs(self, model, prompt, probs, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        self._stash("word_probs")[key] = probs

    def has_word_probs(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        return key in self._stash("word_probs")

    # ── beam word probs (word-level via beam search) ─────────────

    # ---- true_word_probs: exact P(next WORD) by threshold-bounded expansion ----
    # THETA IS IN THE KEY. beam_words put `n` in its key and two beam widths
    # (200 and 1000) then coexisted across 70+ models on different unrecoverable
    # scales, silently mixed by any reader that did not filter. Theta plays the
    # same structural role here, so it is keyed for the same reason -- but unlike
    # a beam width it is a PRINCIPLED floor: expanding every token above theta is
    # complete for every word above theta, and the unexpanded mass is reported as
    # residual rather than divided away.
    # KEY SHAPE CHANGED 2026-07-30, on RH's instruction, while the grid was being
    # rebuilt anyway. Two changes, both of which alter the key and therefore need the
    # migration in scripts/migrate_twp_keys.py run before any read:
    #
    #   `type` REMOVED. It was 'true_word_probs' on all 13,815 entries -- a
    #   discriminator inside a stash of that name, carrying no information. (It is
    #   still present in other stashes' keys and is NOT being removed there; see the
    #   scope note below.)
    #
    #   `mode` ALWAYS PRESENT, no longer omitted when raw. The conditional form
    #   `if mode != "raw": key["mode"] = mode` gave a raw key and a mode key different
    #   SHAPES, which prevents collision but makes raw IMPLICIT -- a four-field entry
    #   was indistinguishable from one written before the mode parameter existed.
    #   Explicit beats inferable.
    #
    # SCOPE, and it is deliberate: ONLY true_word_probs changes. `mode` is keyed in
    # four stashes of twenty-seven and the four acquired it ad hoc, so a general
    # migration is a separate and larger decision. RH: "HOLD OFF ON REKEYING ANY OTHER
    # STASH, it deserves special care."
    #
    # AND MODE IS NOT ONE DIMENSION, which a flat field here will hide: RAW and CHAT
    # are two framings of ONE stimulus, while CONTINUE and THINK prepend
    # "Continue this text:" and therefore measure a DIFFERENT stimulus. See
    # _schema._mode_is_not_one_dimension in data/prompt_categorisation.json. Anything
    # that groups on this field must not pool across that boundary.
    # ── THE GENERIC ENGINE ────────────────────────────────────────────
    #
    # ONE implementation of every operation, reading a DECLARED schema
    # (malign_logits/cache_schema.py). Written for `true_word_probs`; the other
    # twenty-six stashes are undeclared and fall through to the untyped path
    # unchanged, so nothing else moves.
    #
    # The point is not fewer methods. It is that a key SHAPE becomes data that
    # can be validated, filtered on, and migrated -- instead of a dict literal
    # re-expressed in twenty-six method bodies and fourteen call sites.

    def _schema(self, stash):
        from .cache_schema import schema_for
        return schema_for(stash)

    def key_for(self, stash, **kw):
        """Build a key from the declared schema. Refuses undeclared fields."""
        sch = self._schema(stash)
        if sch is None:
            raise KeyError(f"{stash} has no declared schema; use _stash() "
                           f"until it is migrated")
        return sch.build(**kw)

    def get(self, stash, **kw):
        key = self.key_for(stash, **kw)
        s = self._stash(stash)
        return s[key] if key in s else None

    def set(self, stash, payload, **kw):
        self._stash(stash)[self.key_for(stash, **kw)] = payload

    def has(self, stash, **kw):
        return self.key_for(stash, **kw) in self._stash(stash)

    def iter_keys(self, stash, **filters):
        """Every key in `stash`, narrowed by declared filters. Yields dicts.

        `None` means DO NOT FILTER -- not "match the default". A census wants
        `mode=None`; an analysis must say `mode="raw"` and mean it.
        """
        sch = self._schema(stash)
        for k in self._stash(stash).keys():
            d = dict(k) if not isinstance(k, dict) else k
            if sch is None:
                if all(str(d.get(f)) == str(v)
                       for f, v in filters.items() if v is not None):
                    yield d
            elif sch.matches(d, **filters):
                yield d

    def count(self, stash, **filters):
        """CELL count under the filter.

        A count is the reading most exposed to a key change: adding a keyed
        dimension multiplies it while `distinct(stash, "prompt")` stays
        correct, so a script that counts keys and one that collects prompts
        disagree without either looking wrong. Callers who mean "how many
        distinct prompts" should say so with `distinct`.
        """
        return sum(1 for _ in self.iter_keys(stash, **filters))

    def iter_items(self, stash, **filters):
        """(key, value) pairs under the filter.

        Separate from `iter_keys` because reading values is the expensive half
        and most population questions do not need it -- a consumer that only
        wants models or prompts should never pay for a payload fetch. A single
        combined iterator would have made the cheap case cost the dear one.

        WHY KEY-FILTER-FIRST RATHER THAN THE UNDERLYING `.items()`, MEASURED
        on 93,216 lmdb entries 2026-08-02 (a HashStash does support items(),
        so this is a choice and not an oversight):

            unfiltered, 3,000    per-key 0.37s   items() 0.39s    1.0x
            filtered, 979/93,216 per-key 0.78s   items() 12.16s  15.5x

        Equal when you want everything; 15.5x better when you want a slice,
        because filtering keys first fetches 979 payloads instead of
        deserializing all 93,216. Never worse -- so do not "optimise" this into
        a full scan.
        """
        st = self._stash(stash)
        for d in self.iter_keys(stash, **filters):
            try:
                yield d, st[d]
            except Exception:
                continue

    def value_count_by(self, stash, field, default=None, **filters):
        """{value-field -> cell count}, reading the PAYLOAD, not the key.

        For fields that live in the value rather than the key -- `rule_version`
        being the one that matters: it is value-only today, which is exactly
        why two boundary rules can collide in this stash.
        """
        out = {}
        for _, v in self.iter_items(stash, **filters):
            k = (v or {}).get(field, default)
            out[k] = out.get(k, 0) + 1
        return out

    def distinct(self, stash, field, **filters):
        """The set of values `field` takes under the filter."""
        return {d.get(field) for d in self.iter_keys(stash, **filters)}

    def count_by(self, stash, field, **filters):
        """{value: cell count} in ONE pass. The registry's `cells_in_store`."""
        out = {}
        for d in self.iter_keys(stash, **filters):
            v = d.get(field)
            out[v] = out.get(v, 0) + 1
        return out

    # ── true_word_probs: named accessors, three lines each ─────────────
    # Kept because a named method at a call site is worth something, and
    # deliberately NOT reimplementing anything: the schema and the engine are
    # the single source of both shape and behaviour.

    # THE RULE DIMENSION, and why the reads do not take a required parameter.
    #
    # [2959] proved the naive flip dead: every named accessor passes exactly
    # (model, prompt, theta, mode) and CANNOT supply two new required fields,
    # so making them required breaks twelve read sites the moment the schema
    # moves. Making them DEFAULTED would be worse -- a silent default is how
    # `rule_version` ends up on a cell no rule of that version produced.
    #
    # So: RESOLVE, OR REFUSE. A read that names no rule is answerable exactly
    # while the store holds ONE rule, and is ambiguous the instant it holds
    # two. The accessor resolves in the first case and RAISES in the second,
    # naming the rules it found. No call site changes today; no call site can
    # silently pool tomorrow. Enforce by refusal, not by convention.
    #
    # The resolution is cached because it costs a key scan, and invalidated on
    # every write -- a cache that survives the write that invalidates it is the
    # defect this class exists to avoid.

    #: sentinel: the store holds NO rules, so an unnamed read cannot match.
    #: Not an error and not a default -- the exact answer is "nothing is there".
    _NO_RULE = object()

    def _twp_rules(self):
        """{(rule_version, dict_sha)} present in the store. Cached PER INSTANCE.

        [2970].3: this was a CLASS attribute, so two CacheManagers on different
        roots shared one cache and whichever populated first answered for both
        -- an EMPTY store reported the real store's rule. `self._stashes` is
        already per-instance one line above; this follows it.

        [2970].2: populated from the KEYS, not the payloads. After the flip
        `rule_version` and `dict_sha` ARE key fields, so fetching 93,216
        payloads to recover two fields already in the key costs 12.83s against
        0.65s -- the same 15.5x argument this class already books for
        `iter_items`, applied to itself.
        """
        if getattr(self, "_rule_cache", None) is None:
            sch = self._schema("true_word_probs")
            if "rule_version" in sch.fields:
                self._rule_cache = {(d.get("rule_version"), d.get("dict_sha"))
                                    for d in self.iter_keys("true_word_probs")}
            else:
                self._rule_cache = {
                    ((v or {}).get("rule_version"), (v or {}).get("dict_sha"))
                    for _k, v in self.iter_items("true_word_probs")}
        return self._rule_cache

    def _twp_note_rule(self, rv, ds):
        """A write can only ADD a rule, and the writer knows which one.

        [2970].2: the previous version INVALIDATED on every write and the next
        read repopulated by a full scan. The ingest loop is has-then-set per
        record, so that is N^2/2 payload reads -- 3.26e10 at N=255,506, about
        51 DAYS. Extending is O(1) and exactly correct.

        A deletion cannot be noticed this way, so the set can only ever be too
        LARGE -- which yields a spurious "ambiguous, name your rule" refusal.
        THAT FAILS SAFE, which is the right direction for a safety property,
        and it is said here rather than left for the next reader to derive.
        """
        if getattr(self, "_rule_cache", None) is not None:
            self._rule_cache.add((rv, ds))

    def _twp_resolve_rule(self, rule_version, dict_sha):
        """Fill an unnamed rule from the store, refuse if ambiguous, or report
        _NO_RULE when the store is empty."""
        if rule_version is not None and dict_sha is not None:
            return rule_version, dict_sha
        rules = self._twp_rules()
        if len(rules) == 1:
            only = next(iter(rules))
            return (rule_version if rule_version is not None else only[0],
                    dict_sha if dict_sha is not None else only[1])
        if not rules:
            #: [2970].1 -- BOOTSTRAP. Returning (None, None) here made build()
            #: raise on the required field, so `has` threw on an EMPTY store and
            #: the ingest died at record 1: it calls has BEFORE set on every
            #: record, and nothing could write the first cell to bootstrap from.
            #: On a store holding zero rules an unnamed read cannot match
            #: anything -- has is FALSE, get is None. That is the exact answer,
            #: not a default, and conflating it with "you failed to specify" is
            #: the one confusion this design exists to prevent.
            return CacheManager._NO_RULE, CacheManager._NO_RULE
        raise KeyError(
            "true_word_probs holds %d rules %s -- a read that names none is "
            "AMBIGUOUS. Pass rule_version= and dict_sha= explicitly; do not "
            "pool across boundary rules." % (len(rules), sorted(rules)))

    def _twp_key(self, model, prompt, theta, mode,
                 rule_version=None, dict_sha=None):
        """The key, or None when the store holds no rule to resolve against."""
        kw = dict(model=model, prompt=prompt, theta=theta, mode=mode)
        if "rule_version" in self._schema("true_word_probs").fields:
            rv, ds = self._twp_resolve_rule(rule_version, dict_sha)
            if rv is CacheManager._NO_RULE:
                return None                    # empty store: nothing can match
            kw.update(rule_version=rv, dict_sha=ds)
        return self.key_for("true_word_probs", **kw)

    def get_true_word_probs(self, model, prompt, theta=0.001, mode="raw",
                            rule_version=None, dict_sha=None):
        key = self._twp_key(model, prompt, theta, mode, rule_version, dict_sha)
        if key is None:
            return None
        s = self._stash("true_word_probs")
        return s[key] if key in s else None

    def set_true_word_probs(self, model, prompt, payload, theta=0.001, mode="raw"):
        """payload = {"rows": [{word, t1, p}, ...], "residual": {tail, drop, open,
        total}, "batches": int}. One row per (word, FIRST TOKEN): a surface can be
        reached by more than one token path, and t1 is the join key to the
        token-level table and the grouping the masking test needs.

        THE RULE IS READ OFF THE PAYLOAD, NEVER DEFAULTED. A writer that cannot
        say which rule produced a cell must not write it: [2963].2 quarantined
        13,940 rows for exactly this, and defaulting them would have keyed them
        to a rule version that never produced them.
        """
        keyed = "rule_version" in self._schema("true_word_probs").fields
        rv = ds = None
        if keyed:
            rv, ds = (payload or {}).get("rule_version"), (payload or {}).get("dict_sha")
            if rv is None or ds is None:
                raise KeyError(
                    "refusing to write %s / %.40r: payload carries "
                    "rule_version=%r dict_sha=%r and the key requires both. "
                    "A cell whose rule is unknown is quarantined, not guessed."
                    % (model, prompt, rv, ds))
        self._stash("true_word_probs")[
            self._twp_key(model, prompt, theta, mode, rv, ds)] = payload
        if keyed:
            self._twp_note_rule(rv, ds)        # O(1); see _twp_note_rule

    def has_true_word_probs(self, model, prompt, theta=0.001, mode="raw",
                            rule_version=None, dict_sha=None):
        key = self._twp_key(model, prompt, theta, mode, rule_version, dict_sha)
        return False if key is None else key in self._stash("true_word_probs")

    def get_beam_words(self, model, prompt, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        s = self._stash("beam_words")
        return s[key] if key in s else None

    def set_beam_words(self, model, prompt, words, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        self._stash("beam_words")[key] = words

    def has_beam_words(self, model, prompt, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        return key in self._stash("beam_words")

    # ── beams (beam search storylines + cross-model annotations) ──

    def get_beams(self, key):
        s = self._stash("beams")
        return s[key] if key in s else None

    def set_beams(self, key, value):
        self._stash("beams")[key] = value

    def has_beams(self, key):
        return key in self._stash("beams")

    def iter_beam_keys(self):
        for k in self._stash("beams").keys():
            if isinstance(k, dict):
                yield k

    # ── trees (explore_tree results) ───────────────────────────

    def get_tree(self, key):
        s = self._stash("trees")
        return s[key] if key in s else None

    def set_tree(self, key, value):
        self._stash("trees")[key] = value

    def has_tree(self, key):
        return key in self._stash("trees")

    # ── logit lens ──────────────────────────────────────────────

    def get_logit_lens(self, model, prompt, k):
        s = self._stash("logit_lens")
        key = {"model": model, "prompt": prompt, "k": k}
        return s[key] if key in s else None

    def set_logit_lens(self, model, prompt, k, value):
        self._stash("logit_lens")[{"model": model, "prompt": prompt, "k": k}] = value

    def get_logit_lens_raw(self, model, prompt):
        s = self._stash("logit_lens_raw")
        key = {"model": model, "prompt": prompt}
        return s[key] if key in s else None

    def set_logit_lens_raw(self, model, prompt, value):
        self._stash("logit_lens_raw")[{"model": model, "prompt": prompt}] = value

    # ── perplexity ──────────────────────────────────────────────

    def get_perplexity(self, model, prompt):
        s = self._stash("perplexity")
        key = {"model": model, "prompt": prompt}
        return s[key] if key in s else None

    def set_perplexity(self, model, prompt, value):
        self._stash("perplexity")[{"model": model, "prompt": prompt}] = value

    # ── derived (typed routing for psyche.py cache keys) ────────

    def _derived_route(self, key):
        """Map a legacy-style derived key to (getter, setter, checker) thunks."""
        t = key.get("type", "") if isinstance(key, dict) else ""
        m, p = key.get("model"), key.get("prompt")
        if t == "top_words":
            k = key.get("k", 200)
            return (lambda: self.get_top_words(m, p, k),
                    lambda v: self.set_top_words(m, p, v, k),
                    lambda: self.has_top_words(m, p, k))
        if t == "beam_words":
            n, d = key.get("n", 1000), key.get("depth", 3)
            return (lambda: self.get_beam_words(m, p, n, d),
                    lambda v: self.set_beam_words(m, p, v, n, d),
                    lambda: self.has_beam_words(m, p, n, d))
        if t == "score_vocab":
            w = key.get("words")
            return (lambda: self.get_score_vocab(m, p, w),
                    lambda v: self.set_score_vocab(m, p, v, w),
                    lambda: self.has_score_vocab(m, p, w))
        if t in ("beam_annotated_v1", "beam_cross_v1"):
            return (lambda: self.get_beams(key),
                    lambda v: self.set_beams(key, v),
                    lambda: self.has_beams(key))
        if t == "explore_tree_v3":
            return (lambda: self.get_tree(key),
                    lambda v: self.set_tree(key, v),
                    lambda: self.has_tree(key))
        if t == "logit_lens":
            k = key.get("k")
            return (lambda: self.get_logit_lens(m, p, k),
                    lambda v: self.set_logit_lens(m, p, k, v),
                    lambda: self.get_logit_lens(m, p, k) is not None)
        if t == "logit_lens_raw":
            return (lambda: self.get_logit_lens_raw(m, p),
                    lambda v: self.set_logit_lens_raw(m, p, v),
                    lambda: self.get_logit_lens_raw(m, p) is not None)
        if t == "perplexity":
            return (lambda: self.get_perplexity(m, p),
                    lambda v: self.set_perplexity(m, p, v),
                    lambda: self.get_perplexity(m, p) is not None)
        raise ValueError(f"Unknown derived cache key type {t!r}: {key!r} — "
                         f"add a typed stash in CacheManager._derived_route")

    def get_derived(self, key):
        getter, _, _ = self._derived_route(key)
        return getter()

    def set_derived(self, key, value):
        _, setter, _ = self._derived_route(key)
        setter(value)

    def has_derived(self, key):
        _, _, checker = self._derived_route(key)
        return checker()


# Module-level singleton
_cache = None

def get_cache(root=None) -> CacheManager:
    global _cache
    if _cache is None or (root and _cache.root != root):
        _cache = CacheManager(root=root)
    return _cache
