#!/usr/bin/env python
"""Semantic-field counts for any string, from the lexicons already in the repo.

    from malign_logits import fields
    fields.count_all("She felt ashamed and guilty about it.")  # ALL lexicons,
                                                               # one call
    fields.count("She felt ashamed and guilty about it.")   # usas_fine, all
                                                            # tags, content only
    fields.count(text, source="meta", all_tags=False)       # strict 1-token-1-count
    fields.norms("She felt ashamed and guilty about it.")
    python -m malign_logits.fields --selftest

    python -m malign_logits.fields "some text"     # score one string

WHY A MODULE AND NOT A SCRIPT. Six lexicons and two norm sets are on disk, each
with its own lookup policy, and `lexicons/README.md` records that getting those
policies wrong silently returns the WRONG SENSE rather than nothing: surface-form
lookup sends `found` to *establish*, `felt` to the fabric and `saw` to the
cutting tool, and `found` is the corpus's single most frequent riser. A policy
that lives in one importable place is a policy; the same policy retyped in each
analysis script is six policies.

## The field vocabularies

Two granularities, both already built in `lexicons/metafields/`:

    source="meta"       13 fields SHARED BY EVERY LEXICON, so usas, gi and
                        wordnet counts are directly comparable and can be summed
    source="usas_fine"  ~30 finer USAS groups (emotion_and_arousal,
                        danger_caution_and_violence, ...) -- more resolution,
                        one lexicon only
    source="usas" | "gi" | "wordnet" | "rid"    one lexicon, its own namespace

## Coverage is returned with every count, and that is not decoration

A field count without the number of tokens that matched anything is a rate with
no denominator. USAS covers most running text; the General Inquirer is a 1960s
resource whose coverage of explicit violence is thin -- `raped`, `desecrated`
and `stomped` are all absent -- so on this corpus GI silently drops the
transgressive end of the vocabulary. A caller comparing two texts on GI counts
without looking at `coverage` is comparing how much of each text GI happens to
know.

## Norms are trichotomised, and the cuts come from the lexicon

Warriner (valence, arousal, dominance) and Brysbaert (concreteness) are
continuous. They are cut at the TERTILES OF THEIR OWN DISTRIBUTION rather than
at a round number, so "high" means high relative to English and the three bins
are a priori equal. The cuts are computed once from the loaded lexicon and
reported by `norm_cuts()` -- a threshold nobody can see is a free parameter.
"""
import collections
import csv
import functools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEX = os.path.join(ROOT, "meta", "M01_displacement", "lexicons")
MF = os.path.join(LEX, "metafields")
NORMS_DIR = os.path.expanduser(
    "~/Dropbox/Prof/Articles/TheoryMachines/norms_sources")
#: BYU/COCA word database: surface -> (lemma, CLAWS pos) over 86,403 forms.
#: Already the repo's POS authority -- `scripts/build_beam_sample.py` uses its
#: `vv*` test to define R's eligible population, so using it here means
#: "content word" means the same thing in both places.
BYU = os.path.expanduser("~/Dropbox/Prof/Code/osp/worddb.byu.txt")

TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")

#: The 13 shared meta-fields, fixed here so a caller can enumerate them without
#: reading a CSV, and so a source that maps to something outside this set is a
#: loud KeyError rather than a quiet new column.
META_FIELDS = (
    "body_health", "cognition_mental", "communication_speech", "emotion_affect",
    "evaluation_modality", "existence_state", "perception_sensation",
    "physical_action", "possession_exchange", "quantity_degree",
    "social_interpersonal", "time_aspect", "other",
)


#: ── K RATINGS: coder-assigned, k_ prefixed ─────────────────────────────────
#: `meta/M01_displacement/lexicons/k_ratings_{en,zh}.json`, 27,242 English and
#: 20,654 Chinese words on seven 1-7 scales. Findings K.
#:
#: THE k_ PREFIX IS LOAD-BEARING. These are ONE MODEL's judgments at ONE frozen
#: instrument version -- not Warriner, not Brysbaert, not USAS. They agree with
#: human norms where a comparison exists (coder valence vs Warriner 0.81, coder
#: concreteness vs Brysbaert 0.88, Chinese concreteness vs the 9,877-word set
#: |0.81|), which is what licenses using them at all, and especially what
#: licenses the CHINESE half where no human norm exists for five of the seven.
#: But they carry a band and two scales are not established, so the prefix keeps
#: their status visible at the call site rather than in a document nobody opens.
#:
#: WHAT NOT TO DO WITH THEM, each measured rather than asserted:
#:   RANKS, NOT LEVELS. Charge and concreteness shift in level between
#:     instrument versions (2.29->2.58, 3.83->4.06 rated alone) while holding
#:     their order at r 0.88. Never threshold on an absolute value.
#:   register_level IS NOT ESTABLISHED. Weakest on three independent measures:
#:     inter-coder 0.60, rank stability 0.62 when isolated, z 1.0 against its
#:     own null. Usable as a descriptor, not as evidence. (M03 wanting a register
#:     axis should know this is the scale it would be leaning on.)
#:   vulgarity IS A SPARSE INDICATOR. Variance on 463 of 27,242 English words;
#:     r +0.000 with register among non-coarse words and -0.571 among coarse
#:     ones. Its floor effects are NOT nulls.
#:
#: Cross-project by design -- the ratings are a property of the WORD, so any
#: campaign holding word-level outcomes can join them.
K_LEX = os.path.join(LEX, "k_ratings_%s.json")
K_SCALES = ("vulgarity", "register_level", "transgressiveness", "charge",
            "valence", "bodily_harm", "concreteness")


@functools.lru_cache(maxsize=2)
def _k(lang="en"):
    p = K_LEX % lang
    if not os.path.exists(p):
        return {}, {}
    d = json.load(open(p, encoding="utf-8"))
    return d.get("ratings", {}), d.get("_meta", {})


def k_rating(word, scale=None, lang="en"):
    """K's rating(s) for one ALREADY-SEGMENTED word, or None.

    `scale=None` returns a dict of all seven. English keys are lowercase;
    Chinese are the surface as the tokenizer emitted it.
    """
    r, _ = _k(lang)
    v = r.get(word.strip().lower() if lang == "en" else word.strip())
    if v is None:
        return None
    d = dict(zip(K_SCALES, v))
    return d if scale is None else d.get(scale)


def k_meta(lang="en"):
    """Provenance, the inter-annotator band, and the not-established notes.

    Returned rather than buried so a caller can attach the band to whatever it
    reports -- the band travels with the number or the number is unqualified.
    """
    return _k(lang)[1]


def k_coverage(words, lang="en"):
    """(covered, total, fraction) -- report it, as with usas_zh."""
    r, _ = _k(lang)
    ws = [w for w in words if (w or "").strip()]
    key = (lambda w: w.strip().lower()) if lang == "en" else (lambda w: w.strip())
    hit = sum(1 for w in ws if key(w) in r)
    return hit, len(ws), (hit / len(ws) if ws else 0.0)


#: ── CHINESE USAS ────────────────────────────────────────────────────────────
#: UCREL Multilingual-USAS, `Chinese/semantic_lexicon_chi.tsv`, CC BY-NC-SA.
#: 32,540 lemmas on the SAME TAGSET as the English lexicon -- verified 2026-08-12:
#: 272 of 356 Chinese primary tags also occur in `usas_semantic_lexicon_en.txt`,
#: and the 84 that do not are gendered kinship subtags (S4.1.1.2f/m) English does
#: not split. 手 -> B1, 鞋 -> B5, 裤子 -> B5.
#:
#: THIS IS A LOOKUP, NOT A TEXT COUNTER, AND THAT IS DELIBERATE. `TOKEN` is
#: `[A-Za-z][A-Za-z'-]*`, so `tokens()` returns [] for any Chinese string and
#: every text-level entry point here (count, count_all, norms) is silently empty
#: on Chinese. Wiring the lexicon behind those would need a segmenter and a
#: declared segmentation scheme -- a real dependency and a real decision.
#: Callers with an already-segmented word (movement-table rows, K's rating units)
#: need none of that, so they get a lookup and the segmentation question stays
#: open rather than being answered by accident.
#:
#: PROVENANCE CAVEAT, UCREL'S OWN: the Chinese lexicon was generated by
#: translating the English one and "as automatically generated lexicons, they
#: contain errors". Coverage thins on long compounds -- 89% of single characters
#: and 43% of two-character words in K's zh population, 5-9% at 3-4 characters
#: (高跟鞋 and 皮夹克 are both absent). Report coverage with any count.
USAS_ZH = os.path.expanduser(
    "~/Dropbox/Prof/Articles/TheoryMachines/norms_sources/usas_zh/"
    "semantic_lexicon_chi.tsv")


@functools.lru_cache(maxsize=1)
def _usas_zh():
    """word -> (primary tag, all tags). First entry wins, as `_usas` does."""
    out = {}
    if not os.path.exists(USAS_ZH):
        return out
    with open(USAS_ZH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            w = (row.get("lemma") or "").strip()
            tags = (row.get("semantic_tags") or "").split()
            if not w or not tags or w in out:
                continue
            #: strip USAS's confidence/antonym decorations exactly as `_usas`
            #: does, so a zh tag and an en tag compare as strings
            cats = [re.split(r"[+\-%@/]", t)[0] for t in tags]
            out[w] = (cats[0], tuple(cats))
    return out


def usas_zh(word, all_tags=False):
    """USAS tag for one ALREADY-SEGMENTED Chinese word, or None.

    Comparable to `count(text, source="usas")` on English because it is the same
    tagset -- that comparability is the whole reason to prefer this over a
    Chinese-native scheme with its own categories.
    """
    v = _usas_zh().get((word or "").strip())
    if v is None:
        return None
    return v[1] if all_tags else v[0]


def usas_zh_coverage(words):
    """(covered, total, fraction). Coverage is thin on long compounds and a
    count without it invites reading absence as a semantic fact."""
    ws = [w for w in words if (w or "").strip()]
    d = _usas_zh()
    hit = sum(1 for w in ws if w.strip() in d)
    return hit, len(ws), (hit / len(ws) if ws else 0.0)


def _read_map(path, key=0, val=1):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) <= max(key, val) or row[key] in ("category", "group"):
                continue
            out[row[key].strip()] = row[val].strip()
    return out


@functools.lru_cache(maxsize=1)
def _usas():
    """lemma -> (primary tag, meta_field, fine_group).

    USAS lists tags most-likely-first and a token can carry several. Only the
    PRIMARY is used: the trailing tags are alternative readings, and counting
    all of them makes one token contribute to several fields, which turns a
    count into something with no unit.
    """
    meta = _read_map(os.path.join(MF, "usas_map.csv"))
    fine = _read_map(os.path.join(MF, "usas_free.csv"))
    out = {}
    with open(os.path.join(LEX, "usas_semantic_lexicon_en.txt"), encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[0] == "lemma":
                continue
            lemma, tags = p[0].lower(), p[2].split()
            if not tags:
                continue
            #: strip USAS's confidence/antonym decorations (+, -, %, @, /) so
            #: A5.1+ and A5.1- both resolve; the polarity is not in the map.
            cats = [re.split(r"[+\-%@/]", t)[0] for t in tags]
            if lemma in out:
                continue
            out[lemma] = (cats[0], meta.get(cats[0], "other"),
                          fine.get(cats[0], "other"), tuple(cats))
    return out


#: CLAWS content tags. `vv*` is the LEXICAL verb series and deliberately
#: excludes vb*/vd*/vh*/vm* -- be, do, have and the modals are function words
#: wearing verb morphology, and counting "was" as a content verb is how a field
#: count becomes a measure of sentence length.
_CONTENT_POS = ("nn", "vv", "jj", "rr", "np")


@functools.lru_cache(maxsize=1)
def _byu():
    """surface -> (lemma, claws_pos). Empty dict if the file is unreachable.

    31% of BYU's forms have a lemma that differs from the surface, which is the
    gap my suffix-peel was covering badly: the peel handles -s/-ed/-ing and
    silently misses every irregular, and `lexicons/README.md` records that the
    irregulars are exactly where a wrong answer is returned instead of none.
    """
    out = {}
    if not os.path.exists(BYU):
        return out
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        try:
            iw, il, ip = hdr.index("word"), hdr.index("lemma"), hdr.index("pos")
        except ValueError:
            return out
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) <= max(iw, il, ip):
                continue
            w = f[iw].strip().lower()
            #: FIRST ENTRY WINS, and the file is rank-ordered, so the winner is
            #: the most frequent reading of an ambiguous form. That is a real
            #: choice: `felt` resolves to feel rather than the fabric because
            #: the verb is commoner, which is right here and would be wrong in
            #: a textile corpus.
            if w and w not in out:
                out[w] = (f[il].strip().lower() or w, f[ip].strip().lower())
    return out


def is_content_word(tok):
    """CLAWS-based. Unknown forms count as content: the alternative drops every
    proper noun and neologism the corpus invents, and this corpus invents a lot."""
    b = _byu()
    if not b:
        return True
    e = b.get(tok)
    if e is None:
        return True
    return e[1].startswith(_CONTENT_POS)


@functools.lru_cache(maxsize=1)
def _gi():
    g = json.load(open(os.path.join(LEX, "general_inquirer.json")))
    meta = _read_map(os.path.join(MF, "gi_primary_map.csv"))
    return g["words"], meta


@functools.lru_cache(maxsize=1)
def _wordnet():
    w = json.load(open(os.path.join(LEX, "wordnet_verb_supersenses.json")))
    return {k: v.get("first") for k, v in w["words"].items() if v.get("first")}


@functools.lru_cache(maxsize=1)
def _rid():
    """RID as a LOOKUP, not a regex scan. -> (exact, prefixes, maxlen).

    The dictionary ships as 3,151 regexes, but 99.9% of them are `\\bword` or
    `\\bword\\b` -- plain word and word-prefix matches with nothing regex about
    them. Scanning raw text with 3,151 compiled patterns is ~260M searches over
    this corpus and takes about an hour; the same counts come from one dict
    lookup plus a short prefix walk per token, which is the architecture every
    other source in this module already uses.

    The 2 genuinely irregular entries are `test5` and `ttt1`, placeholder rows
    in the source file, and are dropped. Every category label is retained, so
    the category set is unchanged from the regex build.

    `exact` maps a whole token to a SET of labels; `prefixes` likewise. Multiple
    labels per key is required, not incidental: the regex build counts a token
    once per MATCHING PATTERN, so a token whose stem sits in two categories
    contributes to both. A first-hit-only lookup gave 63% category-set
    agreement against the regex build; collecting every match restores it.
    """
    exact, pref, mx = collections.defaultdict(set), collections.defaultdict(set), 0
    E = re.compile(r"^\\b([a-z'-]+)\\b$")
    P = re.compile(r"^\\b([a-z'-]+)$")
    with open(os.path.join(LEX, "rid_regressive_imagery.csv"), newline="",
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rx = row.get("regex") or ""
            lab = "%s:%s" % (row.get("category"), row.get("subcategory"))
            m = E.match(rx)
            if m:
                exact[m.group(1)].add(lab)
                continue
            m = P.match(rx)
            if m:
                w = m.group(1)
                pref[w].add(lab)
                mx = max(mx, len(w))
    return dict(exact), dict(pref), mx


#: RID stems match at a WORD BOUNDARY, and `-` and `'` are boundaries. The
#: module tokenizer deliberately keeps them inside a token (`well-known` is one
#: word), so a RID lookup on the whole token would miss `\bknown`. Split on
#: internal punctuation for this source only.
_RIDSPLIT = re.compile(r"[-']")


def _rid_labels(tok):
    """Every RID label this token carries. Empty set if none."""
    exact, pref, mx = _rid()
    out = set()
    for part in _RIDSPLIT.split(tok):
        if len(part) < 2:
            continue
        out |= exact.get(part, set())
        for L in range(min(len(part), mx), 1, -1):
            out |= pref.get(part[:L], set())
    return out


@functools.lru_cache(maxsize=1)
def _norms():
    """word -> {dim: value}, plus tertile cuts per dimension."""
    vals = collections.defaultdict(dict)
    wp = os.path.join(NORMS_DIR, "BRM-emot-submit.csv")
    if os.path.exists(wp):
        with open(wp, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                w = (row.get("Word") or "").strip().lower()
                if not w:
                    continue
                for dim, col in (("valence", "V.Mean.Sum"), ("arousal", "A.Mean.Sum"),
                                 ("dominance", "D.Mean.Sum")):
                    try:
                        vals[w][dim] = float(row[col])
                    except (KeyError, TypeError, ValueError):
                        pass
    bp = os.path.join(NORMS_DIR, "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    if os.path.exists(bp):
        with open(bp, encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            for row in rd:
                w = (row.get("Word") or "").strip().lower()
                if not w:
                    continue
                try:
                    vals[w]["concreteness"] = float(row["Conc.M"])
                except (KeyError, TypeError, ValueError):
                    pass
    #: EXTREMITY is derived here, not at call time, so its tertiles come from
    #: the same lexicon-wide distribution as everything else. Centre is the
    #: lexicon MEAN rather than the scale midpoint: Warriner runs 1-9 but its
    #: mass is not centred on 5, and using 5 would make "flat" mean "below
    #: average" for any dimension whose mean is off-centre.
    for dim in BASE_DIMS:
        xs = [x[dim] for x in vals.values() if dim in x]
        if len(xs) < 30:
            continue
        mu = sum(xs) / len(xs)
        for w, x in vals.items():
            if dim in x:
                x[dim + "_extremity"] = abs(x[dim] - mu)
    cuts = {}
    for dim in list(BASE_DIMS) + [d + "_extremity" for d in BASE_DIMS]:
        v = sorted(x[dim] for x in vals.values() if dim in x)
        if len(v) > 30:
            cuts[dim] = (v[len(v) // 3], v[2 * len(v) // 3])
    return dict(vals), cuts


#: Per-dimension names for the three bins, so the output reads as English
#: rather than as low/mid/high on four different scales.
BINS = {
    "valence": ("negative", "neutral", "positive"),
    "arousal": ("calm", "neutral", "aroused"),
    "dominance": ("submissive", "neutral", "dominant"),
    "concreteness": ("abstract", "neutral", "concrete"),
    #: EXTREMITY: |value - lexicon mean|. A dimension's SIGN and its INTENSITY
    #: are different questions and signed valence cannot see that `cunt` and
    #: `ecstasy` are the same kind of word -- both far from neutral, opposite
    #: directions. On this corpus that matters: the base arm's taboo nouns and
    #: the aligned arm's rapture nouns both sit at the extremes, so a signed
    #: contrast reports them as a difference and an extremity contrast asks
    #: whether the writing got LESS EMPHATIC at all.
    "valence_extremity": ("flat", "moderate", "extreme"),
    "arousal_extremity": ("flat", "moderate", "extreme"),
    "dominance_extremity": ("flat", "moderate", "extreme"),
    "concreteness_extremity": ("flat", "moderate", "extreme"),
}
BASE_DIMS = ("valence", "arousal", "dominance", "concreteness")


def tokens(text):
    return [t.lower() for t in TOKEN.findall(text or "")]


def _lookup(tok, table):
    """Exact, then BYU's lemma, then a conservative suffix peel.

    The lexicons are lemma-keyed and the text is inflected. BYU supplies a real
    lemma for 86,403 forms including the irregulars, so it goes first and the
    peel is only the fallback for forms BYU does not carry. The peel is kept to
    the regular suffixes because an aggressive stemmer returns the WRONG ENTRY
    rather than no entry -- the failure `lexicons/README.md` documents for
    `found`, `felt` and `saw`.
    """
    if tok in table:
        return table[tok]
    b = _byu().get(tok)
    if b and b[0] != tok and b[0] in table:
        return table[b[0]]
    for suf, add in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"),
                     ("ing", ""), ("ing", "e"), ("ies", "y")):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            cand = tok[:-len(suf)] + add
            if cand in table:
                return table[cand]
    return None


#: USAS's grammatical bins. Not content: pronouns, discourse operators, numbers
#: as tokens, names. They are the bulk of any running text and they swamp a
#: count that is trying to be about subject matter.
_GRAMMATICAL_FINE = {"pronouns", "logical_modal_and_discourse_operators",
                     "personal_names", "grammatical_bin", "other"}


def count(text, source="usas_fine", all_tags=True, content_only=True, _toks=None):
    """-> {"counts", "coverage", "n_tokens", "n_content", "n_counted", "source"}

    THE THREE DEFAULTS ARE CHOICES, not neutral settings, and each has a cost.

    `source="usas_fine"` -- ~30 readable USAS groups rather than the 13 shared
    meta-fields. The coarse map sends all 70 TOPICAL categories to `other`,
    including `G2.1` crime-and-law, so at that granularity guilt-and-law is
    invisible by construction. Pass source="meta" when counts must be
    comparable ACROSS lexicons; usas_fine is one lexicon's namespace.

    `all_tags=True` -- counts every USAS reading a token carries. **Counts no
    longer sum to the token count**: one word can land in three fields, so
    these are tag-instances, not tokens, and a "rate per token" computed from
    them is wrong. The reason it is the default anyway: `guilty` is tagged
    `G2.1- E4.1-`, primarily the legal sense and secondarily sadness, so
    primary-only scores the commonest guilt word in the corpus as crime and
    discards its affect entirely.

    `content_only=True` -- CLAWS content words only (nn/vv/jj/rr/np), where
    `vv*` excludes be/do/have/modals. Function words are ~60% of running prose
    and they swamp everything; unknown forms are KEPT, because dropping them
    would discard every proper noun and coinage, and this corpus coins freely.

    `coverage` is over ALL alphabetic tokens; `n_content` and `n_counted` give
    the denominators that actually apply under the defaults. A field count
    quoted without one of these is a rate with no population.
    """
    toks = _toks if _toks is not None else tokens(text)
    n = len(toks)
    n_content = sum(1 for t in toks if is_content_word(t)) if content_only else n
    c = collections.Counter()
    hit = 0
    counted = 0

    if source in ("meta", "usas", "usas_fine"):
        u = _usas()
        idx = {"usas": 0, "meta": 1, "usas_fine": 2}[source]
        meta_m, fine_m = _read_map(os.path.join(MF, "usas_map.csv")), \
            _read_map(os.path.join(MF, "usas_free.csv"))
        for t in toks:
            v = _lookup(t, u)
            if not v:
                continue
            hit += 1
            if content_only and not is_content_word(t):
                continue
            counted += 1
            if all_tags:
                seen = set()
                for cat in v[3]:
                    lab = (cat if idx == 0 else
                           (meta_m.get(cat, "other") if idx == 1 else fine_m.get(cat, "other")))
                    if lab not in seen:
                        seen.add(lab); c[lab] += 1
            else:
                c[v[idx]] += 1
    elif source == "gi":
        words, meta = _gi()
        for t in toks:
            if content_only and not is_content_word(t):
                continue
            cats = _lookup(t, words)
            if cats:
                hit += 1
                counted += 1
                #: GI gives many tags per word; the FIRST that maps is used, so
                #: one token contributes one count. Unioning them would let a
                #: single word land in five fields.
                for cat in cats:
                    if cat in meta:
                        c[meta[cat]] += 1
                        break
                else:
                    c["other"] += 1
    elif source == "wordnet":
        wn = _wordnet()
        for t in toks:
            if content_only and not is_content_word(t):
                continue
            v = _lookup(t, wn)
            if v:
                hit += 1
                counted += 1
                c[v] += 1
    elif source == "rid":
        for t in toks:
            if content_only and not is_content_word(t):
                continue
            labs = _rid_labels(t)
            if labs:
                hit += 1
                counted += 1
                for lab in labs:
                    c[lab] += 1
        return {"counts": c, "coverage": (hit / n if n else 0.0), "n_tokens": n,
                "n_content": n_content, "n_counted": counted, "source": source}
    else:
        raise ValueError("unknown source %r; try %s" % (
            source, ("meta", "usas", "usas_fine", "gi", "wordnet", "rid")))

    #: A branch that fills `counts` and leaves `counted` at zero produces a
    #: denominator of zero, and every caller that normalises by it silently
    #: drops the row -- which is how `gi` and `wordnet` returned empty tables
    #: that read as "no effects" rather than as "not wired up".
    assert not (c and not counted), (
        "source %r produced counts with n_counted=0; the branch does not "
        "increment `counted`" % source)
    return {"counts": c, "coverage": (hit / n if n else 0.0),
            "n_tokens": n, "n_content": n_content, "n_counted": counted,
            "source": source}


#: Every field vocabulary on disk. `meta` and `usas_fine` and `usas` are three
#: granularities of ONE lexicon and are NOT independent evidence; `gi`,
#: `wordnet` and `rid` are separate resources. A caller treating all six as six
#: votes is counting USAS three times.
ALL_SOURCES = ("meta", "usas_fine", "usas", "gi", "wordnet", "rid")


def count_all(text, all_tags=True, content_only=True, sources=ALL_SOURCES):
    """Every lexicon in ONE call, tokenised once. The function to reach for.

        f = fields.count_all(passage)
        f["flat"]["usas_fine:emotion_and_arousal"]     # namespaced, one dict
        f["by_source"]["gi"]["coverage"]               # per-lexicon detail

    -> {"flat", "by_source", "coverage", "n_tokens", "n_content", "sources"}

    WHY THIS EXISTS. Calling `count()` six times re-tokenises and re-runs the
    CLAWS content test six times, which is the whole cost at corpus scale, and
    every caller that wanted all six wrote its own loop with its own key format.
    `flat` is the shared key format: `source:field`, so counts from different
    lexicons can live in one feature vector without colliding.

    **`flat` IS NOT SUMMABLE ACROSS SOURCES AND NEITHER IS ANYTHING ELSE HERE.**
    `meta`, `usas_fine` and `usas` are three views of the SAME USAS lexicon;
    summing them counts one tagging three times. `gi`, `wordnet` and `rid` are
    genuinely separate resources. Six keys is not six votes -- it is four, and
    only if the three USAS views are collapsed to one first.

    COVERAGE IS RETURNED PER SOURCE AND IT IS NOT DECORATION. GI does not know
    `raped`, `desecrated` or `stomped`; on transgressive text its coverage
    collapses exactly where the signal is. A cross-lexicon comparison that does
    not carry `coverage` is comparing how much of the text each resource knows.
    """
    toks = tokens(text)
    n_content = sum(1 for t in toks if is_content_word(t)) if content_only else len(toks)
    by, flat, cov = {}, {}, {}
    for src in sources:
        r = count(text, source=src, all_tags=all_tags,
                  content_only=content_only, _toks=toks)
        by[src] = r
        cov[src] = r["coverage"]
        for k, v in r["counts"].items():
            flat["%s:%s" % (src, k)] = v
    return {"flat": flat, "by_source": by, "coverage": cov,
            "n_tokens": len(toks), "n_content": n_content,
            "sources": list(sources)}


@functools.lru_cache(maxsize=16)
def residualise(dim, on):
    """Regress `dim` on the dims in `on` across the whole lexicon.

    -> (residuals_by_word, r2, cuts). `r2` is how much of `dim` the controls
    explain, and it is RETURNED rather than logged because a control that
    removes nothing is the most important thing a residualiser can tell you.
    Measured here before use: dominance on concreteness is R2 = 0.0003 over
    13,386 words, so that particular control is a no-op and residualising on it
    would have produced a "corrected" number identical to the original with a
    methods sentence implying otherwise.

    `on` must be a tuple (the cache needs it hashable).
    """
    import numpy as np
    vals, _ = _norms()
    words = [w for w, v in vals.items()
             if dim in v and all(o in v for o in on)]
    if len(words) < 100:
        return {}, 0.0, None
    X = np.column_stack([np.ones(len(words))] +
                        [[vals[w][o] for w in words] for o in on])
    y = np.array([vals[w][dim] for w in words], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else 0.0
    out = {w: float(r) for w, r in zip(words, resid)}
    v = sorted(out.values())
    cuts = (v[len(v) // 3], v[2 * len(v) // 3])
    return out, r2, cuts


def norms(text, residualise_on=None):
    """-> {dim: {"counts": Counter, "coverage": float}}

    Trichotomised at the lexicon's own tertiles, so the bins mean high/mid/low
    RELATIVE TO ENGLISH rather than relative to a round number on a 1-9 scale.

    `residualise_on={"dominance": ("valence",)}` replaces that dimension with
    the part of it the named controls do NOT explain, re-tertiled on the
    residual distribution. The result carries `r2_removed` so a null control is
    visible: if the controls explain nothing, the "residualised" measure is the
    original measure and the reader should be told, not reassured.
    """
    vals, cuts = _norms()
    toks = tokens(text)
    out = {}
    for dim, (lo_name, mid_name, hi_name) in BINS.items():
        rdim = (residualise_on or {}).get(dim)
        if rdim:
            table, r2, rcuts = residualise(dim, tuple(rdim))
            if not table:
                continue
            lo, hi = rcuts
            getter = lambda v, t=None, tab=table: tab.get(t)
        elif dim in cuts:
            lo, hi = cuts[dim]
            table, r2 = None, None
            getter = lambda v, t=None, d=dim: v.get(d)
        else:
            continue
        c = collections.Counter()
        hit = 0
        for t in toks:
            v = _lookup(t, vals)
            if v is None:
                continue
            if table is not None:
                #: residual tables are keyed by WORD, so the lemma has to be
                #: resolved the same way the value lookup does or coverage
                #: silently halves.
                x = table.get(t)
                if x is None:
                    b = _byu().get(t)
                    x = table.get(b[0]) if b else None
            else:
                x = v.get(dim)
            if x is None:
                continue
            hit += 1
            c[lo_name if x < lo else (hi_name if x > hi else mid_name)] += 1
        rec = {"counts": c, "coverage": (hit / len(toks) if toks else 0.0)}
        if rdim:
            rec["r2_removed"] = r2
            rec["controls"] = list(rdim)
        out[dim] = rec
    return out


def norm_cuts():
    """The tertile boundaries actually in use. A threshold nobody can see is a
    free parameter; this makes them quotable."""
    return _norms()[1]


def available():
    """Which sources can actually run on this machine, and why not if not."""
    out = {}
    for name, path in (("byu", BYU),
                       ("usas", os.path.join(LEX, "usas_semantic_lexicon_en.txt")),
                       ("gi", os.path.join(LEX, "general_inquirer.json")),
                       ("wordnet", os.path.join(LEX, "wordnet_verb_supersenses.json")),
                       ("rid", os.path.join(LEX, "rid_regressive_imagery.csv")),
                       ("norms:warriner", os.path.join(NORMS_DIR, "BRM-emot-submit.csv")),
                       ("norms:brysbaert", os.path.join(
                           NORMS_DIR, "Concreteness_ratings_Brysbaert_et_al_BRM.txt"))):
        out[name] = "ok" if os.path.exists(path) else "MISSING: %s" % path
    return out


def _selftest():
    print("AVAILABILITY")
    for k, v in available().items():
        print("   %-18s %s" % (k, v))
    print("\nLEXICON SIZES")
    print("   usas lemmas      %s" % format(len(_usas()), ","))
    print("   gi words         %s" % format(len(_gi()[0]), ","))
    print("   wordnet verbs    %s" % format(len(_wordnet()), ","))
    print("   rid patterns     %s" % format(len(_rid()), ","))
    v, cuts = _norms()
    print("   norm words       %s" % format(len(v), ","))
    print("\nTERTILE CUTS (from the lexicons, not chosen)")
    for d, (lo, hi) in sorted(cuts.items()):
        print("   %-13s low < %.2f  <= neutral <= %.2f < high" % (d, lo, hi))
    probes = [
        ("articulate guilt", "I feel so guilty, but I can't help myself."),
        ("flat guilt", "she felt a little guilty about it"),
        ("explicit", "She knelt down and began to suck his cock."),
    ]
    print("\nPROBES")
    for label, t in probes:
        r = count(t, "meta")
        print("\n  %-18s %r" % (label, t))
        print("     coverage %.0f%% of %d tokens" % (100 * r["coverage"], r["n_tokens"]))
        print("     meta   : %s" % dict(r["counts"].most_common(5)))
        print("     usas   : %s" % dict(count(t, "usas_fine")["counts"].most_common(4)))
        nm = norms(t)
        print("     norms  : %s" % {d: dict(x["counts"]) for d, x in nm.items()
                                    if x["counts"]})
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    txt = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or sys.stdin.read()
    for src in ("meta", "usas_fine", "wordnet"):
        r = count(txt, src)
        print("%-10s cov %5.0f%%  %s" % (src, 100 * (r["coverage"] or 0),
                                         dict(r["counts"].most_common(8))))
    for d, x in norms(txt).items():
        if x["counts"]:
            print("%-10s cov %5.0f%%  %s" % (d, 100 * x["coverage"], dict(x["counts"])))
