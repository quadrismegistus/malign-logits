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
import transformers as _tf
_TFV = _tf.__version__
from transformers import AutoModelForCausalLM, AutoTokenizer

THETA, MAX_DEPTH = 0.001, 6

# THE BOUNDARY RULE IS NOT IN THE CACHE KEY, SO IT GOES IN THE VALUE.
# OVERWRITE makes the END state uniform and does nothing for the INTERMEDIATE
# one: an interrupted run leaves old-rule and new-rule cells coexisting with no
# marker distinguishing them -- beam_words verbatim, two rules in one store
# under a key that does not record which. A per-cell version makes every cell
# self-describing, so a PARTIAL state stays diagnosable without depending on a
# manifest having survived.
#
# v1  ASCII punctuation only; no CJK punctuation, no dictionary, no script
#     transition. Chinese resolves 3-16% of mass against 80-90% for English, and
#     English-prompt cells contain glued cross-script units. Run 1 is v1.
# v2  + fullwidth CJK punctuation            (50e0b13)
#     + dictionary prefix trie on CJK surfaces (6bb9f56)
#     + script transition is a boundary, both directions (68ec402)
RULE_VERSION = 3
RULE_COMMITS_V3_NOTE = ("intra-word punctuation + apostrophe normalisation "
                        "+ mojibake is not a word")
# v3 ALSO EXCLUDES MOJIBAKE. A surface containing U+FFFD is broken bytes, not
# vocabulary, and it went into `words` under v1/v2 -- so RESOLVED MASS IS
# OVERSTATED IN v1/v2 CELLS for any model whose tokenizer cannot represent the
# prompt's script. Measured on amber/Chinese: 0.892 -> 0.467, the difference
# being 0.428 of replacement characters. Its mass now lands in
# residual["mojibake"], a named channel, because it is real mass the model
# assigned and folding it into `drop` would hide it in the sub-theta tail --
# which is how it survived this long.
# v2's KNOWN LIMITATION, FIXED IN v3: punctuation is a boundary
# regardless of context, so digits and contractions split. `$100,000` records as
# `100`, and `$100,000` / `$100,500` / `$100` are indistinguishable. Same for
# `3.14`, `don't`, `state-of-the-art` -- all those characters are in PUNCT.
# Uniform across every cell, so it does not confound within-run comparison, and
# every cell carries rule_version so affected cells are findable. It DOES mean
# numeric-magnitude questions (the wage/salary battery) are not answerable from
# v2 output. A v3 fix is "punctuation flanked by digits or letters is not a
# boundary". DONE -- that is v3, plus one thing the v2 note missed: `didn't` and
# `didn't` with a curly apostrophe were TWO entries in one cell with the mass
# divided between them, because PUNCT held U+0027 and not U+2019. Normalisation
# is required whatever the boundary rule does.
RULE_COMMITS = ["50e0b13", "6bb9f56", "68ec402"]
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


# INTRA-WORD PUNCTUATION. These characters end a word in prose and sit INSIDE
# one when flanked by letters or digits:  don't  100,000  3.14  state-of-the-art
INTRA = set("'\u2019\u02bc,.-\u2010\u2011")

# ONE APOSTROPHE. A model emits U+0027 or U+2019 stochastically, and v2 had only
# the ASCII one in PUNCT -- so `didn't` split to `didn` while `didn't` with a
# curly mark survived whole, and the SAME WORD occupied two entries in one cell
# with its mass divided between them. Normalising is required whatever the
# boundary rule does.
APOS = {"\u2019": "'", "\u02bc": "'", "\u2018": "'"}


def norm_apos(s):
    for k, v in APOS.items():
        s = s.replace(k, v)
    return s


def intra_word(surface, tok_str):
    """Is tok_str's leading punctuation INSIDE a word rather than ending one?

    True only when the surface so far ends alphanumeric AND the character
    immediately after the punctuation is alphanumeric -- `don` + `'t`,
    `100` + `,000`. False for `ran` + `, then` (space follows) and for a token
    that is punctuation alone (nothing follows).

    LIMIT, declared rather than hidden: a tokenizer that emits the punctuation
    as its OWN token gives us nothing to look ahead to, so `3` `.` `14` still
    breaks. Resolving that needs lookahead the expansion does not have at the
    point the mask is applied.
    """
    if not surface or not tok_str:
        return False
    if not surface[-1].isalnum():
        return False
    c = tok_str[0]
    if c not in INTRA:
        return False
    return len(tok_str) > 1 and tok_str[1].isalnum()


REPLACEMENT = "\ufffd"


def is_mojibake(s):
    """A surface containing U+FFFD is BROKEN BYTES, not a word.

    A Latin-oriented tokenizer fragments CJK into byte pieces; an individual
    byte token is not valid UTF-8, so `tok.decode` yields the replacement
    character. The boundary rule then treats it as a perfectly good word --
    non-empty, no boundary character, terminates cleanly -- and it RESOLVES.

    Measured before the fix: Amber, beaver, alpaca and llama-7b put 39-47% of
    their Chinese probability mass on `\ufffd`. Raw resolved mass therefore
    RANKED BROKEN-BYTE MODELS ABOVE FLUENT ONES: Amber scored 0.94 on Chinese
    prompts against Qwen's 0.03, and the difference was that Amber's tokenizer
    could not represent the text at all.

    `is_word` never caught it. That field tests for ASCII-alphabetic surfaces,
    so it flags mojibake only when nothing else does -- and here mojibake WAS
    the top word.
    """
    return REPLACEMENT in s


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
    s = norm_apos(s)
    for i, c in enumerate(s):
        if c.isspace():
            return s[:i]
        if c in PUNCT:
            # keep it if it is flanked by alphanumerics on BOTH sides
            if 0 < i < len(s) - 1 and s[i-1].isalnum() and s[i+1].isalnum() \
                    and c in INTRA:
                continue
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
    ids, strs, latin = [], [], []
    punct_ids, punct_str = [], []
    for i in range(n):
        try:
            t = tok.decode([i])
        except Exception:
            continue
        if not t:
            continue
        if is_cjk(t) and t == t.strip():
            ids.append(i); strs.append(t)
        elif t[0].isascii() and t[0].isalnum():
            latin.append(i)      # word-continuing Latin: boundary after CJK
        tn = norm_apos(t)
        if tn and tn[0] in INTRA and len(tn) > 1 and tn[1].isalnum():
            # a token that COULD be intra-word: `'t`, `,000`, `-of`
            punct_ids.append(i); punct_str.append(tn)
    return (np.array(ids, dtype=int), strs, np.array(latin, dtype=int),
            np.array(punct_ids, dtype=int))


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
    """gc.collect() -> empty_cache(). THE CALLER MUST DROP ITS OWN REFERENCES FIRST.

    THE ARGUMENTS ARE ACCEPTED AND CANNOT HELP. `del o` inside this function
    deletes the LOCAL name bound to the parameter; the caller's `model` still
    holds the object, so `free(model, tok)` released nothing and every model
    stayed resident in VRAM until the caller rebound `model` -- which happens
    DURING the next load, making the peak TWO models rather than one.

    Invisible at 7B (14 + 14 GB against 80) and fatal at 32B (64 + 64 against 93):
    it is why Olmo-3.1-32B-Instruct-SFT reported `126.56 MiB is free` at load
    after the DPO arm had completed 979 cells on the same card, and it is the real
    cause of the 2026-07-30 32B OOM that was booked as "32B at fp16 is marginal on
    80 GB". A single 32B is 64 GB and fits; two at once do not.

    Callers now do `model = tok = None` and call this with no arguments. The
    signature is kept only so an old call site fails loudly at review rather than
    silently at runtime.

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


# THE ADAPTED BATCH SURVIVES THE PROMPT THAT DISCOVERED IT.
# Halving on OOM was right; throwing the answer away afterwards was not. `batch`
# was a per-call default, so EVERY prompt restarted at 64, OOM'd, halved and only
# then succeeded -- one full OOM-and-retry per prompt, forever.
#
# Measured on Falcon-H1-1.5B: 49 OOMs in 58 prompts, 0.083 p/s, which is 3.3 h
# for one arm and 26 h for the eight Falcon arms. The same code ran 7B dense
# transformers at 0.9-2.5 p/s. The models were never the problem.
#
# Reset per MODEL (a new checkpoint gets a fresh ceiling), carried across
# PROMPTS within one model.
_BATCH = {"n": 64}


def reset_batch(start=64):
    _BATCH["n"] = start


@torch.no_grad()
def next_dist(model, tok, pids, prefixes, dev, batch=None):
    """Batch is ADAPTIVE because it is architecture-blind.

    A dense transformer's peak scales with batch x seq x vocab. An SSM's does
    not: Falcon-H1's `torch_forward` materialises B_decay[...,None,:] *
    hidden_states[...,None], which is batch x seq x heads x state x dim and hit
    24 GiB at batch=64 on a 1.5B model -- it OOM'd where a 7B transformer was
    fine. Rather than maintain a per-architecture table that is wrong the moment
    a new family is registered, halve on OOM and carry on -- AND REMEMBER.
    """
    batch = _BATCH["n"] if batch is None else batch
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
            _BATCH["n"] = batch            # remember it for every later prompt
            print(f"    [oom] batch -> {batch}", flush=True)
            continue                       # retry the SAME slice, smaller
        out.append(torch.softmax(lg, -1).cpu().numpy())
        i += len(ch)
    return np.concatenate(out, 0)


# BOS POLICY: THE RUNNER OWNS IT, NOT tokenizer_config.json.
# Amendment 2 to freeze [781]. Inheriting the tokenizer default means the
# conditioning is set by whoever wrote that checkpoint's config -- not a
# research decision anyone made -- and it DEMONSTRABLY VARIES BETWEEN ARMS OF
# ONE FAMILY: internlm2's base does not auto-prepend BOS while both aligned arms
# do, on 763 of 763 prompts, so every base-vs-aligned edge compared different
# conditioning with every position shifted.
#
# DEFAULT IS `inherited`, DELIBERATELY. A global switch to explicit BOS would
# change the encoding for every model already scored and make the whole grid
# mixed -- the harm the amendment exists to prevent. So the policy REPRODUCES
# current behaviour everywhere except families measured to be internally
# inconsistent, which is internlm2 alone.
BOS_POLICY = {"internlm2": "forced"}      # substring match on the model id


# PER-MODEL LOADER TABLE. A DECLARED POLICY, NOT AN INLINE SPECIAL CASE.
#
# transformers v5 (#45488) makes LlamaTokenizer.__init__ install a SentencePiece
# Metaspace pre-tokenizer over the ByteLevel one a repo's tokenizer.json
# declares. On deepseek-llm-7b that deletes every space -- 'She was so angry she
# wanted to' encodes and decodes as 'Shewassoangryshewantedto' -- and drops CJK
# entirely, with unk_token=null so nothing raises. PR #45936 fixed models with
# bespoke architecture tags; #47017 would fix generic model_type=llama repos and
# is NOT merged as of transformers 5.4.0.
#
# The entry retires VISIBLY when upstream lands the generic fix: precondition 6
# (tokenizer_roundtrip_sweep.py) will report the model clean and the row can go.
# An inline `if "deepseek" in mid` would never retire, because nothing would
# ever tell anyone it had become unnecessary.
LOADER_OVERRIDE = {
    "deepseek-ai/deepseek-llm-7b-base": ("PreTrainedTokenizerFast", "#45488/#47017"),
    "deepseek-ai/deepseek-llm-7b-chat": ("PreTrainedTokenizerFast", "#45488/#47017"),
}


def load_tokenizer(mid):
    """Return (tokenizer, loader_id). loader_id is STAMPED ON THE CELL."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    ov = LOADER_OVERRIDE.get(mid)
    if ov and ov[0] == "PreTrainedTokenizerFast":
        # bypasses AutoTokenizer class resolution, which follows
        # tokenizer_config.json's tokenizer_class field and lands on the broken
        # class regardless of use_fast
        return PreTrainedTokenizerFast.from_pretrained(mid), f"override:{ov[1]}"
    return AutoTokenizer.from_pretrained(mid, trust_remote_code=True), "auto"


def bos_policy_for(model_id):
    for k, v in BOS_POLICY.items():
        if k in model_id.lower():
            return v
    return "inherited"


def encode_prompt(tok, prompt, policy):
    """Return ids under an EXPLICIT policy, and the policy actually applied."""
    if policy == "inherited":
        return tok.encode(prompt), "inherited"
    ids = tok.encode(prompt, add_special_tokens=False)
    if policy == "forced":
        b = getattr(tok, "bos_token_id", None)
        if b is None:
            return ids, "none(no_bos_token)"   # declared, never silent
        return [b] + ids, "forced"
    return ids, "none"


# LOGICAL PROMPTS: a surface the runner RESOLVES rather than feeds.
#
# BOS is the first. Its realisation is model-specific, so it cannot be keyed on
# its realisation -- a per-family BOS string scored on 103 models is one
# family's token fed to models where it is a literal, shattering into characters
# (<|begin_of_text|> is 9 tokens on Amber). The sentinel is never tokenized; the
# runner dispatches on it and builds ids DIRECTLY.
#
# ids = [bos_id] AND NOT tok.encode(bos_token). Encoding the string doubles the
# BOS on every tokenizer that auto-prepends one -- amber's tok('<s>') is [1, 1].
# That is amendment 2's rule and this is where it binds.
SENTINEL_BOS = "<<<LOGICAL:BOS>>>"

# 18 families report bos_token=None. A declared fallback or a RECORDED SKIP --
# never the string "None", never a silent crash (registrar, [789].2a).
# PER-FAMILY, FROM EACH FAMILY'S OWN TRAINING CONVENTION -- never one global
# choice (RH). The measurement that proves the rule: Falcon-H1 uses
# <|end_of_text|> where the rest of the Falcon line uses <|endoftext|>, so a
# single global token would be WRONG for two arms while looking right.
#
# Each token is the family's own eos / document separator, read from its
# tokenizer rather than assumed. Longest key wins, so Falcon-H1 is matched
# before the generic falcon prefix.
#
# STRATIFICATION IS FROZEN WITH THE TABLE: F19 reports true-BOS arms and
# fallback arms as SEPARATE STRATA before any pooling. <|endoftext|> signals
# DOCUMENT-START, not sequence-start -- a fallback arm's unconditional
# distribution is conditioned on "a document just ended", which is a different
# state from "nothing precedes this". The resolver stamp is the stratification
# key, so this costs nothing at run time.
#
# Entries retire visibly: if a family later ships a real bos_token, resolve_logical
# takes the bos_token branch first and the row becomes dead code a reader can see.
BOS_FALLBACK = {
    "allenai/olmoe":        "<|endoftext|>",    # OLMo/Dolma document separator
    "allenai/olmo-3":       "<|endoftext|>",    # same lineage
    "allenai/olmo-hybrid":  "<|endoftext|>",    # same lineage
    "tiiuae/falcon-h1":     "<|end_of_text|>",  # H1 DIFFERS from the Falcon line
    "tiiuae/falcon3":       "<|endoftext|>",
    "tiiuae/falcon-mamba":  "<|endoftext|>",
    "zai-org/glm-4":        "<|endoftext|>",    # eos == pad here
    # SmolLM3 IS THE CASE THAT NEEDED A DECISION RATHER THAN A LOOKUP. Its base
    # eos is <|end_of_text|> (a document separator) but its instruct eos is
    # <|im_end|> (a CHAT-TURN terminator). Taking each arm's own eos would
    # condition the two arms of ONE FAMILY on different KINDS of state --
    # document-start vs turn-end -- and every comparison this project makes is
    # WITHIN family. <|end_of_text|> exists at id 128001 on BOTH arms, so the
    # family is conditioned consistently on its document separator.
    "huggingfacetb/smollm3": "<|end_of_text|>",
    "qwen":                 "<|endoftext|>",    # the project's own table
}


def resolve_logical(tok, prompt):
    """(ids, resolved_surface, resolver_id) for a sentinel, else None."""
    if prompt != SENTINEL_BOS:
        return None
    b = getattr(tok, "bos_token_id", None)
    if b is not None:
        return [b], (tok.bos_token or ""), "bos_token"
    name = str(getattr(tok, "name_or_path", "")).lower()
    for key, tokstr in sorted(BOS_FALLBACK.items(), key=lambda kv: -len(kv[0])):
        if key in name:
            ids = tok.convert_tokens_to_ids([tokstr])
            if ids and ids[0] is not None:
                return ids, tokstr, f"fallback:{tokstr}"
    return None, None, "skip:no_bos_token"      # RECORDED, not silent


def purge_model(mid, enabled=True):
    """Delete a model's HF cache. CALLED ON EVERY EXIT PATH, INCLUDING FAILURE.

    THE LOAD-FAILURE PATH USED TO `continue` PAST THE PURGE, and that is the
    second time disk has nearly killed this run. The first was purging AFTER
    completion, so four 65 GB downloads accumulated; the fix moved it before
    each download -- but a model that FAILS TO LOAD never reaches the next
    download, so its cache was never collected.

    Measured on the v3 grid: two 32B arms OOM'd at load and left 123 GB EACH
    resident, with RWKV, mistral-sft and three more failures beside them. Disk
    went 247 GB -> 53 GB free and was still falling. A failed model's weights
    are the LEAST worth keeping and were the only ones kept.
    """
    if not enabled:
        return
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    tag = mid.replace("/", "--")
    if not os.path.isdir(hub):
        return
    for sub in os.listdir(hub):
        if sub.startswith("models--") and tag in sub:
            shutil.rmtree(os.path.join(hub, sub), ignore_errors=True)
            print(f"  purged {sub}", flush=True)


def assert_prompt_survives(tok, prompt, ids):
    """The prompt the model sees must BE the prompt, not a remainder of it.

    Two failures, and the second is the dangerous one:

      empty ids      deepseek-llm-7b maps every CJK character to nothing -- no
                     UNK, no replacement, no error -- so a Chinese prompt
                     becomes []. That crashes downstream with a dtype error
                     naming nothing relevant.
      SILENT TRUNCATION
                     a MIXED prompt losing only its CJK would encode fine,
                     score fine, and resolve fine, on a sentence with part of
                     it removed. Not an error: a plausible number computed on
                     the wrong input.

    An ASSERTION, not a print, per the standing rule -- a warning in a 60,000-
    cell run is a line nobody reads. Swept over the roster: 2 of 110 models
    drop CJK (both deepseek arms), 0 truncate partially.
    """
    if not ids:
        raise ValueError(
            f"tokenizer produced ZERO ids for a non-empty prompt {prompt[:40]!r} "
            f"-- this tokenizer discards the prompt's script entirely")
    # FULL ROUND-TRIP, not a CJK check. The first version of this guard tested
    # only for lost CJK, and it PASSED deepseek-llm-7b on English:
    #
    #     'She was so angry she wanted to' -> 'Shewassoangryshewantedto'
    #
    # Seven ids, non-empty, no CJK to lose. A transformers v5 regression
    # (#45488) installs a SentencePiece Metaspace pre-tokenizer over the
    # ByteLevel one the repo declares; whitespace fails to remap and vanishes,
    # and with `unk_token: null` nothing raises. So a script-specific test was
    # the wrong shape -- the question is whether the model sees THE PROMPT.
    # skip_special_tokens=True, AND THE REASON IS THIS GUARD'S OWN BUG.
    # `ids` comes from encode_prompt() under the BOS POLICY, so on any
    # Mistral/Llama-family tokenizer it carries a leading <s> THAT WE ASKED FOR.
    # Decoding it back without skipping specials rendered '<s> - the agony...'
    # and the guard failed the prompt against its own BOS -- killing
    # zephyr-7b-beta at 0/979 in the v3 grid.
    #
    # THE REAL DEFECT WAS THAT PRECONDITION 7 AND THIS GUARD TESTED DIFFERENT
    # PATHS. prompt_encode_check.py encodes with add_special_tokens=False and
    # passed 100,837 pairs with 0 failures; the runner encodes through the BOS
    # policy. A precondition that green-lights a path the run does not take is
    # the same shape as an encode guarantee established on another library
    # version -- it reads as verified and covers something else.
    #
    # The cost of skipping is real and small: a tokenizer that mangled the
    # prompt INTO special tokens would now pass. Nothing in the roster does
    # that, and the failure it replaces was a false positive that discarded a
    # whole model.
    back = tok.decode(ids, skip_special_tokens=True)
    if back.strip() != prompt.strip():
        # normalise only what a tokenizer may legitimately alter
        a = " ".join(back.split())
        b = " ".join(prompt.split())
        if a != b:
            raise ValueError(
                f"prompt does not survive encoding:\n  sent {prompt[:60]!r}\n"
                f"  got  {back[:60]!r}\n"
                f"the model would score text that is not the prompt")


class SkipPrompt(Exception):
    """This PROMPT cannot be scored on this model. Not a model failure."""


#: expand() computes the logit vector deep inside its beam loop and returns
#: word probabilities. Threading a second return value through every call
#: site and its two exception paths would touch code the run depends on;
#: a one-slot handoff touches none of it. Single-threaded by construction
#: -- one model, one prompt at a time -- and OVERWRITTEN EVERY PROMPT, so
#: a stale read is impossible: the write happens between expand() and the
#: jsonl line for the same prompt.
_LOGIT = {"v": None}


@torch.no_grad()
def expand(model, tok, prompt, dev, bmask, theta=THETA, cjk=None,
           bos_policy="inherited"):
    """cjk=(prefix_trie, ids, strings) enables dictionary word boundaries.

    ONE INSTRUMENT, DISPATCHING ON SCRIPT -- not two runs side by side. A non-CJK
    surface keeps the whitespace/punctuation rule; a CJK surface takes the
    dictionary rule. Mixed-script text is therefore handled correctly WITHIN a
    cell, which two parallel datasets could not do since they would force a
    per-cell choice mixed text cannot make.
    """

    lg_ = resolve_logical(tok, prompt)
    resolved_surface, resolver_id = None, None
    if lg_ is not None:
        pids, resolved_surface, resolver_id = lg_
        if pids is None:
            # A RECORDED SKIP, NOT A CRASH. This raised, and main()'s per-model
            # try/except wraps the WHOLE prompt loop -- so one unresolvable
            # prompt aborted the entire model, which wrote nothing and failed
            # identically on every resume. Measured: 18 of 103 models have
            # bos_token=None and are not covered by the fallback table (all
            # OLMo-Hybrid/OLMoE, the whole Falcon line, glm4), and the sentinel
            # sorts near-first, so ~14,000 cells would have been lost in a way
            # no retry could repair.
            #
            # [789].2a required "a declared fallback or a RECORDED skip". This
            # is the recorded skip; the fallback table is a research decision
            # and is owed separately.
            raise SkipPrompt(resolver_id or "unresolvable_logical")
        # a resolved logical prompt bypasses the survival assert BY DESIGN:
        # there is no surface that was supposed to round-trip
    else:
        pids, _applied = encode_prompt(tok, prompt, bos_policy)
        assert_prompt_survives(tok, prompt, pids)
    lg = model(torch.tensor([pids], device=dev)).logits[0, -1, :].float()
    #: THE LOGIT FOLD (RH, 2026-08-01). `lg` IS ALREADY the full-vocabulary
    #: last-position vector the logit stash wants -- this call is batch-1,
    #: consumes no RNG, and P0/theta/the residual/the beam are all
    #: downstream, so capturing it is a PURE SIDE EFFECT and word-prob
    #: values are bit-identical. RULE_VERSION versions the WORD-BOUNDARY
    #: RULE; a logit vector segments nothing, so no v3->v4.
    #: float16 ON RH's RULING: the existing store is MIXED (49 models f16,
    #: 87 f32) and a uniform store is worth more than a marginally more
    #: precise one that cannot be compared across its own rows.
    _LOGIT["v"] = lg.half().cpu().numpy()
    P0 = torch.softmax(lg, -1).cpu().numpy()
    sel = np.flatnonzero(P0 >= theta)
    live = [((int(t),), float(P0[t]), int(t)) for t in sel]
    words, calls, bcache, intra_cache = {}, 0, {}, {}
    res_moji = 0.0
    res_tail, res_drop = float(1.0 - P0[sel].sum()), 0.0
    for _ in range(MAX_DEPTH):
        if not live:
            break
        dist = next_dist(model, tok, pids, [p for p, _, _ in live], dev); calls += 1
        nxt = []
        for (pref, mass, t1), row in zip(live, dist):
            surf = clean_surface(tok.decode(list(pref)).strip())
            b = bmask
            if cjk is not None and surf and surf[-1].isalnum():
                # INTRA-WORD PUNCTUATION UNMASKED. A token like `'t` or `,000`
                # is a boundary in bmask (it starts with punctuation) but is
                # INSIDE a word when the surface ends alphanumeric. Unmask those
                # ids so the word continues: don + 't -> don't, 100 + ,000 ->
                # 100,000. Surface-dependent, so it cannot live in the static
                # mask -- same reason as the CJK rule.
                b = intra_cache.get(surf)
                if b is None:
                    b = bmask.copy()
                    b[cjk[4]] = False
                    intra_cache[surf] = b
            if cjk is not None and surf:
                trie, cids, cstrs, lids, pids_intra = cjk
                base_b = b
                b = bcache.get(surf)
                if b is None and not is_cjk(surf):
                    # A SCRIPT TRANSITION IS A WORD BOUNDARY. Latin surfaces are
                    # extended by CJK tokens under the static rule -- a CJK token
                    # is neither space-prefixed nor punctuation, so nothing stops
                    # it. The roster contains the result: `mouth什么意思` and
                    # `mouth和He` reported as single words on an ENGLISH prompt.
                    # The dictionary rule alone does not fix this; it only moves
                    # the cut one character later, to `mouth什`.
                    b = base_b.copy()
                    b[cids] = True
                    bcache[surf] = b
                elif b is None:
                    # EVERY CJK token is judged, not just the probable ones.
                    # Termination sums over ALL boundary tokens, so improbable
                    # CJK tokens that should end the word must be marked or the
                    # word runs to the clause end and its mass drains via `drop`.
                    b = base_b.copy()
                    inside = np.fromiter(((surf + t) in trie for t in cstrs),
                                         dtype=bool, count=len(cstrs))
                    b[cids] = ~inside
                    b[lids] = True      # CJK -> Latin is a transition too
                    bcache[surf] = b
            term = float(row[b].sum())
            if surf and not is_mojibake(surf):
                words[(surf, t1)] = words.get((surf, t1), 0.0) + mass * term
            else:
                # A SEPARATE CHANNEL, not silent. Mojibake mass is real mass
                # the model assigned; it is simply not a word, so it must be
                # accounted rather than dropped into the general residual where
                # it would be indistinguishable from sub-theta tail.
                if surf:
                    res_moji += mass * term
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
                       mojibake=res_moji,
                       total=res_tail + res_drop + res_open + res_moji,
                       resolver=resolver_id,
                       resolved_surface=resolved_surface), calls


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
    # THE SPEC GAINED A _meta WRAPPER when the categorisation sha was stamped
    # into it; this read a flat list and got the string "_meta". Accept both --
    # older spec files on disk are still flat, and a runner that only accepts
    # the newest format cannot re-run an archived spec.
    _raw = json.load(open(a.models))
    spec = _raw["spec"] if isinstance(_raw, dict) else _raw
    if isinstance(_raw, dict) and _raw.get("_meta"):
        print(f"spec meta: {_raw['_meta']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trie = None if a.no_dict else load_prefix_trie(a.dict)
    # THE DICTIONARY IS PART OF THE RULE. A different word list is a different
    # boundary rule wearing the same version number, so its hash is stamped per
    # cell alongside the version.
    import hashlib
    dict_sha = None
    if trie is not None and os.path.exists(a.dict):
        h = hashlib.sha256()
        with open(a.dict, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        dict_sha = h.hexdigest()[:16]
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
        # HOISTED, AND IT MATTERS ON EVERY RESTART. This read
        # `p not in done_prompts(path)` inside the comprehension, so the whole
        # jsonl was re-read and re-parsed ONCE PER PROMPT -- 979 x 979 = ~958k
        # JSON parses to decide that a finished model has nothing to do. A
        # completed model took ~2 minutes to SKIP, so resuming past 22 finished
        # models cost ~45 minutes of pure re-parsing before the first new cell.
        # Invisible while the run goes forward and only ever paid on recovery,
        # which is exactly when time is worth most.
        done = done_prompts(path)
        todo = [p for p in prompts if p not in done]
        print(f"\n[{mi}/{len(spec)}] {mid}  {len(todo)}/{len(prompts)} to do", flush=True)
        if not todo:
            continue
        try:
            tok, loader_id = load_tokenizer(mid)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        except Exception as e:
            print(f"  LOAD FAILED: {str(e)[:120]}", flush=True)
            free()                 # the traceback held the partial load
            purge_model(mid, a.purge)   # ITS WEIGHTS ARE NOW DEAD WEIGHT
            continue
        reset_batch()                  # a new checkpoint gets a fresh ceiling
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
                cids, cstrs, lids, pids_intra = cjk_vocab(tok, model.config.vocab_size)
                if len(cids):
                    cjk = (trie, cids, cstrs, lids, pids_intra)
                    print(f"  cjk: {len(cids):,} tokens", flush=True)
        except Exception as e:
            print(f"  MASK FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            model = tok = None          # THE CALLER drops its own references
            free()
            continue
        pol = bos_policy_for(mid)
        if pol != "inherited":
            print(f"  bos_policy: {pol}", flush=True)
        t0, i, skipped = time.time(), 0, 0
        # ONE MODEL MUST NOT END THE ROSTER. The first version guarded only the
        # load, so a mid-run OOM on model 17 of 103 took the other 87 with it.
        # Per-prompt writes are already flushed, so a model that dies partway
        # keeps what it finished and resumes there on the next pass.
        try:
            lpath = os.path.join(a.out, f"{safe}.f16")
            #: Row counter RESUMES from the file's own size, never from a
            #: remembered count -- the file is the record of how many rows it
            #: holds, and a counter that starts at 0 on restart would overwrite
            #: nothing and mis-index everything after it.
            _dim = getattr(model.config, "vocab_size", None)
            logit_n = (os.path.getsize(lpath) // (2 * _dim)
                       if _dim and os.path.exists(lpath) else 0)
            with open(path, "a") as f, open(lpath, "ab") as lf:
                for i, p in enumerate(todo, 1):
                    try:
                        w, res, calls = expand(model, tok, p, dev, bmask,
                                               cjk=cjk, bos_policy=pol)
                    except SkipPrompt as sk:
                        # recorded, flushed, resumable -- and the model lives
                        f.write(json.dumps({
                            "model": mid, "prompt": p, "theta": THETA,
                            "skipped": str(sk), "rows": [], "residual": None,
                            "rule_version": RULE_VERSION,
                            "bos_policy": pol, "loader": loader_id}) + "\n")
                        f.flush()
                        skipped += 1
                        continue
                    tot = sum(w.values()) + res["total"]
                    #: SIDECAR, NOT JSONL. 109 KB of float per cell cannot go in
                    #: a JSON line -- base64 would be ~145 KB of text per row
                    #: plus the parse cost on every resume scan. Raw float16
                    #: appended to a .f16 file; `logit_row` indexes it, so row
                    #: n of the binary IS the nth logit-bearing jsonl line and
                    #: the pairing survives a crash mid-model.
                    _lg = _LOGIT["v"]
                    _row = None
                    if _lg is not None:
                        _row = logit_n
                        lf.write(_lg.tobytes()); lf.flush()
                        logit_n += 1
                        _LOGIT["v"] = None
                    f.write(json.dumps({
                        "model": mid, "prompt": p, "theta": THETA,
                        "logit_row": _row, "logit_dim": (int(_lg.shape[0])
                                                         if _lg is not None else None),
                        "logit_dtype": "float16",
                        "rule_version": RULE_VERSION,
                        "rule_commits": RULE_COMMITS,
                        "dict_sha": dict_sha,
                        "bos_policy": pol,
                        "loader": loader_id,
                        # LIBRARY VERSIONS, ADDED 2026-07-30 BECAUSE THEY TURNED
                        # OUT TO MATTER. transformers refuses .bin weights below
                        # torch 2.6 (check_torch_load_is_safe), which cost this
                        # grid 12 models on a box pinned at 2.5.1 -- so the
                        # recovered arms must be scored under a DIFFERENT torch
                        # than the rest. Local-vs-box on one shared model gave an
                        # IDENTICAL surface set with per-word probabilities
                        # differing by <=1.5e-3, so the effect is small; small is
                        # not zero, and a cell that cannot say which library
                        # produced it cannot be excluded from a comparison later.
                        "torch_version": torch.__version__,
                        "transformers_version": _TFV,
                        "resolver": res.get("resolver"),
                        "resolved_surface": res.get("resolved_surface"),
                        "rows": [{"word": s_, "t1": t_, "p": m_} for (s_, t_), m_ in w.items()],
                        "residual": res, "batches": calls, "conservation": tot}) + "\n")
                    f.flush()                  # crash-safe: complete line on disk
                    if i % 50 == 0:
                        print(f"    {i}/{len(todo)}  {i/(time.time()-t0):.2f} p/s", flush=True)
            print(f"  done {len(todo)} in {(time.time()-t0)/60:.1f} min"
                  + (f"  ({skipped} SKIPPED)" if skipped else ""), flush=True)
        except Exception as e:
            print(f"  RUN FAILED after {i-1}/{len(todo)}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        model = tok = None                 # THE CALLER drops its own references;
        free()                             # passing them to free() released nothing
        purge_model(mid, a.purge)          # download is the binding constraint
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
