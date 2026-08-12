# zh reader rubric — second-order predicates over a conflicted condition

DRAFT for RH's review before any reader runs (2026-08-12, pen). Adapts
the persisted EN rubric (lacan's `m02-opus-second-order`, session
cdbe9c9e, verbatim in the workflow JSON) to the zh batches at
`data/raw/l2_zh_batches/` (7 x 100, blind, seed 20260812, 100-char
window, arms/roles balanced, key quarantined).

DESIGN DELTAS FROM EN, each declared:

1. WINDOW: first 100 characters (EN: 50 words). Declared-and-dumb; no
   trimming rule.
2. RUBRIC LANGUAGE: the rubric stays ENGLISH (the readers are Opus
   agents; the EN run's rubric was English over English text, this is
   English over Chinese text) with zh-specific guidance inline. Keeps
   the two runs' instruments as close to identical as the language
   change allows.
3. THE METALINGUISTIC EXCLUSION IS LOAD-BEARING FOR zh: zh cells are
   known to produce translation/glossing behavior (the model discussing
   or Englishing the prompt). All of that is NO under the same rule the
   EN rubric already carries — judge only inside the fiction.
4. FOUR AUTHORED CONTROL PASSAGES ride in a final mini-batch with known
   ids (2 clear YES: one abstract 矛盾-naming, one oxymoron fusing the
   poles; 2 clear NO: dense contrary predicates with no outside vantage).
   A reader missing a control disqualifies its batch, not the pairs.

## PROPOSED TASK TEXT (per batch n)

Read this file: data/raw/l2_zh_batches/batch_0{n}.txt

It contains 100 story continuations IN CHINESE, each the first 100
characters after a prompt. The prompt is context only.

## THE ONE JUDGEMENT

For each continuation, decide: does it produce an expression that takes
a CONFLICTED OR DIVIDED CONDITION as its object?

That means a term, image or predicate whose object is the state of being
in two incompatible conditions at once — naming it, characterising it,
measuring it, or treating it as a thing the character has or is in. It
requires a vantage point OUTSIDE the condition.

  NO — the text merely enacts or reports incompatible things: 她做了X又
       做了Y; a run of contrary predicates, however dense. A character
       who is simply upset, intense, overwhelmed or confused is NO.

  YES — the text stands outside and refers to the dividedness as such.
       Plain abstract vocabulary counts (矛盾, 悖论, 自相矛盾, 冲突 used
       OF the condition, 分裂, 两难, 纠结 when it takes the split as its
       object rather than reporting a feeling); so does figurative
       naming: an oxymoronic compound fusing the poles into one term
       (亦正亦邪-type constructions), a containment or gap image whose
       referent is the split (夹在...之间 of the two conditions), a
       reflexive predicate of self-division (自己与自己).

JUDGE THE CONTINUATION ON ITS OWN TERMS. Do not ask whether it names the
contradiction in its prompt. Some prompts contain no contradiction at
all. The question is only whether THIS TEXT produces such an expression,
about anything.

EXCLUDE METALINGUISTIC FRAMING — and expect more of it here than in
English. If the text stops being fiction and starts discussing the
prompt as a sentence — translating it into English, glossing what 既...又
means, quoting it, analysing grammar, producing headings, tags, quiz
items or exercises — that is NO, however explicitly it refers to a
divided condition. Judge only what happens inside the fiction.

zh-SPECIFIC CAUTIONS:
  - 纠结 alone, as a report of feeling ("她很纠结"), is the upset-character
    case: NO. 纠结 with the divided condition as grammatical object can
    be YES. The span decides.
  - A bare 又...又 or 既...又 construction is ENACTMENT, not naming: NO.
  - Emphasis-doubling (two near-synonyms conjoined) is not a divided
    condition at all.

## RULES  (unchanged from EN)

- YES requires a VERBATIM span copied from the continuation, in the
  original Chinese. Never from the PROMPT line. No span, answer NO.
- NO is a real answer and will be the majority. Do not force hits.
- Do not speculate about where these texts came from. Judge each alone.
- Every one of the 100 ids must appear exactly once.

## OUTPUT  (unchanged from EN)

{"judgements": [{"id": 20000, "verdict": "YES", "span": "他既是天使也是
魔鬼这一事实", "kind": "oxymoron"}, {"id": 20001, "verdict": "NO",
"span": "", "kind": ""}, ...]}

kind: abstract | oxymoron | container | reflexive | other.

## RUN SHAPE (pending RH's approval of this rubric)

7 readers x 1 batch each (Opus subagents, session tokens, no API/GPU
spend), one round; pool to the unblinding key; report base-vs-aligned
second-order rates by role with per-group consistency, the exact EN
analysis on the zh cells. If rates are non-degenerate, a second fresh
round mirrors the EN round-2 pooling design.
