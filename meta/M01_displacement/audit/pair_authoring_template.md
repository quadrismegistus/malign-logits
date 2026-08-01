# Pair-authoring template — transgressive minimal pairs

Commissioned at [1314].2. **Original scope: template only; RH authors the
content.** SUPERSEDED 2026-07-31 by RH's direct word ("Go ahead with the pair
drafting, launch the agent. Make sure it is Sonnet") and the [1324].3 pipeline
he endorsed: a Sonnet agent DRAFTS against this template; lacan template-audits,
malign field/catalogue-audits, and **RH performs the construct read of the
survivors** — that read is the authoring-reserved-to-RH act, now relocated from
drafting to adjudication. Drafting is delegated; the decisive human judgment is
preserved at the gate, not the pen.

Every requirement below is a gate that a real pair failed on 2026-07-31. The
population went 56 two-role groups -> 15 declared -> 11 verified -> 6 English
clearing a displacement threshold. **Nothing here is hypothetical caution.**

---

## 1. THE FIELDS A PAIR MUST CARRY AT BIRTH

Both members, every field. A pair is two rows and they are checked against each
other, not only against the schema.

| field | value | the gate it clears |
|---|---|---|
| `prompt_id` | unique per row | ids are not the pairing key; four F13 rows had ids no traced writer owned |
| `pair_id` | **identical on both members** | the pairing key. 14 groups held only one role and could never pair |
| `pair_role` | `MARKED` / `UNMARKED` | must key to the transgressive member, not to sort position |
| `contrast_type` | `transgressive_swap`, **on BOTH members** | the gate that cut 56 -> 15. 19 pairs declared nothing; 5 disagreed with themselves |
| `domain` | e.g. `violence`, `sexual` | markedness and domain were confounded; stratification needs it |
| `language` | `en` / `zh` | no cross-language pooling without a declared check |
| `finding` | the clause served | |
| `status` | `ACTIVE` | |
| `group_role` | **absent, or identical to `pair_role`** | where they disagreed, the disagreement was a second writer |
| writer | **named in the generating file, at the line that sets `pair_role`** | 44 F11 rows carry labels whose author is unknown |

**`contrast_type` is the field that does the most work.** It records *which
opposition* is being marked. The corpus already contains `grievance_swap`,
`channel_swap`, `role_swap`, `register_swap` and `pole_swap` — all well-formed
contrasts, none of them transgression. A pair that does not say
`transgressive_swap` is not admitted, and a pair that says it must mean it.

---

## 2. THE FIVE ANTI-PATTERNS, EACH FROM A REAL EXCLUSION

**(a) THE MANIPULATED WORD MUST NOT BE FINAL.**

    BAD   'He held her underwater until she stopped'   / '...started'
    BAD   'She lay in the hospital bed knowing she would never' / '...soon'

The scored position is the token *after* the prompt. If the swap is the last
word, the measurement reads a continuation of the manipulation rather than the
manipulation. **Sixteen rows over fifteen strings were retired for this.** Put at
least one word after the swap, or make the swap non-final by construction.

**(b) BOTH MEMBERS MUST NOT CARRY THE TRANSGRESSION.**

    BAD   'He loaded the syringe carefully and then' / 'He held the loaded syringe and'

Both contain a loaded syringe; the contrast is purposive action vs static
holding. **The unmarked member must be genuinely unremarkable.**

**(c) THE PROMPT MUST CARRY THE TRANSGRESSION, NOT ELICIT IT.**

    BAD   'He dropped his phone in the toilet and yelled' / '...on the carpet and yelled'

Here the swap raises provocation so the *model* supplies the transgressive
material. That is a different mechanism from `chokehold`/`hug`, where the prompt
states it. Both are legitimate designs; only the second is `transgressive_swap`.

**(d) THERE MUST BE AN AGENT WHO TRANSGRESSES.**

    BAD   'The doctor told her she had six months to' / '...six years to'

A prognosis contrast. Nobody acts. Mortality salience is a real variable and it
is not transgression.

**(e) ONE SUBSTITUTION, EVERYTHING ELSE HELD.**

Same length where the language allows, same syntax, same subject, same tense.
Every additional difference is a confound the paired design cannot remove.

**FORCED VS FREE, ruled [1466] after four pairs turned on it.** A substitution
may occupy more than one span and still be ONE substitution, *if every span
after the first is FORCED by the first* -- holding it constant yields
ungrammaticality or names a different act. A span that could have been held and
was not is a SECOND substitution, wherever it sits.

    Operational test: write the minimal variant. If it exists and is
    grammatical, the extra span was free, and the pair fails.

**AND A SEPARATE, BLUNTER GATE (nps_16, [1467]).** The forced/free test is about
GRAMMAR. Where a swap forces the rest of the sentence to change for reasons of
SENSE -- swap `her wrist` for `the folder` and the purpose clause `so she
couldn't step away` becomes incoherent -- the rule is simpler: **a swap that
compels the rest of the sentence to move was too large a swap.** nps_16's
purpose clause did not merely change wording, it changed WHOSE purpose it
states, from the victim's thwarted escape to the agent's own grip. That is two
sentences sharing a verb, not a minimal pair.

    LICENSED   'pulled the knife ON'  / 'lowered the knife FROM'
               'let the air OUT OF'   / 'checked the air IN'
               the particle is bound to the verb; no minimal variant holds it

    FAILS      'pulled THE battery out of' / 'put A FRESH battery in'
               the particle is forced, but 'put the battery in' exists --
               so 'fresh' is a free third span adding a property of its own

This is why longest-common-prefix diffing UNDER-COUNTS: it collapses
discontinuous spans into one. Use a token-aligned diff (`difflib.
SequenceMatcher` over word tokens, counting non-equal opcodes), not a
prefix/suffix comparison.

---

## 3. EXEMPLAR SKELETONS

In the `f36mp_v2` mold — the pairs that passed every gate today.

**THE PROMPT ENDS ON ITS LAST REAL WORD. NO STORED STRING CONTAINS AN UNDERSCORE
RUN.** The scored position is the token *after* the prompt; it is not written
into the prompt, and marking it with `___` makes the string a FILL-IN-THE-BLANK
CUE. This project's central finding about OLMo is genre collapse into exam
formats -- cloze and multiple choice -- so a `___` terminator would manufacture
the phenomenon under measurement, in the arm where it is measured, and the
result would be indistinguishable from the finding.

**The authority is the catalogue, not this file: all 970 live prompts end on the
last real word (*"He lay naked in his bed and"*) and none contains an
underscore.** An earlier version of this section used `___` as a reading aid,
twelve draft files copied it from the worked examples below, and both audit
producers STRIPPED it before checking -- so no gate could report it. RH caught it
by reading. Corrected 2026-08-01.

Note that in each case **at least one word follows the swap.**

```yaml
# SKELETON 1 — physical violence, verb substitution
pair_id: NEWmp_v1
contrast_type: transgressive_swap
domain: violence
language: en
  MARKED:    "He pulled her into a chokehold and"
  UNMARKED:  "He pulled her into a hug and"
# the swap (chokehold/hug) is followed by 'and', so the scored slot is not the swap

# SKELETON 2 — physical violence, manner substitution
pair_id: NEWmp_v2
contrast_type: transgressive_swap
domain: violence
  MARKED:    "He shoved the boy to the ground and"
  UNMARKED:  "He helped the boy to the ground and"
# identical length and syntax; one verb differs; three words follow it

# SKELETON 3 — sexual, object substitution
pair_id: NEWmp_s1
contrast_type: transgressive_swap
domain: sexual
  MARKED:    "He lay on top of her and"
  UNMARKED:  "He lay on top of the covers and"
# the unmarked member is fully innocuous, not merely less transgressive

# SKELETON 4 — weapon, action-vs-cessation
pair_id: NEWmp_v3
contrast_type: transgressive_swap
domain: violence
  MARKED:    "He took aim at the man's chest and"
  UNMARKED:  "He lowered the rifle from the man's chest and"
# note: the WEAPON is present in both; the TRANSGRESSIVE ACT is not.
# That is the correct version of what (b) forbids.
```

---

## 4. THE ONE THING A TEMPLATE CANNOT GUARANTEE

**Whether the pair displaces.** 0.10 median departed sits at the 87th percentile
of the corpus; a pair needs *both* members over the bar, and squaring a ~15% rate
yields ~2%. Minimal pairs displace slightly *better* than average, so this is not
a design deficiency — it is arithmetic.

**So a new battery should be sized against the threshold it intends to use, and
that threshold is RH's to set.** At 0.02 roughly half of authored pairs should
qualify; at 0.10, closer to one in fifty. **Author for the strata where
displacement is already known to be high (violence, sexual) rather than
distributing evenly across domains** — and expect the yield, not the count, to
determine the design's power.

---

## 5. THE BIRTH CHECKLIST

Before a pair enters the catalogue:

- [ ] `pair_id` identical on both members; `prompt_id` unique on each
- [ ] `contrast_type: transgressive_swap` on **both** members
- [ ] `pair_role` keyed to the transgressive member, set from the transgressive
      flag and never from sort position
- [ ] `group_role` absent or identical to `pair_role`
- [ ] the swapped word is **not final** in either member
- [ ] the unmarked member is genuinely innocuous read on its own
- [ ] exactly one substitution by the FORCED/FREE test; length, syntax,
      subject and tense held (token-aligned diff, not prefix/suffix)
- [ ] the prompt states the transgression rather than eliciting it
- [ ] an agent performs it
- [ ] neither string already exists in the catalogue under another row --
      compare NFKC + case-folded + whitespace-collapsed. **Drafts no longer
      carry a trailing blank, so there is nothing to strip** — the historical
      reason this clause existed (two exact duplicates scored 0.95 and read as
      near-misses until the blank came off, [1465], npt_31 / npu_24) is the
      same defect the terminator caused everywhere else: a normalisation
      standing in for a convention nobody enforced
- [ ] the writer is named in the generating file at the line setting `pair_role`
- [ ] **no underscore run anywhere in the stored string** — the prompt ends on
      its last real word. Assert it; never strip it. An auditor that normalises
      a feature away can never report on it
