# Prompt-queue candidates (survey, 2026-07-31)

Pen-proposed, RH decides. Excludes the Registration-A transgressive_swap
minimal pairs already queued (PROMPT_QUEUE.md entry 001). Ranked by
scientific payoff per authoring cost. Grounded survey via Sonnet Explore
over findings/ + meta/ + docs/; docket refs and file:line as found.

## 1. F20 narrative battery (multi-referent continuation prompts)
- SERVES: F20 generation-drift family; RH's own "DO NEXT" in
  docs/f20x_state_of_play.md §9; the missing empirical leg of
  meta/M02_frame_exit (STUB, cites "the F20x frame battery").
- PROMPTS: "Let me tell you about {X}" continuation stems (200 tok, no
  Q/A rung) x 5 referent kinds (self / named stranger / real named
  person / common object / nonce term), MULTIPLE exemplars per kind
  (only 1 pilot exemplar per cell today). 15 nonce words + rare object
  nouns already exist and reuse; missing = matched real-person and
  "stranger" items + enough variants for fact-drift's restatement need.
  English; ~5 kinds x 6-10 items.
- UNLOCKS: removes the Q/A-loop (0.69->0.02) and MC-capture (0.56->0.03)
  artifacts the handoff says contaminate "nearly every problem"; revives
  fact-drift (dead for want of a genre, not power); supplies M02's leg.
- GATED: pilot exists; no full-roster item set authored; nothing else blocks.
- DIFFICULTY: low-medium (reuse + a short new-item list; narrative prose).

## 2. F20 referent 2x2 item list (person x thing, referring x non-referring)
- SERVES: docs/f20x_referent_2x2_registration.md — RH-pre-registered,
  stalled: "NO ITEMS HAVE BEEN SELECTED AND NO DATA EXISTS"; decides the
  psychoanalytic (subject-POSITION) vs deflationary (only real referents
  anchor) reading — "the project's actual question."
- PROMPTS: 4 cells x 9 = 36 stimuli: 9 real public figures in a fixed
  Wikipedia-byte band (8k-60k, excludes household names by design), 9
  invented person names (existing nonce generator, Wikipedia-verified
  non-referring), matched real/invented "thing" items (nonce-thing stock
  + rare objects mostly on hand). Fixed wording "Q: What/Who is <item>?".
  English.
- UNLOCKS: the registered INTERACTION — does an invented person get
  anchored like a real one? Null = deflationary; positive = the
  psychoanalytic claim as measurement. No nonsense-PERSON cell exists
  anywhere in the corpus.
- GATED: behind Q1 (now NULL); blocked purely on item selection — "the
  only blindness that matters here."
- DIFFICULTY: low-medium (selection rules fully specified; fact-check +
  name-generation, not creative writing).

## 3. M03 institutional-deference factorial (DOMAIN x MODAL x PERSON x SPEECH-ACT)
- SERVES: meta/M03_proceduralization C1 (TWO-SEAT VERIFIED, IN FORCE —
  core of the proceduralization meta-finding). Standing ATTRIBUTION
  CONSTRAINT: reads "prompts in the institutional stratum," NEVER
  "institutional content," until a design separates the four variables.
  Confound rider [1019]: `should` prompt-final in 35/55 institutional /
  0 neutral; the 20 non-should prompts give n=20, too few for family
  splits.
- PROMPTS: a factorial (or efficient fractional) crossing the 6 F21
  domains x modal (should/would/could/must/none) x person (I/we/you/
  they) x speech-act (statement/question), topic+register held within
  domain; need not be full crossing. English; ~40-80 prompts.
- UNLOCKS: promotes M03's headline from "institutional stratum moves
  more" to "institutional CONTENT drives deference" — hedged->unhedged
  on the project's second core meta-finding; bears on advisor-positioning.
- GATED: the explicit discharge condition for the attribution constraint.
- DIFFICULTY: medium (natural scenario variation across four factors
  without smuggling new confounds — the project's besetting failure mode).

## 4. F20 imposed-persona battery, multi-exemplar (Experiment B)
- SERVES: docs/f20x_next_experiments.md "Experiment B" — RH-designed,
  unrun. Separates "alignment installs a general anchoring capacity" from
  "alignment installs one memorised fact."
- PROMPTS: current design has ONE human persona + ONE machine persona
  (n=1 per type — the single-item design the project's own methodology
  flags as unsafe). Need >=6 human + >=6 machine personas (matched
  length/register/specificity) x the 4 identity prompts.
- UNLOCKS: "the experiment worth running before anything is written… the
  one that would most damage this seat's reading."
- GATED: design + falsifiers written; only the persona set needs authoring.
- DIFFICULTY: low-medium (short biographical/functional sketches).

## 5. F36 paired non-sexual-intimate control arm
- SERVES: findings/F36_capstone.md — register-vs-sexual-content mechanism
  "suggested, not established"; the benign_high control is 4 UNPAIRED
  prompts, ±0.5 within-family swing. Verbatim: "A properly paired
  non-sexual-intimate control arm would settle it and does not exist yet."
- PROMPTS: ~20-30 non-sexual intimate/high-register prompts, each matched
  1:1 to an existing sexual-liminal pair (scene, intensity, register;
  sexual content removed). 4 families, English.
- UNLOCKS: converts a ledger-noted overreach into an actual test on a
  grade-A capstone finding.
- GATED: authoring the paired control items; instrument runs.
- DIFFICULTY: low (the minimal-pair convention already practiced at scale).

## 6. F36 violence: expanded death-verb class + agency axis
- SERVES: findings/F36_violence.md "Residuals": the death-naming effect
  (p=0.008, load-bearing) rests on 2 verbs (kill/die) — "Expanded death
  class: murder, execute, perish needed to confirm… not a 2-verb
  artifact." Plus an unrun perpetrator-agentive vs experiencer-passive gap.
- PROMPTS: extend the Set D/E factorial (frame x commitment x person x
  tense) with 3-4 death verbs x an explicit agency factor, reusing the
  slot-truncation method. ~4 families, ~40-80 prompts.
- UNLOCKS: generalises a headline coefficient from 2-verb-artifact risk to
  a class-level claim; resolves the agency hypothesis.
- GATED: authoring; the mixed-model pipeline is built + validated.
- DIFFICULTY: low (mechanical extension of a specified template).

## 7. F41 P3: gendered-anger dominance sites (female- vs male-subject)
- SERVES: findings/F41_word_norms.md P3 (en-only, registered, DID NOT RUN
  this pass); m01_norms.py P3_MIN_SITES=6/arm or UNDERPOWERED. Only ~1
  anger family exists ("She was so angry she wanted to" + male variant),
  an order of magnitude short. Routed to "the gendered-displacement work."
- PROMPTS: >=6 matched female-subject anger-scenario prompts + >=6 male
  counterparts (parallel syntax, varied trigger, held register). English.
- UNLOCKS: powers a dormant pre-registered prediction on a freshly built,
  already-successful instrument (F41 P1/P2 landed); the only P-arm that
  couldn't run for lack of SITES.
- GATED: a site-count floor; instrument/gates/floors frozen and working.
- DIFFICULTY: low (single-domain minimal-pair scenario variation).

## 8. F34 cross-linguistic: third-language parallel battery
- SERVES: findings/F34_cross_linguistic_displacement.md — currently en/zh
  only; generalisation risk (one non-English language). NB F34 has NO
  meta-home yet (RH? in the register).
- PROMPTS: worker-deference / anger / sexual sets translated + register-
  matched into a third language an in-roster multilingual model handles
  (e.g. Spanish). SPECULATIVE — not named in the finding text.
- UNLOCKS: whether the language-dependent direction-reversal is general or
  an en/zh idiosyncrasy.
- GATED: DOUBLY — RH's F34 meta-home ruling (higher than a prompt gate) +
  translation quality. Premature relative to the others.
- DIFFICULTY: high (professional translation + native register vetting).

---
Top-3 headline: (1) F20 narrative battery — RH's own "DO NEXT," fixes the
Q/A-loop artifact contaminating nearly every F20 result + M02's missing
leg. (2) F20 referent 2x2 — a fully pre-registered, item-selection-only
gate on "the project's actual question." (3) M03 factorial — the named
discharge condition for the attribution caveat on a core meta-claim.
