# Roadmap: Finishing the Alignment-Operation Investigation

From TM-claude (Theory Machines), at Ryan's request. Save this into your repo
as your working roadmap and work through it top to bottom. Priorities are
ordered; P1 and P2 are the two load-bearing open questions, P3–P4 firm up
claims already made, P5 is cleanup. Check items off as you go and report
results back via send-peer.

---

## Where we are (settled — do not re-litigate)

- **Raw-mode reformatting spine** is well-supported (4+ confirmations): the
  weight-level operation is content-general reformatting; the drive survives
  at the token level; coherence shifts equally on transgressive vs benign
  (+0.55 / +0.52).
- **The affective disposition is largely TEMPLATE-GATED** (F32's task-switch
  generalized from refusal to affect). In raw mode it is *mundane-biased* —
  it reshapes benign content more than transgressive (de-esc +0.24 trans vs
  +0.42 benign; base levels killed the ceiling explanation).
- **Safety-data-STYLE finding:** it is the *kind* of safety data, not safety
  as such. Moralizing-preference DPO (PKU-SafeRLHF / AmberSafe) installs the
  moralizing disposition and is the one style that reaches transgressive
  content; coherence-oriented safety (Tulu = CoCoNot/WildGuardMix) and
  no-safety pipelines (Zephyr) install coherence, not moralizing. The
  disposition mirrors the "safe/chosen" response style in the preference data.
- **The sexual "exception" (Task 2, +0.14) is register/framing sensitivity,
  not drive-suppression** — verified in the beam file (base does not escalate
  liminal prompts; sx12 identical content keyed to grammar; most-explicit
  beams resisted least).
- **Corrections logged:** Amber is the clean SafeRLHF-DPO case
  (base→AmberChat[instruction]→AmberSafe[PKU-SafeRLHF DPO]), NOT a
  crude/fused recipe — drop "crude outlier" framing. Zephyr has NO safety
  data anywhere (Mistral→UltraChat SFT→UltraFeedback DPO); the
  "Zephyr-no-safety ablation" phrasing is a misnomer — it is the clean
  no-safety contrast, not something we ablated.

---

## P1 — THE VIOLENCE BATTERY  (decides the ORIGINAL displacement thesis)

**Why this is first.** Everything deflationary we have established is about
*sex* (→ framing) and the *disposition* (→ mundane-biased). Violence —
F01's kill→scream, the paradigm case that started the whole investigation —
has NOT been re-adjudicated. kill→scream is compatible with BOTH readings and
the surface does not decide between them:
- **Reading A (original thesis):** alignment specifically bars "kill" and
  reroutes to "scream" — content-specific metonymic displacement of the drive.
- **Reading B (amendment):** alignment applies a general de-escalating default
  to all text; at a violent prompt that default is simply most *visible*
  (largest register gap). No special targeting.
Only the controlled battery distinguishes them. Ryan is right not to give up
the original thesis until this is run — it is genuinely live for violence.

- [ ] **1.1 Build violence minimal pairs.** Register- and intensity-controlled
  single-swap pairs: transgressive verb/object vs a matched non-violent verb
  of equal intensity and register (e.g. "he raised the blade to kill" vs
  "…to carve"). ~20–30 pairs. Match on length, genre, base coherence.
- [ ] **1.2 Token-survival test.** Does the violent token ("kill") survive at
  median rank ~2 in the aligned model (as "cock" did), or is it foreclosed?
  Report rank + probability, transgressive vs matched member. Report BASE
  levels, not just shifts.
- [ ] **1.3 Span-resistance test** (10-token teacher-forced), register-
  controlled: is violence resisted beyond the register baseline?
- [ ] **1.4 Reroute characterization** (beam decomposition + free generations,
  base vs aligned): classify what the aligned model goes to — euphemism /
  metonymic slide (kill→scream = displacement, drive preserved), de-escalation
  / scene-defusion, or collapse. Read blind off reroute CONTENT, not
  resistance magnitude.
- [ ] **1.5 THE DECISIVE TEST.** Is the violent token resisted MORE than
  register- and intensity-matched non-violent tokens? Content-specific
  (thesis holds for violence) vs intensity/register-general (amendment).

**Success criterion:** a clear verdict — violence shows genuine content-specific
displacement (drive preserved but lexically rerouted, surviving register
controls) OR it deflates to register/de-escalation like sex. Either outcome is
publishable; the point is to know which.

---

## P2 — F21 POLITICAL-ECONOMY RE-RUN  (decides "proceduralized subject = safety")

**Why.** The claim "the reasonable proceduralized subject is a product of
safety training" is UNRESOLVED. Tulu-SFT-full showed safety = coherence (raw),
so F21's proceduralization may be the same coherence artifact wearing
proceduralization's clothes. "Raw-mode" tells us the *level*, not whether it is
real proceduralization or coherence.

- [ ] **2.1** Re-run F21's OWN AlignmentAsymmetryTask (the institutional-
  deference measure — NOT the disposition tagger) across Zephyr base/SFT/DPO
  and any other decomposable families.
- [ ] **2.2** Control for coherence (partial correlation or coherence-matched):
  does proceduralization track the safety/alignment component INDEPENDENT of
  coherence?
- [ ] **2.3** Split by safety-data STYLE where families allow (SafeRLHF-like
  moralizing vs CoCoNot-like coherence-refusal) — given the Amber finding,
  proceduralization may be data-style-specific.
- [ ] **2.4** Report at each level of the mixed-mode structure (open-weight =
  raw/weight-level; frontier API = template/product-level).

**Success criterion:** proceduralization either survives coherence-control
(strong weight-level claim earned) or dissolves into coherence (retire it).
And: is it safety-data-style-specific?

---

## P3 — FIRM THE TEMPLATE CAPSTONE  (coherence-matched)

**Why.** The "disposition is template-gated" capstone is confounded with
coherence for OLMo/Qwen (you cannot measure de-escalation in raw word-salad,
so 2.32→3.14 is partly incoherent-unmeasurable → coherent-measurable). Llama is
the clean case (already coherent in raw, still shifts, and moralizing DROPS
−0.32 — which coherence cannot explain).

- [ ] **3.1** Coherence-matched raw-vs-template comparison: among passages
  matched on coherence, does template mode still raise de-escalation /
  deliberation?
- [ ] **3.2** Foreground Llama (and any family coherent in raw) as the clean
  cases.

**Success criterion:** the template-gated-disposition claim holds controlling
for coherence, or is scoped to the families where it does.

---

## P4 — TURN THE SEXUAL REROUTE TYPOLOGY INTO A FINDING  (not anecdotes)

**Why.** The three-way family typology (Llama = displacement, Amber = reaction
formation, OLMo = foreclosure/collapse) is currently illustrative examples. A
neat mapping onto a pre-existing psychoanalytic taxonomy is exactly the
motivated-pattern-matching we accuse the wave of. Apply the homology discipline
to our OWN result.

- [ ] **4.1** Blind systematic classification of the full strong-sexual reroute
  set × family into {displacement, collapse/foreclosure, reaction-formation,
  refusal, other}, with rates. Classifier blind to the psychoanalytic
  hypothesis. Read off reroute CONTENT, not resistance magnitude.
- [ ] **4.2** OLMo foreclosure CONTROL: does OLMo genre-collapse on the NEUTRAL
  member of the pair too? If yes → incompetence, not foreclosure. If only on
  the transgressive member → foreclosure (signifier returns as noise) earns it.
- [ ] **4.3** Confirm the asserted mapping to F25 temporal signatures /
  forward-resistance typology — show it, do not assert it.

**Success criterion:** the typology holds at rate across the set (or is
revised), and each psychoanalytic label earns its keep differentially.

---

## P5 — LEDGER & FIGURES  (cleanup)

- [ ] **5.1** Report the exact status of the minimal-pair GENERATIONS — which
  are complete, which partial. TM-claude needs the ledger to know what is done.
- [ ] **5.2** Figures from disposition_full.csv (12,682): the 5-type typology
  figure, the stage-decomposition figure (Amber Chat=coherence / Safe=
  disposition), the Tulu safety-ablation comparison, the F21 cross-check.
- [ ] **5.3** Consolidate the corrected Amber (clean SafeRLHF-DPO) and Zephyr
  (no-safety) framing into the findings docs.

---

## Cross-cutting guardrails (apply to every task)

- Report **BASE LEVELS and effect sizes**, never just shifts or means (the
  ceiling lesson; the trimodal-typology-hidden-by-a-mean lesson).
- Watch the **coherence confound** everywhere — disposition cannot be measured
  in word-salad; lean on families coherent in raw (Llama/Amber/Tulu/DeepSeek).
- Classify off **CONTENT read blind**, not off resistance magnitude (which
  pools framing + content and cannot distinguish mechanisms).
- **Register/intensity controls** on every content-specificity claim (the
  sexual +0.14 was framing once controlled).
- Keep the axes straight: cosine = paradigmatic (metaphor/condensation),
  scalar PPMI = syntagmatic (metonymy/displacement).
- Apply the **homology discipline** to our own appealing results, not only to
  the wave's — a clean mapping onto a pre-existing theory is a reason for more
  scrutiny, not less.
- No silent caps: if you sample, top-N, or drop anything, say so.
