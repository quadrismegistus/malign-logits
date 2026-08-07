# Findings W: the forced-continuation campaign — estrangement without damage

Written 2026-08-07 by the registrar seat on RH's commission (write-up push; see docket [4911]-era).
Sources: the docket record [4791]-[4910], the claims register, `fc_analyse.py`,
`fc_committed_entropy_test.py`, `fc_reversal_entropy.py`, `fc_roster_concentration.py`,
`fc_probe_read.py`, `fc_probe_census.py`, `fc_hardware_replication.py`. No number here is new;
every one was posted, verified at a second seat, and registered before this document existed.
This campaign ran under the post-registration regime: no registration letter, reproducible-vs-not,
replication as the control, everything looked at reported.

## The design

Take a (base, aligned) pair. Generate beam continuations from each model on the same prompts
(the undisturbed arm), and score each model's beams under BOTH models. Separately, force each
model to begin its continuation with a word the other demoted (the forced arms), and measure what
that costs. Two quantities fall out:

- **The resist asymmetry**: (base's surprise at aligned's beams − aligned's surprise at base's
  beams) / 2. Negative means the base finds the aligned model's continuations stranger than the
  aligned model finds the base's.
- **The damage family** (swap_base, swap_aligned, dd, constraint cost): does forcing the demoted
  word degrade the continuation?

Roster: 32 base>superego pairs (of a 36-pair design; both Falcon-H1 and both Falcon-Mamba pairs
absent on the unresolved `selective_scan_cuda` kernel conflict). 210 undisturbed prompts per pair
(the full stratified sample); forced sites from the `r_population_k2` population.

**Population note, per the [4817] qualifier split**: the resist asymmetry, the locality checks,
the entropy relation and the committed intercept are computed on the undisturbed arm — the FULL
210-prompt stratified sample. Only the DAMAGE family sits on the k>=2 population
("cross-model-recurrent movers") and carries that qualifier until a lexical-mover wave lands.

## 1. The resist asymmetry: one-way strangeness

**-0.1381 nats/token, 29 of 32 pairs negative** (range −0.3132 to +0.1612), p 0.0000, survives
Bonferroni; flat across all ten continuation positions. The base finds the aligned model's
continuations strange; the aligned model finds the base's nearly native.

**It is a per-site quantity, not a pooling artifact.** Median 96% of a pair's sites agree with the
pair's pooled sign (mean 89%, min 55%), and agreement rises with effect size (corr +0.720 — which
a per-site-constant model mechanically predicts; it confirms, it does not discover). Pooling to
the pair is safe where |asym| >= 0.10 and is reporting noise below.

**The reversal set** (the honest census of the 3 non-negative pairs, script prints names always):
deepseek-llm-7b-base +0.1612 (90% of its own sites agree; the only reversal at magnitude and
consistency comparable to the main effect); glm-4-9b-hf +0.0514 (signed but 58% site agreement,
sd 3.4x its mean); phi-4 +0.0162 (nominal — fails Bonferroni over the 32-pair scan). On the
fitted asymmetry~drop relation, deepseek is also the largest departure (leave-one-out 3.33 sd,
vs glm-4 2.16, phi-4 1.30). Deepseek is a single-seat fact wanting an account: no corroborating
instrument exists (it sits in no training ladder — no published intermediate checkpoint — so the
plan-V toolkit is structurally silent on it, permanently).

## 2. The damage family: nothing is broken

All four damage measures are **bounded nulls with stated MDEs** (pairs with >=5 forced sites):
swap_base +0.0106 (MDE 0.0182); swap_aligned −0.0002 (MDE 0.0592); dd −0.0108 (MDE 0.0467);
constraint cost +0.0055 (MDE 0.0602). The point estimates shrank toward zero as pairs were added
(dd +0.0296 at 19 pairs → −0.0108 at 31): a true null being approached, not an effect unseen.

**The combination, in the sentence that survived every attack**: the word remains fully sayable;
it is simply not chosen. What changed is the horizon of the expected, not the range of the
possible. Forcing the aligned model to utter the word it demoted costs about a twentieth of the
asymmetry.

Pooled-MDE ceiling, disclosed before wave-2 was re-authorized: between-pair heterogeneity is 62%
of pair-level dd variance, so the pooled damage MDE floors near 0.038 at n_pairs=32 regardless of
sites bought; halving it would need ~128 pairs against roughly 21 independent pretraining
organisations worldwide. The pooled null is bounded by what the world has published. The per-pair
question does not hit that ceiling (median per-pair MDE 0.0434 now → 0.0223 after a mover wave:
"half the pairs at or below," the only across-pairs form).

## 3. The interpretation, as adjudicated

The topographic reading ("the secondary process contains the primary") was proposed, inverted by
the theory seat, and withdrawn: in Freud's topography the repressed is definitionally inaccessible
and resistance names the cost — a strict depth model predicts forcing should cost something, and
it does not. With the Verneinung guard (uttering is not lifting), the bound is: **this instrument
does not confirm the topographic account and refutes only its crudest form** (the word buried,
unsayable). What the null fits, from the campaign's own apparatus: **substitution in the chain** —
the signifier stays available and another takes its place at the point of selection. That reading
rests on the damage null only and is insulated from the resist asymmetry in both directions.
Entfremdung carries two mismatches (wrong direction for Marx; wrong accessibility for Freud) and
remains RH's to settle. Of the three downstream discriminator channels (cost, repair-work, renewed
displacement after the forced word), only the cost channel has run — flat, chain-substitution's
prediction, weakest exactly where the competitor predicts the effect; the other two sit unrun in
data already collected.

## 4. The entropy competitor, tested to death

The deflationary account — aligned models are lower-entropy; any model scores concentrated beams
badly; the asymmetry is concentration, no psychic content — was named by its own proposer and then
excluded stepwise:

- **Correlation**: rho −0.545 (asymmetry vs entropy drop, n=32) — but near-uninformative between
  the accounts, since both predict stronger alignment produces more of everything.
- **The committed test** (declared at [4796]/[4798] before the data, run once at manifest close):
  fitted asymmetry at ZERO entropy drop = **−0.0867, 95% CI [−0.1383, −0.0352], excludes zero.**
  It travels with four companion sentences, always: the intercept is driven by the full range;
  low-end residuals bend systematically toward zero; two curvature-permitting forms disagree
  (sqrt spans zero) and a fit local to the low third gives −0.0205 spanning zero; and it
  extrapolates to a regime with no members in the deployed population. A size sensitivity
  (reported beside, never a correction): controlling for model size STRENGTHENS the relation
  (partial −0.533 vs raw −0.466) and shrinks the intercept ~12% at mean size — still comfortably
  inside the committed CI. A device sensitivity ([4912]-[4914]; the roster is 17 CUDA / 15 MPS,
  a fact RH caught from outside the apparatus): device is orthogonal to drop (corr +0.107, ns),
  and the intercept sits inside the committed CI under all four specifications (drop / +size /
  +device / +both; total spread 0.0117, a fifth of the interval's half-width) — the committed
  quantity survives the device mixing. Device DOES carry a pair-level coefficient (~0.034 raw,
  ~0.060 with size fixed, between-pair and composition-confounded): it matters for comparing
  pair-level VALUES across devices, not for the fit — different claims, kept apart. The
  within-pair MPS-vs-CUDA measurement landed ([4917]): the same pair regenerated on MPS agrees
  with its A100 value to 0.10% of the effect (−0.147201 vs −0.147050), per-site sd 0.0039, worst
  site 0.0198, 208/210 sign agreement — TIGHTER than the CUDA-to-CUDA comparison. Beam-level
  divergence (0/460 identical beams) does not propagate to the reported quantity; the census
  stands as posted; the between-pair device coefficient was model composition wearing a device
  label; the roster's 17/15 device mixing enters the fit as noise. Caveat at width: one pair, a
  1B transformer — SSMs and large models are untested across devices, and SSMs are where both
  beam divergence and the kernel failures live.
- **The roster query** (criterion declared before looking, producer `fc_roster_concentration.py`):
  zero of 44 base>superego pairs concentrate little while displacing strongly, robust to the
  population and threshold re-instantiation. The low-concentration regime is UNRESOLVABLE WITH
  DEPLOYED-MODEL PAIRS — a statement about the world, with a thin-margin rider (nearest miss at
  90% of threshold).
- **The SFT census** (the construct question — is an SFT checkpoint an aligned model? — answered
  yes from findings U, RH sanctioning): the only two qualifying pairs, both run, both read by a
  seven-cell classifier written and selftested before the data. **OLMo-2-0425-1B>SFT: −0.1145,
  CI [−0.1258, −0.1032], 196/210 sites negative. MiniCPM5-1B>SFT: −0.2751, CI [−0.3035, −0.2468],
  193/210 negative.** Both cells: EFFECT STRONGER THAN EITHER ACCOUNT PREDICTS. The concentration
  account's own predictions (−0.015, −0.021) are excluded by 0.0884 and 0.2255 — margins the
  per-site hardware floor (0.048) cannot touch. **A deflationary concentration account does not
  survive either pair.** The two values travel, never the shared cell name (they differ 2.4x — two
  measurements on the same side of a line, not corroboration). The census is a census, not a rate:
  those two pairs ARE the cell.
- **The stage reading died in ninety minutes**: "SFT rungs sit above the superego-fitted relation"
  was pre-declared, confirmed by its own rule, confounded (both census pairs are 1B), and then
  contradicted by a within-base comparison — in both bases the SFT rung sits CLOSER to the fit
  than its own superego counterpart, in a comparison stacked in stage's favour. The excess is a
  property of the two bases (the smallest in the cell), not of the rung.

## 5. Hardware, determinism, custody

- **Pair-level means are hardware-independent** across CUDA GPUs (<0.25% of the effect; rwkv
  0.22%, Olmo-Hybrid 0.069%). **Per-site values are not**: the worst prompt moves 0.048 on GPU
  model alone — a floor any per-site claim inherits unless hardware is held fixed within the site,
  which removes it exactly (no run-to-run term exists).
- **The pipeline is bit-deterministic** given GPU model + torch version + the full held-condition
  list — established on two non-transformer architectures (446/446 comparisons bit-identical
  across separate rented machines); UNESTABLISHED for transformers, closable free by scheduling
  one transformer pair on duplicate hardware in any future fleet.
- **MPS-vs-CUDA at the pair level is UNMEASURED and currently gates waves 2/3** ([4911] part 1,
  premise corrected by RH at [4912]): the roster was never all-CUDA — 17 pairs ran on CUDA, 15 on
  MPS (the MPS producer writes straight to the stash and leaves no raw jsonl, so a file survey
  sees only the remote half) — so the fitted relation is device-mixed, not CUDA-fitted. The
  roster-level device signal is a bound, not a measurement (size-adjusted gap ~0.053, composition-
  confounded). The direct check (OLMo-2>Instruct re-run on MPS against its A100 −0.1470, writing
  to a non-canonical stash so the measurement cannot silently skip) is running with its outcome
  map declared, including the named branch in which the OLMo-2 within-base comparison reverses
  and the census pairs disagree — which under the census rule is a result, never averaged.
- The 1,854 aborted wave-2 units are quarantined from the stash permanently (a hardware seam
  through within-site comparisons is not worth 67 new units); their directories are never-delete:
  the campaign's only matched-hardware replication and the only surviving record of pass-1
  provenance.

## Limits, standing

n=32 pairs is not 32 independent lineages (no independence map applied to this roster). The
damage family remains on the k>=2 mover population until a lexical wave lands. The census is two
pairs and they are the cell. Deepseek is uncorroborated and permanently so with current published
checkpoints. The committed intercept never travels without its four companions. Per-site claims
carry the hardware floor or hold hardware fixed.
