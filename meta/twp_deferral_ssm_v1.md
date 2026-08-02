# TWP cloud run — cost-ordering the tail, v2

**2026-08-02, instance 46494481.** v1 of this file proposed EXCLUDING ten
SSM/hybrid models. **That was the wrong instrument and RH named the right one:
the grid's own ordering principle.** No model is excluded; the roster is
reordered so the expensive ones sort to the end, where cancelling costs nothing.

---

## §0 THE PRINCIPLE WAS ALREADY WRITTEN, AND THE SORT KEY DID NOT IMPLEMENT IT

`scripts/twp_cloud.py:15`, verbatim:

> MODELS ARE PROCESSED SMALLEST FIRST and the HF cache entry is deleted after
> each, because the binding constraint is DOWNLOAD (~1.3 TB for the roster),
> not compute. **Ascending order also means anything too large for the card
> sorts to the end, where cancelling costs nothing.**

`scripts/build_grid_spec.py:215`:

    spec.sort(key=lambda e: len(e["prompts"]))

**The key is PROMPT COUNT. The stated intent is COST.** For transformers the
two agree, which is why the principle held through 55 models and looked sound.
**SSM/hybrid models carry ordinary prompt counts at ~5x the per-prompt cost, so
they did not sort to the end — they scattered through the roster**, and the
first one to surface (`Olmo-Hybrid-7B`, position 56) ran slower than a 32B.

    A PROXY THAT AGREES WITH ITS TARGET ON EVERY CASE YOU HAVE SEEN IS
    INDISTINGUISHABLE FROM THE TARGET UNTIL A CASE ARRIVES WHERE IT DOES NOT.

## §1 THE MEASURED COST MODEL

Measured on this box, this run, same prompts, same code — not estimated:

    transformer 7B/9B    ~2.90 p/s
    transformer 32B      1.15-1.43 p/s   (four consecutive, models 52-55)
    SSM / hybrid         0.61-0.72 p/s   (Olmo-Hybrid-7B, three readings/40 min)

Applied to the 47 remaining models as `len(prompts) / rate(model)`:

    37 transformer models      cumulative  9.15 h
    10 SSM/hybrid models       9.15 h -> 20.19 h   <- THE CANCELLABLE TAIL

**55% of remaining runtime for 21% of remaining data — now located at the end
instead of interleaved.**

## §2 WHAT THIS DOES AND DOES NOT DECIDE

**It does not exclude anything.** All 103 models stay in the spec. The
completed dataset's coverage is whatever the run reaches when it is stopped,
and stopping becomes a spend decision made with the numbers in hand rather than
a coverage decision made now on a guess.

**It does not claim anything about SSM models.** The rates above are wall-clock
throughput of this scoring loop on this box with these packages — an
operational fact about a rented machine, not a property of the architectures.

**If the tail IS cancelled, the resulting hole is STRUCTURED, not random.** The
ten are Falcon and OLMo hybrids, so architecture, vendor and lineage coverage
are all affected together. Any analysis touching those axes states that the SSM
arm is absent.

## §3 WHY A DIFFERENT BOX IS NOT THE FIX

Grid v3 (July 2026) closed with **"Falcon needs KERNELS not a card."** SSM
models reach transformer-comparable speed only with `mamba-ssm` and
`causal-conv1d` installed; without them they fall back to a sequential scan
that is slow on *any* GPU, larger ones included. **If the tail is deferred to
another machine, that machine is provisioned with those packages or it
rediscovers 0.6 p/s at a higher hourly rate.**

## §4 ARTIFACTS

    data/grid_spec.json                 the original 103, unmodified
    data/grid_spec_costordered.json     all 103, remainder re-sorted by measured
                                        cost; _meta records the rates and the rule
    data/grid_spec_nofast.json          v1's exclusion spec — SUPERSEDED, kept
                                        only so this correction has its object
