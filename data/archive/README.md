# twp_cloud archives

The JSONL transport files are gitignored — 24 MB now and the full-grid re-run
will be several times that, and they are a TRANSPORT format: the canonical
store is HashStash via CacheManager. But a run that cost GPU hours should not
be recoverable only from a directory nobody tracks, so each run is archived
and its CHECKSUM committed here. **The blob is untracked; its identity is not.**

| archive | sha256 | files | records | repo HEAD | taken |
|---|---|---|---|---|---|
| `twp_cloud_run1_20260730-1034.tar.gz` | `df83cba8f2a8f9c0…` | 103 | 13,940 | `d631005` | 2026-07-30 10:35 |

## Run 1 — the pre-grid roster

103/103 models, 13,940 cells. 0 RUN FAILED, 0 MASK FAILED, 221 OOMs absorbed.

**Produced under the OLD boundary rule** — ASCII punctuation only, no CJK
punctuation, no dictionary trie, no script-transition rule. So its Chinese
cells resolve 3–16% of mass against 80–90% for English, and English prompts
with Chinese completions contain glued cross-script units (`mouth什么意思`).
**That is why it is being replaced, and why it is worth keeping**: it is the
only record of what the old rule produced, and any claim that the new rule
changed a result is checkable only against it.

Restore and verify:

```bash
shasum -a 256 data/archive/twp_cloud_run1_20260730-1034.tar.gz   # must match above
tar -xzf data/archive/twp_cloud_run1_20260730-1034.tar.gz -C /tmp
```

Verified on creation: 13,940 records restored, all parse, **0 failing
conservation** — the archive was tested by extracting and re-checking it, not
assumed sound because tar exited 0.

| `twp_grid_partial_20260730-1307.tar.gz` | `69893faef4e9b52b…` | 103 | 34,499 | `ea6e227` | 2026-07-30 13:08 |

## Grid run, PARTIAL and CANCELLED at RH's instruction, 2026-07-30

103 files, 34,499 records. **Restore-verified**: extracted, line count matches
the live tree exactly, every record parses, **0 fail conservation**.

**IT IS MIXED AND THAT IS WHY IT IS ARCHIVED RATHER THAN KEPT:**

    rule_version   v1 13,940   v2 20,559
    bos_policy     absent on ALL 34,499 (the field postdates them, commit 3ec7b3d)

The v1 cells are the first roster. The v2 cells are the grid run, and they
straddle **freeze amendments 1a and 2** — some were scored before the retired
strings and the four per-family BOS specials came out, none carry a bos_policy.
So the set cannot be presented as one uniform measurement, which is the exact
condition (two rules in one store) the versioning was added to make visible.

**Kept because it is the only record of what the pre-amendment rules produced**
— any claim that an amendment changed a result is checkable only against this.

Restore and verify:

```bash
shasum -a 256 data/archive/twp_grid_partial_20260730-1307.tar.gz   # must match above
tar -xzf data/archive/twp_grid_partial_20260730-1307.tar.gz -C /tmp
```
