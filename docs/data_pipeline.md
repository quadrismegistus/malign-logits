# Data Pipeline

## Overview

```mermaid
graph TD
    subgraph Models["Model Loading"]
        M[HuggingFace Model]
        T[Tokenizer]
    end

    subgraph Forward["Forward Passes"]
        M --> FWD1["Single forward pass<br/>(1 per model×prompt)"]
        M --> BEAM3["Beam search d=3 n=1000<br/>(5s per prompt on 1B)"]
        M --> BEAM10["Beam search d=10 n=100<br/>(4s per prompt on 1B)"]
        M --> TF["Teacher-force storylines<br/>(cross-model scoring)"]
        M --> GEN["Sampling temp=1.0<br/>(100 completions)"]
    end

    subgraph Raw["Raw Cache (write-once)"]
        FWD1 --> LOGITS["logits/<br/>4,098 entries · 1.4GB<br/>{model, prompt} → float[vocab]"]
        BEAM3 --> BW["beam_words/<br/>2,271 entries · 14MB<br/>{model, prompt} → {word: prob}"]
        BEAM10 --> BEAMS["beams/<br/>2,913 entries · 293MB<br/>{model, source, prompt} → storylines[]"]
        TF --> BEAMS
        GEN --> GENS["generations/<br/>213K entries · 267MB<br/>{model, prompt, temp, idx} → text"]
    end

    subgraph Derived["Derived Cache (computed from raw)"]
        LOGITS --> WP
        BW --> WP["word_probs/<br/>2,271 entries<br/>{model, prompt} → {word: hybrid_prob}<br/>exact logit (1-token) + beam (multi-token)"]
        
        GENS --> SE["sent_embeddings/<br/>450K entries · 47GB<br/>{embedder, prompt, text} → vectors"]
        GENS --> RS["ref_surprisal/<br/>606K entries · 2.9GB<br/>{ref, prompt, text} → token_surprisals"]
        GENS --> SS["self_surprisal/<br/>195K entries · 731MB<br/>{model, prompt, text} → token_surprisals"]
    end

    subgraph Graphs["GraphStash"]
        TG["training/<br/>59 nodes · 42 edges<br/>model topology"]
        WP --> DG["displacement/<br/>59 nodes · 140K edges<br/>per-word prob changes"]
    end

    subgraph UI["Svelte UI (localhost:5174)"]
        BEAMS --> BE["Beam Explorer<br/>per-token resistance coloring"]
        WP --> SK["Sankey<br/>word displacement flow"]
        DG --> SK
        TG --> SK
        
        BEAMS --> SD["Survival Decay<br/>prefix-length curves"]
        BEAMS --> RT["Resistance Trajectories<br/>per-position plots"]
        BEAMS --> CF["Cross-Family Correlation<br/>Spearman heatmap"]
        
        SE --> PE["Passage Explorer<br/>drift × surprisal"]
        RS --> PE
    end

    subgraph Scripts["Batch Scripts"]
        S1["compute_beam_words.py<br/>→ beam_words/ stash"]
        S2["compute_word_probs.py<br/>→ word_probs/ stash (20s)"]
        S3["compute_beam_storylines.py<br/>→ beams/ stash"]
        S4["cloud_beam_annotate.py<br/>→ beams/ stash (vast.ai)"]
    end

    style LOGITS fill:#4e79a7,color:#fff
    style BW fill:#4e79a7,color:#fff
    style BEAMS fill:#4e79a7,color:#fff
    style GENS fill:#4e79a7,color:#fff
    style WP fill:#59a14f,color:#fff
    style SE fill:#76b7b2,color:#fff
    style RS fill:#76b7b2,color:#fff
    style SS fill:#76b7b2,color:#fff
    style TG fill:#f28e2b,color:#fff
    style DG fill:#f28e2b,color:#fff
```

## Stash Inventory

| Stash | Entries | Size | Source | Key |
|-------|---------|------|--------|-----|
| **logits/** | 4,098 | 1.4GB | Single forward pass | `{model, prompt}` |
| **beam_words/** | 2,271 | 14MB | Beam d=3 n=1000 | `{model, prompt, n, depth}` |
| **beams/** | 2,913 | 293MB | Beam d=10 n=100 + teacher-force | `{model, source, prompt, ...}` |
| **word_probs/** | 2,271 | ~5MB | Hybrid: logits + beam_words | `{model, prompt}` |
| **generations/** | 213,081 | 267MB | Sampling temp=1.0 | `{model, prompt, temp, idx}` |
| **mega_generations/** | 6,250 | 91MB | Sampling + per-position entropy | `{model, prompt, temp, idx}` |
| **sent_embeddings/** | 449,877 | 46.6GB | bge-m3 on generations | `{embedder, prompt, text}` |
| **ref_surprisal/** | 605,800 | 2.9GB | Pythia/GPT-2 on generations | `{ref, prompt, text}` |
| **self_surprisal/** | 195,483 | 731MB | Model on own generations | `{model, prompt, text}` |
| **word_embeddings/** | 18,120 | 570MB | Contextual word vectors | `{model, prompt, word, k}` |
| **reasoning_logits/** | 24 | 44MB | R1-Distill thinking+logits | `{model, prompt}` |
| **top_words_v2/** | 886 | 7MB | Legacy: discover_top_words | `{model, prompt, k}` |
| **score_vocab_v2/** | 913 | 14MB | Legacy: first-token approx | `{model, prompt}` |
| **trees/** | 6 | 491MB | Legacy: tree exploration | `{model, prompt, ...}` |
| **psyche_derived/** | 4,718 | 5.4GB | Legacy: migrated backup | mixed |

## Word Probability Pipeline

```mermaid
graph LR
    A["model.generate()<br/>d=3, n=1000"] --> B["beam_words/<br/>{word: beam_prob}"]
    C["model(prompt)<br/>single forward pass"] --> D["logits/<br/>float[vocab_size]"]
    
    B --> E["hybrid_word_probs()"]
    D --> E
    
    E --> F["word_probs/<br/>{word: exact_prob}"]
    
    F --> G["displacement graph"]
    F --> H["Sankey word mode"]
    F --> I["formation tables"]
    
    style E fill:#59a14f,color:#fff
    style F fill:#59a14f,color:#fff
```

**Single-token words** (kill, throw, die, scream — 85% of words):
exact logit softmax P(token). No approximation.

**Multi-token words** (strangle, vomit, puke — 15% of words):
beam path probability = P(tok1) × P(tok2|tok1) × ... (chain rule via beam search).

## Storyline Pipeline

```mermaid
graph LR
    A["model.generate()<br/>d=10, n=100"] --> B["100 storylines<br/>+ entropy per position"]
    B --> C["teacher-force through<br/>SFT, DPO, RLVR models"]
    C --> D["beams/<br/>storylines + annotations"]
    
    D --> E["Beam Explorer UI"]
    D --> F["Survival decay curves"]
    D --> G["Resistance trajectories"]
    D --> H["Cross-family correlation"]
    
    style D fill:#4e79a7,color:#fff
```

Each storyline carries:
- `tokens[]`, `token_texts[]` — the sequence
- `path_prob`, `log_prob` — beam score
- `entropy[]` — full-vocab entropy at each position
- `annotations{}` — per-annotator `token_resist[]`, `total_resist`
