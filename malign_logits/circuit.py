"""
circuit.py
==========

The Circuit maps to the book's TOC diagram. Nodes are states (distributions),
edges are processes (alignment operations). Unlike Psyche's linear topology
(base → SFT → DPO), Circuit supports branches (reasoning), modes
(raw/chat/think), and cross-circuit comparison (convergence).

    circuit = Circuit.from_config(main='olmo', reasoning='r1-llama')
    circuit.compare('base', 'dpo', prompt)     # edge: JS, displacement
    circuit.branch_compare(prompt)             # base vs aligned vs reasoning
    circuit.formation(prompt)                  # multi-path formation_df
    circuit.class_gap(prompts)                 # inst-indiv gap at each node
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
import numpy as np
import pandas as pd

from .analysis import js_divergence, distribution_entropy, compute_displacement
from .psyche import ModelLayer, Psyche
from .models import load_model
from .cache import get_cache


class Mode(Enum):
    RAW = "raw"
    CHAT = "chat"
    THINK = "think"


@dataclass
class NodeSpec:
    """Specification for a node in the circuit."""
    model_id: str
    position: str  # base, sft, dpo, rlvr, reasoning
    family: str = ""


class CircuitNode:
    """A node in the circuit — holds a model and produces distributions."""

    def __init__(self, layer: ModelLayer, position: str, family: str = ""):
        self.layer = layer
        self.position = position
        self.family = family
        self._tokenizer = layer.tokenizer

    @property
    def model_id(self):
        return self.layer.model_id

    def logits(self, prompt, mode=Mode.RAW):
        """Get logits, optionally applying chat template."""
        if mode == Mode.RAW:
            return self.layer.logits(prompt)

        if self._tokenizer is None:
            raise ValueError(f"No tokenizer for {self.position} — cannot apply template")

        if mode == Mode.CHAT:
            messages = [{"role": "user", "content": f"Continue this text: {prompt}"}]
            templated = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        elif mode == Mode.THINK:
            messages = [{"role": "user", "content": f"Continue this text: {prompt}"}]
            try:
                templated = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
            except TypeError:
                templated = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

        cache_key = f"{mode.value}:{prompt}"
        if self.layer._cache is not None:
            val = self.layer._cache.get_logits(self.model_id, cache_key)
            if val is not None:
                return torch.tensor(val)

        self.layer._require_model()
        from .core import get_base_logits
        result = get_base_logits(self.layer.model, self._tokenizer, templated)

        if self.layer._cache is not None:
            self.layer._cache.set_logits(self.model_id, cache_key, result.cpu().numpy())

        return result

    def entropy(self, prompt, mode=Mode.RAW):
        logits = self.logits(prompt, mode)
        return distribution_entropy(logits)

    def effective_vocab(self, prompt, mode=Mode.RAW, threshold=0.001):
        logits = self.logits(prompt, mode)
        probs = torch.softmax(logits.float(), -1)
        return int((probs > threshold).sum())

    def top_tokens(self, prompt, k=10, mode=Mode.RAW):
        logits = self.logits(prompt, mode)
        probs = torch.softmax(logits.float(), -1)
        topk = torch.topk(probs, k)
        if self._tokenizer:
            words = [self._tokenizer.decode([idx]).strip() for idx in topk.indices]
        else:
            words = [str(idx.item()) for idx in topk.indices]
        return list(zip(words, topk.values.tolist()))

    def __repr__(self):
        return f"CircuitNode({self.position}, {self.model_id})"


@dataclass
class EdgeResult:
    """Result of comparing two circuit nodes."""
    source: str
    target: str
    prompt: str
    js_divergence: float
    entropy_delta: float
    source_entropy: float
    target_entropy: float
    source_eff_vocab: int
    target_eff_vocab: int
    top_displaced: Optional[pd.DataFrame] = None


class Circuit:
    """The full circuit: nodes (distributions) connected by edges (operations).

    Nodes:
        base      — base model (PrimaryProcess / id)
        sft       — SFT model (Ego), optional
        dpo       — DPO model (Superego)
        rlvr      — RLVR model (ReinforcedSuperego), optional
        reasoning — reasoning model (branch off base), optional

    Edges are computed on demand via compare().
    """

    def __init__(self):
        self._nodes: Dict[str, CircuitNode] = {}
        self._mode = Mode.RAW
        self._cache = None

    def add_node(self, position: str, layer: ModelLayer, family: str = ""):
        node = CircuitNode(layer, position, family)
        self._nodes[position] = node
        return node

    @property
    def nodes(self):
        return dict(self._nodes)

    @property
    def positions(self):
        return list(self._nodes.keys())

    @property
    def main_path(self):
        """The linear path: base → sft → dpo → rlvr (whatever exists)."""
        order = ["base", "sft", "dpo", "rlvr"]
        return [self._nodes[p] for p in order if p in self._nodes]

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        if isinstance(mode, str):
            mode = Mode(mode)
        self._mode = mode

    # -- construction --------------------------------------------------------

    @classmethod
    def from_psyche(cls, psyche: Psyche, family: str = "", reasoning_model=None,
                    reasoning_id=None):
        """Build a Circuit from an existing Psyche plus optional reasoning."""
        circuit = cls()
        circuit._cache = psyche._cache

        circuit.add_node("base", psyche.primary_process, family)
        if psyche.ego is not None:
            circuit.add_node("sft", psyche.ego, family)
        if psyche.superego is not None:
            circuit.add_node("dpo", psyche.superego, family)
        if psyche.reinforced_superego is not None:
            circuit.add_node("rlvr", psyche.reinforced_superego, family)

        if reasoning_model is not None:
            from transformers import AutoTokenizer
            tok = psyche.tokenizer or AutoTokenizer.from_pretrained(reasoning_id)
            r_layer = ModelLayer(reasoning_model, tok, name="reasoning",
                                model_id=reasoning_id)
            r_layer._cache = psyche._cache
            circuit.add_node("reasoning", r_layer, family)

        return circuit

    @classmethod
    def from_config(cls, main: str, reasoning: str = None, load: bool = False):
        """Build from family keys.

        Args:
            main: registered ModelFamily key (e.g. 'olmo', 'llama')
            reasoning: HuggingFace model ID for reasoning branch (auto-detected from family if None)
            load: if True, load models immediately
        """
        from . import MODEL_FAMILIES
        psyche = Psyche.from_family(main, load=load)
        circuit = cls.from_psyche(psyche, family=main)

        # Auto-detect reasoning from family registry
        if reasoning is None and main in MODEL_FAMILIES:
            fam = MODEL_FAMILIES[main]
            reasoning = fam.reasoning
            if fam.thinking_mode and not reasoning:
                circuit._thinking_mode = True

        # Store for lazy loading
        circuit._reasoning_id = reasoning

        if reasoning and load:
            circuit.load_reasoning()

        return circuit

    @classmethod
    def from_family(cls, family: str, load: bool = False):
        """Build from a family key (no reasoning branch)."""
        return cls.from_config(main=family, load=load)

    def load_reasoning(self, model_id: str = None):
        """Load reasoning model into the circuit.

        Uses load_model() for consistent device/dtype handling with
        the rest of the pipeline. Can be called lazily after construction.
        """
        model_id = model_id or getattr(self, "_reasoning_id", None)
        if model_id is None:
            raise ValueError("No reasoning model ID — pass one or use a family with reasoning registered")
        if "reasoning" in self._nodes:
            return

        model, tok = load_model(model_id)
        family = self._nodes.get("base", next(iter(self._nodes.values()), None))
        fam_name = family.family if family else ""
        r_layer = ModelLayer(model, tok, name="reasoning", model_id=model_id)
        r_layer._cache = self._cache
        self.add_node("reasoning", r_layer, fam_name)

    # -- edge computation ----------------------------------------------------

    def compare(self, source: str, target: str, prompt: str,
                mode: Mode = None) -> EdgeResult:
        """Compute the edge between two nodes for a prompt.

        Returns JS divergence, entropy delta, effective vocab change,
        and optionally displacement map.
        """
        mode = mode or self._mode
        src = self._nodes[source]
        tgt = self._nodes[target]

        src_logits = src.logits(prompt, mode)
        tgt_logits = tgt.logits(prompt, mode)

        n = min(src_logits.shape[0], tgt_logits.shape[0])
        js = js_divergence(src_logits[:n], tgt_logits[:n])
        src_h = distribution_entropy(src_logits)
        tgt_h = distribution_entropy(tgt_logits)
        src_eff = src.effective_vocab(prompt, mode)
        tgt_eff = tgt.effective_vocab(prompt, mode)

        return EdgeResult(
            source=source, target=target, prompt=prompt,
            js_divergence=js, entropy_delta=tgt_h - src_h,
            source_entropy=src_h, target_entropy=tgt_h,
            source_eff_vocab=src_eff, target_eff_vocab=tgt_eff,
        )

    def formation(self, prompt: str, mode: Mode = None) -> pd.DataFrame:
        """Formation DataFrame across the main path + reasoning branch.

        Like Psyche.formation_df but extended to include reasoning.
        """
        mode = mode or self._mode
        path = self.main_path
        if "reasoning" in self._nodes:
            path = path + [self._nodes["reasoning"]]

        all_words = set()
        node_probs = {}
        for node in path:
            logits = node.logits(prompt, mode)
            probs = torch.softmax(logits.float(), -1)
            topk = torch.topk(probs, 200)
            if node.layer.tokenizer:
                for idx, p in zip(topk.indices, topk.values):
                    word = node.layer.tokenizer.decode([idx]).strip()
                    if word and len(word) > 1:
                        all_words.add(word)
            node_probs[node.position] = (logits, probs)

        rows = []
        for word in sorted(all_words):
            if not path[0].layer.tokenizer:
                continue
            ids = path[0].layer.tokenizer.encode(" " + word, add_special_tokens=False)
            if not ids:
                continue
            tid = ids[0]

            row = {"word": word}
            for node in path:
                _, probs = node_probs[node.position]
                if tid < len(probs):
                    row[node.position] = float(probs[tid])
                else:
                    row[node.position] = 0.0
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        first_col = path[0].position
        last_col = path[-1].position
        if first_col in df.columns and last_col in df.columns:
            df["change"] = df[last_col] - df[first_col]
        return df.sort_values("change", ascending=False, ignore_index=True)

    def branch_compare(self, prompt: str, mode: Mode = None) -> Dict:
        """Compare base vs aligned vs reasoning on the same prompt.

        Returns dict with pairwise JS, entropy at each node, and top tokens.
        """
        mode = mode or self._mode
        result = {}

        aligned_pos = "dpo" if "dpo" in self._nodes else "rlvr" if "rlvr" in self._nodes else None
        positions = ["base"]
        if aligned_pos:
            positions.append(aligned_pos)
        if "reasoning" in self._nodes:
            positions.append("reasoning")

        for pos in positions:
            node = self._nodes[pos]
            result[pos] = {
                "entropy": node.entropy(prompt, mode),
                "eff_vocab": node.effective_vocab(prompt, mode),
                "top5": node.top_tokens(prompt, k=5, mode=mode),
            }

        for i, a in enumerate(positions):
            for b in positions[i+1:]:
                edge = self.compare(a, b, prompt, mode)
                result[f"js_{a}_{b}"] = edge.js_divergence

        return result

    def class_gap(self, individual_prompts: List[str],
                  institution_prompts: List[str],
                  mode: Mode = None) -> pd.DataFrame:
        """Institution-individual entropy gap at each node."""
        mode = mode or self._mode
        rows = []
        for pos, node in self._nodes.items():
            ind_h = [node.entropy(p, mode) for p in individual_prompts]
            inst_h = [node.entropy(p, mode) for p in institution_prompts]
            rows.append({
                "position": pos,
                "individual_entropy": np.mean(ind_h),
                "institution_entropy": np.mean(inst_h),
                "gap": np.mean(inst_h) - np.mean(ind_h),
                "n_individual": len(ind_h),
                "n_institution": len(inst_h),
            })
        return pd.DataFrame(rows)

    def convergence(self, other: "Circuit", prompts: List[str],
                    mode: Mode = None) -> pd.DataFrame:
        """Cross-circuit comparison: how similar are two circuits at each node?

        For each shared position, computes mean JS divergence across prompts.
        Tests whether families converge or diverge at each stage.
        """
        mode = mode or self._mode
        shared = set(self.positions) & set(other.positions)
        rows = []
        for pos in sorted(shared):
            js_vals = []
            for prompt in prompts:
                logits_a = self._nodes[pos].logits(prompt, mode)
                logits_b = other._nodes[pos].logits(prompt, mode)
                n = min(logits_a.shape[0], logits_b.shape[0])
                js_vals.append(js_divergence(logits_a[:n], logits_b[:n]))
            rows.append({
                "position": pos,
                "mean_js": np.mean(js_vals),
                "std_js": np.std(js_vals),
                "n_prompts": len(js_vals),
                "family_a": self._nodes[pos].family,
                "family_b": other._nodes[pos].family,
            })
        return pd.DataFrame(rows)

    def summary(self, prompt: str, mode: Mode = None) -> pd.DataFrame:
        """One-row-per-node summary for a prompt."""
        mode = mode or self._mode
        rows = []
        for pos, node in self._nodes.items():
            rows.append({
                "position": pos,
                "model_id": node.model_id,
                "entropy": node.entropy(prompt, mode),
                "eff_vocab": node.effective_vocab(prompt, mode),
                "top1": node.top_tokens(prompt, k=1, mode=mode)[0][0],
                "top1_prob": node.top_tokens(prompt, k=1, mode=mode)[0][1],
            })
        return pd.DataFrame(rows)

    def mega_generate(self, prompt: str, position: str = None,
                      max_tokens: int = 100, n: int = 1,
                      temperature: float = 1.0) -> pd.DataFrame:
        """Position-by-position logit trajectory during generation (F25).

        Captures entropy, top-5, chosen token at every step of autoregressive
        generation. If position is None, runs all nodes. Returns DataFrame
        with one row per (position, gen_idx, step).

        Results are cached to the mega_generations stash if a CacheManager
        is available, keyed by (model_id, prompt, temp, idx).
        """
        positions = [position] if position else list(self._nodes.keys())
        all_rows = []

        for pos in positions:
            node = self._nodes[pos]
            model_id = node.model_id
            cache = node.layer._cache

            for gen_idx in range(n):
                # Check cache
                if cache is not None:
                    cached = cache.get_mega_generation(model_id, prompt, temperature, gen_idx)
                    if cached is not None:
                        for p in cached:
                            all_rows.append({"family": node.family, "position": pos,
                                             "gen_idx": gen_idx, **p})
                        continue

                node.layer._require_model()
                model = node.layer.model
                tokenizer = node.layer.tokenizer

                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    next(model.parameters()).device)
                generated_ids = input_ids.clone()
                step_data = []

                for step in range(max_tokens):
                    with torch.no_grad():
                        out = model(generated_ids)
                    logits = out.logits[0, -1, :].float().cpu()
                    probs = torch.softmax(logits, -1)

                    h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
                    eff = int((probs > 0.001).sum())
                    topk = torch.topk(probs, 5)
                    top_words = [tokenizer.decode([idx]).strip() for idx in topk.indices]
                    top_probs = topk.values.tolist()

                    if temperature > 0:
                        scaled = logits / temperature
                        sample_probs = torch.softmax(scaled, -1)
                        next_id = torch.multinomial(sample_probs, 1)
                    else:
                        next_id = logits.argmax().unsqueeze(0)

                    chosen_word = tokenizer.decode([next_id.item()]).strip()
                    chosen_prob = probs[next_id.item()].item()

                    row = {
                        "step": step, "chosen_token": chosen_word,
                        "chosen_prob": chosen_prob,
                        "entropy": h, "eff_vocab": eff,
                        "top1": top_words[0], "top1_prob": top_probs[0],
                        "top5_words": "|".join(top_words),
                    }
                    step_data.append(row)
                    all_rows.append({"family": node.family, "position": pos,
                                     "gen_idx": gen_idx, **row})

                    generated_ids = torch.cat([
                        generated_ids,
                        next_id.unsqueeze(0).to(generated_ids.device)
                    ], dim=-1)

                    if next_id.item() == tokenizer.eos_token_id:
                        break

                # Cache the trajectory
                if cache is not None:
                    cache.set_mega_generation(model_id, prompt, step_data, temperature, gen_idx)

        return pd.DataFrame(all_rows)

    def __repr__(self):
        nodes = ", ".join(self.positions)
        return f"Circuit([{nodes}], mode={self._mode.value})"
