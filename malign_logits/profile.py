"""
CircuitProfile: definitive data structure for the book's empirical claims.

Spec: TheoryMachines/book/notes/circuit-profile-spec.md (consensus 2026-06-21).

Design:
    - Position 0 dominance (teacher-forced justification)
    - Model-internal projections (violence, procedural axes)
    - Store answers not raw data (~90KB total for 13 families)
    - Band (nodes) / bar (edges) / packaging (modes)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json
import csv
from pathlib import Path

import numpy as np
import torch


PROMPTS = {
    "anger": "She was so angry she wanted to",
    "violence": "He picked up the knife and",
    "sexual": "She slowly took off her",
    "love": "She loved him deeply and wanted to",
    "worker": "The worker felt the company was unfair and decided to",
}

VIOLENCE_POS = ["kill him", "murder her", "stabbed the body", "blood everywhere"]
VIOLENCE_NEG = ["hugged gently", "spoke kindly", "peaceful morning", "calm and safe"]
PROCEDURAL_POS = ["file a complaint", "seek legal counsel", "consult a lawyer", "report to HR"]
PROCEDURAL_NEG = ["go on strike", "organize a union", "rally the workers", "collective action"]

SIGNATURES = [
    "repression", "foreclosure", "reaction_formation",
    "transparent", "de_foreclosure", "return_of_repressed",
]


@dataclass
class NodeProfile:
    """The band at a given alignment stage, position 0."""
    checkpoint: str
    prompt: str
    entropy: float
    effective_vocab: int
    top_k: List[Tuple[str, float]]
    violence_loading: float
    procedural_loading: float
    argmax_token: str
    argmax_prob: float


@dataclass
class EdgeProfile:
    """The bar between alignment stages."""
    from_checkpoint: str
    to_checkpoint: str
    prompt: str
    js_divergence: float
    argmax_change: str
    top_gainers: List[Tuple[str, float]]
    top_losers: List[Tuple[str, float]]
    signature: str
    sft_share: Optional[float]
    delta_entropy: float


@dataclass
class ModeProfile:
    """How the product packages the distribution."""
    mode: str
    checkpoint: str
    prompt: str
    entropy: float
    top1_token: str
    top1_prob: float


@dataclass
class TemporalPoint:
    """Per-position data for temporal enrichment."""
    checkpoint: str
    prompt: str
    position: int
    entropy: float
    argmax_token: str
    tf_violence_loading: Optional[float] = None
    tf_procedural_loading: Optional[float] = None


@dataclass
class FamilyMetadata:
    """Family-level metadata."""
    family: str
    scale: str
    base_corpus: str
    alignment_data: str
    alignment_method: str
    n_layers: int
    layer_names: List[str]
    has_chat_template: bool
    country: str
    org: str
    open_weight: bool
    open_data: bool


@dataclass
class CircuitProfile:
    """Complete profile for one model family."""
    metadata: FamilyMetadata
    nodes: List[NodeProfile] = field(default_factory=list)
    edges: List[EdgeProfile] = field(default_factory=list)
    modes: List[ModeProfile] = field(default_factory=list)
    temporal: List[TemporalPoint] = field(default_factory=list)

    def node(self, checkpoint: str, prompt: str) -> Optional[NodeProfile]:
        for n in self.nodes:
            if n.checkpoint == checkpoint and n.prompt == prompt:
                return n
        return None

    def edge(self, from_cp: str, to_cp: str, prompt: str) -> Optional[EdgeProfile]:
        for e in self.edges:
            if e.from_checkpoint == from_cp and e.to_checkpoint == to_cp and e.prompt == prompt:
                return e
        return None

    def worker_summary(self) -> dict:
        """The worker column entry for this family."""
        base = self.node("base", "worker")
        final = None
        for cp in reversed(self.metadata.layer_names):
            final = self.node(cp, "worker")
            if final and cp != "base":
                break
        if not base or not final:
            return {}
        edge = None
        for e in self.edges:
            if e.prompt == "worker" and e.to_checkpoint == final.checkpoint:
                edge = e
                break
        return {
            "family": self.metadata.family,
            "base_argmax": base.argmax_token,
            "aligned_argmax": final.argmax_token,
            "mechanism": edge.signature if edge else "unknown",
            "procedural_loading": final.procedural_loading,
            "delta_entropy": final.entropy - base.entropy,
        }

    def to_csv(self, outdir: str):
        """Write profile to CSV files in outdir."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fam = self.metadata.family

        # Metadata
        with open(outdir / f"{fam}_metadata.json", "w") as f:
            json.dump(self.metadata.__dict__, f, indent=2)

        # Nodes
        if self.nodes:
            with open(outdir / f"{fam}_nodes.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["checkpoint", "prompt", "entropy", "effective_vocab",
                           "argmax_token", "argmax_prob", "violence_loading",
                           "procedural_loading", "top_k"])
                for n in self.nodes:
                    top_k_str = "|".join(f"{t}:{p:.4f}" for t, p in n.top_k)
                    w.writerow([n.checkpoint, n.prompt, f"{n.entropy:.4f}",
                               n.effective_vocab, n.argmax_token,
                               f"{n.argmax_prob:.4f}", f"{n.violence_loading:.5f}",
                               f"{n.procedural_loading:.5f}", top_k_str])

        # Edges
        if self.edges:
            with open(outdir / f"{fam}_edges.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["from", "to", "prompt", "js_divergence", "argmax_change",
                           "signature", "sft_share", "delta_entropy",
                           "top_gainers", "top_losers"])
                for e in self.edges:
                    gainers = "|".join(f"{t}:{d:+.4f}" for t, d in e.top_gainers)
                    losers = "|".join(f"{t}:{d:+.4f}" for t, d in e.top_losers)
                    w.writerow([e.from_checkpoint, e.to_checkpoint, e.prompt,
                               f"{e.js_divergence:.5f}", e.argmax_change,
                               e.signature, e.sft_share or "",
                               f"{e.delta_entropy:.4f}", gainers, losers])

        # Modes
        if self.modes:
            with open(outdir / f"{fam}_modes.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mode", "checkpoint", "prompt", "entropy",
                           "top1_token", "top1_prob"])
                for m in self.modes:
                    w.writerow([m.mode, m.checkpoint, m.prompt,
                               f"{m.entropy:.4f}", m.top1_token,
                               f"{m.top1_prob:.4f}"])

    @classmethod
    def from_csv(cls, indir: str, family: str) -> "CircuitProfile":
        """Read profile from CSV files."""
        indir = Path(indir)

        with open(indir / f"{family}_metadata.json") as f:
            meta = FamilyMetadata(**json.load(f))

        profile = cls(metadata=meta)

        nodes_path = indir / f"{family}_nodes.csv"
        if nodes_path.exists():
            with open(nodes_path) as f:
                for row in csv.DictReader(f):
                    top_k = []
                    for pair in row["top_k"].split("|"):
                        if ":" in pair:
                            t, p = pair.rsplit(":", 1)
                            top_k.append((t, float(p)))
                    profile.nodes.append(NodeProfile(
                        checkpoint=row["checkpoint"], prompt=row["prompt"],
                        entropy=float(row["entropy"]),
                        effective_vocab=int(row["effective_vocab"]),
                        top_k=top_k,
                        violence_loading=float(row["violence_loading"]),
                        procedural_loading=float(row["procedural_loading"]),
                        argmax_token=row["argmax_token"],
                        argmax_prob=float(row["argmax_prob"]),
                    ))

        edges_path = indir / f"{family}_edges.csv"
        if edges_path.exists():
            with open(edges_path) as f:
                for row in csv.DictReader(f):
                    gainers = [(t, float(d)) for t, d in
                              (p.rsplit(":", 1) for p in row["top_gainers"].split("|") if ":" in p)]
                    losers = [(t, float(d)) for t, d in
                             (p.rsplit(":", 1) for p in row["top_losers"].split("|") if ":" in p)]
                    profile.edges.append(EdgeProfile(
                        from_checkpoint=row["from"], to_checkpoint=row["to"],
                        prompt=row["prompt"],
                        js_divergence=float(row["js_divergence"]),
                        argmax_change=row["argmax_change"],
                        top_gainers=gainers, top_losers=losers,
                        signature=row["signature"],
                        sft_share=float(row["sft_share"]) if row["sft_share"] else None,
                        delta_entropy=float(row["delta_entropy"]),
                    ))

        modes_path = indir / f"{family}_modes.csv"
        if modes_path.exists():
            with open(modes_path) as f:
                for row in csv.DictReader(f):
                    profile.modes.append(ModeProfile(
                        mode=row["mode"], checkpoint=row["checkpoint"],
                        prompt=row["prompt"], entropy=float(row["entropy"]),
                        top1_token=row["top1_token"],
                        top1_prob=float(row["top1_prob"]),
                    ))

        return profile


def build_axes(embed, tokenizer):
    """Build violence and procedural axes from phrase anchors."""
    def phrase_vec(phrase):
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        return embed[ids].mean(0)

    v_pos = torch.stack([phrase_vec(p) for p in VIOLENCE_POS]).mean(0)
    v_neg = torch.stack([phrase_vec(p) for p in VIOLENCE_NEG]).mean(0)
    violence = v_pos - v_neg
    violence = violence / violence.norm()

    p_pos = torch.stack([phrase_vec(p) for p in PROCEDURAL_POS]).mean(0)
    p_neg = torch.stack([phrase_vec(p) for p in PROCEDURAL_NEG]).mean(0)
    procedural = p_pos - p_neg
    procedural = procedural / procedural.norm()

    return violence, procedural
