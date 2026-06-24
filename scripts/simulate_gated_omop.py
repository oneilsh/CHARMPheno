"""Generate synthetic gated background/foreground OMOP data for STM.

Plants a background concept block shared by all patients plus per-group
distinctive concept blocks. A patient's group selects which foreground topics
it may express (background union its group's foreground); a group with no foreground
block (e.g. a large "common" cohort) emits background only. This is the local
analogue of the gated STM the dashboard renders.

Oracle columns true_topic_id / true_block are evaluation-only; the fitter must
not read them. See docs/superpowers/specs/2026-06-23-gated-stm-local-dashboard-design.md
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("data/simulated")


@dataclass(frozen=True)
class GatedBeta:
    beta: np.ndarray            # (K_total, V) row-stochastic
    concept_ids: np.ndarray     # (V,) int
    concept_names: dict         # cid -> "C<cid>"
    topic_blocks: list          # length K_total: "background" or group label
    group_concepts: dict        # group -> list[int] concept_ids
    background_concepts: list    # list[int] concept_ids


def build_gated_beta(*, n_background_concepts, n_group_concepts, background_k,
                     foreground, rng, bleed=0.1) -> GatedBeta:
    """Build a block-structured beta. Background topics live on background
    concepts; each group's foreground topics live on that group's distinctive
    concepts plus a small `bleed` of background mass."""
    groups = [g for g, _ in foreground]
    V = n_background_concepts + n_group_concepts * len(groups)
    concept_ids = np.arange(V, dtype=np.int64)
    concept_names = {int(c): f"C{int(c)}" for c in concept_ids}

    bg_concepts = list(range(n_background_concepts))
    group_concepts = {}
    start = n_background_concepts
    for g in groups:
        group_concepts[g] = list(range(start, start + n_group_concepts))
        start += n_group_concepts

    rows = []
    topic_blocks = []
    # Background topics: Dirichlet over background concepts only.
    for _ in range(background_k):
        v = np.zeros(V)
        v[bg_concepts] = rng.dirichlet(np.full(len(bg_concepts), 0.5))
        rows.append(v)
        topic_blocks.append("background")
    # Foreground topics per group: mostly the group's concepts + `bleed` of bg.
    fg_sizes = dict(foreground)
    for g in groups:
        for _ in range(fg_sizes[g]):
            v = np.zeros(V)
            v[group_concepts[g]] = (1.0 - bleed) * rng.dirichlet(
                np.full(n_group_concepts, 0.5))
            v[bg_concepts] = bleed * rng.dirichlet(np.full(len(bg_concepts), 0.5))
            rows.append(v)
            topic_blocks.append(g)

    beta = np.vstack(rows)
    beta = beta / beta.sum(axis=1, keepdims=True)
    return GatedBeta(beta=beta, concept_ids=concept_ids,
                     concept_names=concept_names, topic_blocks=topic_blocks,
                     group_concepts=group_concepts, background_concepts=bg_concepts)


def _allowed_topic_indices(topic_blocks, group):
    """Background topics union the group's foreground topics (background-only if the
    group has no foreground block)."""
    return [i for i, b in enumerate(topic_blocks)
            if b == "background" or b == group]


def simulate_gated(gb: GatedBeta, *, n_patients, group_props, foreground,
                   visits_per_patient_mean, codes_per_visit_mean, age_means,
                   theta_alpha, seed):
    rng = np.random.default_rng(seed)
    groups = list(group_props.keys())
    probs = np.array([group_props[g] for g in groups], dtype=np.float64)
    probs = probs / probs.sum()
    K_total, V = gb.beta.shape

    ev_rows = []
    person_rows = []
    visit_counter = 0
    for p in range(n_patients):
        g = groups[int(rng.choice(len(groups), p=probs))]
        sex = "M" if rng.random() < 0.5 else "F"
        age = int(max(18, min(95, rng.normal(age_means.get(g, 60.0), 8.0))))
        person_rows.append((p, g, sex, age))

        allowed = _allowed_topic_indices(gb.topic_blocks, g)
        alpha = np.full(len(allowed), theta_alpha)
        theta_allowed = rng.dirichlet(alpha)
        theta = np.zeros(K_total)
        theta[allowed] = theta_allowed

        n_visits = max(1, int(rng.poisson(visits_per_patient_mean)))
        for _ in range(n_visits):
            visit_counter += 1
            n_codes = max(1, int(rng.poisson(codes_per_visit_mean)))
            z = rng.choice(K_total, size=n_codes, p=theta)
            for zi in z:
                w_col = rng.choice(V, p=gb.beta[zi])
                cid = int(gb.concept_ids[w_col])
                ev_rows.append((p, visit_counter, cid, gb.concept_names[cid],
                                g, int(zi), gb.topic_blocks[zi]))

    events = pd.DataFrame(ev_rows, columns=[
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
        "source_cohort", "true_topic_id", "true_block"])
    persons = pd.DataFrame(person_rows, columns=[
        "person_id", "source_cohort", "sex", "age"])
    oracle = {
        "background_concepts": gb.background_concepts,
        "group_concepts": {g: gb.group_concepts[g] for g in gb.group_concepts},
        "topic_blocks": gb.topic_blocks,
        "background_k": sum(1 for b in gb.topic_blocks if b == "background"),
        "foreground": [[g, k] for g, k in foreground],
    }
    return events, persons, oracle


def _parse_pairs(s, valtype):
    out = []
    for piece in s.split(","):
        k, _, v = piece.partition(":")
        out.append((k.strip(), valtype(v)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patients", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--background-k", type=int, default=3)
    p.add_argument("--foreground", default="rare_dx:2",
                   help="per-group foreground sizes 'g:K,g:K'")
    p.add_argument("--group-props", default="common:0.99,rare_dx:0.01",
                   help="group proportions 'g:frac,g:frac' (need not sum to 1)")
    p.add_argument("--age-means", default="common:55,rare_dx:70")
    p.add_argument("--n-background-concepts", type=int, default=40)
    p.add_argument("--n-group-concepts", type=int, default=12)
    p.add_argument("--bleed", type=float, default=0.1)
    p.add_argument("--theta-alpha", type=float, default=0.3)
    p.add_argument("--visits-per-patient-mean", type=float, default=3.0)
    p.add_argument("--codes-per-visit-mean", type=float, default=8.0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    foreground = tuple((g, int(k)) for g, k in _parse_pairs(args.foreground, int))
    group_props = dict(_parse_pairs(args.group_props, float))
    age_means = dict(_parse_pairs(args.age_means, float))

    rng = np.random.default_rng(args.seed)
    gb = build_gated_beta(
        n_background_concepts=args.n_background_concepts,
        n_group_concepts=args.n_group_concepts,
        background_k=args.background_k, foreground=foreground,
        rng=rng, bleed=args.bleed)
    events, persons, oracle = simulate_gated(
        gb, n_patients=args.n_patients, group_props=group_props,
        foreground=foreground,
        visits_per_patient_mean=args.visits_per_patient_mean,
        codes_per_visit_mean=args.codes_per_visit_mean,
        age_means=age_means, theta_alpha=args.theta_alpha, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"N{args.n_patients}_seed{args.seed}"
    events.to_parquet(args.output_dir / f"gated_omop_{stem}.parquet", index=False)
    persons.to_parquet(args.output_dir / f"gated_person_{stem}.parquet", index=False)
    (args.output_dir / f"gated_oracle_{stem}.json").write_text(
        json.dumps(oracle, indent=2))
    log.info("wrote gated sim: %d events, %d patients, groups=%s",
             len(events), len(persons), list(group_props))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
