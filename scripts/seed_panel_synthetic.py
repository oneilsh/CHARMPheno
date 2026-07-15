"""Synthetic method-validation for the seed-panel acceptance test (LOCAL, no
Spark; known ground truth).

Purpose: the seed-panel test (spark_vi.eval.topic.seed_panel) asks whether a
generative concentration scale c over-commits on a tiny 1-2 token seed
prefix -- lands on the right topic but with implausibly total mass, secondary
interests erased. Before trusting that test's verdict on a real, ~5000-word,
60-topic model, this script checks the test itself DETECTS over-commitment on
a small corpus with a KNOWN ground-truth topic-term matrix and correlation
structure.

Ground truth comes from tests/_stm_synth.py:gated_ln_corpus, which returns a
disjoint-vocabulary beta (each topic's signature words are unique -- no other
topic's top codes overlap) and a genuinely unit-diagonal correlation Sigma_true
(bg-bg 0.10, bg-fg 0.25, within-fg 0.30; PD-completed where cross-foreground
pairs are structurally unobserved). No STM fit is run: seed_panel_sweep is a
pure function of (beta, Gamma, R, partition) -- it does not consume documents
-- so the planted beta/R ARE the "real model" for this test, and the s_true=5
mentioned in the task brief is a REFERENCE POINT for interpreting the sweep
(the c value we plan to ship on the real corpus), not a value baked into doc
generation. gated_ln_corpus's own doc-generation draws eta ~ N(0, Sigma_true)
at implicit scale 1; the corpus's docs are unused here (see NOTE below).

The discrimination property under test: conditioned_theta's prior precision is
(1/c) * safe_inverse(R[allowed]) (spark_vi.eval.topic.seed_panel), so INCREASING
c weakens the prior's pull toward the mean, letting a small amount of seed data
dominate the posterior mode MORE -- i.e. higher c should make top_mass rise and
eff_topics/second_mass fall (more concentrated, less secondary structure) as c
grows past the "true" scale. A useful acceptance test must reproduce exactly
that direction on a corpus where we already know the truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark-vi"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark-vi" / "tests"))

from spark_vi.eval.topic.seed_panel import seed_panel_sweep          # noqa: E402
from _stm_synth import gated_ln_corpus                               # noqa: E402

S_TRUE = 5   # reference point for interpretation only -- see module docstring.
C_GRID = [2, 3, 4, 5, 8]
GROUPS = ("A", "B")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "0039-seed-panel"


def main() -> None:
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={g: 1.0 for g in GROUPS},
        fg_per_group=4, bg_k=3, V=400, D=1500, doc_len=40, seed=0,
    )
    K = part.K
    print(f"[synthetic] corpus built: K={K} (bg={part.background_k}, "
          f"fg groups={part.foreground}), V=400, D={len(docs)} docs "
          f"(NOT used -- seed_panel_sweep is beta/R/partition-only)")
    print(f"[synthetic] Sigma_true (planted R) shape={Sigma_true.shape}, "
          f"diag min/max={Sigma_true.diagonal().min():.4f}/"
          f"{Sigma_true.diagonal().max():.4f} (unit-diagonal correlation)")

    Gamma = np.zeros((1, K), dtype=np.float64)   # zero Gamma -> mean eta = 0
    R = Sigma_true

    all_rows = []
    for group in GROUPS:
        rows = seed_panel_sweep(
            beta, Gamma, R, part, group=group, c_grid=C_GRID,
            n_codes=2, x=None, reference=None,
        )
        all_rows.extend(rows)
    print(f"[synthetic] {len(all_rows)} (seed, c) rows across "
          f"{len(all_rows) // len(C_GRID)} seeded topics x {len(C_GRID)} c values")

    # Per-c summary.
    summary = {}
    for c in C_GRID:
        c_rows = [r for r in all_rows if r["c"] == c]
        top_masses = np.array([r["top_mass"] for r in c_rows])
        eff_topics = np.array([r["eff_topics"] for r in c_rows])
        second_masses = np.array([r["second_mass"] for r in c_rows])
        recover_rate = float(np.mean([r["recovers_self"] for r in c_rows]))
        summary[c] = {
            "median_top_mass": float(np.median(top_masses)),
            "median_eff_topics": float(np.median(eff_topics)),
            "median_second_mass": float(np.median(second_masses)),
            "recover_self_rate": recover_rate,
        }

    header = f"{'c':>4} | {'median top_mass':>16} | {'median eff_topics':>18} | {'median second_mass':>19} | {'recover-self rate':>18}"
    print(header)
    print("-" * len(header))
    for c in C_GRID:
        s = summary[c]
        print(f"{c:>4} | {s['median_top_mass']:>16.4f} | {s['median_eff_topics']:>18.4f} | "
              f"{s['median_second_mass']:>19.4f} | {s['recover_self_rate']:>18.4f}")

    # --- Discrimination assertions -------------------------------------
    top_mass_seq = [summary[c]["median_top_mass"] for c in C_GRID]
    eff_topics_seq = [summary[c]["median_eff_topics"] for c in C_GRID]

    monotone_top_mass = all(
        top_mass_seq[i + 1] >= top_mass_seq[i] - 1e-9 for i in range(len(C_GRID) - 1)
    )
    monotone_eff_topics = all(
        eff_topics_seq[i + 1] <= eff_topics_seq[i] + 1e-9 for i in range(len(C_GRID) - 1)
    )
    print(f"\n[synthetic] median top_mass non-decreasing in c: {monotone_top_mass} ({top_mass_seq})")
    print(f"[synthetic] median eff_topics non-increasing in c: {monotone_eff_topics} ({eff_topics_seq})")

    eff_3, eff_8 = summary[3]["median_eff_topics"], summary[8]["median_eff_topics"]
    second_3, second_8 = summary[3]["median_second_mass"], summary[8]["median_second_mass"]
    eff_collapse = eff_8 < eff_3 - 0.05
    second_collapse = second_8 < 0.5 * second_3
    print(f"[synthetic] secondary structure collapses c=8 vs c=3: "
          f"eff_topics {eff_3:.4f}->{eff_8:.4f} (collapse={eff_collapse}), "
          f"second_mass {second_3:.4f}->{second_8:.4f} (collapse={second_collapse})")

    recover_rates_mid = {c: summary[c]["recover_self_rate"] for c in (3, 4, 5)}
    high_recovery = all(v >= 0.8 for v in recover_rates_mid.values())
    print(f"[synthetic] recover-self rate at c in {{3,4,5}}: {recover_rates_mid} "
          f"(all >= 0.8: {high_recovery})")

    assert monotone_top_mass, "median top_mass must be non-decreasing in c"
    assert monotone_eff_topics, "median eff_topics must be non-increasing in c"
    assert eff_collapse or second_collapse, (
        "expected secondary mass to collapse materially between c=3 and c=8"
    )
    assert high_recovery, "expected high self-recovery rate for c in {3,4,5}"
    print("\n[synthetic] ALL DISCRIMINATION ASSERTIONS PASSED")

    _write_results_md(summary, monotone_top_mass, monotone_eff_topics,
                       eff_collapse, second_collapse, recover_rates_mid)


def _write_results_md(summary, monotone_top_mass, monotone_eff_topics,
                       eff_collapse, second_collapse, recover_rates_mid) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Seed-panel synthetic method-validation (exp 0039)")
    lines.append("")
    lines.append(
        "Known-ground-truth check that the seed-panel acceptance test "
        "(spark_vi.eval.topic.seed_panel) DETECTS generative-scale "
        "over-commitment: a planted, disjoint-vocabulary beta and a planted "
        "unit-diagonal correlation R (tests/_stm_synth.py:gated_ln_corpus; "
        "groups=('A','B'), fg_per_group=4, bg_k=3, V=400, doc_len=40) are fed "
        "straight into seed_panel_sweep with a zero Gamma and reference=None. "
        "No STM fit is run -- the sweep is a pure function of (beta, Gamma, "
        "R, partition), so the planted values ARE the model under test. "
        f"s_true={S_TRUE} is a reference point for interpretation (the value "
        "planned for the real corpus), not a scale baked into doc generation "
        "(gated_ln_corpus draws eta ~ N(0, Sigma_true), Sigma_true already "
        "unit-diagonal, i.e. implicit scale 1); this script does not use the "
        "corpus's generated documents at all."
    )
    lines.append("")
    lines.append("| c | median top_mass | median eff_topics | median second_mass | recover-self rate |")
    lines.append("|---|---|---|---|---|")
    for c in C_GRID:
        s = summary[c]
        lines.append(
            f"| {c} | {s['median_top_mass']:.4f} | {s['median_eff_topics']:.4f} | "
            f"{s['median_second_mass']:.4f} | {s['recover_self_rate']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Median top_mass is non-decreasing in c: **{monotone_top_mass}**. "
        f"Median eff_topics is non-increasing in c: **{monotone_eff_topics}**. "
        f"Secondary structure collapses materially between c=3 and c=8 "
        f"(eff_topics collapse={eff_collapse} and/or second_mass collapse="
        f"{second_collapse}). Self-recovery rate at c in {{3,4,5}} is >= 0.8 for "
        f"every value: {recover_rates_mid}."
    )
    lines.append("")
    lines.append(
        "**Interpretation.** As c grows, the prior precision (1/c)*R^-1 over "
        "the allowed topic set weakens, so a 1-2 token seed's likelihood term "
        "dominates the posterior mode more completely -- top_mass rises and "
        "eff_topics/second_mass fall. This is exactly the over-commitment "
        "failure mode the reviewer described, reproduced on a corpus where "
        "the truth is known: the acceptance test's summary statistics move in "
        "the expected direction and the recover-self rate confirms seeds still "
        "land on their planted source topic through c=5. This validates the "
        "METHOD (the statistics DETECT over-commitment as designed) -- it "
        "does not, by itself, decide whether c=5 over-commits on the REAL "
        "corpus; see real_results.md for that."
    )
    (RESULTS_DIR / "synthetic_results.md").write_text("\n".join(lines) + "\n")
    print(f"\n[synthetic] wrote {RESULTS_DIR / 'synthetic_results.md'}")


if __name__ == "__main__":
    main()
