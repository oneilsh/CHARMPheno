# 0080 — Unsupervised gated whole-Mondo shallow topics ARE the known catch-alls (now coherent across the fused multi-domain vocab); the real concerns are sibling redundancy and top-level capacity, not "topics ≠ labels"

**Date:** 2026-09-05
**Topic:** lda
**Status:** Observed

**Relates to / grounded in:** 0019 (LDA at generous K → 3 catch-alls +
~40 phenotypes + graceful tail — the shallow archetypes here are those
catch-alls), 0071 (whole-Mondo abstract meta-nodes near root; and "as ONE fit
K≈3,800 is not viable — decompose by body system into a cascade"), 0063
(spectral↔random init is a null lever — the gate supplies identifiability),
0066 (PC topic-shaping marginal; "No PC runs" is a standing constraint), 0079
(the depth-5 topic-evidence cliff). **This entry corrects its own first draft**,
which framed the shallow topics as "not matching their Mondo labels" and floated
`weight_y>0` — both wrong (see Interpretation).

## Observation

The 0111 episode arm's fed shallow node topics (depths 1–4, the only fed ones —
0079), read via `inspect_topics.py --tour` with all domains named, are **coherent
multi-domain clusters** — and the fused vocabulary works: conditions, labs, and
drugs cohere within a topic.

- `d2` *disease by body system* → **cardiometabolic**: T2DM / HTN / hyperlipidemia
  / CKD-3 + LDL/cholesterol labs + metformin / atorvastatin / insulin / lisinopril.
- `d2` *disease by etiologic mechanism* → **acute-care / ED**: pain / N&V / bleeding
  + vitals + fentanyl / midazolam / propofol / IV fluids / naloxone.
- `d3` *inflammatory disease* → **outpatient respiratory-infection / allergy**;
  `d3` *nervous system disorder* → **psychiatric / substance / HIV**.

These are exactly insight 0019's three catch-alls (acute-presentation,
generic-chronic, cardiometabolic), now **confirmed with measurement + drug** —
0019 was condition-only. Two of the d2 nodes are the abstract Mondo meta-groupings
0071 already flagged to drop via `max_class_fraction<1`.

Two things are genuinely worth flagging:

1. **Sibling redundancy.** ≥3 shallow nodes learned essentially the *same*
   acute-care archetype (IV fluids + opioids + antiemetics + vitals). The effective
   number of distinct shallow topics is a handful, not the 182 fed nodes.
2. **Measurement is `[normal]`-panel-dominated** across nearly every topic and all
   8 background topics (evidence 4–7e6, measurement-dominant). Value-state
   tokenization adds little beyond "labs were drawn."

## Interpretation (corrects the first draft)

- **A general node holding its whole population's common signal, leaving specifics
  to descendants, is the reverse-topo DEFLATION DESIGN WORKING — not a failure.**
  The first draft's "topic content ≠ node label" reading was wrong: "infectious
  disease → acute-care-common" is expected. The right question is whether the
  SPECIFIC descendants receive their specific signal — and 0079 answers no: the
  deep blocks are starved, so the deflation has nothing to hand down.
- **This is `weight_y=0` (unsupervised), and that is the settled regime.** PC /
  co-fit topic-shaping is marginal-to-dead (0066; the 0065–0069 convergence saga;
  "No PC runs" standing constraint). So the topics being unsupervised co-occurrence
  clusters is expected — NOT motivation to turn on the head.
- **Init is a null lever on this engine (0063)** — the DAG gate welds each node's
  topics to its subtree, and 200 SVI iters move off the seed. BUT 0063 was a
  170-block DAG, and the scalable-spectral path is dead code on this branch
  (whole-Mondo handoff §5.6), so whole-Mondo fits are effectively random-init and
  init-at-K≈2714-scale is **untested** — a fair open question, not a closed one.
- **Sibling redundancy + top-level under-capacity point at the topic-block-sizing
  arc** (unresolved; 0015: undersized K regresses crisp topics; 0019: generous K
  works in the FLAT model). With `tpn=1`, each broad node gets ONE topic for a huge
  heterogeneous population, so it can only hold the dominant catch-all, and siblings
  with overlapping populations converge to the same one.

## Implications

- Do **not** read this as motivation for `weight_y>0` (dead / standing constraint)
  or an init change (null lever historically; path dead; untested at scale).
- The known-good direction is 0071's **per-body-system cascade** — each branch fit
  at K~few-hundred, the regime where 0019-style phenotype emergence works — not the
  monolithic K≈2714 whole-Mondo fit 0111 runs.
- Genuinely open, experiment-worthy levers: (a) `tpn>1` / larger `n_bg` at the top
  so broad nodes have capacity beyond one catch-all; (b) quantify sibling redundancy
  (topic-word cosine across siblings) to size the problem; (c) test whether the
  cascade recovers the deep phenotypes the monolith starves; (d) init-at-scale, only
  because 0063's null was measured on a far smaller DAG.

**Setting context:** exp 0111 episode arm — gated-PC **weight_y=0** (unsupervised
topics + separate readout, `skip_unsup_gated: true`), whole-Mondo native DAG
(C=2714, K=2721, n_bg=8, tpn=1), 3 domains, episode index (gap 90d, cap 3, 365d
label), lookback 1825d. Read from the fit's saved globals + bundle meta via the
`inspect_topics.py --tour` tree tour; matched-random control (0112) not yet fit.
