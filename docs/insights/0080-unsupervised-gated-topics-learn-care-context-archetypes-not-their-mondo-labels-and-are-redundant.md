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

1. **Sibling redundancy — MEASURED, and it is NOT a problem.** `inspect_topics
   --redundancy` (per-parent cosine among a parent's FED children, starved ones
   excluded as trivially uniform): of 194 parents with ≥2 fed children, **0 show
   uniform collapse (median fed-cosine > 0.8)**, and **corr(fan-out, median
   fed-cosine) = −0.09** — near-zero and if anything negative, *refuting* the
   "wide parent + `tpn=1` → collapse" hypothesis. Wide parents (heart disorder
   fan-out 18 median 0.23; disease-by-body-system 19, 0.18; respiratory 17, 0.21)
   differentiate their children WELL; only the occasional lone pair is
   near-duplicate (arterial disorder max 0.98, body-system max 0.91 — but median
   low). The top median is diabetes mellitus 0.79 (T1/T2/complicated subtypes that
   *should* share a signature). So partial redundancy is real and harmless; uniform
   collapse does not occur. **Where topics are fed, capacity is fine — starvation
   at depth (0079) is the sole binding wall, not `tpn`/redundancy.**
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
- **This is `weight_y=0` (unsupervised), and that is a DELIBERATE post-PC-arc
  decision, not an oversight.** PC is not "dead" — its scale pathologies were fixed
  (0072/0073, ADR 0044/0045) — it is PARKED with an explicit revival condition
  (PC-arc closeout 2026-08-20 §6): the co-fit head as trained scores 0.567, far
  below the unsupervised gate's own readout (0.739), so shaping topics toward it
  HURTS by construction (exp 0102: readout 0.688 vs 0.7395, negative in all rarity
  quartiles incl. the rare tail). Revive PC only when the co-fit head can match the
  gate (≥~0.74) — which needs a full-K head that beats the O(C·K²) Hessian-collect
  wall (matrix-free L-BFGS, or MI-selected support; both scaffolded, unbuilt). So
  the topics being unsupervised co-occurrence clusters is expected, and turning on
  the head is blocked on head quality, not a quick knob.
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

- Do **not** read this as motivation for `weight_y>0` — it is parked on head
  quality (co-fit 0.567 < gate 0.739; revival needs a full-K head that beats the
  O(C·K²) collect wall, PC-arc closeout §6), not a quick knob. Nor an init change
  (null lever historically; scalable path dead; untested at scale).
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
