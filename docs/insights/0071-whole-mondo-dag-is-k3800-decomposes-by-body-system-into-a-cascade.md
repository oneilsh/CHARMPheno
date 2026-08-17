# 0071 — The whole-Mondo powered DAG is K≈3,800: too big for one co-fit-Newton fit, but it decomposes cleanly by body system into a buildable detection×conditional cascade

**Date:** 2026-08-17
**Topic:** case-finding, ontology, mondo, scale, architecture, prediction-constrained

**Status:** Confirmed on exp 0088 (whole-Mondo powered hierarchy, AoU CDR R2024Q3R8)

> **Correction (2026-08-17, same day).** The "stage as a cascade" conclusion below
> over-stated the piecemeal route. The K³/K² wall is NOT the model size — it is
> specifically the **DENSE co-fit head** (every one of C nodes reads all K topics →
> O(C·K²) memory, O(C·K³) compute). The topic model and the two-stage readout both
> scale. The better answer is a **LOCALIZED DAG-closure head in ONE co-fit**: each
> node's weight is supported only on its gated topic block + ancestors (~O(depth) dims,
> which the gate already defines), so the per-node Fisher/solve is O(depth³) and the
> total head cost is O(C·depth³) — trivial, on ONE joint model that keeps global topic/
> background sharing. The body-system cascade (below) is a *fallback* and a valid
> *inference-time* decomposition, not the required training architecture. Being tested
> on the 41-anchor setup before all-Mondo.

> **Correction 2 (2026-08-17, exp 0090 run 1).** The localized head's O(C·depth²)
> MEMORY claim was, as first implemented, realized only in the SOLVE — the per-node
> Fisher was still *emitted* dense `(C, K, K)` and collected to the driver, so a fit
> OOM'd on `spark.driver.maxResultSize` at K=444 (657 MB × partitions), and would hit
> the same 850 GB wall at K≈3,800. Fixed by emitting the Fisher as a compact padded
> `(C, S, S)` per-node block stack (S = max support), exact (each block is the dense
> Fisher's support sub-block) and ~2 MB instead of 690 MB/partition. Only WITH localized
> emission is the whole-Mondo co-fit actually collectable — the memory win is now real,
> not just in compute. The `cost_report` prints the true collected size (`C*S^2`).

Exp 0087 validated whole-Mondo as a near-complete disease backbone (97.9% of coded
patients placed). Exp 0088 turned it into the actual label DAG and sized it. The size
answer reshapes the fit architecture.

**K≈3,800.** 9,164 Mondo→OMOP anchors → **2,513 powered** (≥100 patients) + **1,306**
compact branch-point class nodes = **3,819 layout nodes** (K = n_bg + nodes·tpn ≈ 3,827
at n_bg=8, tpn=1). ~38× the biggest fit to date (K=101, exp 0081).

**The class tree is clinically clean — Mondo delivered real umbrellas.** The kept branch
points are sensible clinical categories: cardiovascular disorder, nervous/respiratory/
endocrine/immune/psychiatric/hematologic/musculoskeletal system disorder, cancer/neoplasm
families, infectious disease, connective tissue disorder. This is the conditional-
sharpening structure, and it is far cleaner than the SNOMED-derived DAG (which produced
degenerate SLE→SLE near-duplicate nodes, insight 0069/exp 0085) — **vindicating the
"rely on Mondo, less on SNOMED" bet.** (A few kept classes are abstract Mondo meta-
groupings — "disease by body system or component" (2216), "disease by etiologic
mechanism" (995) — near the root; `max_class_fraction < 1` drops those.)

**As ONE fit, K≈3,800 is not viable — and the binding constraint is specifically the
unified co-fit head.** The ridge-Newton head (ADR 0043's unified P(child|parent) model)
is **O(K³·C) compute and O(C·K²) memory**: the per-node Fisher stack is C·K·K ≈ 3819³
floats ≈ **~850 GB**, and each SVI iteration does C solves of K×K systems. The topic
model itself scales (λ is K×V ≈ 19M) and the **two-stage readout LR scales** (C
independent logistic fits on K features), but the *unified Newton head does not*. So at
whole-Mondo scale the ADR-0043 unified-head direction hits a wall that two-stage does not
— a real, scale-dependent tension to record: **unified head at small/medium K, two-stage
(or staging) at whole-Mondo K.**

**But the tree decomposes NATURALLY by body system — the cascade is the answer.** The
~20–30 top-level classes each carry ~100–500 nodes, i.e. a sub-fit at the K~few-hundred
scale already run. This is exactly the **detection-at-top × per-branch-conditional
cascade** (the P(d|x)=P(d|x∈C)·P(x∈C|x) factorization): fit a top-level "which body
system" detector, then per-system conditional models, compose at inference. The
hierarchy shows this decomposition is not just conceptually clean but the **natural
computational partition** — each branch is independently fittable and the O(K³·C) head
cost stays bounded within a branch.

**Implication.** Whole-Mondo is a clean, buildable backbone — as a **staged cascade**, not
a single K≈3,800 model. Next build is the top-level system partition + one per-branch fit
as a template (e.g. cardiovascular: ~273 nodes), not a monolith. Single-fit levers if
ever wanted: raise `min_positives` (drops rare nodes) and lower `max_class_fraction`
(drops abstract top classes).
