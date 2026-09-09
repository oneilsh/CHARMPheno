# Handoff — the discriminability wall, the closed PC line, and the episode frontier (for a fresh strategic read)

**Date:** 2026-09-09
**Branch:** `claude/gated-conditional-voi` (all work below is committed + pushed; tree clean at `e98a226`).
**Purpose:** written for a cold reader (a fresh model asked for its independent read of where this project is stuck and whether the next planned move is the right one). It is deliberately honest about doubt: the point is to enable a real second opinion, not to ratify the current plan.

---

## 0. The struggle in one paragraph

CHARMPheno fits interpretable gated topic models (gated LDA / prediction-constrained PC) over All-of-Us OMOP EHR, one topic-block per Mondo disease node, to do **case-finding** (rank patients by P(node) on a held-out split, macro AUC across nodes) while staying **legible** (each node's topic reads as its syndrome). The recurring wall, now hit from six independent directions, is: **every lever that improves topic legibility/alignment fails to improve — and often mildly hurts — case-finding discriminability.** Legibility and discriminability behave like *different, weakly-anti-correlated* objects. The just-finished experiment (0120) was the strongest test of the strongest version of the "make topics discriminative by supervising them" idea, and it came back negative. The open strategic question is whether this wall is a **representation problem** (static topic mixtures can't carry the signal — fix with a temporal/episode representation, the built-but-unevaluated frontier) or a **data/label problem** (a rare-disease patient's pre-diagnosis EHR genuinely doesn't separate them from controls — in which case no representation fixes case-finding, and the program should re-aim at the interpretable-representation-as-deliverable / VOI framing). This handoff lays out the evidence for both.

---

## 1. Standing constraints (do not relearn these the hard way)

- **Branch:** develop on `claude/gated-conditional-voi`. Cluster commands carry the preamble `cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only` (a fresh cluster clones `main`).
- **Cluster reality:** Dataproc, checked out at `~/repos/CHARMPheno`. Idle-killed/recreated roughly daily (`…-YYYYMMDD-m` hostnames). **Bundle/sidecar caches default to HDFS → wiped with the cluster** ⇒ a bundle+sidecar rebuild every fresh cluster (the expensive ~1h assembly). The user has explicitly **accepted the daily rebuild** and declined moving the cache to persistent GCS. Run dirs live on the workspace **persistent** disk and survive reboots. RUNS_DIR = `/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs`.
- **Egress floor (All-of-Us):** any cell < 20 is non-disclosable. Per-node/patient counts stay workspace-internal; committed docs carry pooled figures + counts-of-nodes only.
- **Cache-key landmine:** bundle/corpus/covariate keys fold the *source hash* of `charmpheno/omop/{cohorts,multi_domain,case_finding_assembly}.py` and the mondo DAG modules; editing them silently invalidates or poisons caches and trips a byte-pinned tripwire suite. Prefer driver-owned seams (`analysis/cloud/{gated_pc_cloud,gated_pc_readout,distributed_readout,…}.py`, `scripts/run_experiment.py`, the Makefile, `analysis/pc/evaluate.py`, tests) — all free to edit. **ADR 0047:** nothing array-shaped / Spark-capturing rides a task closure.
- **Commit trailer (this session):** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01CWHmrp8SXQU127fTYksP1c`. No model identifiers in repo artifacts otherwise.
- **User working style:** dislikes tunable "strength" knobs (wants them derived, e.g. a prior pegged to "one canonical patient document"); wants intellectual honesty and the null called plainly; reads chat as plain text (no LaTeX — use λ, θ, `sum_k`); values crash-resilience (checkpoints) because cluster runs are long and die often. Terminal output is not reliably visible to the user — rely on committed files.

---

## 2. The core struggle: the discriminability wall, six ways

The insight ladder is the evidence. Each rung is an independent lever tried on the *same* representation goal; each buys legibility and not discrimination (`docs/insights/`):

- **0081** — at whole-Mondo scale the *discriminative* multi-domain structure lives in the **8 shared background topics**, not the per-node gated blocks. `n_bg` is not a performance lever. (The signal is shared, not node-specific.)
- **0082 / 0083** — spectral init fixes deep-node **starvation/legibility** but **costs case-finding at every depth**. "Topic evidence is not discriminability." (First clean statement of the wall.)
- **0084** — profile-eta (HPO-profile word prior) at S=1.0 provably reaches the starved floor (legible profile tokens in E[log β]) but **cannot move θ** — a wiring-validated null.
- **0085** — the eta dose-response: strength buys topic-profile **alignment monotonically** (starved top-15 overlap 0.13 → 1.00) with **zero AUC gain**, and starts *costing* globally by 3×. "The prior is an interpretability lever, not an AUC lever." Key mechanism: least-aligned-at-strength nodes are the *common acquired* presentations (hypertensive disorder / CAD) — **HPOA's rare-syndromic evidence is not the corpus's common presentation**.
- **0086** — the matrix-free L-BFGS **co-fit head** (label supervision folded into topic updates) couples cleanly on the local simulator; the trust cap makes the inner knobs non-discriminative. (Engineering greenlight — not the verdict.)
- **0087 (this session, the fair test)** — the scalable strong co-fit head on the aligned target: **negative**. Details below.

Two earlier arcs (0080: unsupervised topics are the known catch-alls, redundant; the whole "profile-eta / PC" program) were attempts to add the label signal the unsupervised topics lack. All roads led to the same wall.

---

## 3. Exp 0120 — the fair PC test, in detail (why it's the strongest null)

Setup: MONDO:0004995 (cardiovascular) branch, tpn=5, C=299 nodes, K=1498 topics; random init + **profile-eta S=1.0** (the aligned target, 0085's operating point) + the **L-BFGS co-fit head** (weight_y=12, `head_trust_move=0.03`). This is the strongest shaping (a right-sized, curvature-aware label pull) on the strongest base (aligned topics). If supervising topics could convert alignment → discrimination, here is where it would.

Results (`docs/experiments/0120-*.md`, `results_readout.json`):

| arm | macro AUC | co-fit-head AUC | top-1% lift | prec@1% |
|---|--:|--:|--:|--:|
| pc_topics_lr (LR on shaped θ) | **0.7555** | — | **7.57** | 0.605 |
| co-fit head (sigmoid w·θ) | — | **0.6397** | — | — |

- **AUC is DOWN, not flat.** 0.7555 < 0116's ~0.7804 (profile-eta, *no* shaping), < the 0.758 revival bar, < 0113 baseline ~0.78. Shaping mildly *lowered* decodability. (No fresh paired `readout-ab`: 0113/0116 aren't scored on the current cluster; comparison is to the recorded macros.)
- **The head's own decoder (0.64) ≪ a fresh LR on its own θ (0.7555).** The head that shaped the topics reads them worse than plain LR — the `self-w ≈ 0` decoupling seen in the decoder view: nearly every node's readout weight sits on **ancestor/shared** "sick-patient" topics, not its own profile-aligned topic. Alignment and decode-weight are different topics, mechanically.
- **It hurt most where aimed.** Credited (36 profiled rare nodes; 14 scored) median AUC **0.734 < 0.767** uncredited (179). AUC falls with depth (d2 0.816 → d7 0.729).
- **The one positive, and its honest limit:** macro **top-1% lift 7.57 / precision 0.605** — the top 1% by score is 60% true cases, 7.6× the majority baseline (a real, deployment-relevant enrichment the macro AUC hides). BUT the lift-leaders (node 253: lift 51, n_pos 211; node 15: 44, n_pos 282; node 4: 30, n_pos 590) all carry **hundreds of positives** — the enrichment lives in **data-rich** nodes and is **absent from the rare credited tail** (64% of nodes are starved at the prior floor). Interpretability is a clean pass (`--profile-align` overlap median 1.00; stage-2 probe: median 45 in-vocab profile tokens/node, 47% of positive cells carry ≥1) — but it sits on topics the decoder doesn't use.

Verdict (pre-registered): "corr healthy + readout down → **PC closes on the merits**." Engineering succeeded (head scales/couples at K=1498, 412 clean batched-L-BFGS passes); science failed (shaping doesn't discriminate). Recorded as **insight 0087**.

---

## 4. What the wall is actually failing against (the deeper diagnosis)

Synthesizing the ladder, three non-exclusive explanations, in rough order of how much evidence supports them:

1. **The label signal isn't in the pre-index features for rare nodes (a DATA/label ceiling).** 0085's least-aligned nodes are common-acquired presentations; 0087's enrichment is entirely in data-rich nodes; 64% of nodes sit at the starvation floor. A rare-disease patient's *pre-diagnosis* EHR may genuinely look like a generic "sick patient" until the diagnosing event — in which case **no representation of these features fixes case-finding**, because the discriminating information isn't present before the label. This is the possibility that would most change strategy, and it is not ruled out.
2. **Static topic mixtures are the wrong representation (a MODEL ceiling — the frontier's bet).** θ is a bag-of-codes summary; it discards *order and timing*. Conditional / per-feature signal ("what would knowing code X now buy") needs sequential structure. This is what the episode/temporal index is built to test.
3. **The discriminative signal is shared, not node-local (a FACTORIZATION issue).** 0081: it lives in the 8 background topics; 0087: decode weight routes through ancestor/shared topics. The per-node gated block may be the wrong place to look for a node's signal; the ancestor cascade / gate structure may be where it is.

These interact: if (1) holds for the rare tail, (2) and (3) can only help the common nodes — where AUC is already fine.

---

## 5. Reusable machinery built this session (available, tested, pushed)

- **`analysis/pc/evaluate.py` — top-k screening metric** (`_score_label`/`_macro`): per-node + macro `prec_at_k` / `recall_at_k` / `lift_at_k` at `topk_frac` (default 0.01). Lift = precision-in-top-1% ÷ prevalence = "× better than predicting the majority." The single source both readout paths inherit (driver `readout_from_proba`→`_bundle_masked`; distributed `per_node_metric_rows`/`_score_group`). Additive, defensive, byte-identical AUC; parity-tested. **Standing recommendation: read every future verdict on top-k lift AND AUC — it discriminated where macro AUC didn't (0087).**
- **The L-BFGS co-fit head** (`spark_vi/models/topic/pc.py` + `batched_lr.py`, `analysis/cloud/distributed_readout.py` provider): matrix-free amortized batched-L-BFGS supervision head; scales to K=1498. Reusable regardless of the PC verdict.
- **`--fit-save-interval`** (gated_pc_cloud, atomic `_save_fit`): periodic fit-only λ checkpoint every N iters (0120 used 5). A crash leaves a readout-able λ; profile-eta fits can't resume (D5 guard), so a dump is the right insurance.
- **`MAX_ITER=` on the `exp` target** (run_experiment `--max-iter`): force a short fit purely to warm a cache-HIT bundle + manifest for the `--emit-eta` probe on a fresh cluster (bootstrap without a full fit).
- **Driver-OOM hardening pattern** (0120 front matter): the long readout / co-fit fit OOMs the *driver* (SparkUI retained state + broadcast backlog on 8g) around iter 30; fix = `spark.ui.enabled:false`, `CHARM_DRIVER_MEMORY=16g`, `head_inner_iters:1` (0086: non-discriminative), `memoryOverhead` bump. The readout's batched solve checkpoints every 10 iters and **resumes** — re-run after an idle-kill, don't restart.

---

## 6. The frontier: episode / temporal representation (built, not yet truly tested)

- **State:** `docs/experiments/0111-episode-anchored-sampling.md` + `0112-episode-random-control.md` are `status: running`; the machinery (episode index, matched-random control index, driver arms, `diag_episode_probe`, the int64 doc-key seam) is **built** (tasks completed). The normative design is `docs/superpowers/specs/2026-09-01-incident-episode-eval-program.md` and plan `2026-09-03-0111-episode-anchored-sampling-plan.md`.
- **The idea:** anchor documents on *episodes* (temporally-honest windows) rather than a single patient bag, so the model sees within-patient sequence/timing; evaluate **incident** case-finding (predict the node *before* its first coded occurrence, pre-index features only) and, ultimately, **feature-level VOI** (spec §3: `P(c | parent, hist)` conditional factors; "what would knowing code X buy here"). This is a *different axis* than 0120's static-θ shaping.
- **The case FOR:** it attacks explanation (2) directly; VOI needs conditional/sequential structure static θ provably lacks; the incident framing also removes the "post-diagnosis codes leak" confound that can flatter prevalent AUC. If discrimination lives anywhere, temporal structure is the most plausible untried place.
- **The case AGAINST / the honest risk:** it may just **relocate the wall**, not break it. If explanation (1) dominates (the pre-index EHR genuinely lacks rare-node signal), episodes make the *evaluation* more honest (incident, pre-index) but won't create signal that isn't there — and incident AUC could come back *worse* than prevalent, precisely because it strips the leaky post-diagnosis codes that were carrying prevalent AUC. The 0087 top-1% result — enrichment only where data is dense — is weak evidence for (1).

---

## 7. The genuinely open questions (what a fresh read should weigh)

1. **Is the wall a data ceiling or a model ceiling?** (§4.1 vs §4.2.) This is THE question. A cheap discriminating test before more model-building: does a *ceiling* model — LR-on-raw-codes per node (no topic bottleneck), or gradient-boosted trees on the code matrix — beat the topic model's 0.76 macro / rare-node AUC? If even an unconstrained classifier can't separate the rare nodes pre-index, the ceiling is data, and no representation (episodes included) fixes case-finding for them. The eval harness already has an LR-on-codes baseline (`analysis/pc/evaluate.py`); it has not been run at this scale on this branch. **This may be the single highest-value next experiment — it could save the whole episode arc from chasing a data ceiling.**
2. **Is case-finding AUC even the right target?** The user has drifted toward top-1% lift / VOI. If the deliverable is "an interpretable, legible phenotype representation + a screening tool that enriches 7×," 0120 is arguably a *success* (legible + real top-1% enrichment), and "macro AUC below baseline" is the wrong scorecard. Reframing the objective may matter more than the next model.
3. **Should legibility be decoupled from the fit entirely?** 0087's `self-w ≈ 0` says the legible topic and the predictive topic are different objects. A clean design: fit the representation purely for prediction (no word-side prior), attach profile alignment as a *post-hoc label* on outputs. This removes the "prior on the state" contamination the user worried about for VOI, and stops paying AUC for legibility.
4. **Is the gated per-node factorization even the right structure**, given the signal keeps showing up in shared/background/ancestor topics (0081, 0087)?

---

## 8. Entry points (read in this order for a cold start)

1. This file.
2. `docs/insights/0087` (freshest verdict) → `0085` → `0083` (the wall, three sharpest statements).
3. `docs/experiments/0120-*.md` Results (the fair PC test).
4. `docs/superpowers/specs/2026-09-01-incident-episode-eval-program.md` §3 (VOI / conditional factors — the endgame the frontier serves).
5. `AGENTS.md` (conventions, the operational invariants).

## 9. Parked / small open items (none blocking)

- **Credited top-k pooling** in `inspect_topics.build_auc_slice`: the credited *AUC* split is done (0.734<0.767); credited/uncredited *lift* pooling is not wired (per-node + macro lift exist). Off-YARN, no re-run. Offered, user hasn't decided — the per-node picture (lift-leaders all high-n_pos) already tells the story.
- **Bookkeeping wart:** a re-readout populates `results_readout.json` but leaves `manifest.partial='fit-only'` (the re-readout path doesn't flip the marker). Harmless; a one-line fix in `gated_pc_readout` if it annoys.
- `0113`/`0116` are not scored on the current cluster — a true paired `readout-ab` vs 0120 needs them re-scored (bundle HIT now, so cheap-ish) if the paired number (not just recorded-macro comparison) is wanted for a write-up.

## 10. Commits of record (this session, on `claude/gated-conditional-voi`)

- `e98a226` — exp 0120 Results + **insight 0087** (the negative verdict; status→done).
- `6f6471a` — **top-1% screening metric** (evaluate + both readout paths + tests).
- `354f014` — `--fit-save-interval` periodic fit checkpoint (atomic `_save_fit`).
- `26fdde6` — 0120 driver-OOM hardening (ui off, 16g driver, inner_iters 1).
- `fe436d2` — `MAX_ITER=` bundle/manifest bootstrap override.
- `5c58d69` — prior handoff (L-BFGS build + PC-revival engineering).

**Bottom line for the next session:** the PC / word-side-prior line is closed for case-finding AUC (0087). Before pouring effort into the episode frontier, the highest-leverage move is probably §7.1 — a ceiling classifier (LR-on-codes / GBT) on the rare nodes — to settle whether the wall is data or model. That answer decides whether episodes are the path or whether the objective itself should be re-aimed at the interpretable-representation / VOI framing the top-1% result quietly supports.
