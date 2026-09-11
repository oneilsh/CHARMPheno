# Handoff — the instrument was broken, the strip was a no-op, α is closed, the lever is β (for a fresh read after compaction)

**Date:** 2026-09-11
**Branch:** `claude/gated-conditional-voi` (all work committed + pushed; tree clean at `0e4699d`).
**Written for:** a cold reader continuing this line. It replaces the 2026-09-09 handoff
(`2026-09-09-discriminability-wall-and-the-episode-frontier-handoff.md`), whose framing —
"legibility levers don't buy discriminability; data vs model ceiling" — was built on two
measurement defects found since. Read this one first; use the old one only for the arc's
history before 0120.

---

## 0. The state of things in one paragraph

Two days of re-analysis on the cardiovascular-branch gated-PC arc (exps 0113–0122) found
that most of what the arc "measured" was instrument, not model: the leakage strip has been
a silent no-op since the native-Mondo switch at 0110 (all prevalent AUCs carry tracking),
and the readout head was under-regularized and under-converged (ridge 1 → 100 lifts the
SAME θ by +0.03, larger than every AUC delta the insight ladder reported). Re-read at a
converged head and paired per node: **profile-eta is a clean AUC null, the co-fit head is
a real uniform −0.015 tax, and the underlying representation supports ~0.81 prevalent
macro on this branch.** But the per-node blocks are not where that signal lives: a
feature ablation shows own block + background 0.67, + siblings 0.72, + ancestors 0.78,
everything 0.81 — a node's identity sits in its closure's SPARE topics because the E-step
lets ancestor blocks capture children's words (the ELBO optimum). The HPO profile tokens
ARE in the patients (76% coverage) but ancestors own that vocabulary, so a word prior on
the child block has nothing to hold. α — learned, or held children-first — cannot move
this: the competition is decided on β (φ ∝ E[θ]·E[β]; a flat child β loses every word).
**The only lever that ever fed every depth is spectral init (0114: 72% → 1% starved), and
its recorded AUC cost was read with the broken instrument.** The user's goal is "fed and
legible per-node topics" — the thing the small shallow-DAG era delivered — and they want
HPO kept as the meaning source. The in-flight run is 0115 (spectral + binary counts)
re-read at ridge 100; the recommended next build is HPO-guided spectral anchors.

---

## 1. Standing constraints (updated)

- **Branch** `claude/gated-conditional-voi`; cluster preamble per AGENTS.md. Cluster
  restarts ~daily; **`~` and `/tmp` are wiped** — logs go in the RUN DIR (`<run>/driver_log.md`
  from the fit, `<run>/readout_log.md` from re-readouts, wrapper output to `<run>/sweep_log.md`).
  RUNS_DIR = `/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs`.
- **Run dir names are fixed per experiment** (`runs/NNNN-slug`). A re-run OVERWRITES —
  the MAX_ITER=2 bootstrap of 0114 destroyed its spectral λ. 0115's dir is intact.
- **The eta TSV** (`data/ontology/profile_eta_MONDO_0004995.tsv`) dies with the cluster;
  regenerate with `hpoa-profile-survey --emit-codes` then `hpoa-stage2-probe ID=116 --emit-eta`
  (bundle HIT needed) before any `CREDITED=1` read.
- **Egress floor** unchanged. **Cache-key landmine** unchanged (nothing hashed was edited).
- **Instrument defaults changed this session** — read cross-date comparisons with care:
  `readout_l2` (default still 1.0 = the record; new experiments set 100 explicitly and
  the manifest records it; `gated-pc-readout` resolves CLI > manifest > 1.0);
  `optimize_doc_concentration` now actually reaches the gated engine (it was inert on
  every PC fit before `578ace5`); heads sidecar carries `W_std`.
- **User preferences, stated this session:** interpretability over conditional-diagnostic
  AUC ("I don't want to throw away interpretability to achieve it"); dislikes strength
  knobs — derived quantities and inits are acceptable, tunables are not; **keep HPO
  usage** ("spectral has no push to be aligned with node meaning"); tpn=5 disliked, tpn=2
  test judged uninformative and dropped; **SAGE cascade on hold** ("feels like a big
  shift"); open to frontier-only gating and to the dismech ribbon / forest-of-anchors
  (https://github.com/monarch-initiative/dismech) as a strategic frame.

---

## 2. What was found (the four insights, in dependency order)

### 2.1 Insight 0088 — the leakage strip is a no-op on the native-Mondo path
`multi_domain.py:296` strips `{vm[c] for c in before_dag.nodes() if c in vm}`; native
node ids are Mondo numerics (`MONDO:0004995 → 4995`), vocab keys are OMOP concept ids.
Audit on 0120: **0/5000 condition, 0/5000 measurement, 0/1601 drug dims** stripped.
Consequence: every 0110+ prevalent number has the disease's own pre-index codes in the
features (lookback window, population index). 0075's tracking share (0.067 on 0110) was
measured under the same no-op. The incident cohort (`preindex_closure: true`) is the only
leakage control and was OFF for the whole CV arc. Also measured: only 834 of 3,135 HPO
profile concepts are in the 5,000-cap condition vocab at all.

### 2.2 Insight 0089 — the readout was the bottleneck
Same 0120 θ: ridge 1 (68/259 heads converged) 0.7555 → ridge 100 **0.7927** → 1e4 0.7909.
Paired re-read at ridge 100: 0113 0.8087 / 0116 0.8098 / 0120 0.7927;
0116−0113 = −0.0008 median (profile-eta: clean null; credited +0.010 n=14);
0120−0116 = **−0.015 median, 144/193 down, every depth, credited ≈ uncredited** (co-fit
head: real, uniform tax — it perturbs the shared representation; NOT "hurt where aimed",
NOT own-topic decoupling). Ridge-1 artefacts retracted: "credited < uncredited"
(converged: 0.847 vs 0.817 on 0116), the steep depth slide, 0083's spectral cost, 0085's
"costs by 3×". The co-fit head is retired for this architecture: −0.015 AUC, own-block
evidence unmoved (65.2 vs 64.3), its own decoder under-trained (0.64), and it pulls on
blocks that hold no residual.

### 2.3 Insight 0090 — where a node's identity actually lives
Feature ablation on 0116 at ridge 100 (heads restricted to a per-node topic mask):

| head may load on | macro AUC |
|---|--:|
| background only | 0.600 |
| own block + background | 0.669 (0.674 at full K: not truncation) |
| + siblings (`family`) | 0.719 |
| + ancestors (`family-closure`) | 0.777 |
| everything except own | 0.796 |
| everything | 0.810 |

Sibling contrast alone is NOT the mechanism; ancestors' spare topics carry the children's
identity (deflation put it there). Standardized decoder weights: own-block share ~0 for
EVERY group, 95% on other nodes' blocks, also among fed topics only. Profile-support read:
HPO tokens present in a median 76% of positives, yet 29/36 credited nodes keep NO
node-specific residual (ancestors own the vocabulary — presence ≠ specificity); the 7 that
do keep one hold clinical context (PAH: cor pulmonale, VSD, tadalafil) or cohort
demographics (pregnancy; older women), not the HPO phenotype; IDF weighting steered the
prior to the rare-syndromic end (PAH's prior topic = JRA, leukemias, autism).

### 2.4 Insight 0091 — α is closed in both directions; the lever is β

| fit | α | starved | fed through depth | depth-3 median ev |
|---|---|--:|--:|--:|
| 0113 | 0.5 uniform, fixed | 72% | 3 | 160 |
| 0121 | equalized init → learned (≈1/K) | 79% | 2 | 65.8 |
| 0122 | equalized, held (~600× leaf/ancestor) | 77% | 2 | 66.8 |
| 0114 | 0.5 + spectral init (0082) | **1%** | all | (d4 1000) |

Learned α collapses to the floor in a few Newton steps from any init and moves the cliff
UP (capture is the ELBO optimum); a held children-first α feeds nothing extra (the prior
on θ can't route a word to a flat β). Do not spend another fit on α.

---

## 3. Machinery shipped (all tested, all driver/engine-side, nothing hashed touched)

- `inspect_topics.py`: `--strip-audit`, `--collinearity` (profile Jaccard, topic cosine
  vs peers/bg/anc/flat, standardized decoder shares own/bg/anc/desc/other + fed-only +
  top-5 relation census), `--profile-support` (prior-shaped topic vs the block's data-only
  siblings, names via `RESOLVE_NAMES=1`); loader prefers `W_std` in the heads sidecar.
- Heads sidecar persists `W_std`; `_apply_feature_mask` (exact reduced solve, masked
  weights pinned at 0, folded into the ckpt fingerprint); `gated_pc_readout
  --readout-feature-mask {own,own-bg,bg,closure,drop-own,drop-closure,family,family-closure}`
  → `results_readout_<mode>.json` + `readout_heads_gated_pc_<mode>.npz` (record untouched);
  `--readout-theta-topm N` override tags outputs `topm<N>`; `--readout-l2` (fit driver +
  manifest + front-matter `readout_l2`, re-readout resolves CLI > manifest > 1.0);
  readout tees to `<run>/readout_log.md`.
- Gated engine: `equalized_alpha(lay, hist, mean)` + `GatedOnlineLDA.set_alpha_policy`;
  PC estimator `alphaInit` Param + histogram from the document RDD (this is what finally
  makes `optimizeDocConcentration` reach an injected engine); driver `--alpha-init`,
  manifest `alpha_init`, front-matter `alpha_init`.
- Makefile help + AGENTS.md cluster-practice line (nothing durable in `~`/`/tmp`).

---

## 4. In flight (the user has the command; results not yet seen)

**0115 re-read at ridge 100** — spectral init + binary counts (fit-only, dir intact, never
read out): full, `own-bg`, `family-closure`, then `readout-ab ID=115 BASE=113`. The full number landed: **0.7898 vs 0113's 0.8087 (−0.019), detection identical** — the
middle of the tree; the own-bg ablation decides. Since landed (exp 0115 doc, Readout):
- **paired AB vs 0113:** median dAUC −0.020, up/down 42/151, uniform d2–d6 (−0.016 to
  −0.027), d7 flat (n=13). The cost is real and broad — the fed deep levels lose as much
  as the shallow ones. That is the "cost real" branch below.
- **family-closure:** 0.7851 (−0.005 vs full; 0116's was −0.033 vs its full) — with fed
  blocks, own+ancestors carry nearly the whole head.
- **own-bg: 0.7309** (0116 unfed: 0.669). Gap to full 0.059 vs 0116's 0.141: a fed leaf
  block carries its own node's signal — **the per-node block is the right unit when fed.**
Decision tree (RESOLVED 2026-09-11, see exp 0115 Readout → Verdict):
- cost vs 0113 small (≲0.01) → spectral is the base; build HPO-guided anchors (§5.1).
- cost real → guided anchors still fix the pregnancy-by-volume problem, but the base
  needs thought; weigh §5.2–5.4.
Either way, read own-bg: a FED leaf block that still carries no own signal would say the
gate's per-node block is the wrong unit even when fed.

**Resolution:** the "cost real" branch, but own-bg clears the block unit. The feeding
mechanism is sound and the −0.019 tax is uniform across depth, i.e. a head-side cost of
WHICH words spectral anchored (volume-driven, the pregnancy-by-volume problem), not of
feeding per se (this run cannot split the tax from binary counts; 0114 was never read at
ridge 100). So the base does not need rethinking — the anchors do. §5.1 (HPO-guided
anchors on the spectral base) is the build that addresses exactly this; §5.2–5.4 remain
on the table but nothing in 0115 argues for them over §5.1. No build has started.

**Next run (user's call, 2026-09-11): exp 0123 = 0115 with raw counts** (`count_transform:
none`, otherwise verbatim). Splits 0115's −0.019 between the binary representation and
spectral's anchor choice — 0114 was this config but lost its λ and was only read at ridge
1. Read: full / own-bg / family-closure at ridge 100, paired AB vs 0115 AND 0113. The
user's stated prior: binary and log1p both discard counts that carry meaning; if binary
costs case-finding, fix the anchors at the anchor search, not by flattening the data.

Also unfinished: 0121's own-bg ablation (low value now); kill stale `nohup` wrappers
(`jobs -l`, `ps -ef | grep -c "[s]park-submit"`).

---

## 5. Design options on the table (ranked by the user's stated priorities)

1. **HPO-guided spectral anchors (recommended next build).** Spectral = choose each
   node's anchor words by co-occurrence geometry, then recover the block's β from data
   given the anchors (`gated_init.py`, `spectral_init.find_anchors`; scalable sketch path
   at gated_init ~600–750). Let the node's in-vocab HPO tokens that clear the co-occurrence
   floor in its OWN documents be the preferred anchor candidates; the corpus recovers the
   rest; unprofiled nodes (or profiles that don't clear the floor) fall back to the plain
   search. "HPO decides WHICH words, data decides WHAT the topic is" — no strength knob;
   deflation (`seed_rows`) already stops an ancestor-claimed token from anchoring a child.
   Build ≈ the eta wiring: a candidate-preference arg on the anchor search threaded through
   the scalable path + the driver's TSV→vocab mapping (`profile_eta.py` already does it).
   Spectral and profile-eta were NEVER combined in any run; the plain stack (0115 config +
   `profile_eta`) is one config line if a quick look is wanted first.
2. **Document-credit weighting in the λ update** (user likes it). A leaf-attested document
   counts fully toward its frontier block and only fractionally toward each ancestor
   block — a β-side lever, unlike α. Open design question: the credit-sharing rule up the
   DAG (one document's shares sum to 1; frontier largest; decay per level or per fan-out).
   Engine change in the gated E-step's sstats scatter (`gated_lda.local_update`). Spec
   before build.
3. **Frontier-only gating** (stop closures). Kills the CLASS nodes' legible topics (native
   Mondo powers many on closure support alone) but NOT conditional diagnostics, which is a
   decoder property needing the children fed, not a parent topic.
4. **Shallow DAGs: 0071's per-body-system cascade / the dismech ribbon.** The engine has
   fed every node in every shallow regime it was run in (EDS, dementia, rare6: all fed,
   legible, subphenotypes recovered) and never fed depth ≥4 without a sharp β start.
   Sidesteps the competition; keeps within-forest conditional diagnostics; aligns with the
   Monarch dismech effort. Strategic frame regardless of 1–3.
5. **SAGE cascade** (β_node ∝ exp(m + Σ_anc dev + dev_node); `docs/references.md` regime
   b; PG-STM engine excised in `ddcf52d`, recoverable; the refuted part was the θ-side
   offset READOUT, insights 0044–0058, β content unaffected). **ON HOLD per the user.**
   The principled structural fix if 1–4 fail; big engine build, pipeline unchanged.

Closed: α (both directions), the co-fit head (this architecture), profile-eta as a word
prior on child blocks (nothing to hold), tpn as a lever (relocates the competition).

---

## 6. Corrections to the earlier record (so nobody re-derives them)

- 0087 "self-w ≈ 0 is the head's mechanism" → universal, not the head's; "hurt most where
  aimed" → retracted (instrument); verdict (−0.015 uniform) stands paired.
- 0090 point 4 originally said "learned α was on all along" → WRONG; it was inert on the
  PC path (α fixed at 0.5 for 0113–0120); corrected in place.
- 0083 spectral "costs case-finding" → unestablished (ridge-1 instrument); being re-read.
- The 09-09 handoff's "data vs model ceiling" framing and its §7.1 ceiling classifier →
  superseded: run any ceiling test on the INCIDENT cohort or it measures tracking.
- My own first read this session ("the strip removed the CV vocabulary") → wrong; the
  strip removed nothing.

---

## 7. Numbers of record (CV branch, MONDO:0004995, tpn=5, K=1498, C=299, 193 scored nodes)

| exp | change | macro AUC @ ridge 1 | @ ridge 100 | starved | notes |
|---|---|--:|--:|--:|---|
| 0113 | baseline (random, α 0.5 fixed) | 0.7813 | 0.8087 | 72% | fed through d3 |
| 0114 | + spectral init | 0.7567 (unconverged) | — (λ overwritten) | 1% | 0082/0083 |
| 0115 | + spectral + binary counts | never read | **0.7898** (det 0.604) | 1%-ish (0114: 1%) | AB vs 0113: median −0.020, 42/151, uniform by depth; own-bg **0.7309** (0116: 0.669); family-closure 0.7851 |
| 0116 | + profile-eta S=1 | 0.7804 | 0.8098 | 72% | ablation ladder §2.3 |
| 0120 | + profile-eta + L-BFGS head | 0.7555 | 0.7927 | — | head −0.015 paired |
| 0121 | equalized α init → learned | — | 0.7895 | 79% | cliff up a level |
| 0122 | equalized α held | — | not read | 77% | no lift |

All prevalent, no strip in force, closure label mask (negatives = siblings).

---

## 8. Entry points

1. This file. 2. Insights 0091 → 0090 → 0089 → 0088 (newest first; each short).
3. `docs/experiments/0115`, `0121`, `0122` (run logs). 4. `docs/references.md` SAGE +
MixEHR entries if §5.5 reopens. 5. `AGENTS.md` (conventions; the new cluster-log rule).

## 9. Commits of record (this session, oldest first)

`568a425` strip-audit + collinearity + insight 0088 · `9a856b4` W_std in heads; 0087/0088
refined · `7b8b573` decoder shares · `2646ef8` fed-only share + `--readout-l2` ·
`56e490e` readout_l2 as a parameter + insight 0089 · `b58280c` paired ridge-100 re-read ·
`cb2b7a9` feature-mask ablation + profile-support · `31328be` topm tag · `141ec67` family
modes · `72201a1` readout_log tee + AGENTS.md rule · `22ec258` insight 0090 · `578ace5`
α policy wired + 0121/0122 + 0090 correction · `6d35131` 0122 re-spec · `3261a90` 0121
digest · `0e4699d` insight 0091.
