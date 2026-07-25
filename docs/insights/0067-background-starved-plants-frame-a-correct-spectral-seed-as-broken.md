# 0067 — A planted corpus with no background documents makes a correct gated spectral seed look broken

**Date:** 2026-07-25
**Topic:** lda | diagnostics | svi
**Status:** Confirmed
**Context:** SP2 of the multi-domain gated DAG LDA arc (branch `multidomain-spectral-init`). Surfaced while building the planted per-domain recovery acceptance test (plan Task 4) on `two_domain_dag_corpus`.

## Finding

`two_domain_dag_corpus` generated every document with a non-root node label, so the corpus contained **zero background documents**. That single omission makes the gated spectral seed (`gated_init.spectral_block_aligned_lambda`) fail in one of two ways, depending on `anchor_scope`, and neither failure looks like a plant problem:

- **`anchor_scope="closure"`** — the background doc pool is *every* document, so background anchors are selected from the full corpus and **steal foreground node signatures**. Measured on a 2-node chain: the two background topics held both nodes' domain-1 signature blocks (top columns at 0.26–0.27 and 0.17) while the node block topics were left holding the shared common pool.
- **`anchor_scope="frontier"`** — the background doc pool is empty-frontier documents only, of which there were none, so `bg_docs == []`, the background block stays pinned at its 1e-9 floor, and the run logs a warning that is easy to swallow in test output.

The consequence read exactly like a model defect. On a 2-deep DAG (`{1:0, 2:1, 3:1}`), per-node/per-domain recovery sat at **~0.28–0.34** against uniform baselines of 0.15/0.125, was stable across corpus seeds 1–8 and model seeds 0–3, and did not move under any plant-strengthening lever (`b_only_signal_boost` to 8, documents to 2000, iterations to 120, `doc_len` to 100, larger vocabularies) — in some configurations it *degraded* with more iterations. Four different init configurations each killed a different node (`frontier/forward` d0 `[0.523, 0.286, 0.455]`; `frontier/reverse` `[0.404, 0.259, 0.488]`; `closure/forward` `[0.150, 0.519, 0.150]`; `closure/reverse` `[0.150, 0.471, 0.430]`).

Two hypotheses were tested and refuted before the real cause was found:

1. **"Dilution caps achievable recovery on a deeper DAG."** The generator does split a document's signature draws across `closure(v)`, so a leaf in a 2-deep DAG gets half the own-signature draws of a leaf in a flat one — a real 2× SNR reduction. But dilution does not lower the *ceiling*: an oracle-initialized fit on the identical corpus, same 800 documents and same 50 iterations, reached **0.9497 / 0.9535**. The plant was fully identifiable all along.
2. **"The deeper DAG needs a different anchor ordering."** Reverse topological order was among the four configurations tried and fixed nothing — consistent with [0063](0063-reverse-topo-spectral-init-is-null-init-geometry-washes-out.md), which had already found that lever null for a different reason.

Adding real background documents (a new `bg_frac` generator parameter, default 0.0 so every existing corpus is byte-identical) moved recovery on the same 3-node DAG to **0.938 / 0.862 / 0.641** (domain 0) and **0.969 / 0.868 / 0.984** (domain 1). Nothing in the engine changed.

## Why it matters

1. **The gated spectral seed has an unstated precondition: the training corpus must contain documents that exercise the background block alone.** Every document here spent `doc_len // 2` on the shared common pool — precisely what a background block should anchor on — but no document carried that pool *without* a node signature, so the background block could not be identified separately from the foreground ones. This is a property of gated anchor recovery, not of this synthetic generator, and it applies to real cohorts: a cohort assembled entirely from labelled cases has the same defect.
2. **A background-starved plant is a silent instrument failure that mimics a model defect.** Two full review rounds were spent treating a correct engine as broken, and a genuine-looking negative was written up before the cause was found. The tell, in hindsight, is *which topics hold the common pool*: if node block topics carry it and the background block does not, suspect the plant before the model.
3. **It questions the premise of [0066](0066-multidomain-random-init-topic-death.md), which has not been re-tested.** 0066 attributed per-seed topic death under random init to a node's signal being "absorbed by the background block" — the same absorption described here — and concluded the spectral seed is load-bearing. All of 0066's measurements predate `bg_frac` and were therefore taken on a background-starved corpus. Whether random multi-domain init still suffers topic death *with* a background pool present is **untested and cheap to check**. Until it is, 0066's conclusion should be read as established for background-starved corpora only.

**Bottom line:** before diagnosing a gated topic model, verify the training corpus contains background-only documents; if it does not, the background block is unidentified and any foreground recovery number is measuring the wrong thing. Recovery of 0.28 became 0.94 on identical engine code.
