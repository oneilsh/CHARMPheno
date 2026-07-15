"""Local ablation: STM hardening toggles (spectral init + Sigma prior).

Sweeps {random-init baseline, +Sigma-prior, +spectral, +spectral+Sigma-prior}
x sigma_init in {1, 5, 20} on a synthetic non-gated corpus (K_rare=8, V=300,
D=1500, doc_len=30, bg_frac=0.7 as specified in the task brief); prints
(recovery/K_rare, Sigma_max) per cell. Then runs a gated minority block
comparing random-init vs block-aware spectral init on foreground_recovers_group
for the rare arm.

Run from repo root:
    .venv/bin/python analysis/local/stm_ablation.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "spark-vi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "spark-vi", "tests"))

import numpy as np
from _stm_synth import (
    synthetic_ehr_corpus,
    synthetic_gated_corpus,
    fit_stm,
    planted_recovery,
    foreground_recovers_group,
    final_sigma_range,
)
from spark_vi.models.topic.spectral_init import spectral_init_beta
from spark_vi.models.topic.partition import TopicBlockPartition


# ─── Non-gated ablation ──────────────────────────────────────────────────────
# Parameters as specified in the task brief.

K_RARE = 8
V = 300
D = 1500
DOC_LEN = 30
BG_FRAC = 0.7
N_ITER = 200
BATCH = 150
SIGMA_INITS = [1, 5, 20]

print("=" * 70)
print("NON-GATED ABLATION: synthetic_ehr_corpus")
print(f"  K_rare={K_RARE}, V={V}, D={D}, doc_len={DOC_LEN}, bg_frac={BG_FRAC}")
print(f"  n_iter={N_ITER}, batch={BATCH}")
print("=" * 70)

docs, planted = synthetic_ehr_corpus(K_rare=K_RARE, V=V, D=D,
                                      doc_len=DOC_LEN, bg_frac=BG_FRAC, seed=7)

# For spectral rows: use an implicit all-background partition (background_k=K,
# no foreground groups) — the degenerate single-pass anchor-word init case.
all_bg_partition = TopicBlockPartition(group_var="", background_k=K_RARE,
                                       foreground=())

print("\nPre-computing spectral beta0 ...")
beta0 = spectral_init_beta(docs, all_bg_partition, V)
print("  done.")

# Note: the inverse-Wishart Sigma-prior ablation rows were removed when the IW
# prior (sigma_prior_scale / sigma_prior_count) was dropped in favor of the
# pd_complete covariance-selection M-step (Task 2). Sigma conditioning is now
# controlled by the completion + min_pair_support, not an IW shrink lever.
configs = [
    ("random-init (baseline)",          dict()),
    ("+spectral",                        dict(_spectral=True)),
    ("+reference",                       dict(_reference=True)),
    ("+spectral+reference",              dict(_spectral=True, _reference=True)),
]

header = (f"{'Config':<32}"
          + "".join(f"  sigma_init={si:<2} (rec/8, Smax)" for si in SIGMA_INITS))
print("\n" + header)
print("-" * len(header))

# Reference runs get one extra topic (the pinned baseline) so they keep K_RARE
# free topics; precompute a matching spectral beta0 at K_RARE+1.
beta0_ref = spectral_init_beta(
    docs, TopicBlockPartition(group_var="", background_k=K_RARE + 1, foreground=()), V)

for config_name, kwargs in configs:
    use_spectral = kwargs.pop("_spectral", False)
    use_reference = kwargs.pop("_reference", False)
    k_fit = K_RARE + 1 if use_reference else K_RARE
    b0 = beta0_ref if use_reference else beta0
    row = f"{config_name:<32}"
    for si in SIGMA_INITS:
        model_kwargs = dict(kwargs)
        if use_reference:
            model_kwargs["reference_topic"] = True
        init_data = {"spectral_beta": b0} if use_spectral else None
        gp = fit_stm(docs, K=k_fit, V=V, sigma_init=si,
                     n_iter=N_ITER, batch=BATCH, seed=42,
                     init_data=init_data, **model_kwargs)
        beta_hat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        rec = planted_recovery(beta_hat, planted, thresh=0.5)
        _, s_max = final_sigma_range(gp)
        row += f"  ({rec}/{K_RARE}, {s_max:.2e})"
    print(row)

# ─── Gated minority block ─────────────────────────────────────────────────────
# A gated corpus with a minority rare arm (bg_frac=0.5 gives enough foreground
# signal at the smaller per-group vocabulary; rare docs thinned to ~10% of maj).

print("\n" + "=" * 70)
print("GATED MINORITY BLOCK: synthetic_gated_corpus")
print("  Groups: maj / rare, rare thinned to minority (~10% of maj count)")
print("=" * 70)

G_FG_PER_GROUP = 2   # 2 foreground topics per group
G_BG_K = 2           # 2 background topics
G_V = 300
G_DOC_LEN = 40
G_BG_FRAC = 0.5
G_D = 800            # total docs (drawn ~equally maj/rare before thinning)
G_N_ITER = 300
G_BATCH = 60

docs_full, planted_g, part = synthetic_gated_corpus(
    groups=("maj", "rare"),
    fg_per_group=G_FG_PER_GROUP,
    bg_k=G_BG_K,
    V=G_V,
    D=G_D,
    doc_len=G_DOC_LEN,
    bg_frac=G_BG_FRAC,
    seed=13,
)

# Thin the rare group to ~10% to simulate minority imbalance.
maj_docs = [d for d in docs_full if "maj" in d.groups]
rare_docs_all = [d for d in docs_full if "rare" in d.groups]
rng_thin = np.random.default_rng(99)
keep_mask = rng_thin.random(len(rare_docs_all)) < 0.10
rare_docs = [d for d, keep in zip(rare_docs_all, keep_mask) if keep]
docs_gated = maj_docs + rare_docs

print(f"  maj docs: {len(maj_docs)}, rare docs (after thinning): {len(rare_docs)}")
print(f"  K total: {part.K}  (bg={G_BG_K}, maj-fg={G_FG_PER_GROUP}, rare-fg={G_FG_PER_GROUP})")
print(f"  n_iter={G_N_ITER}, batch={G_BATCH}")

print("\nPre-computing block-aware spectral beta0 for gated corpus ...")
beta0_gated = spectral_init_beta(docs_gated, part, G_V)
print("  done.")

print("\nGated results  -- foreground_recovers_group('rare', thresh=0.5):")

gp_rand = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=5,
                   n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                   partition=part)
beta_rand = gp_rand["lambda"] / gp_rand["lambda"].sum(axis=1, keepdims=True)
rec_rand = foreground_recovers_group(beta_rand, part, "rare", planted_g, thresh=0.5)
_, smax_rand = final_sigma_range(gp_rand)
print(f"  random-init (sigma_init=5):          recovers rare = {rec_rand},  Sigma_max = {smax_rand:.2e}")

gp_spec = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=5,
                   n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                   partition=part,
                   init_data={"spectral_beta": beta0_gated})
beta_spec = gp_spec["lambda"] / gp_spec["lambda"].sum(axis=1, keepdims=True)
rec_spec = foreground_recovers_group(beta_spec, part, "rare", planted_g, thresh=0.5)
_, smax_spec = final_sigma_range(gp_spec)
print(f"  block-aware spectral (sigma_init=5): recovers rare = {rec_spec},  Sigma_max = {smax_spec:.2e}")

gp_ref = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=1,
                 n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                 partition=part, reference_topic=True)
beta_ref = gp_ref["lambda"] / gp_ref["lambda"].sum(axis=1, keepdims=True)
rec_ref = foreground_recovers_group(beta_ref, part, "rare", planted_g, thresh=0.5)
_, smax_ref = final_sigma_range(gp_ref)
print(f"  reference (sigma_init=1):            recovers rare = {rec_ref},  Sigma_max = {smax_ref:.2e}")

gp_sr = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=1,
                n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                partition=part, reference_topic=True,
                init_data={"spectral_beta": beta0_gated})
beta_sr = gp_sr["lambda"] / gp_sr["lambda"].sum(axis=1, keepdims=True)
rec_sr = foreground_recovers_group(beta_sr, part, "rare", planted_g, thresh=0.5)
_, smax_sr = final_sigma_range(gp_sr)
print(f"  spectral+reference (sigma_init=1):   recovers rare = {rec_sr},  Sigma_max = {smax_sr:.2e}")

print("\nDone.")
