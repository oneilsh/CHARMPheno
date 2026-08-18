"""The compact post-fit report (gated_pc_report.build_report): drops the bulky
per-parent / λ-mass tables + per-iter spam, keeps the signal lines, and surfaces a
weirdness scan + fit-health trajectory."""
from gated_pc_report import build_report

# A miniature summary.md with one of every line SHAPE from a real gated_pc run.
_SUMMARY = """\
[mondo]   powered terminals=273, class nodes=163, branch=MONDO:0004995
[driver]   ledger: {"kept": 437, "dropped": 0, "test_fg_docs": 45992}
[cost]   fan-out (children/parent):  max=14  p90=4  p99=12
[cost]   head Fisher MEMORY:  dense C*K^2=657.3MB   collected C*S^2=1.9MB (S=24)
[driver]   iter 1/100: ELBO=-59238899.0114, batch=22086, rho=0.1190, 51.5s
[driver]     α[min=0.002252 max=0.002252 mean=0.002252], |w_CK|max=32.9, weight_y=50, corr_relΔλ=4.10e-03
[driver]   iter 100/100: ELBO=-25944399.7250, batch=22037, rho=0.0742, 31.5s
[driver]     α[min=0.002252 max=0.002252 mean=0.002252], |w_CK|max=273, weight_y=50, corr_relΔλ=1.00e-06
[driver]   gated_pc (pc_topics_lr): macro AUC=0.7365 AP=0.5182 (over 177 nodes)
[driver]   gated_pc (pc_topics_lr): detection (case vs bg) AUC=0.5000 AP=0.7777 prev=0.778
[conditional sharpening: gated_pc]  P(child|parent), by DAG depth  ECE=0.0057
  depth  #edges  cond_AUC  |  cond_AP  marg_AP  lift  (context)  top1
      1      10  0.7604  |  0.4418  0.3786  0.0632            0.8486
      8       4  0.7226  |  0.6682  0.2203  0.4479            0.8349
  per-node reliability (ECE over 177 nodes): mean=0.0588  max=0.3211 (worst X->Y)  vs pooled=0.0057
    d1 cardiovascular disorde top1=0.849 (majority=0.800) bal_acc=0.134  (n=12868, 10 children)
    d2 vascular disorder      top1=0.851 (majority=0.825) bal_acc=0.096  (n=8221, 13 children)
[driver]   HEAD-FORMULATION LADDER  cond_AUC (frozen θ, localized support):
[driver]     co-fit head (as TRAINED)                       0.523
[driver]     engine Newton [rel-ridge, no-icpt, CONVERGED]  0.610
[per-node domain λ-mass]  node                      condition  measuremen        drug
  atrial fibrillation           0.242       0.049       0.709
  optic choroid disorder        0.427       0.025       0.548
[driver]   HEADLINE (gated_pc vs unsup_gated):
[driver]     pc_topics_lr  AUC 0.7365 vs 0.7365 (Δ-0.0000)
26/08/18 02:01:13 WARN YarnAllocator: Container from a bad node ... Exit status: 143
"""


def test_report_drops_bulk_keeps_signal_and_flags():
    r = build_report(_SUMMARY, title="0091-x")

    # keeps the signal lines verbatim
    for keep in ("[mondo]", '"kept": 437', "[cost]", "macro AUC=0.7365",
                 "detection (case vs bg)", "per-node reliability",
                 "cond_AUC", "0.7604", "HEAD-FORMULATION LADDER",
                 "co-fit head (as TRAINED)", "HEADLINE", "Δ-0.0000"):
        assert keep in r, f"missing signal: {keep!r}"

    # drops the bulk
    assert "top1=0.849" not in r                        # per-parent table row
    assert "atrial fibrillation" not in r               # λ-mass table row
    assert "α[min=" not in r                            # per-iter diag spam
    assert "iter 1/100: ELBO=" not in r                 # per-iter ELBO line (echo)

    # fit-health trajectory (extracted from the dropped per-iter lines)
    assert "FIT-HEALTH" in r
    assert "ELBO" in r and "|w_CK|max" in r and "corr_relΔλ" in r
    # weirdness: head blowup (|w|=273) + starved correction (corr max 4.1e-3? -> not
    # starved since >1e-3) + the worker death.
    assert "head-blowup" in r and "273" in r
    assert "executor-loss" in r                         # the WARN bad-node line
    # corr max is 4.1e-3 (> 1e-3) so NOT flagged starved here
    assert "pc-no-op" not in r


def test_report_digests_only_latest_appended_run():
    """summary.md is append-only; the report defaults to the most recent ## section."""
    multi = (
        "## Fit 2026-08-17 (old run)\n"
        "[driver]   gated_pc (pc_topics_lr): macro AUC=0.5000 AP=0.1111 (over 9 nodes)\n"
        "[driver]   |w_CK|max=999, weight_y=50, corr_relΔλ=9.9e-01\n"
        "## Fit 2026-08-18 (new run)\n"
        "[driver]   gated_pc (pc_topics_lr): macro AUC=0.7365 AP=0.5182 (over 177 nodes)\n"
        "[driver]   |w_CK|max=42, weight_y=50, corr_relΔλ=5.0e-03\n"
    )
    r = build_report(multi)
    assert "new run" in r and "0.7365" in r             # latest section digested
    assert "0.5000" not in r and "999" not in r         # old section excluded
    assert "head-blowup" not in r                        # 999 was the OLD run only

    # --all-sections opts into the whole file.
    r_all = build_report(multi, all_sections=True)
    assert "0.5000" in r_all and "0.7365" in r_all


def test_report_flags_starved_correction_and_falling_elbo():
    s = ("[driver]  iter 1/2: ELBO=-9\n"
         "[driver]   |w_CK|max=5, weight_y=50, corr_relΔλ=2.0e-07\n"
         "[driver]  iter 2/2: ELBO=-10\n"       # ELBO FELL (-9 -> -10): divergence
         "[driver]   |w_CK|max=6, weight_y=50, corr_relΔλ=1.0e-07\n")
    r = build_report(s)
    assert "pc-no-op" in r                              # corr max 2e-7 < 1e-3
    assert "STARVED" in r
    assert "FELL" in r                                  # ELBO should rise; it fell
