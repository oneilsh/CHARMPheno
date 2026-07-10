"""5-plant bias inversion for the held-out generative-scale calibration (insight 0044, closing sec 5).

Indirect inference: at the production regime (K=60, V=5000, len=44) plant at several
KNOWN scales, measure each estimator's recovered c* as a function of the true scale per
holdout fraction (the "bias map"), then invert the real-corpus readings (exp 0047)
through the measured maps to recover the bias-corrected true scale. Answers: do the MAP
and marginalized estimators reconcile to a common corrected scale, does the corrected
scale still drift across holdout (genuine misspecification), and is the shipped MAP scale
too high/low. Reuses the exp 0046 harness; pure numpy, no Spark. Caveat: the bias map is
computed on a synthetic beta (make_shared_beta), a MODEL of the real beta -- the inversion
assumes that bias transfers.
"""
import json, numpy as np
from spark_vi.eval.topic.concentration_recovery import (
    make_shared_beta, plant_corpus, sweep_heldout, sweep_heldout_marginalized)
from spark_vi.mllib.topic.stm import smooth_scale_log_quadratic

# Production regime, matching exp 0047's real run knobs
K, V, DL, D, S = 60, 5000, 44, 1000, 64
GRID = [round(x,4) for x in np.geomspace(0.5, 32.0, 13)]
HOLDOUTS = [0.5, 0.8, 0.95]
PLANTS = [2.0, 3.5, 5.0, 7.0, 10.0]
beta = make_shared_beta(K=K, V=V, seed=0)

def cstar(res):
    return smooth_scale_log_quadratic(res["lls"])["c_star"]

# bias maps: est_map[estimator][f] = list of (c_true, chat) over PLANTS
rows = {f: {"MAP": [], "MARG": []} for f in HOLDOUTS}
for ct in PLANTS:
    docs, _ = plant_corpus(beta, D=D, doc_len=DL, mechanism="logistic_normal", level=ct, seed=1)
    for f in HOLDOUTS:
        m = cstar(sweep_heldout(docs, beta, method="stm", knobs=GRID, holdout_frac=f, seed=0))
        g = cstar(sweep_heldout_marginalized(docs, beta, knobs=GRID, holdout_frac=f, seed=0, n_samples=S))
        rows[f]["MAP"].append((ct, m)); rows[f]["MARG"].append((ct, g))
    print(f"[plant c_true={ct}] MAP={[round(rows[f]['MAP'][-1][1],2) for f in HOLDOUTS]} "
          f"MARG={[round(rows[f]['MARG'][-1][1],2) for f in HOLDOUTS]}", flush=True)

def invert(pairs, chat_real):
    # pairs: list of (c_true, chat) ; invert chat_real -> implied c_true via log-log interp
    ct = np.array([p[0] for p in pairs]); ch = np.array([p[1] for p in pairs])
    order = np.argsort(ch)  # need chat increasing
    lch, lct = np.log(ch[order]), np.log(ct[order])
    outside = chat_real < ch.min() or chat_real > ch.max()
    return float(np.exp(np.interp(np.log(chat_real), lch, lct))), outside, (float(ch.min()), float(ch.max()))

# Real readings from exp 0047
REAL = {
  "MAP_full":  {0.5:4.61, 0.8:3.75, 0.95:3.65},
  "MAP_samp":  {0.5:5.30, 0.8:3.90, 0.95:3.80},
  "MARG_samp": {0.5:2.36, 0.8:2.65, 0.95:3.76},
}
print("\n=== BIAS MAPS  chat_est(c_true) per holdout ===")
for f in HOLDOUTS:
    print(f"f={f}:")
    print(f"   c_true : {[p[0] for p in rows[f]['MAP']]}")
    print(f"   MAP  ch: {[round(p[1],2) for p in rows[f]['MAP']]}")
    print(f"   MARG ch: {[round(p[1],2) for p in rows[f]['MARG']]}")

print("\n=== INVERSION: implied true c_true from each real reading (bias-corrected) ===")
print(f"{'f':>5} | {'MAP_full->':>12} {'MAP_samp->':>12} {'MARG_samp->':>12}")
inv = {}
for f in HOLDOUTS:
    a,ao,ar = invert(rows[f]["MAP"], REAL["MAP_full"][f])
    b,bo,br = invert(rows[f]["MAP"], REAL["MAP_samp"][f])
    c,co,cr = invert(rows[f]["MARG"], REAL["MARG_samp"][f])
    inv[f] = (a,b,c)
    flag = lambda o: "*" if o else " "
    print(f"{f:>5} | {a:>11.2f}{flag(ao)} {b:>11.2f}{flag(bo)} {c:>11.2f}{flag(co)}")
print("(* = real reading outside the measured synthetic range -> extrapolated, unreliable)")

def drift(vals): return max(vals)-min(vals)
print("\n=== corrected-scale DRIFT across f (estimator bias removed; residual = misspecification) ===")
for name,idx in [("MAP_full",0),("MAP_samp",1),("MARG_samp",2)]:
    vals=[inv[f][idx] for f in HOLDOUTS]
    print(f"  {name:>10}: implied c_true across f = {[round(v,2) for v in vals]}  drift={drift(vals):.2f}")
print("\n=== reconciliation: MAP_samp vs MARG_samp implied c_true per f ===")
for f in HOLDOUTS:
    print(f"  f={f}: MAP->{inv[f][1]:.2f}  MARG->{inv[f][2]:.2f}  ratio={inv[f][1]/inv[f][2]:.2f}")
json.dump({"rows":{str(f):{k:rows[f][k] for k in rows[f]} for f in HOLDOUTS},
           "inversion":{str(f):inv[f] for f in HOLDOUTS}}, open(f"{__import__('os').environ.get('SCR','/tmp')}/bias_inv_out.json","w"))
