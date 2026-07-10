"""Real-beta transfer check for the bias inversion (insight 0044, Fable's prerequisite).

Tests whether the MAP under-recovery ratio (~0.66) measured under SYNTHETIC beta
transfers to the DEPLOYED beta. Loads the real fitted beta from a dashboard bundle's
model.json, plants at known scales over it (non-gated, mean-0, Sigma=c*I -- same
instrument as scripts/marginalized_scale_bias_inversion.py, only beta swapped), recovers
via the MAP held-out sweep, and prints the recovered c* and the c*/c_true ratio side by
side with the synthetic-beta ratio. If the ratio matches, the 0.66 is a property of the
estimator geometry (K, doc length, MAP-under-prior), not of the synthetic beta -- so the
bias correction is earned and has provenance under the deployed emission matrix.

Run: BUNDLE=/path/to/unzipped/bundle python scripts/realbeta_bias_transfer_check.py
"""
import json, os, numpy as np
from spark_vi.eval.topic.concentration_recovery import make_shared_beta, plant_corpus, sweep_heldout
from spark_vi.mllib.topic.stm import smooth_scale_log_quadratic

root=os.environ["BUNDLE"]
beta_real=np.array(json.load(open(os.path.join(root,"model.json")))["beta"], dtype=float)
beta_real=beta_real/beta_real.sum(axis=1, keepdims=True)   # renormalize rows to exact simplex
K,V=beta_real.shape
DL,D=44,1000
GRID=[round(x,4) for x in np.geomspace(0.5,32.0,13)]
HOLDOUTS=[0.5,0.8,0.95]
PLANTS=[2.0,3.5,5.0,7.0]
beta_syn=make_shared_beta(K=K,V=V,seed=0)

def cstar(docs,beta,f):
    return smooth_scale_log_quadratic(sweep_heldout(docs,beta,method="stm",knobs=GRID,holdout_frac=f,seed=0)["lls"])["c_star"]

print(f"# real-beta bias map  K={K} V={V} D={D} len={DL}  (MAP only)")
print(f"{'c_true':>6} | {'REAL-beta MAP c* (f=.5/.8/.95)':>34} | ratio c*/c_true (f=.5) || {'SYNTH-beta MAP c*':>22} | ratio")
for ct in PLANTS:
    docs_r,_=plant_corpus(beta_real,D=D,doc_len=DL,mechanism="logistic_normal",level=ct,seed=1)
    docs_s,_=plant_corpus(beta_syn ,D=D,doc_len=DL,mechanism="logistic_normal",level=ct,seed=1)
    r=[cstar(docs_r,beta_real,f) for f in HOLDOUTS]
    s=[cstar(docs_s,beta_syn ,f) for f in HOLDOUTS]
    print(f"{ct:>6} | {r[0]:>10.2f} {r[1]:>10.2f} {r[2]:>10.2f} | {r[0]/ct:>18.3f} || "
          f"{s[0]:>7.2f} {s[1]:>6.2f} {s[2]:>6.2f} | {s[0]/ct:.3f}", flush=True)
print("DONE")
