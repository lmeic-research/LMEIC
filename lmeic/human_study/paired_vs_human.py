"""E1c: paired model comparisons against the human majority label (thread-cluster bootstrap).

Moved into the repo from a scratch script on 2026-10-26 (audit finding: the output
e1b_supplement/paired_vs_human.csv had no registered generator). Logic unchanged; adds a
"no_overlap" subset that drops cross-role items whose thread is in the training pool, where the
models have seen the same thread (audit P1).

Usage: python -m lmeic.human_study.paired_vs_human

Requires the confidential EMAIL-3R human-study files (items_with_labels.jsonl and the
annotation database); it is included to document the analysis and does not run without them.

"""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from lmeic.human_study.agreement import SYSTEMS, load_system, macro_f1, LAB, _REPO_ROOT
it=pd.read_json("datasets/email3r_runs/rev1_e1_human_r2_20261025_130349/items_with_labels.jsonl", lines=True)
it["pred_id"]=it.pred_id.astype(str)
d=it[it.human_majority.notna()].reset_index(drop=True)
y=d.human_majority.map(LAB).to_numpy()
P={n:load_system(k,_REPO_ROOT/p) for n,(k,p) in SYSTEMS.items()}
keys=list(zip(d.pred_set,d.pred_id)); rng=np.random.default_rng(20261025); out=[]
for src in ["core_test","fresh_holdout","cross_role","all","no_overlap"]:
    if src=="all": m=np.ones(len(d),bool)
    elif src=="no_overlap": m=(d.eval_tier!="thread_in_training_pool").to_numpy()
    else: m=(d.source==src).to_numpy()
    idx=np.where(m)[0]
    names=[n for n in P if all(keys[i] in P[n] for i in idx)]
    pr={n:np.array([P[n][keys[i]] for i in idx]) for n in names}
    yy=y[idx]; th=d.thread_id.to_numpy()[idx]; u=np.unique(th); mem={t:np.where(th==t)[0] for t in u}
    B=[np.concatenate([mem[t] for t in rng.choice(u,len(u))]) for _ in range(2000)]
    for x,z in [("R3_full","E11_lora_qwen25_role"),("R3_full","R2_explicit"),("E14_both_full","E11_lora_qwen25_role"),("E11_lora_qwen25_role","R2_explicit")]:
        if x in pr and z in pr:
            dd=np.array([macro_f1(yy[b],pr[x][b])-macro_f1(yy[b],pr[z][b]) for b in B]); obs=macro_f1(yy,pr[x])-macro_f1(yy,pr[z])
            r=dict(source=src,n=len(idx),a=x,b=z,diff_pp=round(obs*100,2),lo=round(np.percentile(dd,2.5)*100,2),hi=round(np.percentile(dd,97.5)*100,2),p=round(2*min((dd<=0).mean(),(dd>=0).mean()),3))
            out.append(r); print(r)
pd.DataFrame(out).to_csv("datasets/email3r_runs/rev1_e1_human_r2_20261025_130349/e1b_supplement/paired_vs_human.csv",index=False)
