# -*- coding: utf-8 -*-
# VENDORED VERBATIM from agents/L16/src/feats2.py @ origin/agent/L16.
# The ONLY edits are the removal of the module-level absolute paths and of the two data
# loaders that used them (`common.load`, `feats2.load2`); every arithmetic line is L16's,
# because a 5.7e-14 difference in one feature column is enough to flip a GBDT split and
# stop `router_s.npy` reproducing bit-for-bit (measured -- a reimplementation drifted by
# 0.0118 in router score on 7,500 rows).
# -*- coding: utf-8 -*-
"""Gate-agnostic cheap features for the router.  build2(d, gate_key, extra_gates=[...]).

`extra_gates` adds a cross-gate block (each extra gate is another LLM pass -- price it).
Everything else is free: the gate's own posterior, the two GBDT encoders, text/group statistics.
"""
import numpy as np, pandas as pd
from .common import probs, norm_log

def _stats(L,p):
    P=probs(L); S=np.sort(P,1); SL=np.sort(L,1)
    return {p+'_p0':P[:,0],p+'_p1':P[:,1],p+'_p2':P[:,2],p+'_pmax':S[:,-1],p+'_p2nd':S[:,-2],
            p+'_pmin':S[:,0],p+'_marg':S[:,-1]-S[:,-2],
            p+'_marg_nats':np.clip(SL[:,-1]-SL[:,-2],0,60),p+'_marg23':S[:,-2]-S[:,0],
            p+'_ent':-(P*np.log(np.clip(P,1e-12,None))).sum(1),p+'_argmax':L.argmax(1).astype(float)}

def _kl(A,B): return (probs(A)*(A-B)).sum(1)

def build2(d, gate_key='c3_tcbg', extra_gates=()):
    G=d['gate_'+gate_key] if 'gate_'+gate_key in d else d[gate_key]
    te=d['te']; n=len(G); ga=G.argmax(1); f={}
    f.update(_stats(G,'g'))
    for c in range(3): f['g_is_%d'%c]=(ga==c).astype(float)
    encs={'e1':d['e_tyonly_cal'],'e2':d['e_s1_ty']}
    for k,E in encs.items():
        f.update(_stats(E,k))
        f[k+'_agree']=(E.argmax(1)==ga).astype(float)
        f[k+'_kl_ge']=_kl(G,E); f[k+'_kl_eg']=_kl(E,G)
        f[k+'_p_gatepick']=probs(E)[np.arange(n),ga]
        f[k+'_dot']=(probs(E)*probs(G)).sum(1)
    f['e12_agree']=(encs['e1'].argmax(1)==encs['e2'].argmax(1)).astype(float)
    EB=norm_log(np.log((probs(encs['e1'])+probs(encs['e2']))/2))
    f.update(_stats(EB,'eb')); f['eb_agree']=(EB.argmax(1)==ga).astype(float)
    f['eb_p_gatepick']=probs(EB)[np.arange(n),ga]
    f['gate_minus_eb']=probs(G)[np.arange(n),ga]-f['eb_p_gatepick']
    f['eb_rank_of_gatepick']=(probs(EB)>probs(EB)[np.arange(n),ga][:,None]).sum(1).astype(float)
    # ---- cross-gate block: each entry is another paid LLM pass ------------------
    for xg in extra_gates:
        X=d['gate_'+xg]; xa=X.argmax(1); p=xg[:6]
        f.update(_stats(X,p))
        f[p+'_agree']=(xa==ga).astype(float)
        f[p+'_p_gatepick']=probs(X)[np.arange(n),ga]
        f[p+'_kl']=_kl(G,X); f[p+'_kl_r']=_kl(X,G)
        f[p+'_js']=0.5*(_kl(G,np.log((probs(G)+probs(X))/2))+_kl(X,np.log((probs(G)+probs(X))/2)))
        f[p+'_margdiff']=f['g_marg']-f[p+'_marg']
        # 3-way: does the extra gate side with the encoders against this gate?
        f[p+'_sides_with_eb']=((xa==EB.argmax(1))&(xa!=ga)).astype(float)
    # ---- free text / group statistics ------------------------------------------
    q=te['query'].fillna('').astype(str); t=te['title'].fillna('').astype(str)
    cat=te['category'].fillna('').astype(str); br=te['brand'].fillna('').astype(str)
    at=te['attributes'].fillna('').astype(str)
    f['q_len']=q.str.len().values.astype(float); f['q_tok']=q.str.split().str.len().values.astype(float)
    f['t_len']=t.str.len().values.astype(float); f['t_tok']=t.str.split().str.len().values.astype(float)
    f['cat_depth']=cat.str.count('/').values.astype(float)
    f['n_attr']=at.str.count(',').values.astype(float)
    f['brand_unknown']=br.isin(['','unknown']).values.astype(float)
    f['gender_unknown']=(te['gender'].astype(str)=='unknown').values.astype(float)
    f['grp_size']=q.map(q.value_counts()).values.astype(float)
    f['cat_size']=cat.map(cat.value_counts()).values.astype(float)
    qt=q.str.split().apply(set); tt=t.str.split().apply(set)
    f['tok_overlap']=np.array([len(a&b)/max(len(a),1) for a,b in zip(qt,tt)])
    f['tok_jacc']=np.array([len(a&b)/max(len(a|b),1) for a,b in zip(qt,tt)])
    f['q_in_t']=np.array([float(x in y) for x,y in zip(q,t)])
    df=pd.DataFrame({'q':q.values,'m':f['g_marg_nats'],'a':ga})
    f['grp_mean_marg']=df.groupby('q')['m'].transform('mean').values
    f['grp_n_classes']=df.groupby('q')['a'].transform('nunique').values.astype(float)
    f['marg_minus_grp']=f['g_marg_nats']-f['grp_mean_marg']
    dfc=pd.DataFrame({'c':cat.values,'m':f['g_marg_nats']})
    f['marg_minus_cat']=f['g_marg_nats']-dfc.groupby('c')['m'].transform('mean').values
    return pd.DataFrame(f).replace([np.inf,-np.inf],0).fillna(0)
