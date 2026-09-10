from __future__ import annotations
import argparse, re, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAT=re.compile(r"L(?P<L>\d+)_b(?P<beta>\d+\.\d+)_seed(?P<seed>\d+)_(?P<mode>hot|cold)$")

def acov(x):
    x=np.asarray(x,float); n=len(x); x=x-x.mean()
    m=1<<(2*n-1).bit_length()
    f=np.fft.rfft(x,n=m)
    a=np.fft.irfft(f*np.conjugate(f),n=m)[:n]
    return a/np.arange(n,0,-1)

def tau_int(x):
    a=acov(x)
    if a[0]<=0: return 0.5
    rho=a/a[0]; s=0.0
    for k in range(1,len(rho)-1,2):
        pair=rho[k]+rho[k+1]
        if pair<=0: break
        s+=pair
    return max(0.5,0.5+s)

def split_rhat(chains):
    parts=[]
    for x in chains:
        x=np.asarray(x,float); n=len(x)//2
        parts.extend([x[:n],x[-n:]])
    X=np.stack(parts); n=X.shape[1]
    W=np.mean(X.var(axis=1,ddof=1)); B=n*np.var(X.mean(axis=1),ddof=1)
    return float(np.sqrt(((n-1)/n*W+B/n)/W)) if W>0 else 1.0

def chain_mean_var(x):
    return 2*tau_int(x)*np.var(x,ddof=1)/len(x)

def hotcold_z(chains,modes):
    H=[]; C=[]
    for x,m in zip(chains,modes):
        item=(float(np.mean(x)),float(chain_mean_var(x)))
        (H if m=="hot" else C).append(item)
    if not H or not C: return np.nan
    delta=np.mean([m for m,_ in H])-np.mean([m for m,_ in C])
    var=sum(v for _,v in H)/len(H)**2 + sum(v for _,v in C)/len(C)**2
    return float(delta/math.sqrt(var)) if var>0 else np.nan

def load(root):
    ens={}
    for d in Path(root).iterdir():
        if not d.is_dir(): continue
        m=PAT.match(d.name)
        if not m: continue
        f=d/"measurements.csv"
        if not f.exists(): continue
        k=(int(m.group("L")),float(m.group("beta")))
        ens.setdefault(k,[]).append((int(m.group("seed")),m.group("mode"),pd.read_csv(f)))
    for k in ens: ens[k].sort(key=lambda x:x[0])
    return ens

def circular_block_indices(n,B,block,rng):
    nb=math.ceil(n/block)
    starts=rng.integers(0,n,size=(B,nb))
    idx=(starts[:,:,None]+np.arange(block)[None,None,:])%n
    return idx.reshape(B,-1)[:,:n]

def bootstrap_ensemble(items,B,block,rng):
    xi=[]; m2=[]; m4=[]
    for _,_,d in items:
        idx=circular_block_indices(len(d),B,block,rng)
        xi.append(d.xi_over_L.to_numpy()[idx].mean(axis=1))
        m2.append(d.m2.to_numpy()[idx].mean(axis=1))
        m4.append(d.m4.to_numpy()[idx].mean(axis=1))
    xi=np.mean(xi,axis=0); m2=np.mean(m2,axis=0); m4=np.mean(m4,axis=0)
    return {"xi_over_L":xi,"binder_Q":m2*m2/m4}

def common_support_root(b1,y1,b2,y2):
    b1=np.asarray(b1,float); b2=np.asarray(b2,float)
    y1=np.asarray(y1,float); y2=np.asarray(y2,float)
    lo=max(float(np.min(b1)),float(np.min(b2))); hi=min(float(np.max(b1)),float(np.max(b2)))
    if not lo < hi: return np.nan, None
    grid=sorted(set([float(x) for x in b1 if lo<=x<=hi]+[float(x) for x in b2 if lo<=x<=hi]+[lo,hi]))
    for a,b in zip(grid[:-1],grid[1:]):
        da=np.interp(a,b1,y1)-np.interp(a,b2,y2)
        db=np.interp(b,b1,y1)-np.interp(b,b2,y2)
        if da==0: return float(a),(a,a)
        if da*db<=0:
            root=float(a-da*(b-a)/(db-da))
            if root < a-1e-12 or root > b+1e-12: raise RuntimeError("Root escaped interpolation bracket")
            return root,(a,b)
    return np.nan,None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="results/equilibrium")
    ap.add_argument("--out",default="results/analysis")
    ap.add_argument("--bootstrap",type=int,default=4000)
    ap.add_argument("--block",type=int,default=12)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ens=load(args.root)
    if not ens: raise SystemExit("No production chains found")

    B=args.bootstrap; block=args.block; rng=np.random.default_rng(20260821)
    boot={}; rows=[]
    for (L,beta),items in sorted(ens.items()):
        modes=[m for _,m,_ in items]
        rec={"L":L,"beta":beta,"nchains":len(items),"nmeasurements_per_chain":len(items[0][2])}
        for obs in ["action_density_per_link","m2","xi_over_L"]:
            ch=[d[obs].to_numpy() for _,_,d in items]
            rec[f"{obs}_mean"]=float(np.mean([x.mean() for x in ch]))
            rec[f"{obs}_rhat"]=split_rhat(ch)
            rec[f"{obs}_tau_max"]=float(max(tau_int(x) for x in ch))
            rec[f"{obs}_ess_total"]=float(sum(len(x)/(2*tau_int(x)) for x in ch))
            rec[f"{obs}_hotcold_z"]=hotcold_z(ch,modes)
        b=bootstrap_ensemble(items,B,block,rng); boot[(L,beta)]=b
        rec["xi_over_L_boot_sem"]=float(np.std(b["xi_over_L"],ddof=1))
        m2=np.concatenate([d.m2.to_numpy() for _,_,d in items]); m4=np.concatenate([d.m4.to_numpy() for _,_,d in items])
        rec["binder_Q"]=float(m2.mean()**2/m4.mean())
        rec["binder_Q_boot_sem"]=float(np.std(b["binder_Q"],ddof=1))
        rows.append(rec)

    df=pd.DataFrame(rows).sort_values(["L","beta"])
    df.to_csv(out/"ensemble_diagnostics.csv",index=False)
    Ls=sorted(df.L.unique())
    pairs=[(Ls[i],Ls[i+1]) for i in range(len(Ls)-1)]
    if 16 in Ls:
        for L1 in [10,12]:
            if L1 in Ls: pairs.append((L1,16))
    pairs=list(dict.fromkeys(pairs))

    cross=[]
    for obs,col in [("xi_over_L","xi_over_L_mean"),("binder_Q","binder_Q")]:
        for L1,L2 in pairs:
            r1=df[df.L==L1].sort_values("beta"); r2=df[df.L==L2].sort_values("beta")
            root,bracket=common_support_root(r1.beta.to_numpy(),r1[col].to_numpy(),r2.beta.to_numpy(),r2[col].to_numpy())
            if not np.isfinite(root): continue
            vals=[]; b1=r1.beta.to_numpy(); b2=r2.beta.to_numpy()
            for i in range(B):
                y1=np.array([boot[(L1,float(b))][obs][i] for b in b1]); y2=np.array([boot[(L2,float(b))][obs][i] for b in b2])
                rr,_=common_support_root(b1,y1,b2,y2)
                if np.isfinite(rr): vals.append(rr)
            vals=np.asarray(vals,float)
            a,b=bracket
            end_rows=pd.concat([df[(df.L==L1)&(df.beta.isin([a,b]))],df[(df.L==L2)&(df.beta.isin([a,b]))]])
            max_rhat=float(end_rows.xi_over_L_rhat.max()) if len(end_rows) else np.nan
            max_z=float(np.max(np.abs(end_rows.xi_over_L_hotcold_z))) if len(end_rows) else np.nan
            cross.append({
                "observable":obs,"L1":L1,"L2":L2,"beta_cross":root,
                "bracket_lo":a,"bracket_hi":b,"bracket_max_xi_rhat":max_rhat,
                "bracket_max_abs_xi_hotcold_z":max_z,"bracket_valid":bool(max_rhat<1.02 and max_z<3.0),
                "bootstrap_valid":len(vals),"bootstrap_sd":float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,
                "ci025":float(np.quantile(vals,.025)) if len(vals)>20 else np.nan,
                "ci975":float(np.quantile(vals,.975)) if len(vals)>20 else np.nan,
            })

    cdf=pd.DataFrame(cross)
    cdf.to_csv(out/"crossings_with_bootstrap.csv",index=False)

    fig=plt.figure(figsize=(6.5,4.6)); ax=fig.add_subplot(111)
    for L in Ls:
        r=df[df.L==L].sort_values("beta")
        ax.errorbar(r.beta,r.xi_over_L_mean,yerr=r.xi_over_L_boot_sem,marker="o",capsize=2,label=f"L={L}")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\xi/L$"); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(out/"FIG_xi_over_L.png",dpi=300); fig.savefig(out/"FIG_xi_over_L.pdf"); plt.close(fig)

    fig=plt.figure(figsize=(6.5,4.6)); ax=fig.add_subplot(111)
    for L in Ls:
        r=df[df.L==L].sort_values("beta")
        ax.errorbar(r.beta,r.binder_Q,yerr=r.binder_Q_boot_sem,marker="o",capsize=2,label=f"L={L}")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$Q_B=\langle m^2\rangle^2/\langle m^4\rangle$"); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(out/"FIG_binder_Q.png",dpi=300); fig.savefig(out/"FIG_binder_Q.pdf"); plt.close(fig)

    critical=df[(df.beta>=0.600)&(df.beta<=0.610)]
    summary={
        "chains":int(df.nchains.sum()),"ensembles":int(len(df)),"bootstrap_replicates":B,
        "moving_block_length_measurements":block,
        "max_xi_rhat_0p600_to_0p610":float(critical.xi_over_L_rhat.max()),
        "max_abs_xi_hotcold_z_0p600_to_0p610":float(np.max(np.abs(critical.xi_over_L_hotcold_z))),
        "max_action_rhat_0p600_to_0p610":float(critical.action_density_per_link_rhat.max()),
        "max_m2_rhat_0p600_to_0p610":float(critical.m2_rhat.max()),
        "all_reported_xi_crossing_brackets_valid":bool(cdf[cdf.observable=="xi_over_L"].bracket_valid.all()) if len(cdf[cdf.observable=="xi_over_L"]) else True,
        "interpretation":"finite-size dynamics demonstration only; no precision infinite-volume beta_c or critical exponent is extracted",
    }
    summary["status"]="PASS" if (
        summary["max_xi_rhat_0p600_to_0p610"]<1.02 and summary["max_abs_xi_hotcold_z_0p600_to_0p610"]<3.0
        and summary["max_action_rhat_0p600_to_0p610"]<1.05 and summary["max_m2_rhat_0p600_to_0p610"]<1.03
        and summary["all_reported_xi_crossing_brackets_valid"]
    ) else "REVIEW"
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(cdf.to_string(index=False) if len(cdf) else "No crossings")

if __name__=="__main__":
    main()
