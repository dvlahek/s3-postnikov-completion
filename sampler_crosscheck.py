from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from equilibrium_core_base import (
    periodic_neighbors, hot_spins, seed_numba,
    heatbath_checkerboard_sweep, metropolis_checkerboard_sweep,
    link_observables, _pilot_adaptive_burnin, _advance_fixed_clusters,
)


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


def summarize(x):
    x=np.asarray(x,float); t=tau_int(x)
    sem=math.sqrt(2*t*np.var(x,ddof=1)/len(x))
    return {"mean":float(x.mean()),"sem":float(sem),"tau_int":float(t),"ess":float(len(x)/(2*t))}


def fixed_wolff_series(L,beta,seed,nbetween,nmeas,burn=200.0,washout_intervals=20):
    alln,plus,_=periodic_neighbors(L); V=L**4
    spins=hot_spins(L,seed+991); seed_numba(seed)
    marks=np.zeros(V,np.int32); stack=np.empty(V,np.int32); members=np.empty(V,np.int32)
    mark=0
    _,_,mark=_pilot_adaptive_burnin(spins,alln,beta,burn,marks,stack,members,mark)
    _,_,mark=_advance_fixed_clusters(spins,alln,beta,washout_intervals*nbetween,marks,stack,members,mark)
    out=[]
    for _ in range(nmeas):
        _,_,mark=_advance_fixed_clusters(spins,alln,beta,nbetween,marks,stack,members,mark)
        out.append(link_observables(spins,plus)["action_density_per_link"])
    return np.asarray(out)


def adaptive_negative(L,beta,seed,nmeas):
    alln,plus,_=periodic_neighbors(L); V=L**4
    spins=hot_spins(L,seed+991); seed_numba(seed)
    marks=np.zeros(V,np.int32); stack=np.empty(V,np.int32); members=np.empty(V,np.int32)
    mark=0
    _,_,mark=_pilot_adaptive_burnin(spins,alln,beta,50.0,marks,stack,members,mark)
    out=[]
    for _ in range(nmeas):
        _,_,mark=_pilot_adaptive_burnin(spins,alln,beta,1.0,marks,stack,members,mark)
        out.append(link_observables(spins,plus)["action_density_per_link"])
    return np.asarray(out)


def heatbath_series(L,beta,seed,burn,nmeas):
    alln,plus,parity=periodic_neighbors(L)
    spins=hot_spins(L,seed+991); seed_numba(seed)
    for _ in range(burn): heatbath_checkerboard_sweep(spins,alln,parity,beta)
    out=[]
    for _ in range(nmeas):
        heatbath_checkerboard_sweep(spins,alln,parity,beta)
        out.append(link_observables(spins,plus)["action_density_per_link"])
    return np.asarray(out)


def metropolis_series(L,beta,seed,burn,nmeas):
    alln,plus,parity=periodic_neighbors(L)
    spins=hot_spins(L,seed+991); seed_numba(seed)
    acc=[]
    for _ in range(burn): acc.append(metropolis_checkerboard_sweep(spins,alln,parity,beta))
    out=[]
    for _ in range(nmeas):
        for _ in range(2): acc.append(metropolis_checkerboard_sweep(spins,alln,parity,beta))
        out.append(link_observables(spins,plus)["action_density_per_link"])
    return np.asarray(out),float(np.mean(acc))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--schedule",default="results/fixed_cluster_schedule.json")
    ap.add_argument("--out",default="results/sampler_crosscheck.json")
    ap.add_argument("--measurements",type=int,default=6000)
    args=ap.parse_args()

    sched=json.loads(Path(args.schedule).read_text())
    lookup={(int(e["L"]),round(float(e["beta"]),6)):e for e in sched["entries"]}
    tests=[]
    for beta in [0.600,0.610]:
        L=4
        nb=int(lookup[(L,round(beta,6))]["fixed_clusters_between_measurements"])
        w=fixed_wolff_series(L,beta,41000+int(beta*1000),nb,args.measurements)
        a=adaptive_negative(L,beta,46000+int(beta*1000),args.measurements)
        h=heatbath_series(L,beta,51000+int(beta*1000),1000,args.measurements)
        m,macc=metropolis_series(L,beta,61000+int(beta*1000),2500,args.measurements)
        sw,sa,sh,sm=map(summarize,[w,a,h,m])
        def z(x,y): return (x["mean"]-y["mean"])/math.sqrt(x["sem"]**2+y["sem"]**2)
        tests.append({
            "L":L,"beta":beta,"fixed_clusters_between":nb,
            "fixed_count_wolff":sw,
            "adaptive_measurement_negative_control":sa,
            "exact_heatbath":sh,
            "metropolis":sm,
            "metropolis_acceptance":macc,
            "z_wolff_minus_heatbath":z(sw,sh),
            "z_adaptive_minus_heatbath":z(sa,sh),
            "z_metropolis_minus_heatbath":z(sm,sh),
        })
    status="PASS" if all(abs(t["z_wolff_minus_heatbath"])<=3 for t in tests) else "FAIL"
    result={
        "purpose":"stationary-measure validation of fixed-count production sampling",
        "tests":tests,
        "pass_rule":"|z(Wolff - exact heat bath)| <= 3 at both reference couplings",
        "status":status,
    }
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame([{
        "L":t["L"],"beta":t["beta"],
        "wolff":t["fixed_count_wolff"]["mean"],"wolff_sem":t["fixed_count_wolff"]["sem"],
        "heatbath":t["exact_heatbath"]["mean"],"heatbath_sem":t["exact_heatbath"]["sem"],
        "metropolis":t["metropolis"]["mean"],"metropolis_sem":t["metropolis"]["sem"],
        "adaptive":t["adaptive_measurement_negative_control"]["mean"],
        "adaptive_sem":t["adaptive_measurement_negative_control"]["sem"],
        "z_wolff_heatbath":t["z_wolff_minus_heatbath"],
        "z_adaptive_heatbath":t["z_adaptive_minus_heatbath"],
        "z_metropolis_heatbath":t["z_metropolis_minus_heatbath"],
    } for t in tests]).to_csv(p.with_suffix(".csv"),index=False)
    print(json.dumps(result,indent=2))
    if status!="PASS": raise SystemExit(2)

if __name__=="__main__":
    main()
