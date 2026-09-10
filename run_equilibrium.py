from __future__ import annotations
import argparse, json
from pathlib import Path
from equilibrium_core import run_chain

SEEDS=[17,71,113,197]

def start_for(seed):
    return "cold" if seed in (17,71) else "hot"

def tag_beta(beta):
    return f"{beta:.4f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--schedule",default="results/fixed_cluster_schedule.json")
    ap.add_argument("--out",default="results/equilibrium")
    ap.add_argument("--measurements",type=int,default=None)
    ap.add_argument("--burnin-sweeps",type=float,default=None)
    ap.add_argument("--washout-intervals",type=int,default=None)
    args=ap.parse_args()

    sched=json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    profile=sched.get("profile","paper")
    measurements=args.measurements or (60 if profile=="quick" else 800)
    burnin=args.burnin_sweeps or (20.0 if profile=="quick" else 200.0)
    washout=args.washout_intervals or (5 if profile=="quick" else 20)

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    lookup={(int(e["L"]),round(float(e["beta"]),6)):e for e in sched["entries"]}
    seeds=[17,113] if profile=="quick" else SEEDS
    summaries=[]

    for L,betas in sched["grid"].items():
        L=int(L)
        for beta in betas:
            beta=float(beta)
            e=lookup[(L,round(beta,6))]
            nb=int(e["fixed_clusters_between_measurements"])
            for seed in seeds:
                start=start_for(seed)
                d=out/f"L{L}_b{tag_beta(beta)}_seed{seed}_{start}"
                print(f"RUN L={L} beta={beta:.4f} seed={seed} {start} burn={burnin} washout={washout} between={nb}",flush=True)
                summaries.append(run_chain(
                    L,beta,seed,start,d,
                    fixed_clusters_between_measurements=nb,
                    measurements=measurements,
                    adaptive_burnin_sweeps=burnin,
                    fixed_washout_intervals=washout,
                ))

    campaign={
        "profile":profile,
        "sampler":"adaptive discarded burn-in + deterministic fixed-count washout + fixed-count production",
        "measurements_per_chain":measurements,
        "adaptive_burnin_target_sweeps":burnin,
        "fixed_washout_intervals":washout,
        "seeds":seeds,
        "summaries":summaries,
    }
    (out/"campaign_summary.json").write_text(json.dumps(campaign,indent=2),encoding="utf-8")
    print(f"WROTE {out/'campaign_summary.json'}")

if __name__=="__main__":
    main()
