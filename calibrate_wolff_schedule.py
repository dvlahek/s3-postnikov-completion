from __future__ import annotations
import argparse, json
from pathlib import Path
from equilibrium_core_base import calibrate_fixed_schedule, reproducibility_self_test


def campaign_grid(profile):
    if profile == "quick":
        return {4:[0.600,0.610]}
    return {
        4:[0.595,0.600,0.605,0.6075,0.610,0.615],
        6:[0.595,0.600,0.605,0.6075,0.610,0.615],
        8:[0.600,0.605,0.6075,0.610,0.615],
        10:[0.600,0.605,0.6075,0.610,0.615],
        12:[0.600,0.605,0.6075,0.610,0.615],
        16:[0.6025,0.605,0.6075,0.610],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",choices=["quick","paper"],default="paper")
    ap.add_argument("--out",default="results/fixed_cluster_schedule.json")
    args=ap.parse_args()

    grid=campaign_grid(args.profile)
    pilot_clusters=200 if args.profile=="quick" else 600
    pilot_burnin=10.0 if args.profile=="quick" else 50.0
    between=1.0 if args.profile=="quick" else 2.0

    entries=[]
    for L,betas in grid.items():
        for beta in betas:
            print(f"CALIBRATE L={L} beta={beta:.4f}", flush=True)
            e=calibrate_fixed_schedule(
                L, beta,
                pilot_burnin_sweeps=pilot_burnin,
                pilot_clusters=pilot_clusters,
                target_between_sweep_equiv=between,
                target_therm_sweep_equiv=1.0,
            )
            e.pop("fixed_thermalization_clusters", None)
            e.pop("target_therm_sweep_equiv", None)
            entries.append(e)

    result={
        "profile":args.profile,
        "method":"discarded equilibrated pilots -> frozen deterministic production cluster counts",
        "burnin_rule":"adaptive sweep coverage allowed only in discarded burn-in",
        "post_burnin_rule":"deterministic fixed-count washout before first measurement",
        "production_measurement_rule":"fixed number of Wolff clusters; realized cluster sizes never determine a measurement time",
        "target_between_sweep_equiv_from_pilot":between,
        "reproducibility_self_test":reproducibility_self_test(),
        "grid":grid,
        "entries":entries,
    }
    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"WROTE {p}")

if __name__=="__main__":
    main()
