from __future__ import annotations
from pathlib import Path
import csv, json
import numpy as np

from equilibrium_core_base import (
    periodic_neighbors, cold_spins, hot_spins, seed_numba,
    _pilot_adaptive_burnin, _advance_fixed_clusters,
    link_observables, magnetic_observables, NUMBA_AVAILABLE,
)


def run_chain(
    L: int,
    beta: float,
    seed: int,
    start: str,
    outdir: Path,
    fixed_clusters_between_measurements: int,
    measurements: int = 800,
    adaptive_burnin_sweeps: float = 200.0,
    fixed_washout_intervals: int = 20,
):
    """Run one production chain with fixed-count measurement intervals."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    V = L**4
    alln, plus, _ = periodic_neighbors(L)
    spins = cold_spins(L) if start == "cold" else hot_spins(L, seed + 991)
    seed_numba(int(seed))

    marks = np.zeros(V, dtype=np.int32)
    stack = np.empty(V, dtype=np.int32)
    members = np.empty(V, dtype=np.int32)
    mark = 0

    burn_clusters, burn_flipped, mark = _pilot_adaptive_burnin(
        spins, alln, beta, adaptive_burnin_sweeps,
        marks, stack, members, mark
    )

    washout_clusters = int(fixed_washout_intervals) * int(fixed_clusters_between_measurements)
    wash_flipped, _, mark = _advance_fixed_clusters(
        spins, alln, beta, washout_clusters,
        marks, stack, members, mark
    )

    rows = []
    for m in range(int(measurements)):
        flipped, sizes, mark = _advance_fixed_clusters(
            spins, alln, beta, int(fixed_clusters_between_measurements),
            marks, stack, members, mark
        )
        obs = {
            "measurement": m,
            "fixed_clusters_interval": int(fixed_clusters_between_measurements),
            "site_flips_interval": int(flipped),
            "realized_sweep_coverage": float(flipped / V),
            "mean_cluster_fraction_interval": float(np.mean(sizes) / V),
        }
        obs.update(link_observables(spins, plus))
        obs.update(magnetic_observables(spins, L))
        rows.append(obs)

    with (outdir/"measurements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "sampler": "wolff_adaptive_discarded_burnin_then_fixed_count_production",
        "L": int(L),
        "beta": float(beta),
        "seed": int(seed),
        "start": start,
        "adaptive_burnin_target_sweeps": float(adaptive_burnin_sweeps),
        "adaptive_burnin_clusters": int(burn_clusters),
        "adaptive_burnin_realized_coverage": float(burn_flipped / V),
        "fixed_washout_intervals": int(fixed_washout_intervals),
        "fixed_washout_clusters": int(washout_clusters),
        "fixed_washout_realized_coverage": float(wash_flipped / V),
        "fixed_clusters_between_measurements": int(fixed_clusters_between_measurements),
        "measurements": int(measurements),
        "mean_realized_sweep_coverage_between": float(np.mean([r["realized_sweep_coverage"] for r in rows])),
        "mean_action_density_per_link": float(np.mean([r["action_density_per_link"] for r in rows])),
        "mean_m2": float(np.mean([r["m2"] for r in rows])),
        "mean_xi_over_L": float(np.mean([r["xi_over_L"] for r in rows])),
        "numba_available": bool(NUMBA_AVAILABLE),
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
