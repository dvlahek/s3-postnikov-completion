from __future__ import annotations
import math
from pathlib import Path
import csv, json
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


def periodic_neighbors(L: int):
    V = L**4
    alln = np.empty((V, 8), dtype=np.int32)
    parity = np.empty(V, dtype=np.int8)

    def idx(c):
        return (((c[0] * L + c[1]) * L + c[2]) * L + c[3])

    k = 0
    for x0 in range(L):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    c = [x0, x1, x2, x3]
                    parity[k] = (x0 + x1 + x2 + x3) & 1
                    for mu in range(4):
                        cp = c.copy(); cm = c.copy()
                        cp[mu] = (cp[mu] + 1) % L
                        cm[mu] = (cm[mu] - 1) % L
                        alln[k, mu] = idx(cp)
                        alln[k, 4 + mu] = idx(cm)
                    k += 1
    return alln, alln[:, :4].copy(), parity


@njit(cache=True)
def seed_numba(seed: int):
    np.random.seed(seed)


@njit(cache=True)
def wolff_update(spins, all_neighbors, beta, marks, stack, members, mark):
    V = spins.shape[0]
    r = np.empty(4, dtype=np.float64)
    nr = 0.0
    for a in range(4):
        r[a] = np.random.normal(); nr += r[a] * r[a]
    nr = math.sqrt(nr)
    for a in range(4): r[a] /= nr

    seed = np.random.randint(0, V)
    top = 1; nmem = 0
    stack[0] = seed; marks[seed] = mark
    while top > 0:
        top -= 1
        i = stack[top]
        members[nmem] = i; nmem += 1
        si = 0.0
        for a in range(4): si += spins[i, a] * r[a]
        for kk in range(8):
            j = all_neighbors[i, kk]
            if marks[j] == mark: continue
            sj = 0.0
            for a in range(4): sj += spins[j, a] * r[a]
            prod = si * sj
            if prod > 0.0:
                p = 1.0 - math.exp(-2.0 * beta * prod)
                if np.random.random() < p:
                    marks[j] = mark; stack[top] = j; top += 1

    for kk in range(nmem):
        i = members[kk]
        d = 0.0
        for a in range(4): d += spins[i, a] * r[a]
        for a in range(4): spins[i, a] -= 2.0 * d * r[a]
    return nmem


def cold_spins(L: int):
    s = np.zeros((L**4, 4), dtype=np.float64); s[:, 0] = 1.0
    return s


def hot_spins(L: int, seed: int):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(L**4, 4)); s /= np.linalg.norm(s, axis=1, keepdims=True)
    return s


def link_observables(spins, plus):
    dots = np.stack([np.sum(spins * spins[plus[:, mu]], axis=1) for mu in range(4)], axis=1)
    md = float(np.mean(dots))
    return {"mean_link_dot": md, "action_density_per_link": 1.0 - md}


def magnetic_observables(spins, L: int):
    V = spins.shape[0]
    M = np.sum(spins, axis=0)
    M2 = float(np.dot(M, M))
    m2 = M2 / (V * V); m4 = m2 * m2; S0 = M2 / V
    arr = spins.reshape((L, L, L, L, 4))
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    sk = []
    for mu in range(4):
        shape = [1, 1, 1, 1, 1]; shape[mu] = L
        w = phase.reshape(shape)
        F = np.sum(arr * w, axis=(0, 1, 2, 3))
        sk.append(float(np.vdot(F, F).real / V))
    Sk = float(np.mean(sk))
    ratio = S0 / Sk - 1.0 if Sk > 0 else -1.0
    xi = math.sqrt(max(ratio, 0.0)) / (2.0 * math.sin(math.pi / L)) if ratio > 0 else 0.0
    return {"m2": m2, "m4": m4, "S0": S0, "Skmin": Sk, "xi": xi, "xi_over_L": xi / L}


def _advance_fixed_clusters(spins, alln, beta, n_clusters, marks, stack, members, mark_counter):
    mark = mark_counter; flipped = 0; sizes = []
    for _ in range(int(n_clusters)):
        mark += 1
        if mark >= 2_000_000_000:
            marks.fill(0); mark = 1
        n = int(wolff_update(spins, alln, beta, marks, stack, members, mark))
        flipped += n; sizes.append(n)
    return flipped, sizes, mark


def _pilot_adaptive_burnin(spins, alln, beta, target_sweep_equiv, marks, stack, members, mark_counter):
    V = spins.shape[0]
    target = max(1, int(math.ceil(target_sweep_equiv * V)))
    flipped = 0; mark = mark_counter; clusters = 0
    while flipped < target:
        mark += 1
        if mark >= 2_000_000_000:
            marks.fill(0); mark = 1
        n = int(wolff_update(spins, alln, beta, marks, stack, members, mark))
        flipped += n; clusters += 1
    return clusters, flipped, mark


def calibrate_fixed_schedule(L: int, beta: float, pilot_seeds=(9001, 9002), pilot_starts=("cold", "hot"), pilot_burnin_sweeps: float = 30.0, pilot_clusters: int = 500, target_between_sweep_equiv: float = 1.0, target_therm_sweep_equiv: float = 150.0):
    V = L**4
    alln, _, _ = periodic_neighbors(L)
    pooled_sizes = []; pilot_records = []
    for seed, start in zip(pilot_seeds, pilot_starts):
        spins = cold_spins(L) if start == "cold" else hot_spins(L, seed + 991)
        seed_numba(int(seed))
        marks = np.zeros(V, dtype=np.int32); stack = np.empty(V, dtype=np.int32); members = np.empty(V, dtype=np.int32)
        mark = 0
        bc, bf, mark = _pilot_adaptive_burnin(spins, alln, beta, pilot_burnin_sweeps, marks, stack, members, mark)
        sizes = []
        for _ in range(int(pilot_clusters)):
            mark += 1
            n = int(wolff_update(spins, alln, beta, marks, stack, members, mark))
            sizes.append(n)
        pooled_sizes.extend(sizes)
        pilot_records.append({"seed": int(seed), "start": start, "adaptive_burnin_clusters": int(bc), "adaptive_burnin_effective_coverage": float(bf / V), "sampled_clusters": int(pilot_clusters), "mean_cluster_size": float(np.mean(sizes)), "median_cluster_size": float(np.median(sizes))})
    mean_cluster_size = float(np.mean(pooled_sizes))
    n_between = max(1, int(math.ceil(target_between_sweep_equiv * V / mean_cluster_size)))
    n_therm = max(n_between, int(math.ceil(target_therm_sweep_equiv * V / mean_cluster_size)))
    return {"L": int(L), "beta": float(beta), "pilot_mean_cluster_size": mean_cluster_size, "pilot_mean_cluster_fraction": mean_cluster_size / V, "target_between_sweep_equiv": float(target_between_sweep_equiv), "target_therm_sweep_equiv": float(target_therm_sweep_equiv), "fixed_clusters_between_measurements": int(n_between), "fixed_thermalization_clusters": int(n_therm), "pilot_records": pilot_records}


@njit(cache=True)
def _sample_uniform_s3():
    x = np.empty(4, dtype=np.float64); n = 0.0
    for a in range(4):
        x[a] = np.random.normal(); n += x[a] * x[a]
    n = math.sqrt(n)
    for a in range(4): x[a] /= n
    return x


@njit(cache=True)
def _sample_vmf4(mu, kappa):
    if kappa < 1e-12:
        return _sample_uniform_s3()
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + 9.0)) / 3.0
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + 3.0 * math.log(1.0 - x0 * x0)
    while True:
        z = np.random.beta(1.5, 1.5)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.random()
        if kappa * w + 3.0 * math.log(1.0 - x0 * w) - c >= math.log(u): break
    v = np.empty(3, dtype=np.float64); nv = 0.0
    for a in range(3):
        v[a] = np.random.normal(); nv += v[a] * v[a]
    nv = math.sqrt(nv)
    for a in range(3): v[a] /= nv
    y = np.empty(4, dtype=np.float64); y[0] = w
    s = math.sqrt(max(0.0, 1.0 - w * w))
    for a in range(3): y[a + 1] = s * v[a]
    d2 = (1.0 - mu[0]) * (1.0 - mu[0])
    for a in range(1, 4): d2 += mu[a] * mu[a]
    if d2 < 1e-24: return y
    inv = 1.0 / math.sqrt(d2)
    vh = np.empty(4, dtype=np.float64); vh[0] = (1.0 - mu[0]) * inv
    for a in range(1, 4): vh[a] = -mu[a] * inv
    dot = 0.0
    for a in range(4): dot += vh[a] * y[a]
    out = np.empty(4, dtype=np.float64)
    for a in range(4): out[a] = y[a] - 2.0 * vh[a] * dot
    return out


@njit(cache=True)
def heatbath_checkerboard_sweep(spins, alln, parity, beta):
    V = spins.shape[0]
    h = np.empty(4, dtype=np.float64); mu = np.empty(4, dtype=np.float64)
    for target_parity in (0, 1):
        for i in range(V):
            if parity[i] != target_parity: continue
            for a in range(4): h[a] = 0.0
            for kk in range(8):
                j = alln[i, kk]
                for a in range(4): h[a] += spins[j, a]
            nh = 0.0
            for a in range(4): nh += h[a] * h[a]
            nh = math.sqrt(nh)
            if nh < 1e-14:
                mu[0] = 1.0; mu[1] = mu[2] = mu[3] = 0.0; kappa = 0.0
            else:
                for a in range(4): mu[a] = h[a] / nh
                kappa = beta * nh
            x = _sample_vmf4(mu, kappa)
            for a in range(4): spins[i, a] = x[a]


@njit(cache=True)
def metropolis_checkerboard_sweep(spins, alln, parity, beta):
    V = spins.shape[0]; accepted = 0
    h = np.empty(4, dtype=np.float64)
    for target_parity in (0, 1):
        for i in range(V):
            if parity[i] != target_parity: continue
            for a in range(4): h[a] = 0.0
            for kk in range(8):
                j = alln[i, kk]
                for a in range(4): h[a] += spins[j, a]
            prop = _sample_uniform_s3(); delta = 0.0
            for a in range(4): delta += h[a] * (prop[a] - spins[i, a])
            if delta >= 0.0 or np.random.random() < math.exp(beta * delta):
                for a in range(4): spins[i, a] = prop[a]
                accepted += 1
    return accepted / V


def reproducibility_self_test():
    L = 3; alln, _, _ = periodic_neighbors(L); V = L**4
    s1 = cold_spins(L); s2 = cold_spins(L)
    b1 = (np.zeros(V, np.int32), np.empty(V, np.int32), np.empty(V, np.int32))
    b2 = (np.zeros(V, np.int32), np.empty(V, np.int32), np.empty(V, np.int32))
    seed_numba(12345)
    for mark in range(1, 8): wolff_update(s1, alln, 0.6, *b1, mark)
    seed_numba(12345)
    for mark in range(1, 8): wolff_update(s2, alln, 0.6, *b2, mark)
    return {"numba_available": NUMBA_AVAILABLE, "reproducibility_max_abs_diff": float(np.max(np.abs(s1 - s2)))}
