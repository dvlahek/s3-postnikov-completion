from __future__ import annotations
from itertools import combinations
import argparse, json
from pathlib import Path
import numpy as np

VERTICES = tuple(range(6))
SIMPLICES = {d: list(combinations(VERTICES, d + 1)) for d in range(6)}
INDEX = {d: {s: i for i, s in enumerate(SIMPLICES[d])} for d in SIMPLICES}


def delta(x: np.ndarray, degree: int) -> np.ndarray:
    out = np.zeros(len(SIMPLICES[degree + 1]), dtype=np.uint8)
    for r, simplex in enumerate(SIMPLICES[degree + 1]):
        v = 0
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            v ^= int(x[INDEX[degree][face]])
        out[r] = v
    return out


def cup_i_term_indices(simplex, p: int, q: int, i: int):
    """Medina-Mardones simplicial cup-i formula over F2."""
    n = len(simplex) - 1
    if n != p + q - i:
        raise ValueError("degree mismatch")
    terms = []
    for U in combinations(range(n + 1), n - i):
        U0, U1 = [], []
        for j, u in enumerate(U, start=1):
            (U0 if ((j - u) & 1) == 0 else U1).append(u)
        f0 = tuple(v for pos, v in enumerate(simplex) if pos not in U0)
        f1 = tuple(v for pos, v in enumerate(simplex) if pos not in U1)
        if len(f0) == p + 1 and len(f1) == q + 1:
            terms.append((INDEX[p][f0], INDEX[q][f1]))
    return terms


OP = {}
for key in [(3, 3, 1), (3, 3, 2), (2, 3, 1), (2, 2, 0), (2, 3, 0), (3, 2, 0)]:
    p, q, i = key
    d = p + q - i
    OP[key] = [cup_i_term_indices(s, p, q, i) for s in SIMPLICES[d]]


def cup_i(x: np.ndarray, p: int, y: np.ndarray, q: int, i: int) -> np.ndarray:
    key = (p, q, i)
    if key not in OP:
        d = p + q - i
        OP[key] = [cup_i_term_indices(s, p, q, i) for s in SIMPLICES[d]]
    out = np.zeros(len(SIMPLICES[p + q - i]), dtype=np.uint8)
    for r, terms in enumerate(OP[key]):
        v = 0
        for a, b in terms:
            v ^= int(x[a] & y[b])
        out[r] = v
    return out


def gf2_pivot_columns(A):
    A = A.copy()
    m, n = A.shape
    row = 0
    pivots = []
    for c in range(n):
        rr = np.where(A[row:, c] == 1)[0]
        if len(rr) == 0:
            continue
        r = row + int(rr[0])
        A[[row, r]] = A[[r, row]]
        for j in range(m):
            if j != row and A[j, c]:
                A[j] ^= A[row]
        pivots.append(c)
        row += 1
        if row == m:
            break
    return pivots


def cocycle_basis_z3():
    D2 = np.zeros((len(SIMPLICES[3]), len(SIMPLICES[2])), dtype=np.uint8)
    for c in range(len(SIMPLICES[2])):
        e = np.zeros(len(SIMPLICES[2]), dtype=np.uint8)
        e[c] = 1
        D2[:, c] = delta(e, 2)
    piv = gf2_pivot_columns(D2)
    basis = D2[:, piv].T.copy()
    assert basis.shape == (10, 15)
    return basis


def all_z3_cocycles():
    basis = cocycle_basis_z3()
    out = []
    for mask in range(1 << 10):
        a = np.zeros(15, dtype=np.uint8)
        for i in range(10):
            if (mask >> i) & 1:
                a ^= basis[i]
        assert np.all(delta(a, 3) == 0)
        out.append(a)
    return out


def explicit_sq2_5simplex(alpha):
    I = INDEX[3]
    A = lambda s: int(alpha[I[tuple(s)]])
    return (
        A((0, 3, 4, 5)) * A((0, 1, 2, 3))
        ^ A((0, 1, 4, 5)) * A((1, 2, 3, 4))
        ^ A((0, 1, 2, 5)) * A((2, 3, 4, 5))
    )


def systematic_b_suite():
    n = len(SIMPLICES[2])
    suite = [np.zeros(n, dtype=np.uint8)]
    for i in range(n):
        b = np.zeros(n, dtype=np.uint8)
        b[i] = 1
        suite.append(b)
    for i in range(n):
        for j in range(i + 1, n):
            b = np.zeros(n, dtype=np.uint8)
            b[i] = b[j] = 1
            suite.append(b)
    assert len(suite) == 211
    return suite


def run_checks():
    alphas = all_z3_cocycles()
    explicit_matches = 0
    sq1_count = 0
    zeta_pairs = 0
    for a in alphas:
        sq_generic = int(cup_i(a, 3, a, 3, 1)[0])
        sq_explicit = explicit_sq2_5simplex(a)
        if sq_generic != sq_explicit:
            raise AssertionError("explicit Sq^2 formula disagrees with generic cup_1")
        explicit_matches += 1
        sq1_count += sq_generic
        solutions = 0
        for zmask in range(1 << 6):
            dz = zmask.bit_count() & 1
            if dz == sq_generic:
                solutions += 1
        if solutions != 32:
            raise AssertionError("wrong number of zeta trivializations")
        zeta_pairs += solutions

    b_suite = systematic_b_suite()
    descendant_pairs = 0
    for a in alphas:
        sq_a = cup_i(a, 3, a, 3, 1)
        for b in b_suite:
            db = delta(b, 2)
            ap = a ^ db
            rhs = cup_i(ap, 3, ap, 3, 1) ^ sq_a
            Q = (
                cup_i(a, 3, db, 3, 2)
                ^ cup_i(b, 2, db, 3, 1)
                ^ cup_i(b, 2, b, 2, 0)
            )
            lhs = delta(Q, 4)
            if not np.array_equal(lhs, rhs):
                raise AssertionError("descendant identity failed")
            descendant_pairs += 1

    rng = np.random.default_rng(20260821)
    cupi_identity_trials = 0
    for _ in range(2000):
        b = rng.integers(0, 2, len(SIMPLICES[2]), dtype=np.uint8)
        c = rng.integers(0, 2, len(SIMPLICES[3]), dtype=np.uint8)
        lhs = delta(cup_i(b, 2, c, 3, 1), 4)
        rhs = (
            cup_i(delta(b, 2), 3, c, 3, 1)
            ^ cup_i(b, 2, delta(c, 3), 4, 1)
            ^ cup_i(b, 2, c, 3, 0)
            ^ cup_i(c, 3, b, 2, 0)
        )
        if not np.array_equal(lhs, rhs):
            raise AssertionError("cup-i coboundary identity failed")
        cupi_identity_trials += 1

    return {
        "test_name": "local cochain consistency, not an independent Gu-Wen evaluation",
        "z3_dimension": 10,
        "z3_cocycles": len(alphas),
        "generic_vs_explicit_sq2_matches": explicit_matches,
        "sq2_equal_one_cases": int(sq1_count),
        "compatible_alpha_zeta_pairs": int(zeta_pairs),
        "zeta_solutions_per_alpha": 32,
        "descendant_identity_pairs_tested": int(descendant_pairs),
        "descendant_b_suite": "all C^2 cochains of Hamming weight 0, 1, or 2",
        "cup_i_coboundary_random_trials": int(cupi_identity_trials),
        "status": "PASS",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cochain_checks.json")
    args = ap.parse_args()
    result = run_checks()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
