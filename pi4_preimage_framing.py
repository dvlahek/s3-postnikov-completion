from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def suspended_hopf(X):
    """Suspension Sigma h: S^4 -> S^3 in R^5 -> R^4 coordinates."""
    X = np.asarray(X, dtype=float)
    X = X / np.linalg.norm(X)
    X0 = X[0]
    A, B, C, D = X[1:]
    r2 = A*A + B*B + C*C + D*D
    r = math.sqrt(max(r2, 0.0))
    if r < 1e-14:
        return np.array([1.0 if X0 >= 0 else -1.0, 0.0, 0.0, 0.0])
    h1 = 2.0 * (A*C + B*D) / r2
    h2 = 2.0 * (B*C - A*D) / r2
    h3 = (A*A + B*B - C*C - D*D) / r2
    y = np.array([X0, r*h1, r*h2, r*h3], dtype=float)
    return y / np.linalg.norm(y)


def preimage_point(t):
    """Preimage circle of regular value y*=(0,0,0,1)."""
    return np.array([0.0, math.cos(t), math.sin(t), 0.0, 0.0])


def polar_so3(A):
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def derivative_framing_matrix(t, eps=1e-6):
    """Differential D(Sigma h) on an oriented normal frame of the preimage."""
    X = preimage_point(t)
    normals = [
        np.array([1.,0.,0.,0.,0.]),
        np.array([0.,0.,0.,0.,1.]),
        np.array([0.,0.,0.,1.,0.]),
    ]
    target = [
        np.array([1.,0.,0.,0.]),
        np.array([0.,1.,0.,0.]),
        np.array([0.,0.,1.,0.]),
    ]
    A = np.zeros((3,3), dtype=float)
    for j, v in enumerate(normals):
        xp = X + eps*v
        xm = X - eps*v
        xp /= np.linalg.norm(xp)
        xm /= np.linalg.norm(xm)
        d = (suspended_hopf(xp) - suspended_hopf(xm)) / (2.0*eps)
        for i, w in enumerate(target):
            A[i,j] = float(np.dot(d,w))
    return A, polar_so3(A)


def quat_mul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ], dtype=float)


def quat_from_rotation(R):
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25*S,(R[2,1]-R[1,2])/S,(R[0,2]-R[2,0])/S,(R[1,0]-R[0,1])/S])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        q = np.array([(R[2,1]-R[1,2])/S,0.25*S,(R[0,1]+R[1,0])/S,(R[0,2]+R[2,0])/S])
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        q = np.array([(R[0,2]-R[2,0])/S,(R[0,1]+R[1,0])/S,0.25*S,(R[1,2]+R[2,1])/S])
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        q = np.array([(R[1,0]-R[0,1])/S,(R[0,2]+R[2,0])/S,(R[1,2]+R[2,1])/S,0.25*S])
    return q / np.linalg.norm(q)


def lift_framing_loop(n_segments=512, eps=1e-6):
    ts = np.linspace(0.0, 2.0*np.pi, n_segments+1)
    Rs=[]; dets=[]; minsv=[]
    for t in ts:
        A,R = derivative_framing_matrix(float(t), eps=eps)
        Rs.append(R)
        dets.append(float(np.linalg.det(A)))
        minsv.append(float(np.min(np.linalg.svd(A, compute_uv=False))))

    q=np.array([1.,0.,0.,0.])
    max_step=0.0
    for i in range(n_segments):
        dR=Rs[i].T @ Rs[i+1]
        dq=quat_from_rotation(dR)
        if dq[0] < 0:
            dq=-dq
        max_step=max(max_step, 2.0*math.acos(float(np.clip(dq[0],-1,1))))
        q=quat_mul(q,dq)
        q/=np.linalg.norm(q)

    angles=[]
    for R in Rs:
        angles.append(math.atan2(R[2,1], R[1,1]))
    angles=np.unwrap(np.array(angles))
    winding=float((angles[-1]-angles[0])/(2*np.pi))

    return {
        "map": "suspended Hopf Sigma h",
        "regular_value": [0.0,0.0,0.0,1.0],
        "preimage": "X(t)=(0,cos t,sin t,0,0)",
        "segments": int(n_segments),
        "minimum_abs_det_differential": float(np.min(np.abs(dets))),
        "minimum_singular_value": float(np.min(minsv)),
        "SO3_loop_closure_max_abs": float(np.max(np.abs(Rs[0]-Rs[-1]))),
        "SO2_block_winding": winding,
        "spin_lift_endpoint": [float(x) for x in q],
        "spin_lift_distance_to_minus_one": float(np.linalg.norm(q-np.array([-1.,0,0,0]))),
        "max_increment_rotation_angle": float(max_step),
        "W_sigma_h": -1 if np.linalg.norm(q-np.array([-1.,0,0,0])) < 1e-8 else None,
        "status": "PASS" if abs(abs(winding)-1.0)<1e-8 and np.linalg.norm(q-np.array([-1.,0,0,0]))<1e-8 else "FAIL",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--segments",type=int,default=512)
    ap.add_argument("--out",default="results/pi4_preimage_framing.json")
    args=ap.parse_args()
    result=lift_framing_loop(args.segments)
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":
        raise SystemExit(2)

if __name__=="__main__":
    main()
