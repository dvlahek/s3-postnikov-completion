from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def cdiff(arr, axis, a):
    return (
        -np.roll(arr, -2, axis=axis)
        + 8.0*np.roll(arr, -1, axis=axis)
        - 8.0*np.roll(arr, 1, axis=axis)
        + np.roll(arr, 2, axis=axis)
    )/(12.0*a)


def local_orders(rows, key):
    out=[]
    for r1,r2 in zip(rows[:-1],rows[1:]):
        p=math.log(r1[key]/r2[key])/math.log(r2["L"]/r1["L"])
        out.append({"L1":r1["L"],"L2":r2["L"],"order":p})
    return out


def smooth_skyrmion_field(L, R=0.32):
    """C-infinity compactly supported degree-one field on a periodic 3-torus."""
    a=1.0/L
    c=(np.arange(L)+0.5)/L-0.5
    x,y,z=np.meshgrid(c,c,c,indexing="ij")
    r2=x*x+y*y+z*z
    s2=r2/(R*R)
    q=np.zeros_like(r2)
    mask=s2<1.0
    q[mask]=np.exp(-1.0/(1.0-s2[mask]))
    den=r2+q*q
    den_safe=np.where(den>1e-300,den,1.0)
    n=np.empty((L,L,L,4),dtype=float)
    n[...,0]=(r2-q*q)/den_safe
    n[...,1]=2*q*x/den_safe
    n[...,2]=2*q*y/den_safe
    n[...,3]=2*q*z/den_safe
    zero=den<=1e-300
    if np.any(zero):
        n[zero,0]=-1.0
        n[zero,1:]=0.0
    return n,a


def skyrmion_charge(n,a):
    d=[cdiff(n,mu,a) for mu in range(3)]
    mats=np.stack([n,d[0],d[1],d[2]],axis=-2)
    det=np.linalg.det(mats)
    return float(np.sum(det)*a**3/(2*np.pi**2))


def skyrmion_series(Ls):
    rows=[]
    for L in Ls:
        n,a=smooth_skyrmion_field(L)
        Q=skyrmion_charge(n,a)
        rows.append({"L":int(L),"spacing":a,"Q":Q,"integer_error":abs(abs(Q)-1.0)})
    return rows


def smooth_patch_field(L):
    x=np.arange(L)/L
    X0,X1,X2,X3=np.meshgrid(x,x,x,x,indexing="ij")
    chi=0.35+0.18*np.sin(2*np.pi*X0)*np.cos(2*np.pi*X1)
    theta=1.1+0.25*np.sin(2*np.pi*X2)
    phi=2*np.pi*X3+0.2*np.sin(2*np.pi*X0)
    u=np.stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta),
    ],axis=-1)
    n=np.empty(u.shape[:-1]+(4,),dtype=float)
    n[...,0]=np.cos(chi)
    n[...,1:]=np.sin(chi)[...,None]*u
    return n,chi,u,1.0/L


def gerbe_error(L):
    n,chi,u,a=smooth_patch_field(L)
    axes=(0,2,3)
    du={mu:cdiff(u,mu,a) for mu in axes}
    dn={mu:cdiff(n,mu,a) for mu in axes}
    f=(2*chi-np.sin(2*chi))/(8*np.pi**2)
    def B(mu,nu):
        return f*np.sum(u*np.cross(du[mu],du[nu]),axis=-1)
    B23=B(2,3); B03=B(0,3); B02=B(0,2)
    dB=cdiff(B23,0,a)-cdiff(B03,2,a)+cdiff(B02,3,a)
    mats=np.stack([n,dn[0],dn[2],dn[3]],axis=-2)
    H=np.linalg.det(mats)/(2*np.pi**2)
    rms=float(np.sqrt(np.mean((dB-H)**2)))
    hrms=float(np.sqrt(np.mean(H**2)))
    return {"L":int(L),"spacing":a,"relative_rms":rms/(hrms+1e-300)}


def gerbe_series(Ls):
    return [gerbe_error(L) for L in Ls]


def pi4_field(L,rho=1.0):
    ys=np.linspace(-1.0,1.0,L)
    Y=np.meshgrid(ys,ys,ys,ys,indexing="ij")
    boundary=np.zeros((L,L,L,L),bool)
    for mu in range(4):
        boundary |= np.isclose(np.abs(Y[mu]),1.0)
    X=[]
    for mu in range(4):
        v=np.zeros_like(Y[mu])
        mask=~boundary
        v[mask]=rho*np.tan(0.5*np.pi*Y[mu][mask])
        X.append(v)
    r2=sum(v*v for v in X)
    den=r2+rho*rho
    X0=(r2-rho*rho)/den
    XX=[2*rho*v/den for v in X]
    rr=np.sqrt(sum(v*v for v in XX))
    rrsq=np.where(rr>1e-14,rr*rr,1.0)
    A,B,C,D=XX
    h1=2*(A*C+B*D)/rrsq
    h2=2*(B*C-A*D)/rrsq
    h3=(A*A+B*B-C*C-D*D)/rrsq
    out=np.zeros((L,L,L,L,4),float)
    out[...,0]=X0
    out[...,1]=rr*h1; out[...,2]=rr*h2; out[...,3]=rr*h3
    out[boundary,0]=1.0; out[boundary,1:]=0.0
    out/=np.linalg.norm(out,axis=-1,keepdims=True)
    return out


def pi4_metrics(g):
    L=g.shape[0]
    normerr=float(np.max(np.abs(np.sum(g*g,axis=-1)-1.0)))
    b=np.zeros(g.shape[:-1],bool)
    for mu in range(4):
        s=[slice(None)]*4; s[mu]=0; b[tuple(s)]=True
        s[mu]=-1; b[tuple(s)]=True
    target=np.array([1.,0,0,0])
    berr=float(np.max(np.linalg.norm(g[b]-target,axis=-1)))
    dots=[]
    for mu in range(4):
        a=[slice(None)]*4; c=[slice(None)]*4
        a[mu]=slice(0,-1); c[mu]=slice(1,None)
        dots.append(np.sum(g[tuple(a)]*g[tuple(c)],axis=-1).ravel())
    dots=np.concatenate(dots)
    return {
        "L":int(L),"norm_max_error":normerr,"boundary_max_error":berr,
        "max_link_angle":float(np.max(np.arccos(np.clip(dots,-1,1))))
    }


def make_plots(out,sk,gerbe):
    for name,rows,key,ylabel in [
        ("skyrmion",sk,"integer_error",r"$||Q|-1|$"),
        ("gerbe",gerbe,"relative_rms",r"relative RMS $|dB-H|$"),
    ]:
        a=np.array([r["spacing"] for r in rows],float)
        e=np.array([r[key] for r in rows],float)
        guide=e[-1]*(a/a[-1])**4
        fig=plt.figure(figsize=(6.2,4.4)); ax=fig.add_subplot(111)
        ax.loglog(a,e,marker="o",label="numerical error")
        ax.loglog(a,guide,linestyle="--",label=r"$O(a^4)$ guide")
        ax.set_xlabel(r"lattice spacing $a$")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out/f"FIG_{name}_fourth_order.png",dpi=300)
        fig.savefig(out/f"FIG_{name}_fourth_order.pdf")
        plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--out",default="results/topology")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    if args.quick:
        sk_L=[16,24,32]
        ge_L=[8,12,16]
        p4_L=[9,13]
    else:
        sk_L=[16,24,32,40,48,64]
        ge_L=[8,12,16,24,32]
        p4_L=[9,13,17,25]

    sk=skyrmion_series(sk_L)
    ge=gerbe_series(ge_L)
    p4=[pi4_metrics(pi4_field(L)) for L in p4_L]
    result={
        "skyrmion":sk,
        "skyrmion_local_orders":local_orders(sk,"integer_error"),
        "gerbe":ge,
        "gerbe_local_orders":local_orders(ge,"relative_rms"),
        "pi4_grid":p4,
        "interpretation":"errors approach the expected fourth-order finite-difference accuracy; no asymptotic exponent is fitted",
    }
    (out/"topology_benchmarks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    make_plots(out,sk,ge)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
