from __future__ import annotations
import argparse, subprocess, sys, shutil, zipfile, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(cmd):
    print("\n>>> "+" ".join(str(x) for x in cmd),flush=True)
    subprocess.run([str(x) for x in cmd],check=True,cwd=HERE)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",choices=["quick","paper"],default="paper")
    ap.add_argument("--fresh",action="store_true")
    ap.add_argument("--results",default="results")
    args=ap.parse_args()

    results=HERE/args.results
    if args.fresh and results.exists(): shutil.rmtree(results)
    results.mkdir(parents=True,exist_ok=True)
    py=sys.executable

    run([py,"cochain_checks.py","--out",results/"cochain_checks.json"])
    run([py,"pi4_preimage_framing.py","--out",results/"pi4_preimage_framing.json"])
    topo=[py,"topology_benchmarks.py","--out",results/"topology"]
    if args.profile=="quick": topo.append("--quick")
    run(topo)

    schedule=results/"fixed_cluster_schedule.json"
    run([py,"calibrate_wolff_schedule.py","--profile",args.profile,"--out",schedule])
    cross_n=1500 if args.profile=="quick" else 6000
    run([py,"sampler_crosscheck.py","--schedule",schedule,"--out",results/"sampler_crosscheck.json","--measurements",str(cross_n)])
    run([py,"run_equilibrium.py","--schedule",schedule,"--out",results/"equilibrium"])
    boot=800 if args.profile=="quick" else 4000
    run([py,"analyze_equilibrium.py","--root",results/"equilibrium","--out",results/"analysis","--bootstrap",str(boot),"--block","12"])

    manifest={}
    for f in sorted(HERE.glob("*.py")): manifest[f.name]=sha256(f)
    (results/"code_sha256.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    outzip=HERE/("full_s3_results.zip" if args.profile=="paper" else "full_s3_quick_results.zip")
    if outzip.exists(): outzip.unlink()
    with zipfile.ZipFile(outzip,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(results.rglob("*")):
            if f.is_file(): z.write(f,f.relative_to(HERE))
    print(f"\nDONE: {outzip}")

if __name__=="__main__":
    main()
