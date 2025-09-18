import argparse, os, sys, subprocess, glob

def main(ckpt, out):
    os.makedirs(out, exist_ok=True)
    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", ckpt,
        "--task", "text-generation-with-past",
        "--weight-format", "fp16",
        "--ov_config", "PERFORMANCE_HINT=LATENCY",
        "--output", out,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    xmls = glob.glob(os.path.join(out, "*.xml"))
    if not xmls:
        raise RuntimeError(f"No XML produced under {out}. Export failed.")
    print("Exported:", xmls[0], flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    a = ap.parse_args()
    main(a.ckpt, a.out)
