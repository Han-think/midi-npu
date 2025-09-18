import argparse
import glob
import os
import subprocess
import sys


def main(ckpt: str, out: str) -> None:
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    os.makedirs(out, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.openvino",
        "--model",
        ckpt,
        "--task",
        "text-generation-with-past",
        "--weight-format",
        "fp16",
        "--ov_config",
        "PERFORMANCE_HINT=LATENCY",
        "--output",
        out,
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    xmls = glob.glob(os.path.join(out, "*.xml")) or glob.glob(
        os.path.join(out, "**", "*.xml"), recursive=True
    )
    if not xmls:
        raise RuntimeError(f"No XML produced under {out}. Export failed.")
    print("Exported:", xmls[0], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.ckpt, args.out)
