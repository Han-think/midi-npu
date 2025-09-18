import argparse
import glob
import os
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"exporter failed rc={proc.returncode}")


def _find_xml(out: str) -> str | None:
    xmls = glob.glob(os.path.join(out, "*.xml")) or glob.glob(
        os.path.join(out, "**", "*.xml"), recursive=True
    )
    return xmls[0] if xmls else None


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

    try:
        _run(cmd)
        xml = _find_xml(out)
        if xml:
            print("Exported:", xml)
            return

        task_idx = cmd.index("text-generation-with-past")
        cmd[task_idx] = "text-generation"
        _run(cmd)
        xml = _find_xml(out)
        if xml:
            print("Exported:", xml)
            return
    except Exception as exc:
        print("Exporter exception:", exc)

    raise RuntimeError(f"No XML produced under {out}. Export failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.ckpt, args.out)
