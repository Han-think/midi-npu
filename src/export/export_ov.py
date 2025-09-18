from __future__ import annotations

import glob
import os
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    print(proc.stdout)
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
        # retry plain
        cmd[cmd.index("text-generation-with-past")] = "text-generation"
        _run(cmd)
        xml = _find_xml(out)
        if xml:
            print("Exported:", xml)
            return
    except Exception as e:  # pragma: no cover
        print("Exporter exception:", e)

    raise RuntimeError(f"No XML produced under {out}. Export failed.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.ckpt, a.out)
