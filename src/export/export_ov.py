"""Export checkpoints to OpenVINO IR using Optimum."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def export_model(checkpoint: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "optimum.exporters.openvino",
        f"--model={checkpoint}",
        "--task=text-generation",
        "--weight-format=fp16",
        "--ov_config=PERFORMANCE_HINT=LATENCY",
        str(output_dir),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print("Exported:", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to OpenVINO IR")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the saved checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Directory where the IR will be written")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.ckpt, args.out)
