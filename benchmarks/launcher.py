#!/usr/bin/env python3
"""Source-isolated quadrotor diagnostics and fixed-budget full-solve timing."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True

from source_isolation import select_source


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("count must be at least 1")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("imports", "diagnose", "benchmark"), default="imports")
    parser.add_argument("--variant", choices=("original", "robust"), default="original")
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--trials", type=positive_int, default=3)
    parser.add_argument("--max-solve-seconds", type=positive_seconds, default=30.0,
                        help="Stop after a completed slow solve; does not interrupt GPU work")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory inside the harness checkout; never overwritten")
    args = parser.parse_args()
    if args.mode == "benchmark" and args.variant != "original":
        parser.error("benchmark mode preserves the original variant; no parameter changes")
    try:
        if args.mode in ("diagnose", "benchmark"):
            harness_root = Path(__file__).resolve().parent.parent
            if args.output_dir is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                args.output_dir = harness_root / "benchmarks/results" / (
                    f"{stamp}-{args.mode}-{args.variant}-{uuid.uuid4().hex[:8]}")
            args.output_dir = args.output_dir.resolve()
            if not args.output_dir.is_relative_to(harness_root / "benchmarks/results"):
                raise RuntimeError("Output directory must be inside this harness checkout's benchmarks/results")
            args.output_dir.mkdir(parents=True, exist_ok=False)
            # GPU execution is required explicitly in experiment.py. JAX 0.8.1
            # also needs a CPU backend for host diagnostic callback operands.
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"
            os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
            for key, name in (("CUDA_CACHE_PATH", "cuda_cache"),
                              ("TRITON_CACHE_DIR", "triton_cache"),
                              ("XDG_CACHE_HOME", "cache")):
                os.environ[key] = str(args.output_dir / name)
        provenance = select_source(args.source_root)
        if args.mode == "imports":
            print(json.dumps(provenance, indent=2))
            return 0
        if args.mode == "benchmark":
            from full_solve_benchmark import run_benchmark
            return run_benchmark(args, provenance)
        from experiment import run_diagnostic
        return run_diagnostic(args, provenance)
    except Exception as exc:
        print(f"Benchmark stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
