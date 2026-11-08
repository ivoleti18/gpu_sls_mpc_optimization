"""Single GPU diagnostic solve, with artifacts outside tracked example paths."""

from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from quadrotor_scenario import OUTPUT_NAMES, build_scenario, input_digest
from runtime_diagnostics import RuntimeDiagnostics, json_safe
from source_isolation import verify_origins
from validation import validate


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        self.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_hashes(checkout):
    files = list((checkout / "src/gpu_sls").rglob("*.py"))
    files += [checkout / "examples/quadrotor/quadrotor.py"]
    harness = Path(__file__).parent
    files += list(harness.glob("*.py")) + list((harness / "scenarios").glob("*.json"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def gpu_lock_path():
    # A nested/custom run directory must not create a separate GPU lock.
    return Path(__file__).resolve().parent / "results/.gpu.lock"


def gpu_preflight():
    """Read-only GPU checks shared by diagnostic and timing modes, under lock."""
    import jax
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"
    ], text=True).strip()
    if active:
        raise RuntimeError(f"Other GPU compute processes are active: {active}")
    status = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader"
    ], text=True).strip()
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one GPU, found {devices}")
    return devices[0], status


def run_diagnostic(args, provenance):
    import jax
    import numpy as np

    output = args.output_dir
    report = {
        "status": "started", "mode": "diagnose", "variant": args.variant,
        "provenance": provenance, "python": sys.version, "executable": sys.executable,
        "versions": {name: metadata.version(name) for name in
                     ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "scipy", "trajax")},
        "file_sha256": file_hashes(args.source_root.resolve()),
        "timing": None,
        "logging_policy": "Original ADMM prints retained; extra ordered runtime callbacks; not timed",
        "cache_policy": "Persistent JAX cache disabled; incidental caches confined to this run directory",
    }
    observer = None
    with (output / "solve.log").open("x") as log, \
         redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), \
         gpu_lock_path().open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            device, report["gpu_status"] = gpu_preflight()
            report["device"] = str(device)
            # This mode never silently falls back to a CPU execution.
            with jax.default_device(device), (output / "events.jsonl").open("x") as events:
                scenario = build_scenario(args.source_root, args.variant)
                report["scenario"] = scenario.definition
                jax.block_until_ready(scenario.inputs)
                report["input_sha256_before"] = input_digest(scenario.inputs)
                observer = RuntimeDiagnostics(events, scenario.definition["sqp"])
                observer.install()
                print(f"Starting diagnostic GPU solve: {args.variant}", flush=True)
                outputs = scenario.controller._solve(*scenario.inputs)
                jax.block_until_ready(outputs)
                jax.effects_barrier()
                report["gpu_solve_completed"] = True
                report["diagnostics"] = observer.summary()
                observer.restore()
                report["input_sha256_after"] = input_digest(scenario.inputs)
                report["validation"] = validate(scenario, outputs, report["diagnostics"])
                report["validation"]["gates"]["frozen_inputs_unchanged"] = (
                    report["input_sha256_before"] == report["input_sha256_after"])
                report["validation"]["gates"]["outputs_on_gpu"] = all(
                    all(d.platform == "gpu" for d in a.devices()) for a in outputs)
                report["validation"]["accepted"] = all(report["validation"]["gates"].values())
                np.savez_compressed(output / "outputs.npz", **dict(zip(OUTPUT_NAMES,
                                    jax.device_get(outputs), strict=True)))
                np.savez_compressed(output / "inputs.npz", **{
                    f"input_{i:02d}": np.asarray(a) for i, a in enumerate(scenario.inputs)})
                report["provenance"]["module_origins"] = verify_origins(args.source_root)
                report["status"] = "passed" if report["validation"]["accepted"] else "validation_failed"
                print(json.dumps(json_safe({"status": report["status"],
                                           "diagnostics": report["diagnostics"],
                                           "validation": report["validation"]}), indent=2), flush=True)
        except Exception as exc:
            report["status"] = "execution_failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if observer:
                report["diagnostics"] = observer.summary()
            traceback.print_exc()
        finally:
            if observer:
                observer.restore()
            write_json(output / "report.json", report)
            print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 2
