"""Full compiled solve latency for the NONCONVERGED / FIXED-BUDGET workload."""

from collections import Counter
from contextlib import redirect_stdout
import csv
import fcntl
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

from experiment import file_hashes, gpu_lock_path, gpu_preflight, write_json
from quadrotor_scenario import OUTPUT_NAMES, build_scenario, input_digest
from source_isolation import verify_origins
from validation import validate


LABEL = "NONCONVERGED / FIXED-BUDGET"
METHOD = (
    "Already-compiled GenericMPC._solve with the same frozen GPU argument tuple; "
    "perf_counter -> dispatch -> block_until_ready(complete 11-output tree) -> "
    "effects_barrier -> perf_counter. Compilation, preparation, validation and file "
    "writes excluded. Original ADMM debug prints retained and formatted into an "
    "in-memory buffer during execution; log files written only after stopping timer. "
    "No additional Milestone 1 observer callbacks in the timed executable."
)


def synchronized_call(compiled, inputs, jax_api, clock=None):
    """Return completed host-observed solve milliseconds and the full output tree."""
    if clock is None:
        clock = time.perf_counter
    jax_api.block_until_ready(inputs)
    jax_api.effects_barrier()
    start = clock()
    outputs = compiled(*inputs)
    jax_api.block_until_ready(outputs)
    jax_api.effects_barrier()
    elapsed_ms = (clock() - start) * 1000
    return elapsed_ms, outputs


def freeze_on_device(inputs, device):
    """Materialize even the weakly typed dt scalar without changing JAX avals."""
    import jax
    frozen = tuple(jax.device_put(value, device) for value in inputs)
    if any(jax.core.get_aval(a) != jax.core.get_aval(b)
           for a, b in zip(inputs, frozen, strict=True)):
        raise RuntimeError("Device placement changed input abstract values")
    if not all(value.devices() == {device} for value in frozen):
        raise RuntimeError("Frozen arguments must all reside on the selected GPU")
    jax.block_until_ready(frozen)
    return frozen


def latency_statistics(samples):
    return {
        "mean_ms": statistics.fmean(samples) if samples else None,
        "median_ms": statistics.median(samples) if samples else None,
        "minimum_ms": min(samples) if samples else None,
        "maximum_ms": max(samples) if samples else None,
        "raw_timings_ms": list(samples),
    }


def check_fixed_budget_proof(proof):
    """Permit SQP nonconvergence explicitly, without relaxing Milestone 1 gates."""
    if not proof.get("gpu_solve_completed"):
        raise RuntimeError("Untimed diagnostic GPU solve did not complete")
    gates = proof["validation"]["gates"]
    failures = [name for name, passed in gates.items()
                if not passed and name != "sqp_stopping_predicate"]
    if failures:
        raise RuntimeError(f"Untimed diagnostic failed workload checks: {failures}")
    if "sqp_stopping_predicate" not in gates or gates["sqp_stopping_predicate"]:
        raise RuntimeError("Expected the original nonconverged workload; convergence behavior changed")
    spec, diagnostics = proof["scenario"], proof["diagnostics"]
    budget = spec["sqp"]["max_sqp_iterations"] + spec["sls"]["max_initial_sqp_iterations"]
    counts = diagnostics["counts"]
    if counts.get("sqp_iteration") != budget:
        raise RuntimeError("Diagnostic did not execute the complete original SQP budget")
    if not all(counts.get(k, 0) for k in
               ("sls_solve_gpu", "get_controller", "calculate_phis", "admm_robust")):
        raise RuntimeError("Untimed runtime proof of the robust SLS path is missing")


def admm_signature(text):
    """Observe existing runtime prints, adding no effects to the timed graph."""
    return [(int(iterations), converged == "True") for iterations, converged in
            re.findall(r"ADMM done: Total Iterations=(\d+) converged=(True|False)", text)]


def verify_result(scenario, frozen, frozen_hash, controller_hash, outputs,
                  diagnostics, log_text, expected_admm, reference_outputs):
    import jax
    import numpy as np
    result = validate(scenario, outputs, diagnostics)
    gates = result["gates"]
    gates["frozen_inputs_unchanged"] = input_digest(frozen) == frozen_hash
    gates["controller_state_unchanged"] = input_digest(scenario.inputs) == controller_hash
    device = next(iter(frozen[0].devices()))
    gates["outputs_on_gpu"] = all(a.devices() == {device} for a in outputs)
    observed_admm = admm_signature(log_text)
    spec = scenario.definition
    admm_budget = (spec["sls"]["max_initial_sqp_iterations"] +
                   spec["sqp"]["max_sqp_iterations"] * spec["sls"]["max_sls_iterations"])
    gates["full_sqp_budget_observed"] = len(observed_admm) == admm_budget
    # Compare repeated calls to the SAME uninstrumented executable. Additional
    # diagnostic effects can change XLA code generation and the float32 path;
    # their inner-iteration history is not an execution-time reference.
    gates["admm_work_matches_warmup"] = (expected_admm is None or
                                        Counter(observed_admm) == Counter(expected_admm))
    host = dict(zip(OUTPUT_NAMES, jax.device_get(outputs), strict=True))
    result["diagnostic_output_comparison"] = {name: {
        "allclose_rtol_1e_3_atol_1e_4": bool(np.allclose(
            host[name], reference_outputs[name], rtol=1e-3, atol=1e-4)),
        "max_abs_difference": float(np.max(np.abs(host[name] - reference_outputs[name]))),
    } for name in OUTPUT_NAMES}
    result["diagnostic_comparison_policy"] = (
        "Informational comparison between different executables; workload acceptance "
        "uses physical checks and repeated execution of the uninstrumented warmup reference")
    result["observed_admm_signature"] = observed_admm
    # Public SQP outputs discard status. Report the separate runtime proof's
    # status with explicit attribution; do not claim per-trial SQP flag access.
    result["diagnostic_status_source"] = "Separate untimed process, same source/configuration/JAX input avals"
    result["workload_checks_passed"] = all(passed for name, passed in gates.items()
                                           if name != "sqp_stopping_predicate")
    result["accepted"] = False  # Never a validated converged control solution.
    return result, host


def run_benchmark(args, provenance):
    import jax
    import numpy as np

    output = args.output_dir
    hashes = file_hashes(args.source_root.resolve())
    manifest = {
        "baseline_label": LABEL, "provenance": provenance, "file_sha256": hashes,
        "python": sys.version, "executable": sys.executable,
        "versions": {name: metadata.version(name) for name in
                     ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "scipy", "trajax")},
        "methodology": METHOD, "warmup_count": args.warmup, "trial_count": args.trials,
        "max_solve_seconds": args.max_solve_seconds,
        "cache_policy": "Persistent JAX cache disabled; run-local incidental caches",
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
            "JAX_ENABLE_COMPILATION_CACHE", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR")},
    }
    summary = {
        "baseline_label": LABEL, "status": "started", "validated_converged_mpc": False,
        "solver_converged": None, "robust_sls_executed": False,
        "convergence_status_source": "Separate untimed instrumented diagnostic",
        "timed_sqp_convergence_flag": "Not exposed by the unchanged public solver return",
        "warmup_count_requested": args.warmup, "trial_count_requested": args.trials,
        "warmups_completed": 0, "trials_completed": 0,
    }
    samples, records = [], []
    try:
        print(f"{LABEL}: running one separate, untimed runtime diagnostic", flush=True)
        command = [sys.executable, "-B", str(Path(__file__).parent / "launcher.py"),
                   "--source-root", str(args.source_root.resolve()), "--mode", "diagnose",
                   "--variant", "original", "--output-dir", str(output / "diagnostic")]
        with (output / "diagnostic_console.log").open("x") as console:
            probe = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT,
                                   timeout=300, check=False)
        proof_path = output / "diagnostic/report.json"
        proof = json.loads(proof_path.read_text())
        if probe.returncode not in (0, 2):
            raise RuntimeError(f"Diagnostic launcher failed: exit {probe.returncode}")
        check_fixed_budget_proof(proof)
        if proof["file_sha256"] != hashes:
            raise RuntimeError("Source changed between timing setup and runtime proof")
        with (output / "diagnostic/events.jsonl").open() as stream:
            events = [json.loads(line) for line in stream]
        expected_admm = [(e["iterations"], e["converged"]) for e in events
                         if e["event"] == "admm_summary"]
        if len(expected_admm) != proof["diagnostics"]["counts"]["admm_summary"]:
            raise RuntimeError("Incomplete diagnostic ADMM events")
        with np.load(output / "diagnostic/outputs.npz", allow_pickle=False) as archive:
            reference_outputs = {name: archive[name] for name in OUTPUT_NAMES}
        summary.update(solver_converged=False, robust_sls_executed=True,
                       diagnostic_counts=proof["diagnostics"]["counts"],
                       convergence_diagnostics=proof["diagnostics"],
                       diagnostic_report="diagnostic/report.json")

        with gpu_lock_path().open("a") as lock, (output / "solve.log").open("x") as log:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            device, manifest["gpu_model_driver_uuid"] = gpu_preflight()
            manifest["device"] = str(device)
            with jax.default_device(device):
                scenario = build_scenario(args.source_root, "original")
                if scenario.definition != proof["scenario"]:
                    raise RuntimeError("Scenario differs from the untimed diagnostic")
                controller_hash = input_digest(scenario.inputs)
                if controller_hash != proof["input_sha256_before"]:
                    raise RuntimeError("Initial arguments/state differ from the untimed diagnostic")
                frozen = freeze_on_device(scenario.inputs, device)
                frozen_hash = input_digest(frozen)
                manifest.update(scenario=scenario.definition, input_sha256=frozen_hash,
                                controller_input_sha256=controller_hash,
                                frozen_input_signatures=[{
                                    "shape": list(a.shape), "dtype": str(a.dtype),
                                    "weak_type": bool(a.weak_type), "device": str(device),
                                } for a in frozen])
                print("Compiling original _solve without the extra diagnostic observers", flush=True)
                compiled = scenario.controller._solve.lower(*frozen).compile()
                warm_reference, warm_admm = None, None
                for phase, count in (("warmup", args.warmup), ("trial", args.trials)):
                    for index in range(1, count + 1):
                        capture = io.StringIO()
                        with redirect_stdout(capture):
                            elapsed_ms, outputs = synchronized_call(compiled, frozen, jax)
                        text = capture.getvalue()
                        # Every write/check below is outside the timed region.
                        log.write(f"\n--- {phase} {index} ---\n{text}")
                        log.flush()
                        validation, host = verify_result(
                            scenario, frozen, frozen_hash, controller_hash, outputs,
                            proof["diagnostics"], text, warm_admm, reference_outputs)
                        if warm_reference is None:
                            warm_reference = host
                            warm_admm = admm_signature(text)
                            summary["diagnostic_vs_timed_admm_history_matches"] = (
                                Counter(warm_admm) == Counter(expected_admm))
                        validation["gates"]["repeat_outputs_agree"] = all(
                            np.allclose(host[name], warm_reference[name], rtol=1e-5, atol=1e-6)
                            for name in OUTPUT_NAMES)
                        validation["workload_checks_passed"] &= validation["gates"]["repeat_outputs_agree"]
                        records.append({"phase": phase, "index": index,
                                        "elapsed_ms": elapsed_ms, "validation": validation})
                        if phase == "trial":
                            samples.append(elapsed_ms)
                            summary["trials_completed"] += 1
                        else:
                            summary["warmups_completed"] += 1
                        if not validation["workload_checks_passed"]:
                            failed = [k for k, v in validation["gates"].items()
                                      if not v and k != "sqp_stopping_predicate"]
                            raise RuntimeError(f"{phase} {index} failed workload checks: {failed}")
                        print(f"{phase} {index}/{count}: {elapsed_ms:.3f} ms ({LABEL})", flush=True)
                        if elapsed_ms > args.max_solve_seconds * 1000:
                            raise RuntimeError("Completed solve exceeded --max-solve-seconds; stopping repetitions")
                manifest["provenance"]["module_origins"] = verify_origins(args.source_root)
                if file_hashes(args.source_root.resolve()) != hashes:
                    raise RuntimeError("Source changed during benchmark")
                summary["status"] = "completed_nonconverged_fixed_budget"
                summary["workload_checks_passed"] = True
                summary["timed_full_sqp_budget_executed"] = True
                summary["timed_robust_path_evidence"] = (
                    "Separate runtime proof; timed executable returns nonzero responses from zero "
                    "response inputs and emits the maximum 120 ADMM call summaries, requiring all "
                    "110 SQP iterations and 10 robust SLS calls. Repeated timed ADMM histories and "
                    "outputs match the uninstrumented warmup. Exact timed SQP flag is not exposed.")
    except Exception as exc:
        summary["status"] = "benchmark_failed"
        summary["workload_checks_passed"] = False
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        summary.update(latency_statistics(samples))
        summary["runs"] = records
        manifest["status"] = summary["status"]
        with (output / "timings.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("trial", "elapsed_ms"))
            writer.writerows(enumerate(samples, start=1))
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", summary)
        print(f"{summary['status']}: {output}", flush=True)
    return 0 if summary["status"] == "completed_nonconverged_fixed_budget" else 2
