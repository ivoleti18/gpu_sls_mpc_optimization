"""Timing contract tests; no quadrotor optimization or GPU time required."""

import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"

from full_solve_benchmark import (
    admm_signature, check_fixed_budget_proof, freeze_on_device,
    latency_statistics, synchronized_call, verify_result,
)


def proof_fixture():
    return {
        "gpu_solve_completed": True,
        "validation": {"gates": {
            "finite_outputs": True, "robust_runtime_execution": True,
            "sqp_stopping_predicate": False,
        }},
        "scenario": {"sqp": {"max_sqp_iterations": 10},
                     "sls": {"max_initial_sqp_iterations": 100}},
        "diagnostics": {"counts": {
            "sqp_iteration": 110, "sls_solve_gpu": 10, "get_controller": 20,
            "calculate_phis": 20, "admm_robust": 20,
        }},
    }


class TimingContractTests(unittest.TestCase):
    def test_timer_encloses_dispatch_complete_tree_and_effects(self):
        events = []
        inputs = (object(), object())
        outputs = (object(), {"independent_leaf": object()})

        class Jax:
            @staticmethod
            def block_until_ready(tree):
                events.append(("ready", tree))

            @staticmethod
            def effects_barrier():
                events.append(("effects",))

        def compiled(*args):
            self.assertTrue(all(a is b for a, b in zip(args, inputs, strict=True)))
            events.append(("dispatch",))
            return outputs

        values = iter((10.0, 10.025))

        def clock():
            events.append(("clock",))
            return next(values)

        elapsed, actual = synchronized_call(compiled, inputs, Jax, clock)
        self.assertIs(actual, outputs)
        self.assertAlmostEqual(elapsed, 25.0)
        self.assertEqual(events, [("ready", inputs), ("effects",), ("clock",),
                                  ("dispatch",), ("ready", outputs), ("effects",), ("clock",)])

    def test_freezing_preserves_weak_scalar_aval_and_input_values(self):
        import jax
        import jax.numpy as jnp
        from quadrotor_scenario import input_digest
        inputs = (jnp.arange(4, dtype=jnp.float32), 0.03)
        before = input_digest(inputs)
        frozen = freeze_on_device(inputs, jax.devices("cpu")[0])
        self.assertTrue(frozen[1].weak_type)
        self.assertEqual(jax.core.get_aval(inputs[1]), jax.core.get_aval(frozen[1]))
        self.assertEqual(input_digest(inputs), before)

    def test_real_jax_callback_finishes_before_stop_clock(self):
        import jax
        import jax.numpy as jnp
        finished = []

        def work(x):
            jax.debug.callback(lambda value: finished.append(True), x, ordered=True)
            return x + 1, {"other": x * 2}

        inputs = (jnp.ones((2,), dtype=jnp.float32),)
        compiled = jax.jit(work).lower(*inputs).compile()
        self.assertEqual(finished, [])
        clock_calls = []

        def clock():
            clock_calls.append(len(finished))
            return float(len(clock_calls))

        synchronized_call(compiled, inputs, jax, clock)
        self.assertEqual(clock_calls, [0, 1])

    def test_only_expected_sqp_nonconvergence_is_permitted(self):
        proof = proof_fixture()
        check_fixed_budget_proof(proof)
        for gate in ("finite_outputs", "robust_runtime_execution"):
            bad = copy.deepcopy(proof)
            bad["validation"]["gates"][gate] = False
            with self.assertRaisesRegex(RuntimeError, "failed workload checks"):
                check_fixed_budget_proof(bad)
        proof["validation"]["gates"]["sqp_stopping_predicate"] = True
        with self.assertRaisesRegex(RuntimeError, "convergence behavior changed"):
            check_fixed_budget_proof(proof)

    def test_missing_runtime_sls_evidence_is_rejected(self):
        proof = proof_fixture()
        proof["diagnostics"]["counts"]["get_controller"] = 0
        with self.assertRaisesRegex(RuntimeError, "runtime proof"):
            check_fixed_budget_proof(proof)

    def test_statistics_and_existing_print_parser(self):
        stats = latency_statistics([10.0, 40.0, 10.0])
        self.assertEqual(stats, {"mean_ms": 20.0, "median_ms": 10.0,
                                "minimum_ms": 10.0, "maximum_ms": 40.0,
                                "raw_timings_ms": [10.0, 40.0, 10.0]})
        self.assertIsNone(latency_statistics([])["mean_ms"])
        self.assertEqual(admm_signature(
            "ADMM done: Total Iterations=399 converged=False rho=1\n"
            "ADMM done: Total Iterations=59 converged=True rho=4\n"),
            [(399, False), (59, True)])

    def test_repetitions_use_uninstrumented_work_reference(self):
        import jax.numpy as jnp
        import numpy as np
        from quadrotor_scenario import OUTPUT_NAMES, input_digest
        frozen = (jnp.ones((1,), dtype=jnp.float32),)
        scenario = SimpleNamespace(inputs=frozen, definition={
            "sls": {"max_initial_sqp_iterations": 1, "max_sls_iterations": 1},
            "sqp": {"max_sqp_iterations": 1},
        })
        outputs = tuple(jnp.ones((1,), dtype=jnp.float32) for _ in OUTPUT_NAMES)
        other_executable = {name: np.zeros((1,), dtype=np.float32) for name in OUTPUT_NAMES}
        diagnostic_validation = {"gates": {"finite_outputs": True,
                                           "sqp_stopping_predicate": False}}
        log = ("ADMM done: Total Iterations=2 converged=True\n"
               "ADMM done: Total Iterations=1 converged=True\n")
        digest = input_digest(frozen)
        with patch("full_solve_benchmark.validate", return_value=copy.deepcopy(diagnostic_validation)):
            first, _ = verify_result(scenario, frozen, digest, digest, outputs, {},
                                     log, None, other_executable)
        self.assertTrue(first["workload_checks_passed"])
        self.assertFalse(first["diagnostic_output_comparison"]["X"]["allclose_rtol_1e_3_atol_1e_4"])
        with patch("full_solve_benchmark.validate", return_value=copy.deepcopy(diagnostic_validation)):
            changed, _ = verify_result(scenario, frozen, digest, digest, outputs, {},
                                       log, [(1, True), (1, True)], other_executable)
        self.assertFalse(changed["workload_checks_passed"])
        self.assertFalse(changed["gates"]["admm_work_matches_warmup"])

    def test_cli_rejects_parameter_variant_and_invalid_counts(self):
        base = [sys.executable, "-B", str(ROOT / "benchmarks/launcher.py"),
                "--source-root", str(ROOT), "--mode", "benchmark"]
        for extra in (("--variant", "robust"), ("--warmup", "0"),
                      ("--trials", "-1"), ("--max-solve-seconds", "nan")):
            proc = subprocess.run([*base, *extra], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
