"""Run on CPU: python -B -m unittest discover -s benchmarks/tests -v."""

import ast
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"

import source_isolation


class SourceIsolationTests(unittest.TestCase):
    def test_real_launcher_removes_upstream_and_checks_all_origins(self):
        env = dict(os.environ, PYTHONPATH="/home/ors/gpu_sls/src", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "benchmarks/launcher.py"),
                               "--source-root", str(ROOT), "--mode", "imports"],
                              env=env, text=True, capture_output=True, check=True)
        report = json.loads(proc.stdout)
        self.assertIn("/home/ors/gpu_sls/src", report["removed_import_paths"])
        self.assertEqual(report["submodule_commit"], source_isolation.PINNED_ILQR)
        self.assertIn("gpu_sls.external", report["module_origins"])
        for entry in report["module_origins"].values():
            paths = ([entry["file"]] if entry["file"] else []) + entry["search_paths"]
            self.assertTrue(all(Path(p).is_relative_to(ROOT / "src") for p in paths))

    def test_foreign_namespace_is_rejected(self):
        foreign = types.ModuleType("gpu_sls.foreign")
        foreign.__path__ = ["/home/ors/gpu_sls/src/gpu_sls"]
        with patch.dict(sys.modules, {foreign.__name__: foreign}):
            with self.assertRaisesRegex(RuntimeError, "Foreign solver import"):
                source_isolation.verify_origins(ROOT)

    def test_preloaded_solver_is_rejected(self):
        with patch.dict(sys.modules, {"gpu_sls": types.ModuleType("gpu_sls")}):
            with self.assertRaisesRegex(RuntimeError, "before solver imports"):
                source_isolation.select_source(ROOT)

    def test_wrong_gitlink_is_rejected_before_importing(self):
        with patch.object(source_isolation, "is_solver_module", return_value=False), \
             patch.object(source_isolation, "git", return_value="160000 commit wrong path"):
            with self.assertRaisesRegex(RuntimeError, "Unexpected.*gitlink"):
                source_isolation.select_source(ROOT)

    def test_wrong_submodule_head_is_rejected(self):
        gitlink = f"160000 commit {source_isolation.PINNED_ILQR} path"
        with patch.object(source_isolation, "is_solver_module", return_value=False), \
             patch.object(source_isolation, "git", side_effect=[gitlink, "wrong"]):
            with self.assertRaisesRegex(RuntimeError, "must be clean"):
                source_isolation.select_source(ROOT)


class ScenarioAndDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fresh source selection for this test process, before its first imports.
        source_isolation.select_source(ROOT)
        import jax
        jax.config.update("jax_enable_x64", False)
        from quadrotor_scenario import build_scenario
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.original = build_scenario(ROOT, "original")
            cls.robust = build_scenario(ROOT, "robust")

    def test_scenario_matches_original_main_before_first_solve(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from gpu_sls.generic_mpc import MPCConfig
        from gpu_sls.gpu_admm import ADMMConfig
        from gpu_sls.gpu_sls import SLSConfig
        from gpu_sls.gpu_sqp import SQPConfig
        import gpu_sls.utils.constraint_utils as constraints

        captured = {}

        class Captured(Exception):
            pass

        class Controller:
            def __init__(self, *args, **kwargs):
                captured.update(configs=args, construction=kwargs)

            def run(self, **kwargs):
                captured["run"] = kwargs
                raise Captured

        source = ROOT / "examples/quadrotor/quadrotor.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        namespace = dict(vars(self.original.model), jax=jax, jnp=jnp,
                         MPCConfig=MPCConfig, ADMMConfig=ADMMConfig,
                         SLSConfig=SLSConfig, SQPConfig=SQPConfig, GenericMPC=Controller,
                         NUM_RANDOM=5, NUM_ADV=26)
        namespace.update({name: getattr(constraints, name) for name in (
            "make_control_box_constraints", "make_state_box_constraints",
            "combine_constraints", "make_constant_disturbance")})
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with warnings.catch_warnings(), self.assertRaises(Captured):
            warnings.simplefilter("ignore", UserWarning)
            namespace["main"]()

        scenario = self.original
        for old, current in zip(captured["configs"], (
                scenario.controller.sls_config, scenario.controller.sqp_config,
                scenario.controller.admm_config), strict=True):
            self.assertEqual(asdict(old), asdict(current))
        original = captured["construction"]
        for name in ("n", "nu", "N", "dt"):
            self.assertEqual(getattr(original["config"], name), getattr(scenario.controller.config, name))
        for name in ("W", "u_ref"):
            np.testing.assert_array_equal(getattr(original["config"], name),
                                          getattr(scenario.controller.config, name))
        for key, current in (("X_in", scenario.controller.X0), ("U_in", scenario.controller.U0),
                             ("obstacles", scenario.controller.obstacles)):
            np.testing.assert_array_equal(original[key], current)
        for name, index in (("x0", 3), ("reference", 0), ("parameter", 1)):
            np.testing.assert_array_equal(captured["run"][name], scenario.inputs[index])
        for name in ("constraints", "disturbance"):
            old, current = original[name], getattr(scenario.controller, name)
            args = ((scenario.inputs[3], scenario.controller.config.u_ref, 0)
                    if name == "constraints" else (scenario.controller.X0,))
            np.testing.assert_array_equal(old(*args), current(*args))
        self.assertEqual(original["num_constraints"], 33)
        self.assertEqual(original["shift"], scenario.controller.shift)

    def test_candidate_changes_only_nominal_iteration_threshold(self):
        from quadrotor_scenario import input_digest
        self.assertEqual(input_digest(self.original.inputs), input_digest(self.robust.inputs))
        old, new = dict(self.original.definition), dict(self.robust.definition)
        for key in ("variant", "changes_from_original"):
            old.pop(key)
            new.pop(key)
        new["sls"] = dict(new["sls"], max_initial_sqp_iterations=100)
        self.assertEqual(old, new)

    def test_diagnostic_events_come_from_runtime_not_tracing(self):
        import jax
        import jax.numpy as jnp
        import gpu_sls.gpu_sls as sls
        from runtime_diagnostics import RuntimeDiagnostics

        observer = RuntimeDiagnostics(io.StringIO(), self.original.definition["sqp"])
        # Tiny stand-ins test callback placement/branch behavior without solving.
        with patch.object(sls, "calculate_phis", jax.jit(lambda x: (x, x))), \
             patch.object(sls, "get_controller", jax.jit(lambda x: sls.calculate_phis(x))):
            observer.install()
            try:
                work = jax.jit(lambda flag, x: jax.lax.cond(
                    flag, lambda a: sls.get_controller(a), lambda a: (a, a), x))
                x = jnp.ones((2,), dtype=jnp.float32)
                compiled = work.lower(True, x).compile()
                self.assertEqual(observer.events, [])
                jax.block_until_ready(compiled(False, x))
                jax.effects_barrier()
                self.assertEqual(observer.events, [])
                for _ in range(2):
                    jax.block_until_ready(compiled(True, x))
                    jax.effects_barrier()
                self.assertEqual(observer.summary()["counts"],
                                 {"calculate_phis": 2, "get_controller": 2})
            finally:
                observer.restore()


if __name__ == "__main__":
    unittest.main()
