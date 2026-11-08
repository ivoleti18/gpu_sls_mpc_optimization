"""Reuse the example's exact model without its plotting/cache side effects."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MODEL_CONSTANTS = {"MASS", "GRAVITY", "JX", "JY", "JZ", "J", "J_INV"}
MODEL_FUNCTIONS = {
    "wrap_to_pi", "rotation_matrix", "euler_angle_rates_matrix",
    "rigid_body_3d_step", "dynamics", "cost", "build_piecewise_reference",
}
OUTPUT_NAMES = ("X", "U", "V", "w", "y", "rho", "backoffs", "Phi_x",
                "Phi_u", "beta", "mu")


def load_example_model(checkout):
    """Compile unchanged AST nodes, retaining source filenames/line numbers.

    Importing the entire script would import plotting and set /tmp/jax_cache.
    Only physical constants and the named pure model definitions are loaded.
    No function body is rewritten, and main() is never executed.
    """
    import jax
    import jax.numpy as jnp
    source = Path(checkout) / "examples/quadrotor/quadrotor.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                            level=0)]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in MODEL_CONSTANTS:
                nodes.append(node)
                found.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name in MODEL_FUNCTIONS:
            nodes.append(node)
            found.add(node.name)
    if found != MODEL_CONSTANTS | MODEL_FUNCTIONS:
        raise RuntimeError(f"Example model interface changed: {found}")
    namespace = {"jax": jax, "jnp": jnp, "Any": Any}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(**{k: namespace[k] for k in found})


@dataclass
class Scenario:
    definition: dict
    model: Any
    controller: Any
    inputs: tuple
    goal: Any
    x_min: Any
    x_max: Any
    u_min: Any
    u_max: Any


def build_scenario(checkout, variant):
    import jax
    import jax.numpy as jnp
    from gpu_sls.generic_mpc import GenericMPC, MPCConfig
    from gpu_sls.gpu_admm import ADMMConfig
    from gpu_sls.gpu_sls import SLSConfig
    from gpu_sls.gpu_sqp import SQPConfig
    from gpu_sls.utils.constraint_utils import (
        combine_constraints, make_constant_disturbance,
        make_control_box_constraints, make_state_box_constraints,
    )

    config_path = Path(__file__).parent / "scenarios/quadrotor_reference_v0.json"
    spec = json.loads(config_path.read_text())
    spec["variant"] = variant
    spec["changes_from_original"] = {}
    if variant == "robust":
        spec["sls"]["max_initial_sqp_iterations"] = 0
        spec["changes_from_original"] = {
            "sls.max_initial_sqp_iterations": {"original": 100, "candidate": 0}
        }
    elif variant != "original":
        raise ValueError(variant)
    jax.config.update("jax_enable_x64", spec["jax_enable_x64"])
    model = load_example_model(checkout)
    array = lambda value: jnp.asarray(value, dtype=jnp.float32)
    n, nu, N, dt = (spec[k] for k in ("n", "nu", "N", "dt"))
    hover = model.MASS * model.GRAVITY
    x0, goal = array(spec["x0"]), array(spec["goal"])
    x_min, x_max = array(spec["x_min"]), array(spec["x_max"])
    u_min, u_max = array(spec["u_min"]), array(spec["u_max"])
    obstacles = array(spec["obstacles"])
    cfg = MPCConfig(n=n, nu=nu, N=N, dt=dt, W=array(spec["weights"]),
                    u_ref=array([hover, 0, 0, 0]))
    controller = GenericMPC(
        SLSConfig(**spec["sls"]), SQPConfig(**spec["sqp"]),
        ADMMConfig(**spec["admm"]), config=cfg,
        dynamics=model.dynamics,
        constraints=combine_constraints(make_state_box_constraints(x_min, x_max),
                                        make_control_box_constraints(u_min, u_max)),
        obstacles=obstacles, cost=model.cost,
        num_constraints=2 * n + 2 * nu + obstacles.shape[0],
        disturbance=make_constant_disturbance(n, spec["disturbance_magnitude"] * dt),
        shift=spec["shift"], X_in=jnp.zeros((N + 1, n), dtype=jnp.float32),
        U_in=jnp.zeros((N, nu), dtype=jnp.float32).at[:, 0].set(hover),
    )
    reference = model.build_piecewise_reference(x0, goal, N, dt)
    # Match GenericMPC.run's _solve call exactly, without its state mutation.
    inputs = (
        reference, dt, cfg.W, x0, controller.X0, controller.U0, controller.V0,
        controller.w, controller.y, controller.rho, obstacles,
        controller.h_ct_ws, controller.beta_ws, controller.mu_ws,
        controller.Phi_x_ws, controller.Phi_u_ws,
    )
    spec["physical_parameters"] = {
        k: getattr(model, k) for k in ("MASS", "GRAVITY", "JX", "JY", "JZ")
    }
    return Scenario(spec, model, controller, inputs, goal, x_min, x_max, u_min, u_max)


def input_digest(inputs):
    """Hash values, shapes, dtypes and Python scalar types of frozen inputs."""
    import numpy as np
    digest = hashlib.sha256()
    for value in inputs:
        array = np.asarray(value)
        digest.update(repr((type(value).__name__, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
