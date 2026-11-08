"""Correctness gates for the preserved v0 scenario, outside solve timing."""

import numpy as np

from quadrotor_scenario import OUTPUT_NAMES


def validate(scenario, outputs, diagnostics):
    import jax
    spec = scenario.definition
    values = dict(zip(OUTPUT_NAMES, jax.device_get(outputs), strict=True))
    N, n, nu = (spec[k] for k in ("N", "n", "nu"))
    nc = 2 * n + 2 * nu + len(spec["obstacles"])
    expected = {
        "X": (N + 1, n), "U": (N, nu), "V": (N + 1, n),
        "w": (N + 1, nc), "y": (N + 1, nc), "rho": (),
        "backoffs": (N + 1, nc - len(spec["obstacles"])),
        "Phi_x": (N + 1, N + 1, n, n), "Phi_u": (N, N + 1, nu, n),
        "beta": (N + 1, N + 1, nc - len(spec["obstacles"])), "mu": (N + 1, nc),
    }
    details = {name: {"shape": list(np.shape(a)), "dtype": str(a.dtype),
                      "finite": bool(np.isfinite(a).all())} for name, a in values.items()}
    gates = {
        "finite_outputs": all(a["finite"] for a in details.values()),
        "shapes": all(np.shape(values[name]) == shape for name, shape in expected.items()),
        "dtypes": all(a["dtype"] == spec["dtype"] for a in details.values()),
    }
    counts = diagnostics["counts"]
    gates["robust_runtime_execution"] = all(counts.get(k, 0) > 0 for k in
        ("sls_solve_gpu", "get_controller", "calculate_phis", "admm_robust"))
    last_sqp = diagnostics["last_sqp_iteration"]
    last_sls = diagnostics["last_sls_solve"]
    last_admm = diagnostics["last_admm"]
    gates["sqp_stopping_predicate"] = bool(last_sqp and last_sqp["stopping_predicate"])
    gates["final_sls_converged"] = bool(last_sls and last_sls["converged"])
    gates["final_admm_converged"] = bool(last_admm and last_admm["converged"])
    report = {"gates": gates, "outputs": details, "metrics": {}}
    if not all(gates[k] for k in ("finite_outputs", "shapes", "dtypes")):
        report["accepted"] = False
        report["skipped"] = "Geometric/residual evaluation requires finite, correctly shaped float32 outputs"
        return report

    X, U = values["X"], values["U"]
    predicted = jax.vmap(scenario.model.rigid_body_3d_step, in_axes=(0, 0, None))(
        outputs[0][:-1], outputs[1], spec["dt"])
    residual = np.asarray(jax.device_get(predicted)) - X[1:]
    initial_residual = float(np.max(np.abs(X[0] - np.asarray(spec["x0"]))))
    dynamics_residual = float(np.max(np.abs(residual)))
    xmin, xmax = np.asarray(scenario.x_min), np.asarray(scenario.x_max)
    umin, umax = np.asarray(scenario.u_min), np.asarray(scenario.u_max)

    def violation(lower, value, upper):
        return float(max(0, np.max(lower - value), np.max(value - upper)))

    state_violation = violation(xmin, X, xmax)
    control_violation = violation(umin, U, umax)
    obstacles = np.asarray(spec["obstacles"])
    delta = X[:, None, :2] - obstacles[None, :, :2]
    clearance = np.linalg.norm(delta, axis=-1) - obstacles[None, :, 2]
    segment = X[1:, :2] - X[:-1, :2]
    offset = X[:-1, None, :2] - obstacles[None, :, :2]
    projection = np.clip(-np.sum(offset * segment[:, None, :], axis=-1) /
                         np.maximum(np.sum(segment * segment, axis=-1)[:, None], 1e-30), 0, 1)
    closest = offset + projection[:, :, None] * segment[:, None, :]
    segment_clearance = np.linalg.norm(closest, axis=-1) - obstacles[None, :, 2]
    goal_delta = X[-1] - np.asarray(spec["goal"])
    goal_delta[3:6] = (goal_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    goal_error = float(np.linalg.norm(goal_delta[:3]))
    tube_x = np.linalg.norm(values["Phi_x"], axis=-1).sum(axis=1)
    tube_u = np.linalg.norm(values["Phi_u"], axis=-1).sum(axis=1)
    state_tube_violation = float(max(0, np.max(xmin - (X - tube_x)),
                                    np.max(X + tube_x - xmax)))
    control_tube_violation = float(max(0, np.max(umin - (U - tube_u)),
                                      np.max(U + tube_u - umax)))
    tolerance = spec["validation"]
    gates.update({
        "nonlinear_dynamics": dynamics_residual <= spec["sqp"]["feas_tol"],
        "initial_state": initial_residual <= spec["sqp"]["feas_tol"],
        "state_bounds": state_violation <= tolerance["bounds_tolerance"],
        "control_bounds": control_violation <= tolerance["bounds_tolerance"],
        "obstacle_at_nodes": float(clearance.min()) >= -tolerance["obstacle_tolerance"],
        "obstacle_along_linear_segments": float(segment_clearance.min()) >= -tolerance["obstacle_tolerance"],
        "goal_position": goal_error <= tolerance["goal_position_tolerance"],
        "nonzero_response": bool(np.any(values["Phi_x"]) and np.any(values["Phi_u"])),
        "state_tube_bounds": state_tube_violation <= tolerance["bounds_tolerance"],
        "control_tube_bounds": control_tube_violation <= tolerance["bounds_tolerance"],
    })
    report["metrics"] = {
        "nonlinear_dynamics_max_abs": dynamics_residual,
        "initial_state_max_abs": initial_residual,
        "state_max_violation": state_violation,
        "control_max_violation": control_violation,
        "obstacle_min_node_clearance_m": float(clearance.min()),
        "obstacle_min_linear_segment_clearance_m": float(segment_clearance.min()),
        "goal_position_error_m": goal_error,
        "terminal_state": X[-1].tolist(), "terminal_goal_delta": goal_delta.tolist(),
        "terminal_speed_m_s": float(np.linalg.norm(X[-1, 6:9])),
        "terminal_body_rate_norm_rad_s": float(np.linalg.norm(X[-1, 9:12])),
        "state_tube_max_violation": state_tube_violation,
        "control_tube_max_violation": control_tube_violation,
        "Phi_x_max_abs": float(np.max(np.abs(values["Phi_x"]))),
        "Phi_u_max_abs": float(np.max(np.abs(values["Phi_u"]))),
    }
    report["limits"] = (
        "No disturbance rollouts or continuous-time safety proof. Segment clearance checks "
        "linear interpolation of discrete positions. Terminal speed/rates are inspected, "
        "not acceptance gates, because the unchanged reference has nonzero terminal velocity. "
        "These conservative diagnostic gates are not a frozen team baseline."
    )
    report["accepted"] = all(gates.values())
    return report
