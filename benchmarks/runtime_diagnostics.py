"""Opt-in runtime callbacks; no solver file or mathematical output is changed."""

from collections import Counter
from functools import wraps
import json
import math


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray) or (hasattr(value, "__array__")
                                        and not isinstance(value, np.generic)):
        # JAX 0.8.1 supplies CPU JAX arrays to debug callbacks, not necessarily
        # NumPy arrays. Convert only here on the host, after runtime execution.
        return json_safe(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class RuntimeDiagnostics:
    """Install before tracing, and run one diagnostic solve per fresh process.

    Callbacks depend on runtime results and execute on the host only when the
    device executes that branch. Python tracing alone produces no events.
    Diagnostic executables must never be used as performance measurements.
    """

    def __init__(self, stream, sqp_config):
        self.stream = stream
        self.sqp_config = sqp_config
        self.events = []
        self.patches = []

    def record(self, event, **fields):
        item = json_safe({"sequence": len(self.events), "event": event, **fields})
        self.events.append(item)
        self.stream.write(json.dumps(item, allow_nan=False) + "\n")
        self.stream.flush()

    def patch(self, module, name, replacement):
        self.patches.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def install(self):
        import jax
        import jax.numpy as jnp
        import gpu_sls.gpu_sls as sls
        import gpu_sls.gpu_sqp as sqp

        def mark(event, result, **fields):
            def receive(max_abs_output, **values):
                self.record(event, max_abs_output=max_abs_output, **values)
            jax.debug.callback(receive, jnp.max(jnp.abs(result[0])),
                               ordered=True, **fields)

        def observed(module, name, event, field_extractor=None):
            original = getattr(module, name)

            @wraps(original)
            def wrapper(*args, **kwargs):
                result = original(*args, **kwargs)
                fields = field_extractor(result) if field_extractor else {}
                mark(event, result, **fields)
                return result
            self.patch(module, name, wrapper)

        observed(sls, "calculate_phis", "calculate_phis")
        observed(sls, "get_controller", "get_controller")
        observed(sqp, "sls_solve_gpu", "sls_solve_gpu", lambda r: {
            "converged": r[6], "last_admm_converged": r[7],
        })
        for module, event in ((sqp, "admm_nominal"), (sls, "admm_robust")):
            observed(module, "constrained_solve", event,
                     lambda r: {"converged": r[7], "rho": r[5]})

        original_direction = sqp.compute_search_direction

        @wraps(original_direction)
        def direction(*args, **kwargs):
            result = original_direction(*args, **kwargs)
            feasibility = jnp.max(jnp.abs(args[12]))
            step = jnp.maximum(jnp.max(jnp.abs(result[0])), jnp.max(jnp.abs(result[1])))
            threshold = self.sqp_config["step_tol"] * (
                1 + jnp.maximum(jnp.max(jnp.abs(args[9])), jnp.max(jnp.abs(args[10]))))
            predicate = (feasibility <= self.sqp_config["feas_tol"]) & (step <= threshold)
            mark("sqp_iteration", result, iteration=args[21], feasibility=feasibility,
                 step=step, step_threshold=threshold, stopping_predicate=predicate)
            return result
        self.patch(sqp, "compute_search_direction", direction)

        # Observe operands of the existing final ADMM debug print, retaining
        # the original print and exact termination residuals/iteration count.
        original_print = jax.debug.print

        def debug_print(fmt, *args, **kwargs):
            original_print(fmt, *args, **kwargs)
            if fmt.startswith("ADMM done: Total Iterations="):
                names = ("iterations", "converged", "rho", "primal_residual",
                         "primal_threshold", "dual_residual", "dual_threshold", "initial_rho")
                if len(args) != len(names):
                    raise RuntimeError("ADMM diagnostic format changed")
                jax.debug.callback(lambda *v: self.record("admm_summary", **dict(zip(names, v))),
                                   *args, ordered=True)
        self.patch(jax.debug, "print", debug_print)

    def restore(self):
        for module, name, original in reversed(self.patches):
            setattr(module, name, original)
        self.patches.clear()

    def summary(self):
        def last(event):
            return next((e for e in reversed(self.events) if e["event"] == event), None)
        return {
            "counts": dict(Counter(e["event"] for e in self.events)),
            "last_sqp_iteration": last("sqp_iteration"),
            "last_sls_solve": last("sls_solve_gpu"),
            "last_admm": last("admm_summary"),
            "admm_failed_calls": sum(e["event"] == "admm_summary" and not e["converged"]
                                     for e in self.events),
            "sqp_status_method": "stopping predicate reconstructed from unchanged search-direction operands",
        }
