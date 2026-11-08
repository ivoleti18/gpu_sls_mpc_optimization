# Quadrotor runtime reference

Milestone 2 adds synchronized full-solve timing of the unchanged original
scenario. This reference is explicitly **NONCONVERGED / FIXED-BUDGET**. It is
not a validated converged MPC solution. Solver algorithms, tolerances and iteration
limits are unchanged. See [Milestone 1](MILESTONE1_REPORT.md) for the diagnostic
findings and [Milestone 2](MILESTONE2_REPORT.md) for actual timing results.

## Reproduce

From a shell on the lab PC:

```bash
source /home/ors/.local/share/gpu_sls-env-report/activate.sh
cd /home/ors/gpu_sls_mpc_optimization

# Imports only; does not initialize a GPU or create an experiment directory.
python -B benchmarks/launcher.py \
  --source-root /home/ors/gpu_sls_mpc_optimization --mode imports

# One diagnostic solve, using exactly the original numerical configuration.
python -B benchmarks/launcher.py \
  --source-root /home/ors/gpu_sls_mpc_optimization \
  --mode diagnose --variant original

# Runtime smoke test: separate diagnostic, compilation, one warm-up, three trials.
python -B benchmarks/launcher.py \
  --source-root /home/ors/gpu_sls_mpc_optimization \
  --mode benchmark --warmup 1 --trials 3

# A larger run, when an exclusive GPU window is available.
python -B benchmarks/launcher.py \
  --source-root /home/ors/gpu_sls_mpc_optimization \
  --mode benchmark --warmup 5 --trials 30

# CPU regression checks; no optimization solve.
python -B -m unittest discover -s benchmarks/tests -v
```

The diagnostic currently exits **2** because its SQP convergence gate fails.
Benchmark mode permits exactly that expected nonconvergence, while requiring
the other workload checks to pass. It exits **0** when the requested benchmark
completes, recording `completed_nonconverged_fixed_budget` and
`validated_converged_mpc: false`. This exit code indicates a successful timing
experiment, not a successful converged control solve. Launcher failures exit 1;
experiment/verification failures exit 2 and preserve failure artifacts.

Counts must be positive; benchmark mode only accepts `--variant original`.
`--max-solve-seconds` defaults to 30 and stops repetitions after a completed
warm-up/trial exceeds that limit. It does not interrupt in-flight GPU work.

Each diagnostic creates a unique directory under `benchmarks/results/` containing
`report.json`, runtime `events.jsonl`, `solve.log`, `inputs.npz`, and `outputs.npz`.
The report records source hashes, actual module origins, revisions, input hashes,
configuration, device, convergence diagnostics and correctness gates. JSON uses
strings for nonfinite diagnostic values rather than invalid JSON NaN literals.
Results and incidental caches are ignored by Git. Existing directories are
rejected, not overwritten. An optional `--output-dir` must name a new directory
inside this checkout's `benchmarks/results/`.

The runner obtains the same nonblocking GPU lock at
`benchmarks/results/.gpu.lock` even for nested/custom run directories and rejects
other active GPU compute processes. Everyone using this harness path shares that
lock. It cannot prevent unrelated GPU jobs from starting afterward.
Both CUDA and CPU JAX backends are enabled because JAX host callbacks need CPU
devices. The solve explicitly requires a GPU, places its inputs on that GPU,
and checks all output placements; there is no CPU solve fallback.

## Source selection

Before importing GPU-SLS, `source_isolation.py` removes import-path entries
containing conflicting solver source packages and puts the selected `src` first.
It verifies the parent gitlink, the clean initialized submodule at
`273d78dec439ed270af6f9cb8340af801fefd332`, and every imported solver file and
namespace-package search path. Preloaded solver modules and foreign module
origins are rejected. Installed dependencies and the editable installation are
unchanged.

For a future developer clone, use the same launcher with that clone's absolute
`--source-root`. It must contain the pinned initialized submodule. All imports
and namespace paths must resolve inside that clone's `src`; no editable reinstall
is needed. The clean pinned-submodule restriction is deliberate for this
milestone; future submodule development needs a reviewed provenance policy.
No developer clones were created here. To benchmark a future development checkout
using the shared harness, change only the source-root argument:

```bash
source /home/ors/.local/share/gpu_sls-env-report/activate.sh
python -B /home/ors/gpu_sls_mpc_optimization/benchmarks/launcher.py \
  --source-root /absolute/path/to/development-checkout \
  --mode benchmark --warmup 1 --trials 3
```

That checkout must have its initialized pinned submodule. The scenario JSON and
runner come from this harness; solver/model functions come from the selected
source. Results stay under this harness checkout, not in the selected source.
Use `--mode imports` to inspect origins first. This invocation contract is tested
in the team checkout; separate developer checkouts have not been created/tested.

## Preserved scenario

`scenarios/quadrotor_reference_v0.json` records the original horizon (`110`),
timestep (`0.03 s`), dimensions (`12`, `4`), cost weights, start, goal, state and
control bounds, cylinder, disturbance settings and all solver configurations.
It preserves the original zero state guess and hover input guess. The XY bounds
remain +/-15 m; no 5x5 workspace, waypoint or physical-parameter change is made.

`quadrotor_scenario.py` compiles the **unchanged AST nodes** for the original
physical constants, nonlinear dynamics, cost and reference builder from the
selected checkout's `examples/quadrotor/quadrotor.py`. It does not import plotting,
execute `main()`, or execute the example's shared-cache configuration. This avoids
copying or rewriting the mathematical model. Source filename/line information is
retained. A regression test executes the original `main()` only until an
intercepted first controller call and checks the scenario/configuration parity.

Effective JAX precision remains float32 with x64 disabled, including the original
model's float64-to-float32 warnings. Disturbance metadata is retained, but this
milestone does not execute rollouts. JAX persistent compilation caching is
disabled for diagnostics and incidental caches are redirected into the run
directory. These are execution-policy differences, not numerical scenario changes.

`--variant robust` is an available **untested** candidate whose only numerical
configuration change is `sls.max_initial_sqp_iterations: 100 -> 0`. There is no
automatic fallback or parameter search. It was not run: the original already
executes SLS, and its failed convergence gate requires review before further
numerical experiments.

## Runtime proof and validation

`runtime_diagnostics.py` temporarily wraps functions in this process, before
tracing, with ordered `jax.debug.callback` calls depending on runtime results.
It observes `sls_solve_gpu`, `get_controller`, `calculate_phis`, nominal and robust
constrained ADMM calls. It additionally observes the operands of the solver's
existing final ADMM print, retaining that print, to record exact residuals,
thresholds, convergence and iteration counts. No solver file is edited and all
mathematical outputs are returned unchanged.

SQP's public return discards its convergence flag. The diagnostic reconstructs
the same stopping predicate from the search-direction operands and records each
executed SQP iteration. This is explicitly labeled in the report. Tests verify
that tracing/compilation and an untaken JIT branch produce no events, while two
runtime executions produce two sets of events.

The complete 11-output tree is synchronized, then `jax.effects_barrier()` drains
callbacks before inspecting or serializing results. The frozen `_solve` arguments
are used directly; `GenericMPC.run()` state is never mutated. These extra callbacks
affect execution and are **not** a timing implementation.

Validation checks all output shapes/dtypes/finiteness, GPU placement, unchanged
input hashes, runtime SLS evidence, stopping/convergence diagnostics, nonlinear
dynamics and initial-state residuals, original bounds, cylinder clearance at
nodes and along linearly interpolated segments, goal position, response presence,
and componentwise state/control tube bounds. Bound tolerance is 0.001 in each
component's units; obstacle numerical tolerance is 0.00001 m; goal position
tolerance is the example's 0.25 m; dynamics tolerance is SQP's 0.01.

These are provisional diagnostic gates, not a frozen baseline. Terminal velocity
and rates are reported separately because the original reference ends with
nonzero velocity. There are no disturbance-rollout or continuous-time robust
safety claims. Componentwise tube-bound checks do not certify nonlinear obstacle
avoidance under every disturbance.

## Full-solve timing methodology

The milestone 2 mode reuses the source-isolating launcher, scenario builder,
input hashing, validation and diagnostic implementation. A separate child
process performs one untimed diagnostic using the original parameters and records
runtime proof of SLS/controller/response/ADMM execution. Its expected SQP failure
is reported, not removed from the Milestone 1 validation result. No repeated
diagnostic search or convergence tuning occurs.

The timing process has none of the extra Milestone 1 observer callbacks. It
reconstructs the exact original `_solve` arguments, compares source/configuration
and input hashes with the diagnostic, then places the complete argument tuple on
the GPU. This includes the timestep scalar, preserving its weak float32 JAX
abstract value. The same tuple and original controller state are retained for
all calls; no outputs are fed back and `GenericMPC.run()` is never called.

`_solve.lower(*frozen_inputs).compile()` completes before warm-up. At least one
successful warm-up precedes measured trials. Each latency uses:

```text
Pre-timer: input readiness and pending-effect barriers
perf_counter()
already_compiled_solve(*same_frozen_inputs)
jax.block_until_ready(complete_11_output_tree)
jax.effects_barrier()
perf_counter()
```

The solver's original ADMM debug prints remain active. During each call they
format into an in-memory buffer; log files are written only after the timer
stops. Latencies therefore include original diagnostic callback/formatting
overhead and GPU dispatch/synchronization, with no terminal or file I/O in the
timed region. They do not measure a hypothetical logging-free solver or pure GPU
kernel duration. Compilation, preparation, plotting, rollout simulation,
validation, hashes and artifact serialization are excluded.

After every call, checks cover the full output tree's dimensions/dtypes/finiteness,
GPU placement, unchanged frozen inputs/controller state, and the existing
geometric/residual gates. Repeated outputs must agree with the first uninstrumented
warm-up (`rtol=1e-5`, `atol=1e-6`). The complete multiset of ADMM iteration counts
and convergence flags, read from the existing prints, must also match that warm-up.
Cross-executable differences against the instrumented diagnostic are quantified
and reported as information, not treated as repeatability failures: the extra
effects can affect code generation and the float32 numerical path.

Every call must emit the maximum 120 ADMM summaries permitted by this unchanged
configuration: 100 nominal calls plus 10 robust SQP iterations with two SLS
iterations each. Together with nonzero returned responses from zero response
inputs, this confirms that the timed executable reaches the robust path. Actual
SQP convergence flags are attributed to the separate instrumented run because the
unchanged public solver return discards them; the exact last-iteration timed SQP
flag is not claimed to have been observed.

No early-exit rule is disabled to force a fixed operation count. "FIXED-BUDGET"
means the original iteration limits, source configuration and initial state are
fixed; the current workload is observed to exhaust its SQP budget. Inner ADMM
and SLS stopping behavior remains unchanged and is checked against the probe.

## Timing artifacts and viewing results

Each benchmark prints its unique results directory. It contains:

| File | Contents |
|---|---|
| `timings.csv` | Trial number and elapsed milliseconds; measured trials only |
| `summary.json` | Mean/median/minimum/maximum, all raw timings, requested/completed counts, nonconvergence label, diagnostic evidence and per-call validation |
| `manifest.json` | Revisions and source hashes, scenario, actual import origins, GPU identity, environment/versions, methodology, frozen-input signatures, warm-up/trial counts |
| `solve.log` | Original ADMM output from warm-ups and measured trials; written outside timers |
| `diagnostic/` | Separate untimed runtime proof, arrays, event log and report |
| `diagnostic_console.log` | Child diagnostic console output |

Use the exact directory printed by the launcher:

```bash
benchmark_run=benchmarks/results/<printed-run-directory>
cat "$benchmark_run/timings.csv"
python -m json.tool "$benchmark_run/summary.json"
python -m json.tool "$benchmark_run/manifest.json"
```

Generated outputs remain ignored by Git. Existing results are never overwritten.
Three trials establish a smoke measurement, not a stable statistical comparison.
A converged control benchmark still requires separate review of inner accuracy,
SQP stopping behavior, terminal-reference semantics and robust nonlinear rollout
validation. None of those mathematical changes is part of this milestone.
