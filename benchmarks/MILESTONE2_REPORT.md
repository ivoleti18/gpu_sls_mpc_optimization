# Milestone 2: shared full-solve runtime reference

**GPU timing completed. This is a NONCONVERGED / FIXED-BUDGET workload, not a
validated converged MPC solution. No convergence tuning was performed.**

## Actual RTX 3060 smoke results

Completed run:
[`results/20261108T213021Z-benchmark-original-a5a0fe6a/`](results/20261108T213021Z-benchmark-original-a5a0fe6a/).
The artifacts are local and Git-ignored:

- [timings.csv](results/20261108T213021Z-benchmark-original-a5a0fe6a/timings.csv)
- [summary.json](results/20261108T213021Z-benchmark-original-a5a0fe6a/summary.json)
- [manifest.json](results/20261108T213021Z-benchmark-original-a5a0fe6a/manifest.json)
- [untimed diagnostic](results/20261108T213021Z-benchmark-original-a5a0fe6a/diagnostic/report.json)
- [warm-up/trial ADMM log](results/20261108T213021Z-benchmark-original-a5a0fe6a/solve.log)

| Completed call | Host-observed latency (ms) |
|---|---:|
| Warm-up, excluded from statistics | 1274.979577 |
| Trial 1 | 1181.356333 |
| Trial 2 | 1187.953896 |
| Trial 3 | 1181.365701 |
| Mean of timed trials | **1183.558643** |
| Median of timed trials | **1181.365701** |
| Minimum / maximum | 1181.356333 / 1187.953896 |

These measure complete compiled optimization dispatch, completed GPU outputs,
and original ADMM diagnostic effects. They are not pure device-kernel durations.
Three trials are a smoke result, not a statistically established performance claim.

## Source, environment and unchanged workload

- Existing branch: `benchmark/quadrotor-runtime`; no new branch was created.
- Parent commit: `b2572d95884643f8c1ea941adbe984e2f3ac8b55`.
- Clean iLQR submodule: `273d78dec439ed270af6f9cb8340af801fefd332`.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB;
  UUID `GPU-edf3690c-e792-546f-1ef3-86bbec73ee8e`; driver 535.183.01.
- Python 3.11.18, JAX/JAXlib/CUDA plugin 0.8.1.
- Original `quadrotor_reference_v0` scenario: `N=110`, `dt=0.03`, 12 states,
  four controls; original dynamics, reference, obstacle, bounds, cost and guesses.
- ADMM `eps_abs=0.1`; SLS `max_initial_sqp_iterations=100`. All original solver
  limits, tolerances, warm-start settings and mathematical operations are retained.
- All 15 loaded GPU-SLS module/package origins resolve under the selected team `src`.
- Original controller-input hash matches Milestone 1:
  `94e440549c1c7f22336ec5565f3da5c11180974f2ef40df2c7e701dc3380994b`.
- Frozen all-device argument hash:
  `a3af6af5dc2c31f57af3f31d9620de0d6dedefdf9605b020084d01a32997e996`.

The hashes differ because the timing tuple materializes the Python timestep as a
GPU scalar with the same weak float32 JAX abstract value. The launcher verifies
abstract-value equality, and all arguments reside on the GPU before timing.

The benchmark files are still working-tree files, not a newly committed release.
The manifest records dirty status and SHA-256 hashes of all solver/model/harness
source files, rather than implying that the parent commit identifies this harness.

## Timing boundary and state handling

The clean existing boundary is `GenericMPC._solve`, the jitted partial of `sqp`.
Its 16 arguments include the reference, parameter, weights, initial state,
trajectory guesses, duals, rho, obstacles, backoffs, beta, mu and response arrays.
The harness reuses that tuple for every call and never invokes the state-mutating
`GenericMPC.run()` or feeds previous outputs into the next trial.

Compilation through `.lower(...).compile()` and one successful warm-up precede
the three timed calls. Each sample uses `time.perf_counter()` around dispatch,
`jax.block_until_ready()` on the **entire 11-output tree**, and
`jax.effects_barrier()`. Pre-timer barriers drain prior input/effect work.

Existing ADMM print callbacks remain in the executable. They format into a
`StringIO` buffer during each call; the buffer is written to `solve.log` only
after the timer stops. Preparation, compilation, validation, hashing, plotting,
rollouts and file output are excluded. The additional Milestone 1 observer
callbacks execute only in a separate untimed process.

## Verification and nonconvergence

The separate runtime diagnostic recorded:

| Runtime event | Count |
|---|---:|
| SQP iterations | 110 |
| Nominal ADMM calls | 100 |
| Robust ADMM calls | 20 |
| SLS calls | 10 |
| Controller / response calculations | 20 / 20 |

Every warm-up/trial passed dimensions, float32 precision, finiteness, GPU
placement, unchanged frozen arguments/controller state, original bounds,
geometric/residual checks, nonzero responses and repeated-output agreement.
Every call emitted the same 120 ADMM summaries and the same per-call iteration /
convergence multiset. Since 120 is the maximum allowed by the original outer/SLS
budgets, this requires all 110 SQP iterations, including the robust phase.
Returning nonzero response arrays from zero response inputs corroborates that
the timed SLS path executed. All warm-up/trial outputs also matched this run's
separate diagnostic exactly in the recorded comparison.

The timed results retain the Milestone 1 behavior:

- Nonlinear dynamics max residual: 0.00472625345.
- State/control and componentwise tube-bound violations: zero.
- Minimum nominal linear-segment cylinder clearance: 0.0550151698 m.
- Terminal position error: 0.00399796941 m; speed: 0.730722666 m/s.
- One initial ADMM call reaches 399 iterations without convergence; the other
  119 report convergence. The final call reports convergence after 59 iterations.
- The instrumented SQP stopping predicate remains false after 110 iterations.

The unchanged SQP public return discards its convergence flag. `solver_converged:
false` in the summary is explicitly attributed to the separate diagnostic;
the exact final timed SQP flag is **not directly observed**. No early-exit rule is
disabled. Fixed-budget means fixed original limits and initial conditions, with
budget exhaustion observed for this workload. `validated_converged_mpc` is false.

Sixteen regression tests passed, including the original eight. New tests cover
timer/barrier ordering, complete output trees, real JAX callback completion before
the stop clock, scalar placement preserving JAX abstract values, expected
nonconvergence policy, missing SLS evidence, statistics, ADMM log parsing,
same-executable work comparisons and invalid CLI arguments.

## Recorded harness correction

The earlier attempt in
`results/20261108T212548Z-benchmark-original-086e0584/` stopped after one completed
1262.876 ms warm-up and before measured trials. A check incorrectly required the
separate instrumented executable's inner ADMM history and all outputs to match
the uninstrumented executable. In that attempt, the instrumented nominal phase
mostly used 46 ADMM iterations, while the uninstrumented nominal phase mostly
used one. Both retained the same parameters, executed 120 ADMM calls and passed
the physical checks. The exact cause of the cross-compilation numerical-path
difference was not isolated; extra effects/code generation and float32 sensitivity
are relevant possibilities.

The correction makes the first **uninstrumented** warm-up the work/output
reference for repetitions. Diagnostic-versus-timed differences remain quantified
in JSON and are not hidden or treated as convergence fixes. The completed run
matches even across executables, but cross-compilation identity is not assumed.
No solver parameter was changed and no result directory was deleted or overwritten.

## Exact files changed in this milestone

| File | Change |
|---|---|
| `benchmarks/launcher.py` | Extend existing launcher with benchmark mode and positive warm-up/trial/slow-call limits; reject non-original benchmark variants |
| `benchmarks/experiment.py` | Share GPU preflight and use one canonical lock even for nested result directories; retain diagnostic behavior |
| `benchmarks/full_solve_benchmark.py` | Add compilation, frozen GPU inputs, timing, checks and CSV/JSON artifacts |
| `benchmarks/tests/test_timing.py` | Add eight timing/verification regression tests |
| `benchmarks/README.md` | Document activation, smoke/larger invocations, source selection, result viewing and limitations |
| `benchmarks/MILESTONE2_REPORT.md` | Record this implementation and actual results |

The scenario JSON, Milestone 1 report/tests, source-isolation implementation,
solver files, submodule files, original examples, tracked figures, upstream
checkout, installed dependencies and shared activation/environment were not
modified. No developer clones, CUDA kernels or branch creation were performed.
Generated run artifacts remain under the existing ignored results directory.

## Remaining limits

There is no blocker to using this harness as a **nonconverged fixed-budget timing
reference**. Larger statistical runs, comparisons across processes/compilations,
and independent development checkouts remain unverified. The shared lock
coordinates this harness but cannot prevent unrelated GPU work starting later.
Retained diagnostic overhead is part of the reference and must remain consistent
in comparisons.

A converged control benchmark requires separately reviewed work on SQP/inner
solve behavior, terminal-reference/goal semantics and nonlinear disturbed-rollout
validation. No such tuning or follow-on milestone was started. Exact commands
are in [README.md](README.md).
