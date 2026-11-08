# Milestone 1 actual results

**Source isolation and runtime SLS proof succeeded. Baseline acceptance is blocked
by SQP nonconvergence. No solve timing was collected.**

The working tree was initially clean on the existing
`benchmark/quadrotor-runtime` branch, parent commit
`b2572d95884643f8c1ea941adbe984e2f3ac8b55`. No branch was created. The iLQR
submodule remains clean at `273d78dec439ed270af6f9cb8340af801fefd332`.

## Recorded experiment

The completed diagnostic is
[`results/20261108T210436Z-original-14bfce09/report.json`](results/20261108T210436Z-original-14bfce09/report.json),
with [runtime events](results/20261108T210436Z-original-14bfce09/events.jsonl),
[solver log](results/20261108T210436Z-original-14bfce09/solve.log), and saved input/output
arrays in that directory. Generated artifacts remain local and Git-ignored.

- Device: `cuda:0`, NVIDIA GeForce RTX 3060 **Laptop GPU**, 6144 MiB.
- GPU UUID: `GPU-edf3690c-e792-546f-1ef3-86bbec73ee8e`.
- Driver: 535.183.01; JAX/JAXlib: 0.8.1; Python: 3.11.18.
- Configuration: `quadrotor_reference_v0`, `original` variant, **no numerical changes**.
- All 15 loaded solver module/package origins resolve under the selected team `src`.
- Input SHA-256 before and after the solve is identical:
  `94e440549c1c7f22336ec5565f3da5c11180974f2ef40df2c7e701dc3380994b`.

## Runtime evidence

| Observed runtime event | Count |
|---|---:|
| SQP search-direction iterations | 110 |
| Nominal constrained ADMM | 100 |
| Robust constrained ADMM | 20 |
| `sls_solve_gpu` | 10 |
| `get_controller` | 20 |
| `calculate_phis` | 20 |
| Final ADMM summaries | 120 |

The original `max_initial_sqp_iterations=100` **does not skip robust SLS in this
run**. It executes the entire nominal phase followed by ten robust SQP iterations.
The original audit identified a possible early-convergence branch, not an observed
fact that SLS was always skipped. Runtime callbacks now resolve that uncertainty
for this experiment. No `max_initial_sqp_iterations=0` experiment was needed to
prove execution, and none was run after the convergence failure.

## Correctness results

| Check | Actual result |
|---|---|
| GPU solve completes | Yes; all 11 outputs reside on GPU |
| All important outputs finite | Yes |
| Shapes / effective dtype | Correct; all float32 |
| Initial-state max residual | 0 |
| Final nonlinear dynamics max residual | 0.00472625345, below configured 0.01 |
| Original state / control bound violations | 0 / 0 |
| Nominal cylinder clearance at nodes | Minimum 0.0553372369 m |
| Nominal clearance along linear position segments | Minimum 0.0550151698 m |
| Final position error | 0.00399796941 m, below original 0.25 m tolerance |
| Terminal speed | 0.730722666 m/s; the vehicle is not at rest |
| Terminal body-rate norm | 0.000426032 rad/s |
| Componentwise state / control tube-bound violations | 0 / 0 |
| Final SLS convergence flag | True |
| Final ADMM convergence flag | True |
| SQP stopping predicate | **False; iteration budget exhausted** |

At SQP iteration 109, the stopping predicate uses pre-update feasibility
`0.0169996917 > 0.01` and search-direction infinity norm
`1.822990894 > 0.00122776022`. None of the 110 observed SQP iterations satisfies
the stopping predicate. The returned trajectory's residual is lower because the
last iteration applies another update; a small final dynamics residual does not
establish optimization convergence.

The first nominal ADMM solve reaches 399 iterations with `converged=False`:
primal residual `0.449989974 > 0.136520520`. The other 119 ADMM calls report
convergence. The final ADMM call takes 59 iterations:

- Primal residual `0.113222152 <= 0.118229911`.
- Dual residual `0.096402884 <= 0.427972019`.
- Final ADMM rho `4.0`.

Terminal position is approximately `(1.00397515, 0.80038291, 0.50018787)`.
Terminal velocity is approximately `(0.54092294, 0.48530886, 0.07637573)` m/s.
The preserved reference requests nonzero velocity even though the supplied goal
state has zero velocity. Goal **position** is reached, but stationary goal
attainment is not established. No new terminal constraint or reference was added.

## Stop condition and next experiment

The convergence gate fails, so this is **not an accepted benchmark baseline**.
No warm steady-state timing, parameter sweep, disturbance rollout, or additional
GPU solve was performed after observing that failure. No timing entry point was
added under the conditional timing task.

The smallest proposed follow-up is one explicitly recorded run that changes only
ADMM `eps_abs` from `0.1` to `0.01`, retaining the original scenario, nominal phase,
iteration budgets and other settings. The hypothesis is that coarse inner-solve
accuracy prevents the much tighter SQP step test from being reached. This is a
testable hypothesis, not a diagnosed solver fix or a guarantee of convergence.
Do not increase iteration limits or change the initialization/reference in the
same experiment. Review terminal-goal semantics separately.

## Harness verification and changes

Eight CPU regression tests pass. They verify source selection despite the shared
upstream editable path, rejection of foreign namespace paths and preloaded
modules, rejection of incorrect gitlink/submodule revisions, exact parity with
the original example's pre-solve construction, the candidate's single configuration
delta, and runtime-versus-tracing behavior of diagnostic callbacks.

An earlier diagnostic attempt is retained in
`results/20261108T210232Z-original-49d248c3/`. It failed in the harness before
producing numerical evidence because GPU-only backend selection prevented JAX
host callbacks from obtaining a CPU device. A CPU unit test also found that JAX
callback operands require conversion before JSON serialization. Both harness
issues were fixed before the completed experiment. The numerical configuration
was not tuned between these attempts.

New files are confined to `benchmarks/`: the source-isolating launcher and helper,
scenario JSON and model adapter, runtime observer, validation and experiment
modules, regression tests, documentation, and a local results ignore rule.
Solver files, submodule files, original examples, existing plots, installed
packages, drivers, activation script and the upstream checkout were not modified.
No clones, CUDA kernels, dependencies, or final repeated-trial infrastructure were
created. Exact reproduction commands are in [README.md](README.md).
