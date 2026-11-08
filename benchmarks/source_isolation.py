"""Process-local source selection. This module must not import JAX or GPU-SLS."""

from pathlib import Path
import importlib
import subprocess
import sys


PINNED_ILQR = "273d78dec439ed270af6f9cb8340af801fefd332"
SUBMODULE = Path("src/gpu_sls/external/primal_dual_ilqr")


def git(checkout, *args):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def is_solver_module(name):
    return any(name == p or name.startswith(p + ".")
               for p in ("gpu_sls", "primal_dual_ilqr"))


def verify_origins(checkout):
    """Reject both foreign files and foreign namespace-package search paths."""
    src = Path(checkout).resolve() / "src"
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if not is_solver_module(name) or module is None:
            continue
        filename = getattr(module, "__file__", None)
        paths = [str(Path(p).resolve()) for p in getattr(module, "__path__", [])]
        if filename:
            filename = str(Path(filename).resolve())
        for location in ([filename] if filename else []) + paths:
            if not Path(location).is_relative_to(src):
                raise RuntimeError(f"Foreign solver import: {name}: {location}")
        if not filename and not paths:
            raise RuntimeError(f"Solver module has no verifiable origin: {name}")
        origins[name] = {"file": filename, "search_paths": paths}
    return origins


def select_source(checkout):
    checkout = Path(checkout).resolve(strict=True)
    src = checkout / "src"
    if not (src / "gpu_sls/generic_mpc.py").is_file():
        raise RuntimeError(f"Not a GPU-SLS source checkout: {checkout}")
    loaded = [name for name in sys.modules if is_solver_module(name)]
    if loaded:
        raise RuntimeError(f"Source must be selected before solver imports: {loaded}")

    submodule = checkout / SUBMODULE
    # Also check the superproject gitlink: a different local submodule HEAD must
    # not masquerade as the expected dependency of the selected parent commit.
    gitlink = git(checkout, "ls-tree", "HEAD", str(SUBMODULE)).split()
    if len(gitlink) < 3 or gitlink[0] != "160000" or gitlink[2] != PINNED_ILQR:
        raise RuntimeError(f"Unexpected Primal-Dual iLQR gitlink: {gitlink}")
    if not (submodule / ".git").exists():
        raise RuntimeError("Primal-Dual iLQR is not initialized in selected checkout")
    revision = git(submodule, "rev-parse", "HEAD")
    if revision != PINNED_ILQR or git(submodule, "status", "--porcelain"):
        raise RuntimeError("Primal-Dual iLQR must be clean at " + PINNED_ILQR)

    kept, removed = [], []
    for entry in sys.path:
        path = Path(entry or ".").resolve()
        contains_solver = any((path / p).exists() for p in
                              ("gpu_sls", "gpu_sls.py", "primal_dual_ilqr"))
        if path == src or contains_solver:
            removed.append(str(path))
        else:
            kept.append(entry)
    sys.path[:] = [str(src), *kept]
    importlib.invalidate_caches()

    for name in (
        "gpu_sls.generic_mpc", "gpu_sls.gpu_sqp", "gpu_sls.gpu_sls",
        "gpu_sls.gpu_admm", "gpu_sls.utils.constraint_utils",
        "gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr",
        "gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers",
    ):
        importlib.import_module(name)
    return {
        "checkout": str(checkout), "commit": git(checkout, "rev-parse", "HEAD"),
        "branch": git(checkout, "branch", "--show-current"),
        "git_status": git(checkout, "status", "--porcelain"),
        "submodule_commit": revision, "removed_import_paths": removed,
        "module_origins": verify_origins(checkout),
    }
