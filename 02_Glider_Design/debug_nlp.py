from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import aerosandbox as asb


@dataclass(frozen=True, slots=True)
class SmokeTestResult:
    """Outcome of a quick IPOPT evaluation at the current initial guess."""
    ok: bool
    stage: str
    message: str
    hint: str


def variable_count(opti: asb.Opti) -> int:
    """Number of scalar decision variables currently in Opti (best-effort)."""
    try:
        return int(opti.x.shape[0])
    except Exception:
        return -1


def constraint_count(opti: asb.Opti) -> int:
    """Number of scalar constraints currently in Opti (best-effort)."""
    try:
        return int(opti.g.shape[0])
    except Exception:
        return -1


def _extract_useful_hint(error_text: str) -> str:
    """Extract the most actionable IPOPT/CasADi hint from an exception string."""
    lines = [ln.strip() for ln in error_text.splitlines() if ln.strip()]
    needles = (
        "NaN detected",
        "Inf detected",
        "Invalid number",
        "Error in Function",
        "Evaluation error",
        "return_status",
    )
    for needle in needles:
        for ln in lines:
            if needle in ln:
                return ln
    return lines[-1] if lines else error_text


def _set_ipopt_solver(
    opti: asb.Opti,
    *,
    max_iter: int,
    max_cpu_time: float,
    silent: bool,
    extra_ipopt_opts: Optional[Mapping[str, Any]],
) -> None:
    """Apply IPOPT options to an Opti instance."""
    plugin_opts: dict[str, Any] = {
        "print_time": False,
        "verbose": False,
    }

    ipopt_opts: dict[str, Any] = {
        "max_iter": int(max_iter),
        "max_cpu_time": float(max_cpu_time),
        "check_derivatives_for_naninf": "yes",
    }

    if silent:
        # IPOPT-level silence (still not enough in some builds without fd redirection).
        ipopt_opts.update(
            {
                "print_level": 0,
                "sb": "yes",
                "print_timing_statistics": "no",
                "file_print_level": 0,
                "output_file": "",
            }
        )

    if extra_ipopt_opts:
        ipopt_opts.update(dict(extra_ipopt_opts))

    opti.solver("ipopt", plugin_opts, ipopt_opts)


def _silence_everything(enabled: bool):
    """Context manager to silence stdout/stderr as aggressively as possible."""
    if not enabled:
        class _NoOp:
            def __enter__(self):  # type: ignore[no-untyped-def]
                return None

            def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
                return False

        return _NoOp()

    import io
    import os
    import sys
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        buf = io.StringIO()

        # Save Python-level streams
        old_py_out = sys.stdout
        old_py_err = sys.stderr

        # Save OS-level fds
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_out_fd = os.dup(1)
        old_err_fd = os.dup(2)

        # Best-effort CasADi log suppression
        old_casadi_level: Optional[int] = None
        casadi = None
        try:
            import casadi as casadi  # type: ignore[no-redef]
        except Exception:
            casadi = None

        try:
            # Redirect OS-level
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)

            # Redirect Python-level
            sys.stdout = buf
            sys.stderr = buf

            # Reduce CasADi logging if available
            if casadi is not None:
                try:
                    old_casadi_level = int(casadi.get_log_level())
                    casadi.set_log_level(0)
                except Exception:
                    old_casadi_level = None

            yield
        finally:
            # Restore CasADi logging
            if casadi is not None and old_casadi_level is not None:
                try:
                    casadi.set_log_level(old_casadi_level)
                except Exception:
                    pass

            # Restore Python-level
            sys.stdout = old_py_out
            sys.stderr = old_py_err

            # Restore OS-level
            os.dup2(old_out_fd, 1)
            os.dup2(old_err_fd, 2)

            os.close(old_out_fd)
            os.close(old_err_fd)
            os.close(devnull_fd)

    return _ctx()


def ipopt_smoke_test(
    opti: asb.Opti,
    *,
    stage: str = "unknown",
    max_iter: int = 0,
    max_cpu_time: float = 1e-3,
    raise_on_fail: bool = False,
    silent: bool = True,
    hard_silence: bool = True,
    extra_ipopt_opts: Optional[Mapping[str, Any]] = None,
) -> SmokeTestResult:
    """ Run a quick IPOPT evaluation to catch NaN/Inf at the current initial guess."""
    try:
        with _silence_everything(hard_silence):
            # Important: set solver INSIDE the silence context to prevent IPOPT
            # plugin-load banners from leaking.
            _set_ipopt_solver(
                opti,
                max_iter=max_iter,
                max_cpu_time=max_cpu_time,
                silent=silent,
                extra_ipopt_opts=extra_ipopt_opts,
            )
            opti.solve()

        return SmokeTestResult(ok=True, stage=stage, message="OK", hint="OK")

    except RuntimeError as exc:
        msg = str(exc)
        hint = _extract_useful_hint(msg)
        result = SmokeTestResult(ok=False, stage=stage, message=msg, hint=hint)
        if raise_on_fail:
            raise
        return result