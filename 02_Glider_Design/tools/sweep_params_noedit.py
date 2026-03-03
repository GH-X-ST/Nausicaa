import contextlib
import importlib.util
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "nausicaa.py"
OUTDIR = ROOT / "outputs"
TMP = OUTDIR / "_tmp_nausicaa_run.py"

BEST_LOG = OUTDIR / "best_run_log.txt"
BEST_PARAMS = OUTDIR / "best_params.json"
ALL_CSV = OUTDIR / "all_runs.csv"

# -------------------------
# Speed strategy (edit)
# -------------------------
# "adaptive" = coarse pass, then local refinement around best coarse points.
# "grid"     = evaluate all candidates.
SEARCH_STRATEGY = "adaptive"

COARSE_POINTS_V_NOM = 3
COARSE_POINTS_V_TURN = 3
COARSE_POINTS_K_LEVEL = 3
LOCAL_TOP_K = 6
LOCAL_NEIGHBOR_RADIUS = 1
MAX_LOCAL_CANDIDATES = 80

# Parallel evaluation of candidate batches.
# Keep STOP_AT_FIRST_FEASIBLE=False if you want parallel to engage.
N_WORKERS = max(1, min(4, os.cpu_count() or 1))
PARALLEL_MIN_BATCH = 4

# "inprocess" is much faster because it avoids launching a new Python process
# and uses legacy_single_run_main(..., export_outputs=False) directly.
EVAL_BACKEND = "inprocess"  # "inprocess" | "subprocess"
FAST_SWEEP_MODE = True

FAST_IPOPT_OPTIONS = {
    "max_iter": 120,
    "tol": 5e-3,
    "acceptable_tol": 2e-2,
    "acceptable_iter": 4,
    "print_level": 0,
    "sb": "yes",
    "hessian_approximation": "limited-memory",
}

DISABLE_INITIAL_DESIGN_CHECK = True
DISABLE_PLOTS = True
EXPORT_OUTPUTS_IN_SWEEP = False
INPROCESS_WARM_START = True

# -------------------------
# Sweep ranges (edit freely)
# -------------------------
V_NOM_LIST = [4.6, 4.8, 5.0, 5.2, 5.4, 5.6]
V_TURN_LIST = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
K_LEVEL_LIST = [0.80, 0.85, 0.90, 0.95, 1.00]

# -------------------------
# Sanity thresholds (edit)
# -------------------------
MAX_SPAN = 0.70
MAX_CHORD = 0.20

# "tail not ridiculously large" proxies (tune)
MAX_TAIL_ARM = 0.70
MAX_HTAIL_AREA = 0.050
MAX_VTAIL_AREA = 0.020

# -------------------------
# Analytical pruning (fast skip of impossible turn settings)
# -------------------------
USE_TURN_PRUNING = True

# Conservative estimates used only for pruning (not for final validation)
MASS_KG_EST = 0.10
RHO_KG_M3 = 1.225

TURN_BANK_DEG_EST = 45.0
TURN_CL_CAP_EST = 1.25
# In current nausicaa.py, turn kinematics use R = V^2/(g*tan(phi)); K only scales lift requirement.

ARENA_WIDTH_M_EST = 4.8
WALL_CLEARANCE_M_EST = 0.30

# Assumed geometry used in footprint/CL pruning:
# - MAX_SPAN is conservative (may skip feasible smaller-span designs)
# - If you want less aggressive pruning, set span/chord lower
PRUNE_SPAN_ASSUMED_M = MAX_SPAN
PRUNE_CHORD_ASSUMED_M = MAX_CHORD

# Feasibility-first sweep behavior
FEASIBILITY_FIRST_ORDER = True
STOP_AT_FIRST_FEASIBLE = False

# -------------------------
# Output parsing patterns
# -------------------------
FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

PATTERNS = {
    "wing_span_m": re.compile(rf"\bwing_span_m\b\s*[:=]\s*({FLOAT})"),
    "wing_chord_m": re.compile(rf"\bwing_chord_m\b\s*[:=]\s*({FLOAT})"),
    "tail_arm_m": re.compile(rf"\btail_arm_m\b\s*[:=]\s*({FLOAT})"),
    "htail_area_m2": re.compile(rf"\bhtail_area_m2\b\s*[:=]\s*({FLOAT})"),
    "vtail_area_m2": re.compile(rf"\bvtail_area_m2\b\s*[:=]\s*({FLOAT})"),
    "objective": re.compile(rf"\bobjective\b\s*[:=]\s*({FLOAT})"),
    "sink_rate_mps": re.compile(rf"\bsink(?:_rate)?(?:_mps)?\b\s*[:=]\s*({FLOAT})"),
}

FAIL_MARKERS = [
    "No feasible design was found",
    "Solver failed",
    "NonIpopt_Exception_Thrown",
    "return_status",
    "Infeasible",
]

_NAUSICAA_MOD: Any = None
_INPROCESS_AVAILABLE = True
_WARM_START_INIT: Dict[str, float] = {}


@dataclass
class RunResult:
    v_nom: float
    v_turn: float
    k_level: float
    success: bool
    metrics: Dict[str, float]
    score: Optional[float]  # lower is better
    elapsed_s: float
    stdout: str
    stderr: str
    returncode: int


def replace_constant(text: str, name: str, value: float) -> Tuple[str, bool]:
    """
    Replace a top-level constant assignment like:
        NAME = 3.5
    using regex anchored to line start.
    """
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*).*$", re.MULTILINE)
    new_text, n = pattern.subn(lambda m: f"{m.group(1)}{value}", text)
    return new_text, (n > 0)


def patch_source(v_nom: float, v_turn: float, k_level: float) -> str:
    src = SRC.read_text(encoding="utf-8")

    src, ok1 = replace_constant(src, "V_NOM_MPS", float(v_nom))
    src, ok2 = replace_constant(src, "V_TURN_MPS", float(v_turn))
    src, ok3 = replace_constant(src, "K_LEVEL_TURN", float(k_level))

    missing = [
        n
        for n, ok in [("V_NOM_MPS", ok1), ("V_TURN_MPS", ok2), ("K_LEVEL_TURN", ok3)]
        if not ok
    ]
    if missing:
        raise RuntimeError(
            f"Could not find constant assignment(s) in nausicaa.py: {missing}. "
            f"Make sure they exist as top-level lines like 'V_NOM_MPS = 3.5'."
        )
    return src


def parse_metrics(stdout: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for key, pat in PATTERNS.items():
        hit = pat.search(stdout)
        if hit:
            try:
                m[key] = float(hit.group(1))
            except Exception:
                pass
    return m


def decide_success(returncode: int, stdout: str, stderr: str) -> bool:
    blob = (stdout + "\n" + stderr).lower()
    for s in FAIL_MARKERS:
        if s.lower() in blob:
            return False
    if returncode != 0:
        return False
    return True


def passes_sanity(metrics: Dict[str, float]) -> bool:
    # If a metric isn't available, do not hard-fail; just skip that check.
    if "wing_span_m" in metrics and metrics["wing_span_m"] > MAX_SPAN:
        return False
    if "wing_chord_m" in metrics and metrics["wing_chord_m"] > MAX_CHORD:
        return False
    if "tail_arm_m" in metrics and metrics["tail_arm_m"] > MAX_TAIL_ARM:
        return False
    if "htail_area_m2" in metrics and metrics["htail_area_m2"] > MAX_HTAIL_AREA:
        return False
    if "vtail_area_m2" in metrics and metrics["vtail_area_m2"] > MAX_VTAIL_AREA:
        return False
    return True


def compute_score(metrics: Dict[str, float]) -> Optional[float]:
    # Prefer objective if printed; else sink rate; else None.
    if "objective" in metrics:
        return metrics["objective"]
    if "sink_rate_mps" in metrics:
        return metrics["sink_rate_mps"]
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        val = float(x)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def _load_nausicaa_module() -> Any:
    global _NAUSICAA_MOD
    if _NAUSICAA_MOD is not None:
        return _NAUSICAA_MOD

    spec = importlib.util.spec_from_file_location("nausicaa_sweep_runtime", str(SRC))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {SRC}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _NAUSICAA_MOD = mod
    return mod


def _candidate_metrics_from_object(mod: Any, candidate: Any) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in [
        "objective",
        "sink_rate_mps",
        "wing_span_m",
        "wing_chord_m",
        "tail_arm_m",
        "htail_span_m",
        "vtail_height_m",
        "roll_tau_s",
        "mass_total_kg",
        "l_over_d",
        "static_margin",
    ]:
        if hasattr(candidate, key):
            v = _safe_float(getattr(candidate, key))
            if v is not None:
                metrics[key] = v

    htail_span = metrics.get("htail_span_m")
    if htail_span is not None and hasattr(mod, "HT_AR"):
        ht_ar = max(_safe_float(getattr(mod, "HT_AR", 0.0)) or 0.0, 1e-8)
        metrics["htail_area_m2"] = htail_span * htail_span / ht_ar

    vtail_h = metrics.get("vtail_height_m")
    if vtail_h is not None and hasattr(mod, "VT_AR"):
        vt_ar = max(_safe_float(getattr(mod, "VT_AR", 0.0)) or 0.0, 1e-8)
        metrics["vtail_area_m2"] = vtail_h * vtail_h / vt_ar

    return metrics


def _warm_start_from_candidate(candidate: Any) -> Dict[str, float]:
    warm: Dict[str, float] = {}

    # Keep warm-start conservative: only variables with known one-to-one fields.
    mappings = {
        "alpha_nom_deg": "alpha_deg",
        "delta_a_nom_deg": "delta_a_deg",
        "delta_e_nom_deg": "delta_e_deg",
        "delta_r_nom_deg": "delta_r_deg",
        "wing_span_m": "wing_span_m",
        "wing_chord_m": "wing_chord_m",
        "tail_arm_m": "tail_arm_m",
        "htail_span_m": "htail_span_m",
        "vtail_height_m": "vtail_height_m",
    }
    for init_key, cand_key in mappings.items():
        if hasattr(candidate, cand_key):
            v = _safe_float(getattr(candidate, cand_key))
            if v is not None:
                warm[init_key] = v
    return warm


def estimate_turn_pruning(v_turn: float, k_level: float) -> Dict[str, float]:
    v = max(float(v_turn), 1e-8)
    k = max(float(k_level), 1e-8)

    phi_rad = math.radians(float(TURN_BANK_DEG_EST))
    tan_phi = math.tan(phi_rad)
    cos_phi = math.cos(phi_rad)
    if tan_phi <= 1e-8 or cos_phi <= 1e-8:
        return {
            "n_load_factor_est": float("inf"),
            "q_dyn_turn_est": float("inf"),
            "area_est_m2": 0.0,
            "turn_cl_required_est": float("inf"),
            "turn_cl_margin_est": float("-inf"),
            "turn_radius_m_est": float("inf"),
            "footprint_margin_m_est": float("-inf"),
        }

    n_load_factor = 1.0 / cos_phi
    area_est_m2 = max(float(PRUNE_SPAN_ASSUMED_M) * float(PRUNE_CHORD_ASSUMED_M), 1e-8)
    q_dyn_turn = 0.5 * float(RHO_KG_M3) * v * v

    turn_lift_required_n = k * n_load_factor * float(MASS_KG_EST) * 9.81
    turn_cl_required = turn_lift_required_n / max(q_dyn_turn * area_est_m2, 1e-8)
    # Kinematics for commanded bank: K does not change turn radius.
    radius_denom = 9.81 * tan_phi
    turn_radius_m = (v * v) / max(radius_denom, 1e-8)

    half_arena_width = 0.5 * float(ARENA_WIDTH_M_EST)
    footprint_need_m = turn_radius_m + 0.5 * float(PRUNE_SPAN_ASSUMED_M) + float(WALL_CLEARANCE_M_EST)
    footprint_margin_m = half_arena_width - footprint_need_m

    return {
        "n_load_factor_est": n_load_factor,
        "q_dyn_turn_est": q_dyn_turn,
        "area_est_m2": area_est_m2,
        "turn_cl_required_est": turn_cl_required,
        "turn_cl_margin_est": float(TURN_CL_CAP_EST) - turn_cl_required,
        "turn_radius_m_est": turn_radius_m,
        "footprint_margin_m_est": footprint_margin_m,
    }


def check_turn_pair_possible(v_turn: float, k_level: float) -> Tuple[bool, Dict[str, float], str]:
    est = estimate_turn_pruning(v_turn, k_level)
    reasons: List[str] = []

    if est["turn_cl_margin_est"] < 0.0:
        reasons.append(
            f"CL cap fail: need {est['turn_cl_required_est']:.3f} > cap {TURN_CL_CAP_EST:.3f}"
        )
    if est["footprint_margin_m_est"] < 0.0:
        reasons.append(
            f"Footprint fail: margin {est['footprint_margin_m_est']:.3f} m < 0"
        )

    return (len(reasons) == 0), est, "; ".join(reasons)


def feasibility_sort_key(v_nom: float, v_turn: float, k_level: float, est: Dict[str, float]) -> Tuple[float, ...]:
    cl_margin = est.get("turn_cl_margin_est", float("-inf"))
    fp_margin = est.get("footprint_margin_m_est", float("-inf"))

    cl_deficit = max(0.0, -cl_margin)
    fp_deficit = max(0.0, -fp_margin)
    any_deficit = 1 if (cl_deficit > 0.0 or fp_deficit > 0.0) else 0

    # deficits dominate ordering; then prefer larger positive margins
    deficit_score = (cl_deficit / max(float(TURN_CL_CAP_EST), 1e-8)) + (
        fp_deficit / max(0.5 * float(ARENA_WIDTH_M_EST), 1e-8)
    )
    robust_slack = min(
        cl_margin / max(float(TURN_CL_CAP_EST), 1e-8),
        fp_margin / max(0.5 * float(ARENA_WIDTH_M_EST), 1e-8),
    )

    return (
        any_deficit,
        deficit_score,
        -robust_slack,
        abs(float(v_nom) - float(v_turn)),
        -float(k_level),
        float(v_turn),
        float(v_nom),
    )


def build_sweep_candidates() -> Tuple[List[Tuple[float, float, float, Dict[str, float]]], List[str]]:
    pair_eval: Dict[Tuple[float, float], Tuple[bool, Dict[str, float], str]] = {}
    skipped_msgs: List[str] = []

    for v_turn, k_level in itertools.product(V_TURN_LIST, K_LEVEL_LIST):
        pair_eval[(v_turn, k_level)] = check_turn_pair_possible(v_turn, k_level)

    candidates: List[Tuple[float, float, float, Dict[str, float]]] = []
    for v_nom, v_turn, k_level in itertools.product(V_NOM_LIST, V_TURN_LIST, K_LEVEL_LIST):
        possible, est, _ = pair_eval[(v_turn, k_level)]
        if USE_TURN_PRUNING and not possible:
            continue
        candidates.append((v_nom, v_turn, k_level, est))

    if USE_TURN_PRUNING:
        for v_turn, k_level in itertools.product(V_TURN_LIST, K_LEVEL_LIST):
            possible, _, reason = pair_eval[(v_turn, k_level)]
            if not possible:
                skipped_msgs.append(
                    f"skip (V_TURN={v_turn}, K={k_level}) -> {reason}"
                )

    if FEASIBILITY_FIRST_ORDER:
        candidates.sort(key=lambda c: feasibility_sort_key(c[0], c[1], c[2], c[3]))

    return candidates, skipped_msgs


def run_once_inprocess(v_nom: float, v_turn: float, k_level: float) -> RunResult:
    global _WARM_START_INIT

    t0 = time.perf_counter()
    out = io.StringIO()
    err = io.StringIO()
    returncode = 0

    mod = _load_nausicaa_module()
    mod.V_NOM_MPS = float(v_nom)
    mod.V_TURN_MPS = float(v_turn)
    mod.K_LEVEL_TURN = float(k_level)
    if hasattr(mod, "DESIGN_SPEED_MPS"):
        mod.DESIGN_SPEED_MPS = float(v_nom)
    if DISABLE_INITIAL_DESIGN_CHECK and hasattr(mod, "ENABLE_INITIAL_DESIGN_CHECK"):
        mod.ENABLE_INITIAL_DESIGN_CHECK = False
    if hasattr(mod, "ENABLE_OPTIMIZATION_AFTER_INITIAL_CHECK"):
        mod.ENABLE_OPTIMIZATION_AFTER_INITIAL_CHECK = True
    if DISABLE_PLOTS and hasattr(mod, "MAKE_PLOTS"):
        mod.MAKE_PLOTS = False
    if hasattr(mod, "RUN_WORKFLOW"):
        mod.RUN_WORKFLOW = False
    if hasattr(mod, "MANUAL_RUN_NOTE_PRINT"):
        mod.MANUAL_RUN_NOTE_PRINT = False

    init_override: Dict[str, float] = {}
    if hasattr(mod, "default_initial_guess"):
        try:
            init_override = dict(mod.default_initial_guess())
        except Exception:
            init_override = {}
    if INPROCESS_WARM_START and _WARM_START_INIT:
        init_override.update(_WARM_START_INIT)

    ipopt_options = FAST_IPOPT_OPTIONS if FAST_SWEEP_MODE else None
    candidate = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            candidate = mod.legacy_single_run_main(
                init_override=init_override,
                ipopt_options=ipopt_options,
                export_outputs=bool(EXPORT_OUTPUTS_IN_SWEEP),
            )
    except Exception as exc:
        returncode = 1
        err.write(f"\n[SWEEP ERROR] {exc}\n")

    stdout = out.getvalue()
    stderr = err.getvalue()
    metrics = parse_metrics(stdout)
    success = (returncode == 0) and (candidate is not None)

    if candidate is not None:
        metrics.update(_candidate_metrics_from_object(mod, candidate))

    if success and not passes_sanity(metrics):
        success = False

    score = compute_score(metrics) if success else None
    elapsed_s = time.perf_counter() - t0

    if success and candidate is not None and INPROCESS_WARM_START:
        _WARM_START_INIT = _warm_start_from_candidate(candidate)

    return RunResult(
        v_nom=v_nom,
        v_turn=v_turn,
        k_level=k_level,
        success=success,
        metrics=metrics,
        score=score,
        elapsed_s=elapsed_s,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def run_once_subprocess(v_nom: float, v_turn: float, k_level: float) -> RunResult:
    t0 = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    patched = patch_source(v_nom, v_turn, k_level)
    tmp_path = OUTDIR / f"_tmp_nausicaa_run_{os.getpid()}_{time.time_ns()}.py"
    tmp_path.write_text(patched, encoding="utf-8")

    try:
        cmd = ["python", str(tmp_path)]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    finally:
        with contextlib.suppress(Exception):
            tmp_path.unlink()

    stdout = p.stdout or ""
    stderr = p.stderr or ""

    metrics = parse_metrics(stdout)
    success = decide_success(p.returncode, stdout, stderr)

    # sanity filter only applied if success; otherwise irrelevant
    if success and not passes_sanity(metrics):
        success = False

    score = compute_score(metrics) if success else None
    elapsed_s = time.perf_counter() - t0

    return RunResult(
        v_nom=v_nom,
        v_turn=v_turn,
        k_level=k_level,
        success=success,
        metrics=metrics,
        score=score,
        elapsed_s=elapsed_s,
        stdout=stdout,
        stderr=stderr,
        returncode=p.returncode,
    )


def _run_once_with_backend(v_nom: float, v_turn: float, k_level: float, backend: str) -> RunResult:
    if backend == "inprocess":
        try:
            return run_once_inprocess(v_nom, v_turn, k_level)
        except Exception as exc:
            fallback = run_once_subprocess(v_nom, v_turn, k_level)
            fallback.stderr = (fallback.stderr or "") + f"\n[WARN] inprocess failed: {exc}"
            return fallback
    return run_once_subprocess(v_nom, v_turn, k_level)


def run_once(v_nom: float, v_turn: float, k_level: float) -> RunResult:
    global _INPROCESS_AVAILABLE
    if EVAL_BACKEND == "inprocess" and _INPROCESS_AVAILABLE:
        try:
            return run_once_inprocess(v_nom, v_turn, k_level)
        except Exception as exc:
            _INPROCESS_AVAILABLE = False
            print(f"[WARN] inprocess backend failed ({exc}); falling back to subprocess.")
            return run_once_subprocess(v_nom, v_turn, k_level)
    return run_once_subprocess(v_nom, v_turn, k_level)


def _make_worker_exception_result(v_nom: float, v_turn: float, k_level: float, error_text: str) -> RunResult:
    return RunResult(
        v_nom=v_nom,
        v_turn=v_turn,
        k_level=k_level,
        success=False,
        metrics={},
        score=None,
        elapsed_s=0.0,
        stdout="",
        stderr=f"[WORKER ERROR] {error_text}",
        returncode=1,
    )


def _run_batch(
    batch: List[Tuple[float, float, float, Dict[str, float]]],
    rows: List[RunResult],
    evaluated: set[Tuple[float, float, float]],
    best: Optional[RunResult],
    counter: int,
    stage_label: str,
) -> Tuple[Optional[RunResult], int, bool, List[RunResult]]:
    pending = [c for c in batch if (c[0], c[1], c[2]) not in evaluated]
    stage_results: List[RunResult] = []
    if not pending:
        return best, counter, False, stage_results

    stage_total = len(pending)
    use_parallel = (
        N_WORKERS > 1
        and stage_total >= PARALLEL_MIN_BATCH
        and not STOP_AT_FIRST_FEASIBLE
    )

    if use_parallel:
        print(
            f"{stage_label}: parallel batch with workers={N_WORKERS}, backend={EVAL_BACKEND}, n={stage_total}"
        )
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            future_map = {
                pool.submit(_run_once_with_backend, v_nom, v_turn, k, EVAL_BACKEND): (v_nom, v_turn, k)
                for (v_nom, v_turn, k, _) in pending
            }

            stage_done = 0
            for fut in as_completed(future_map):
                v_nom, v_turn, k = future_map[fut]
                stage_done += 1
                counter += 1
                key = (v_nom, v_turn, k)

                try:
                    r = fut.result()
                except Exception as exc:
                    r = _make_worker_exception_result(v_nom, v_turn, k, str(exc))

                rows.append(r)
                stage_results.append(r)
                evaluated.add(key)

                if _better(best, r):
                    best = r
                    print(
                        f"[{stage_label} {stage_done}/{stage_total}] NEW BEST: "
                        f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}, t={r.elapsed_s:.2f}s"
                    )
                elif r.success:
                    print(
                        f"[{stage_label} {stage_done}/{stage_total}] ok: "
                        f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}, t={r.elapsed_s:.2f}s"
                    )
                else:
                    print(
                        f"[{stage_label} {stage_done}/{stage_total}] fail: "
                        f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, t={r.elapsed_s:.2f}s"
                    )

        return best, counter, False, stage_results

    print(f"{stage_label}: serial batch (n={stage_total})")
    stage_done = 0
    for v_nom, v_turn, k, _ in pending:
        stage_done += 1
        counter += 1
        key = (v_nom, v_turn, k)

        r = run_once(v_nom, v_turn, k)
        rows.append(r)
        stage_results.append(r)
        evaluated.add(key)

        if _better(best, r):
            best = r
            print(
                f"[{stage_label} {stage_done}/{stage_total}] NEW BEST: "
                f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}, t={r.elapsed_s:.2f}s"
            )
        elif r.success:
            print(
                f"[{stage_label} {stage_done}/{stage_total}] ok: "
                f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}, t={r.elapsed_s:.2f}s"
            )
        else:
            print(
                f"[{stage_label} {stage_done}/{stage_total}] fail: "
                f"V_NOM={v_nom}, V_TURN={v_turn}, K={k}, t={r.elapsed_s:.2f}s"
            )

        if STOP_AT_FIRST_FEASIBLE and r.success:
            print(f"[{stage_label} {stage_done}/{stage_total}] stop: first feasible found")
            return best, counter, True, stage_results

    return best, counter, False, stage_results


def _coarse_values(values: List[float], n_points: int) -> List[float]:
    uniq = list(dict.fromkeys(values))
    if not uniq:
        return []
    if n_points <= 1:
        return [uniq[len(uniq) // 2]]
    if n_points >= len(uniq):
        return uniq

    idx = [round(i * (len(uniq) - 1) / (n_points - 1)) for i in range(n_points)]
    return [uniq[i] for i in sorted(set(idx))]


def _pick_seed_triples(
    stage1_results: List[RunResult],
    stage1_candidates: List[Tuple[float, float, float, Dict[str, float]]],
) -> List[Tuple[float, float, float]]:
    feasible = [r for r in stage1_results if r.success]
    if feasible:
        feasible.sort(
            key=lambda r: (
                1 if r.score is None else 0,
                float("inf") if r.score is None else float(r.score),
            )
        )
        chosen = feasible[: max(1, min(LOCAL_TOP_K, len(feasible)))]
        return [(r.v_nom, r.v_turn, r.k_level) for r in chosen]

    fallback = stage1_candidates[: max(1, min(LOCAL_TOP_K, len(stage1_candidates)))]
    return [(c[0], c[1], c[2]) for c in fallback]


def _build_local_refinement_candidates(
    candidate_lookup: Dict[Tuple[float, float, float], Dict[str, float]],
    seed_triples: List[Tuple[float, float, float]],
    already_evaluated: set[Tuple[float, float, float]],
) -> List[Tuple[float, float, float, Dict[str, float]]]:
    v_nom_idx = {v: i for i, v in enumerate(V_NOM_LIST)}
    v_turn_idx = {v: i for i, v in enumerate(V_TURN_LIST)}
    k_idx = {k: i for i, k in enumerate(K_LEVEL_LIST)}

    picks: set[Tuple[float, float, float]] = set()
    for v_nom, v_turn, k_level in seed_triples:
        i0 = v_nom_idx.get(v_nom)
        j0 = v_turn_idx.get(v_turn)
        k0 = k_idx.get(k_level)
        if i0 is None or j0 is None or k0 is None:
            continue

        for ii in range(max(0, i0 - LOCAL_NEIGHBOR_RADIUS), min(len(V_NOM_LIST), i0 + LOCAL_NEIGHBOR_RADIUS + 1)):
            for jj in range(max(0, j0 - LOCAL_NEIGHBOR_RADIUS), min(len(V_TURN_LIST), j0 + LOCAL_NEIGHBOR_RADIUS + 1)):
                for kk in range(max(0, k0 - LOCAL_NEIGHBOR_RADIUS), min(len(K_LEVEL_LIST), k0 + LOCAL_NEIGHBOR_RADIUS + 1)):
                    key = (V_NOM_LIST[ii], V_TURN_LIST[jj], K_LEVEL_LIST[kk])
                    if key in already_evaluated:
                        continue
                    if key not in candidate_lookup:
                        continue
                    picks.add(key)

    ordered = sorted(
        picks,
        key=lambda key: feasibility_sort_key(
            key[0], key[1], key[2], candidate_lookup[key]
        ),
    )
    if MAX_LOCAL_CANDIDATES > 0:
        ordered = ordered[:MAX_LOCAL_CANDIDATES]
    return [(k[0], k[1], k[2], candidate_lookup[k]) for k in ordered]


def _better(best: Optional[RunResult], cand: RunResult) -> bool:
    if not cand.success:
        return False
    if best is None:
        return True
    if best.score is None and cand.score is not None:
        return True
    if best.score is not None and cand.score is not None and cand.score < best.score:
        return True
    return False


def write_csv(rows: List[RunResult]) -> None:
    # Collect all metric keys seen
    keys = sorted({k for r in rows for k in r.metrics.keys()})
    header = ["V_NOM_MPS", "V_TURN_MPS", "K_LEVEL_TURN", "success", "score", "elapsed_s", "returncode"] + keys
    lines = [",".join(header)]

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bool):
            return "1" if x else "0"
        if isinstance(x, float):
            return f"{x:.8g}"
        return str(x)

    for r in rows:
        vals = [
            fmt(r.v_nom), fmt(r.v_turn), fmt(r.k_level),
            fmt(r.success), fmt(r.score), fmt(r.elapsed_s), fmt(r.returncode),
        ] + [fmt(r.metrics.get(k, None)) for k in keys]
        lines.append(",".join(vals))

    ALL_CSV.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows: List[RunResult] = []
    best: Optional[RunResult] = None
    run_start = time.perf_counter()

    candidates, skipped_msgs = build_sweep_candidates()
    candidate_lookup: Dict[Tuple[float, float, float], Dict[str, float]] = {
        (v_nom, v_turn, k): est for (v_nom, v_turn, k, est) in candidates
    }
    evaluated: set[Tuple[float, float, float]] = set()

    total_unpruned = len(V_NOM_LIST) * len(V_TURN_LIST) * len(K_LEVEL_LIST)
    total = len(candidates)
    counter = 0

    print(
        f"Sweep backend={EVAL_BACKEND} | strategy={SEARCH_STRATEGY} | fast_mode={FAST_SWEEP_MODE} | workers={N_WORKERS}"
    )

    if USE_TURN_PRUNING:
        skipped_runs = total_unpruned - total
        print(
            f"Turn pruning active: skipping ~{skipped_runs} runs; evaluating {total}/{total_unpruned}."
        )
        if skipped_msgs:
            preview_n = min(10, len(skipped_msgs))
            for line in skipped_msgs[:preview_n]:
                print(f"  {line}")
            if len(skipped_msgs) > preview_n:
                print(f"  ... and {len(skipped_msgs) - preview_n} more skipped pairs")

    if total == 0:
        print("\nNo sweep candidates left after analytical pruning.")
        print("Try reducing PRUNE_SPAN_ASSUMED_M/PRUNE_CHORD_ASSUMED_M or disabling USE_TURN_PRUNING.")
        return

    stop_now = False
    if SEARCH_STRATEGY == "adaptive":
        coarse_v_nom = set(_coarse_values(V_NOM_LIST, COARSE_POINTS_V_NOM))
        coarse_v_turn = set(_coarse_values(V_TURN_LIST, COARSE_POINTS_V_TURN))
        coarse_k = set(_coarse_values(K_LEVEL_LIST, COARSE_POINTS_K_LEVEL))

        stage1_candidates = [
            c for c in candidates
            if c[0] in coarse_v_nom and c[1] in coarse_v_turn and c[2] in coarse_k
        ]
        if not stage1_candidates:
            stage1_candidates = candidates[: min(20, len(candidates))]

        print(
            f"Adaptive stage 1: evaluating {len(stage1_candidates)} coarse candidates."
        )
        best, counter, stop_now, stage1_results = _run_batch(
            batch=stage1_candidates,
            rows=rows,
            evaluated=evaluated,
            best=best,
            counter=counter,
            stage_label="stage1",
        )

        if not stop_now:
            seed_triples = _pick_seed_triples(stage1_results, stage1_candidates)
            stage2_candidates = _build_local_refinement_candidates(
                candidate_lookup=candidate_lookup,
                seed_triples=seed_triples,
                already_evaluated=evaluated,
            )
            print(
                f"Adaptive stage 2: evaluating {len(stage2_candidates)} local-refinement candidates."
            )
            best, counter, stop_now, _ = _run_batch(
                batch=stage2_candidates,
                rows=rows,
                evaluated=evaluated,
                best=best,
                counter=counter,
                stage_label="stage2",
            )
    else:
        print(f"Grid mode: evaluating {len(candidates)} candidates.")
        best, counter, stop_now, _ = _run_batch(
            batch=candidates,
            rows=rows,
            evaluated=evaluated,
            best=best,
            counter=counter,
            stage_label="grid",
        )

    write_csv(rows)
    elapsed_total = time.perf_counter() - run_start
    print(f"\nSweep runtime: {elapsed_total:.2f} s for {len(rows)} evaluated runs.")

    if best is not None:
        BEST_LOG.write_text(best.stdout, encoding="utf-8")
        payload = {
            "V_NOM_MPS": best.v_nom,
            "V_TURN_MPS": best.v_turn,
            "K_LEVEL_TURN": best.k_level,
            "score": best.score,
            "metrics": best.metrics,
            "backend": EVAL_BACKEND,
            "strategy": SEARCH_STRATEGY,
            "workers": N_WORKERS,
            "evaluated_runs": len(rows),
            "total_runtime_s": elapsed_total,
        }
        BEST_PARAMS.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print("\nBest feasible run saved:")
        print(f"  {BEST_PARAMS}")
        print(f"  {BEST_LOG}")
        print(f"  {ALL_CSV}")
    else:
        print("\nNo feasible run found under current sweep ranges + sanity filters.")
        print("Next actions:")
        print("  - Widen V_TURN_LIST slightly (if arena footprint allows).")
        print("  - Relax tail thresholds temporarily.")
        print("  - Try SEARCH_STRATEGY='grid' if adaptive search is too aggressive.")
        print("  - Try EVAL_BACKEND='subprocess' if inprocess import fails.")


if __name__ == "__main__":
    main()
