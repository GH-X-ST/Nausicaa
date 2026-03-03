import itertools
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "nausicaa.py"
OUTDIR = ROOT / "outputs"
TMP = OUTDIR / "_tmp_nausicaa_run.py"

BEST_LOG = OUTDIR / "best_run_log.txt"
BEST_PARAMS = OUTDIR / "best_params.json"
ALL_CSV = OUTDIR / "all_runs.csv"

# -------------------------
# Sweep ranges (edit freely)
# -------------------------
V_NOM_LIST = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
V_TURN_LIST = [3.6, 3.7, 3.8, 3.9, 4.0, 4.1]
K_LEVEL_LIST = [0.80, 0.85, 0.90, 0.95, 1.00]

# -------------------------
# Sanity thresholds (edit)
# -------------------------
MAX_SPAN = 0.70
MAX_CHORD = 0.20

# "tail not ridiculously large" proxies (tune)
MAX_TAIL_ARM = 0.65
MAX_HTAIL_AREA = 0.030
MAX_VTAIL_AREA = 0.020

# -------------------------
# Analytical pruning (fast skip of impossible turn settings)
# -------------------------
USE_TURN_PRUNING = True

# Conservative estimates used only for pruning (not for final validation)
MASS_KG_EST = 0.10          # set 0.10 or 0.11 depending on your current target
RHO_KG_M3 = 1.225

TURN_BANK_DEG_EST = 45.0    # must match TURN_BANK_DEG in nausicaa.py
TURN_CL_CAP_EST = 1.25      # must match TURN_CL_CAP in nausicaa.py
K_LEVEL_IS_RADIUS_FACTOR = True  # code uses R = V^2/(K*g*tan(phi))

ARENA_WIDTH_M_EST = 4.8     # must match ARENA_WIDTH_M in nausicaa.py
WALL_CLEARANCE_M_EST = 0.30 # must match WALL_CLEARANCE_M in nausicaa.py

# Assumed geometry used in footprint/CL pruning:
# - MAX_SPAN is conservative (may skip feasible smaller-span designs)
# - If you want less aggressive pruning, set span/chord lower
PRUNE_SPAN_ASSUMED_M = MAX_SPAN
PRUNE_CHORD_ASSUMED_M = MAX_CHORD  # for S estimate

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

SUCCESS_MARKERS = [
    # If your script prints something like this, it helps.
    # Leave as-is; runner also uses return code.
    "Solved",
    "Optimal",
    "feasible",
]

@dataclass
class RunResult:
    v_nom: float
    v_turn: float
    k_level: float
    success: bool
    metrics: Dict[str, float]
    score: Optional[float]   # lower is better
    stdout: str
    stderr: str
    returncode: int

def replace_constant(text: str, name: str, value: float) -> Tuple[str, bool]:
    """
    Replace a top-level constant assignment like:
        NAME = 3.5
    using regex anchored to line start.
    """
    # Match: optional spaces, NAME, optional spaces, =, rest of line
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*).*$", re.MULTILINE)
    new_text, n = pattern.subn(rf"\1{value}", text)
    return new_text, (n > 0)

def patch_source(v_nom: float, v_turn: float, k_level: float) -> str:
    src = SRC.read_text(encoding="utf-8")

    src, ok1 = replace_constant(src, "V_NOM_MPS", float(v_nom))
    src, ok2 = replace_constant(src, "V_TURN_MPS", float(v_turn))
    src, ok3 = replace_constant(src, "K_LEVEL_TURN", float(k_level))

    missing = [n for n, ok in [("V_NOM_MPS", ok1), ("V_TURN_MPS", ok2), ("K_LEVEL_TURN", ok3)] if not ok]
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
    # if nonzero return code, usually failure
    if returncode != 0:
        return False
    # otherwise treat as success
    return True

def passes_sanity(metrics: Dict[str, float]) -> bool:
    # If a metric isn't available, do not hard-fail; just skip that check.
    # (This is necessary because we are not modifying nausicaa.py to guarantee prints.)
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

    radius_denom = 9.81 * tan_phi
    if K_LEVEL_IS_RADIUS_FACTOR:
        radius_denom *= k
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

def run_once(v_nom: float, v_turn: float, k_level: float) -> RunResult:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    patched = patch_source(v_nom, v_turn, k_level)
    TMP.write_text(patched, encoding="utf-8")

    cmd = ["python", str(TMP)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    stdout = p.stdout or ""
    stderr = p.stderr or ""

    metrics = parse_metrics(stdout)
    success = decide_success(p.returncode, stdout, stderr)

    # sanity filter only applied if success; otherwise irrelevant
    if success and not passes_sanity(metrics):
        success = False

    score = compute_score(metrics) if success else None

    return RunResult(
        v_nom=v_nom,
        v_turn=v_turn,
        k_level=k_level,
        success=success,
        metrics=metrics,
        score=score,
        stdout=stdout,
        stderr=stderr,
        returncode=p.returncode,
    )

def write_csv(rows: List[RunResult]) -> None:
    # Collect all metric keys seen
    keys = sorted({k for r in rows for k in r.metrics.keys()})
    header = ["V_NOM_MPS", "V_TURN_MPS", "K_LEVEL_TURN", "success", "score", "returncode"] + keys
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
            fmt(r.success), fmt(r.score), fmt(r.returncode),
        ] + [fmt(r.metrics.get(k, None)) for k in keys]
        lines.append(",".join(vals))

    ALL_CSV.write_text("\n".join(lines), encoding="utf-8")

def main():
    rows: List[RunResult] = []
    best: Optional[RunResult] = None

    candidates, skipped_msgs = build_sweep_candidates()
    total_unpruned = len(V_NOM_LIST) * len(V_TURN_LIST) * len(K_LEVEL_LIST)
    total = len(candidates)
    i = 0

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

    for v_nom, v_turn, k, _ in candidates:
        i += 1
        r = run_once(v_nom, v_turn, k)
        rows.append(r)

        if r.success:
            # Choose best by score if possible; otherwise first feasible.
            if best is None:
                best = r
                print(f"[{i}/{total}] NEW BEST (first feasible): V_NOM={v_nom}, V_TURN={v_turn}, K={k}, metrics={r.metrics}")
            else:
                if best.score is None and r.score is not None:
                    best = r
                    print(f"[{i}/{total}] NEW BEST (has score): V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}")
                elif (best.score is not None) and (r.score is not None) and (r.score < best.score):
                    best = r
                    print(f"[{i}/{total}] NEW BEST: V_NOM={v_nom}, V_TURN={v_turn}, K={k}, score={r.score}")
                elif best.score is None and r.score is None:
                    # no change
                    pass
            if STOP_AT_FIRST_FEASIBLE:
                print(f"[{i}/{total}] stop: first feasible found (STOP_AT_FIRST_FEASIBLE=True)")
                break
        else:
            # keep log light
            print(f"[{i}/{total}] fail: V_NOM={v_nom}, V_TURN={v_turn}, K={k}")

    write_csv(rows)

    if best is not None:
        BEST_LOG.write_text(best.stdout, encoding="utf-8")
        payload = {
            "V_NOM_MPS": best.v_nom,
            "V_TURN_MPS": best.v_turn,
            "K_LEVEL_TURN": best.k_level,
            "score": best.score,
            "metrics": best.metrics,
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
        print("  - Ensure nausicaa.py prints wing_span_m, wing_chord_m, tail metrics if you want strict filtering.")

if __name__ == "__main__":
    main()
