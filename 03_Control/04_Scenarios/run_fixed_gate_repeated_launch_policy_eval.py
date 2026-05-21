from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from episode_logging import write_episode_log  # noqa: E402
from fixed_gate_code_path_map import active_code_path_text  # noqa: E402
from fixed_gate_contract import FIXED_LAUNCH_GATE  # noqa: E402
from fixed_gate_policies import policy_table  # noqa: E402
from repeated_launch_episode import RepeatedLaunchEpisodeConfig, run_repeated_launch_episode  # noqa: E402


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_fixed_gate_repeated_launch_policy_eval(
    *,
    governor_candidate_package_csv: Path,
    run_id: int,
    episodes_per_policy: int = 1,
    seed: int = 20260521,
    allow_diagnostic_source: bool = False,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "repeated_launch_policy_eval"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(governor_candidate_package_csv)
    eligible_candidates = _eligible_package_candidates(candidates, allow_diagnostic_source=allow_diagnostic_source)
    policies = policy_table()
    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    query_frames: list[pd.DataFrame] = []
    belief_frames: list[pd.DataFrame] = []
    for _, policy in policies.iterrows():
        for fan_branch in ("single_fan_branch", "four_fan_branch"):
            branch_candidates = _branch_candidates(eligible_candidates, fan_branch)
            for episode_index in range(int(episodes_per_policy)):
                episode_id = f"fg_policy_s{int(run_id):03d}_{policy['policy_id']}_{fan_branch}_{episode_index:03d}"
                result = run_repeated_launch_episode(
                    _gate_centre_state(episode_index, seed),
                    branch_candidates,
                    RepeatedLaunchEpisodeConfig(
                        episode_id=episode_id,
                        policy_id=str(policy["policy_id"]),
                        fan_branch=fan_branch,
                        W_layer="W1",
                        claim_status="simulation_only",
                    ),
                )
                episode_frames.append(result["episode_summary"])
                step_frames.append(result["primitive_steps"])
                query_frames.append(result["governor_queries"])
                belief_frames.append(result["belief_updates"])
    episode_summary = _concat(episode_frames)
    primitive_steps = _concat(step_frames)
    governor_queries = _concat(query_frames)
    belief_updates = _concat(belief_frames)
    comparison = _policy_comparison(episode_summary)
    log_outputs = write_episode_log(
        root / "episode_logs",
        episode_summary=episode_summary,
        primitive_steps=primitive_steps,
        governor_queries=governor_queries,
        belief_updates=belief_updates,
        manifest_extra={
            "campaign": CAMPAIGN,
            "pass_name": "fixed_gate_repeated_launch_policy_eval",
            "governor_candidate_package_csv": str(governor_candidate_package_csv),
            "candidate_package_source": "fixed_gate_cluster_selection",
            "default_toy_candidates_used": False,
            "policy_readiness_status": _policy_readiness_status(candidates, eligible_candidates, allow_diagnostic_source),
        },
        overwrite=overwrite,
    )
    paths = log_outputs.as_dict()
    paths.update(
        {
            "policy_comparison_csv": root / "metrics" / "repeated_launch_policy_comparison.csv",
            "manifest_json": root / "manifests" / "repeated_launch_policy_eval_manifest.json",
            "report_md": root / "reports" / "repeated_launch_policy_eval_report.md",
        }
    )
    for directory in (root / "manifests", root / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(paths["policy_comparison_csv"], index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": "fixed_gate_repeated_launch_policy_eval",
        "active_mission_path": active_code_path_text(),
        "governor_candidate_package_csv": str(governor_candidate_package_csv),
        "candidate_package_source": "fixed_gate_cluster_selection",
        "default_toy_candidates_used": False,
        "candidate_row_count": int(len(candidates)),
        "eligible_candidate_row_count": int(len(eligible_candidates)),
        "diagnostic_sources_allowed": bool(allow_diagnostic_source),
        "episode_count": int(len(episode_summary)),
        "policy_readiness_status": _policy_readiness_status(candidates, eligible_candidates, allow_diagnostic_source),
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Repeated-launch policy evaluation consumes the clustering-derived governor candidate package; "
            "no real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, "
            "or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    paths["report_md"].write_text(_report(manifest), encoding="ascii")
    return paths


def _branch_candidates(candidates: pd.DataFrame, fan_branch: str) -> pd.DataFrame:
    if candidates.empty or "fan_branch" not in candidates.columns:
        return candidates.copy()
    return candidates[candidates["fan_branch"].astype(str).eq(str(fan_branch))].copy()


def _eligible_package_candidates(candidates: pd.DataFrame, *, allow_diagnostic_source: bool) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    frame = candidates.copy()
    if "evidence_role" in frame.columns:
        roles = {"mission_candidate", "partial_feedback"}
        if bool(allow_diagnostic_source):
            roles.update({"ablation_diagnostic", "boundary_diagnostic"})
        frame = frame[frame["evidence_role"].astype(str).isin(roles)]
    if "recommended_use" in frame.columns:
        uses = {"simulation_candidate", "hardware_candidate", "thesis", "hardware"}
        if bool(allow_diagnostic_source):
            uses.update({"diagnostic", "diagnostic_only"})
        frame = frame[frame["recommended_use"].astype(str).isin(uses)]
    return frame.reset_index(drop=True)


def _policy_readiness_status(
    candidates: pd.DataFrame,
    eligible_candidates: pd.DataFrame,
    allow_diagnostic_source: bool,
) -> str:
    if candidates.empty:
        return "blocked_no_clustering_candidate_package"
    if eligible_candidates.empty:
        return "diagnostic_only_package_blocked" if not allow_diagnostic_source else "blocked_no_eligible_package_rows"
    if allow_diagnostic_source:
        return "diagnostic_only_policy_eval_not_mission_evidence"
    return "ready"


def _gate_centre_state(episode_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + int(episode_index))
    state = np.zeros(15, dtype=float)
    state[0] = 0.5 * sum(FIXED_LAUNCH_GATE.x_w_m)
    state[1] = 0.5 * sum(FIXED_LAUNCH_GATE.y_w_m)
    state[2] = 0.5 * sum(FIXED_LAUNCH_GATE.z_w_m)
    state[5] = float(rng.uniform(-0.02, 0.02))
    state[6] = 5.5
    return state


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _policy_comparison(episode_summary: pd.DataFrame) -> pd.DataFrame:
    if episode_summary.empty:
        return pd.DataFrame(columns=["policy_id", "episode_count", "mean_energy_residual_m", "mean_lift_dwell_time_s"])
    return (
        episode_summary.groupby("policy_id", dropna=False)
        .agg(
            episode_count=("episode_id", "count"),
            mean_energy_residual_m=("energy_residual_m", "mean"),
            mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
            governor_accept_count=("governor_accept_count", "sum"),
            governor_reject_count=("governor_reject_count", "sum"),
        )
        .reset_index()
    )


def _report(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Fixed-Gate Repeated-Launch Policy Evaluation",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Candidate rows from clustering package: `{manifest['candidate_row_count']}`",
            f"- Eligible mission/partial candidate rows: `{manifest['eligible_candidate_row_count']}`",
            f"- Episode rows: `{manifest['episode_count']}`",
            f"- Policy readiness status: `{manifest['policy_readiness_status']}`",
            "- Default toy candidates used: `False`",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--governor-candidate-package-csv", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--episodes-per-policy", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--allow-diagnostic-source", action="store_true")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_repeated_launch_policy_eval(
        governor_candidate_package_csv=args.governor_candidate_package_csv,
        run_id=args.run_id,
        episodes_per_policy=args.episodes_per_policy,
        seed=args.seed,
        allow_diagnostic_source=args.allow_diagnostic_source,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
