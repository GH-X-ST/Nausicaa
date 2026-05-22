from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

import four_fan_annular_gaussian_bemt as base


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Tuning Configuration and Data Sources
# 2) Data Containers
# 3) Tuning Table Builders
# 4) Tuning Report Export
# =============================================================================

# =============================================================================
# 1) Tuning Configuration and Data Sources
# =============================================================================
# Workbook, parameter, and output paths below define the data-provenance boundary for this run.


TUNING_REPORT_XLSX = Path("B_results/four_annular_bemt_tuning_report.xlsx")
TUNING_SHEET_NAME = "tuning"
WRMSE_TIE_TOL = 1.0e-4


@dataclass(frozen=True)

# =============================================================================
# 2) Data Containers
# =============================================================================
# Small containers keep fitted parameters, diagnostics, and uncertainty assumptions explicit at module boundaries.

class JointTuneCandidate:
    name: str
    fan_fourier_order: Tuple[int, ...]
    fan_robust_loss: Tuple[str, ...]
    fan_robust_f_scale: Tuple[float, ...]
    fan_harmonic_ridge_lambda: Tuple[float, ...]
    fan_harmonic_rel_cap_lambda: Tuple[float, ...]
    fan_harmonic_rel_max_to_a0: Tuple[float, ...]
    fan_harmonic_order_weight_exp: Tuple[float, ...]


@dataclass
class JointTuneTrial:
    candidate: JointTuneCandidate
    fit_results: List[base.JointFitResult]
    overall_metrics: base.RawMetrics
    per_fan_metrics: Sequence[base.RawMetrics]


JOINT_AUTO_TUNE_CANDIDATES = (
    JointTuneCandidate(
        name="baseline_uniform_N1",
        fan_fourier_order=(1, 1, 1, 1),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.0, 1.0, 1.0, 1.0),
        fan_harmonic_ridge_lambda=(0.02, 0.02, 0.02, 0.02),
        fan_harmonic_rel_cap_lambda=(0.10, 0.10, 0.10, 0.10),
        fan_harmonic_rel_max_to_a0=(0.80, 0.80, 0.80, 0.80),
        fan_harmonic_order_weight_exp=(1.0, 1.0, 1.0, 1.0),
    ),
    JointTuneCandidate(
        name="uniform_N2_soft_bal",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.0, 1.0, 1.0, 1.0),
        fan_harmonic_ridge_lambda=(0.05, 0.05, 0.05, 0.05),
        fan_harmonic_rel_cap_lambda=(0.20, 0.20, 0.20, 0.20),
        fan_harmonic_rel_max_to_a0=(0.70, 0.70, 0.70, 0.70),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="uniform_N2_soft_rmse",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.5, 1.5, 1.5, 1.5),
        fan_harmonic_ridge_lambda=(0.05, 0.05, 0.05, 0.05),
        fan_harmonic_rel_cap_lambda=(0.20, 0.20, 0.20, 0.20),
        fan_harmonic_rel_max_to_a0=(0.70, 0.70, 0.70, 0.70),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="uniform_N2_soft_cons",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.5, 1.5, 1.5, 1.5),
        fan_harmonic_ridge_lambda=(0.03, 0.03, 0.03, 0.03),
        fan_harmonic_rel_cap_lambda=(0.10, 0.10, 0.10, 0.10),
        fan_harmonic_rel_max_to_a0=(0.80, 0.80, 0.80, 0.80),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="uniform_N2_huber_mae",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("huber", "huber", "huber", "huber"),
        fan_robust_f_scale=(0.5, 0.5, 0.5, 0.5),
        fan_harmonic_ridge_lambda=(0.05, 0.05, 0.05, 0.05),
        fan_harmonic_rel_cap_lambda=(0.05, 0.05, 0.05, 0.05),
        fan_harmonic_rel_max_to_a0=(0.70, 0.70, 0.70, 0.70),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="mixed_soft_l1_split",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.0, 1.5, 1.0, 1.5),
        fan_harmonic_ridge_lambda=(0.05, 0.03, 0.05, 0.03),
        fan_harmonic_rel_cap_lambda=(0.20, 0.10, 0.20, 0.10),
        fan_harmonic_rel_max_to_a0=(0.70, 0.80, 0.70, 0.80),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="mixed_soft_l1_cons_split",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.0, 1.5, 1.0, 1.5),
        fan_harmonic_ridge_lambda=(0.03, 0.03, 0.03, 0.03),
        fan_harmonic_rel_cap_lambda=(0.10, 0.10, 0.10, 0.10),
        fan_harmonic_rel_max_to_a0=(0.80, 0.80, 0.80, 0.80),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
    JointTuneCandidate(
        name="mixed_soft_l1_alt_bias",
        fan_fourier_order=(2, 2, 2, 2),
        fan_robust_loss=("soft_l1", "soft_l1", "soft_l1", "soft_l1"),
        fan_robust_f_scale=(1.5, 1.0, 1.5, 1.0),
        fan_harmonic_ridge_lambda=(0.03, 0.05, 0.03, 0.05),
        fan_harmonic_rel_cap_lambda=(0.10, 0.20, 0.10, 0.20),
        fan_harmonic_rel_max_to_a0=(0.80, 0.70, 0.80, 0.70),
        fan_harmonic_order_weight_exp=(1.5, 1.5, 1.5, 1.5),
    ),
)

# =============================================================================
# 3) Tuning Table Builders
# =============================================================================
# Table builders record each candidate assumption before selecting a preferred fit.


# Candidate application changes only the declared tuning parameters for comparison.
def apply_candidate(candidate: JointTuneCandidate) -> None:
    base.FAN_FOURIER_ORDER = tuple(int(v) for v in candidate.fan_fourier_order)
    base.FAN_ROBUST_LOSS = tuple(str(v) for v in candidate.fan_robust_loss)
    base.FAN_ROBUST_F_SCALE = tuple(
        float(v) for v in candidate.fan_robust_f_scale
    )
    base.FAN_HARMONIC_RIDGE_LAMBDA = tuple(
        float(v) for v in candidate.fan_harmonic_ridge_lambda
    )
    base.FAN_HARMONIC_REL_CAP_LAMBDA = tuple(
        float(v) for v in candidate.fan_harmonic_rel_cap_lambda
    )
    base.FAN_HARMONIC_REL_MAX_TO_A0 = tuple(
        float(v) for v in candidate.fan_harmonic_rel_max_to_a0
    )
    base.FAN_HARMONIC_ORDER_WEIGHT_EXP = tuple(
        float(v) for v in candidate.fan_harmonic_order_weight_exp
    )


# Tuning tables record candidate settings beside the resulting diagnostics.
def build_tuning_table(
    selected: JointTuneCandidate,
    trials: Sequence[JointTuneTrial],
) -> pd.DataFrame:
    rows: List[Dict[str, float | str | int | bool]] = []
    for trial in trials:
        rows.append(
            {
                "name": trial.candidate.name,
                "raw_wrmse": float(trial.overall_metrics.total_wrmse),
                "raw_sae": float(trial.overall_metrics.total_sae),
                "n_samples": int(trial.overall_metrics.n_samples),
                "fan_fourier_order": str(trial.candidate.fan_fourier_order),
                "fan_robust_loss": str(trial.candidate.fan_robust_loss),
                "fan_robust_f_scale": str(trial.candidate.fan_robust_f_scale),
                "fan_harmonic_ridge_lambda": str(
                    trial.candidate.fan_harmonic_ridge_lambda
                ),
                "fan_harmonic_rel_cap_lambda": str(
                    trial.candidate.fan_harmonic_rel_cap_lambda
                ),
                "fan_harmonic_rel_max_to_a0": str(
                    trial.candidate.fan_harmonic_rel_max_to_a0
                ),
                "fan_harmonic_order_weight_exp": str(
                    trial.candidate.fan_harmonic_order_weight_exp
                ),
                "is_selected": trial.candidate.name == selected.name,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["raw_wrmse", "raw_sae"],
        ascending=[True, True],
    )


# Joint autotuning evaluates one candidate set across all fitted height planes.
def autotune_joint_candidate() -> Tuple[
    JointTuneCandidate,
    JointTuneTrial,
    List[JointTuneTrial],
]:
    trials: List[JointTuneTrial] = []

    for candidate in JOINT_AUTO_TUNE_CANDIDATES:
        apply_candidate(candidate)
        base.validate_joint_fan_settings()
        fit_results = base.fit_all_heights_joint(base.SHEETS)
        overall_metrics, per_fan_metrics = base.evaluate_joint_fit_on_raw_maps(
            fit_results
        )
        trials.append(
            JointTuneTrial(
                candidate=candidate,
                fit_results=fit_results,
                overall_metrics=overall_metrics,
                per_fan_metrics=per_fan_metrics,
            )
        )

    min_wrmse = min(
        float(trial.overall_metrics.total_wrmse) for trial in trials
    )
    finalists = [
        trial
        for trial in trials
        if float(trial.overall_metrics.total_wrmse)
        <= min_wrmse + WRMSE_TIE_TOL
    ]
    selected_trial = min(
        finalists,
        key=lambda trial: float(trial.overall_metrics.total_sae),
    )

    print("Auto-tuning four-fan joint BEMT candidates:")
    print(" name                     raw_WRMSE   raw_SAE")
    for trial in trials:
        marker = "*"
        if trial.candidate.name != selected_trial.candidate.name:
            marker = " "
        print(
            f"{marker}{trial.candidate.name:24s}  "
            f"{trial.overall_metrics.total_wrmse:9.5f}  "
            f"{trial.overall_metrics.total_sae:8.4f}"
        )
    print(f"Selected four-fan candidate: {selected_trial.candidate.name}")
    return selected_trial.candidate, selected_trial, trials

# =============================================================================
# 4) Tuning Report Export
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.


# Main execution keeps data loading, evaluation, and export order deterministic.
def main() -> None:
    selected_candidate, selected_trial, trials = autotune_joint_candidate()
    apply_candidate(selected_candidate)
    base.write_joint_fit_tables(
        fit_results=selected_trial.fit_results,
        per_fan_metrics=selected_trial.per_fan_metrics,
    )

    tuning_df = build_tuning_table(
        selected=selected_candidate,
        trials=trials,
    )
    TUNING_REPORT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        TUNING_REPORT_XLSX,
        engine="openpyxl",
        mode="w",
    ) as writer:
        tuning_df.to_excel(writer, index=False, sheet_name=TUNING_SHEET_NAME)

    print("\nSelected four-fan harmonic annular hyper-parameters")
    print(
        f"name={selected_candidate.name}, "
        f"orders={selected_candidate.fan_fourier_order}, "
        f"losses={selected_candidate.fan_robust_loss}, "
        f"f_scales={selected_candidate.fan_robust_f_scale}"
    )
    print(
        f"ridge={selected_candidate.fan_harmonic_ridge_lambda}, "
        f"relcap={selected_candidate.fan_harmonic_rel_cap_lambda}, "
        f"relmax={selected_candidate.fan_harmonic_rel_max_to_a0}, "
        f"order_exp={selected_candidate.fan_harmonic_order_weight_exp}"
    )
    print(
        f"raw_WRMSE={selected_trial.overall_metrics.total_wrmse:.6f}, "
        f"raw_SAE={selected_trial.overall_metrics.total_sae:.6f}"
    )
    print(f"Saved fit parameters to: {base.OUT_AZ_PARAMS_XLSX.resolve()}")
    print(f"Saved tuning report to: {TUNING_REPORT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
