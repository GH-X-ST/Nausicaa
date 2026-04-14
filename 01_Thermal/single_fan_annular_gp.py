from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import single_fan_gp as base_gp
from annular_gp_models import (
    AnnularGPModelBundle,
    SingleFanBEMTMeanModel,
    build_autotune_cv_table,
    make_grid_prediction_tables,
    make_training_prediction_table,
)


XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
FAN_CENTER_XY = (4.2, 2.4)

FEATURE_MODE = "cartesian"
KERNEL_FAMILY = "rbf_ard"
ALPHA_SCALE = 1.0
SIGMA_FALLBACK = 0.14
SIGMA_MIN = 0.03

ENABLE_AUTO_TUNE = True
AUTO_TUNE_CV_N_SPLITS = 3
AUTO_TUNE_CV_RESTARTS_OPTIMIZER = 0
AUTO_TUNE_RMSE_TIE_TOL = 1e-4
AUTO_TUNE_CANDIDATES = (
    ("baseline", "cartesian", "rbf_ard", 1.0),
    ("cartesian_rbf_ard_a0p8", "cartesian", "rbf_ard", 0.8),
    ("cartesian_rbf_ard_a1p2", "cartesian", "rbf_ard", 1.2),
    ("cartesian_matern32_ard_a1", "cartesian", "matern32_ard", 1.0),
    ("cartesian_matern52_ard_a1", "cartesian", "matern52_ard", 1.0),
)
N_RESTARTS_OPTIMIZER = 0
RANDOM_STATE = 42

MEAN_PARAMS_XLSX = Path("B_results/single_annular_bemt_params_pchip.xlsx")
MEAN_PARAMS_SHEET = "single_bemt_az_pchip"
MEAN_FUNCTION_NAME = "harmonic_annular_bemt"

OUT_DIR = Path("B_results/Single_Fan_Annular_GP")
TRAIN_PRED_CSV_PATH = OUT_DIR / "single_annular_gp_training_predictions.csv"
SUMMARY_XLSX_PATH = OUT_DIR / "single_annular_gp_summary.xlsx"
GRID_PRED_XLSX_PATH = OUT_DIR / "single_annular_gp_grid_predictions.xlsx"
ANALYSIS_XLSX_PATH = Path("B_results/single_annular_gp_analysis.xlsx")
ANALYSIS_SHEET_NAME = "single_annular_gp_analysis"


def build_tune_candidates() -> List[base_gp.GPTuneCandidate]:
    candidates: List[base_gp.GPTuneCandidate] = []
    for name, feature_mode, kernel_family, alpha_scale in AUTO_TUNE_CANDIDATES:
        candidates.append(
            base_gp.GPTuneCandidate(
                name=str(name),
                feature_mode=base_gp.normalize_feature_mode_name(feature_mode),
                kernel_family=base_gp.normalize_kernel_family_name(
                    kernel_family
                ),
                alpha_scale=float(alpha_scale),
            )
        )
    return candidates


def select_gp_candidate(
    train_df: pd.DataFrame,
    fan_center_xy: Tuple[float, float],
) -> Tuple[base_gp.GPTuneCandidate, List[base_gp.GPTuneTrial]]:
    candidates = build_tune_candidates()
    trials: List[base_gp.GPTuneTrial] = []

    for idx, candidate in enumerate(candidates):
        try:
            cv_metrics, n_folds = base_gp.evaluate_candidate_group_cv(
                train_df=train_df,
                candidate=candidate,
                fan_center_xy=fan_center_xy,
                n_splits=AUTO_TUNE_CV_N_SPLITS,
                n_restarts_optimizer=AUTO_TUNE_CV_RESTARTS_OPTIMIZER,
                random_state=RANDOM_STATE + 17 * idx,
            )
            trials.append(
                base_gp.GPTuneTrial(
                    candidate=candidate,
                    cv_metrics=cv_metrics,
                    n_folds=n_folds,
                )
            )
        except Exception as exc:
            print(f"Auto-tune candidate '{candidate.name}' failed: {exc}")

    if not trials:
        return (
            base_gp.GPTuneCandidate(
                name="fallback",
                feature_mode=FEATURE_MODE,
                kernel_family=KERNEL_FAMILY,
                alpha_scale=float(ALPHA_SCALE),
            ),
            [],
        )

    min_wrmse = min(float(trial.cv_metrics["wrmse_mps"]) for trial in trials)
    finalists = [
        trial
        for trial in trials
        if float(trial.cv_metrics["wrmse_mps"])
        <= min_wrmse + float(AUTO_TUNE_RMSE_TIE_TOL)
    ]
    best_trial = min(
        finalists,
        key=lambda trial: (
            float(trial.cv_metrics["mae_mps"]),
            float(trial.cv_metrics["rmse_mps"]),
        ),
    )

    print("Auto-tuning annular-GP candidates (grouped CV by sheet):")
    print(
        " name               mode       kernel        alpha  "
        "CV_WRMSE  CV_RMSE   CV_MAE"
    )
    for trial in trials:
        marker = (
            "*" if trial.candidate.name == best_trial.candidate.name else " "
        )
        metrics = trial.cv_metrics
        print(
            f"{marker}{trial.candidate.name:17s}  "
            f"{trial.candidate.feature_mode:9s}  "
            f"{trial.candidate.kernel_family:12s}  "
            f"{trial.candidate.alpha_scale:5.2f}  "
            f"{metrics['wrmse_mps']:8.5f}  "
            f"{metrics['rmse_mps']:8.5f}  "
            f"{metrics['mae_mps']:8.5f}"
        )
    print(f"Selected annular-GP candidate: {best_trial.candidate.name}")

    return best_trial.candidate, trials


def build_residual_training_table(
    train_df: pd.DataFrame,
    mean_model: SingleFanBEMTMeanModel,
) -> pd.DataFrame:
    residual_df = train_df.copy()
    w_prior = mean_model.evaluate(
        x_m=residual_df["x_m"].to_numpy(dtype=float),
        y_m=residual_df["y_m"].to_numpy(dtype=float),
        z_m=residual_df["z_m"].to_numpy(dtype=float),
    )
    residual_df["w_prior_mps"] = w_prior
    residual_df["w_residual_obs_mps"] = (
        residual_df["w_obs_mps"].to_numpy(dtype=float) - w_prior
    )
    residual_df["w_obs_mps"] = residual_df["w_residual_obs_mps"]
    return residual_df


def build_summary_metrics_table(
    overall_metrics: Dict[str, float],
    selected_candidate: base_gp.GPTuneCandidate,
    model: AnnularGPModelBundle,
) -> pd.DataFrame:
    summary_df = pd.DataFrame([overall_metrics])
    summary_df["feature_mode"] = selected_candidate.feature_mode
    summary_df["kernel_family"] = selected_candidate.kernel_family
    summary_df["alpha_scale"] = float(selected_candidate.alpha_scale)
    summary_df["autotune_enabled"] = bool(ENABLE_AUTO_TUNE)
    summary_df["autotune_selected"] = str(selected_candidate.name)
    summary_df["mean_function"] = MEAN_FUNCTION_NAME
    summary_df["mean_params_xlsx"] = str(model.mean_model.source_path)
    summary_df["mean_params_sheet"] = model.mean_model.sheet_name
    summary_df["log_marginal_likelihood"] = float(
        model.residual_gp_model.gp.log_marginal_likelihood_value_
    )
    summary_df["kernel"] = str(model.residual_gp_model.gp.kernel_)
    return summary_df


def build_hyperparameter_table(
    selected_candidate: base_gp.GPTuneCandidate,
    model: AnnularGPModelBundle,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parameter": "xlsx_path", "value": XLSX_PATH},
            {"parameter": "sheet_count", "value": len(SHEETS)},
            {"parameter": "fan_center_x_m", "value": float(FAN_CENTER_XY[0])},
            {"parameter": "fan_center_y_m", "value": float(FAN_CENTER_XY[1])},
            {
                "parameter": "feature_mode",
                "value": selected_candidate.feature_mode,
            },
            {
                "parameter": "kernel_family",
                "value": selected_candidate.kernel_family,
            },
            {
                "parameter": "alpha_scale",
                "value": float(selected_candidate.alpha_scale),
            },
            {
                "parameter": "autotune_enabled",
                "value": bool(ENABLE_AUTO_TUNE),
            },
            {
                "parameter": "autotune_selected",
                "value": str(selected_candidate.name),
            },
            {"parameter": "mean_function", "value": MEAN_FUNCTION_NAME},
            {
                "parameter": "mean_params_xlsx",
                "value": str(model.mean_model.source_path),
            },
            {
                "parameter": "mean_params_sheet",
                "value": model.mean_model.sheet_name,
            },
            {
                "parameter": "sigma_fallback_mps",
                "value": float(SIGMA_FALLBACK),
            },
            {"parameter": "sigma_min_mps", "value": float(SIGMA_MIN)},
            {
                "parameter": "n_restarts_optimizer",
                "value": int(N_RESTARTS_OPTIMIZER),
            },
            {"parameter": "random_state", "value": int(RANDOM_STATE)},
            {
                "parameter": "autotune_cv_n_splits",
                "value": int(AUTO_TUNE_CV_N_SPLITS),
            },
            {
                "parameter": "autotune_cv_restarts_optimizer",
                "value": int(AUTO_TUNE_CV_RESTARTS_OPTIMIZER),
            },
            {
                "parameter": "kernel_fitted",
                "value": str(model.residual_gp_model.gp.kernel_),
            },
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mean_model = SingleFanBEMTMeanModel.from_workbook(
        xlsx_path=MEAN_PARAMS_XLSX,
        sheet_name=MEAN_PARAMS_SHEET,
        fan_center_xy=FAN_CENTER_XY,
    )
    train_df = base_gp.build_training_table(
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
        fan_center_xy=FAN_CENTER_XY,
        sigma_fallback=SIGMA_FALLBACK,
        sigma_min=SIGMA_MIN,
    )
    residual_train_df = build_residual_training_table(
        train_df=train_df,
        mean_model=mean_model,
    )

    if ENABLE_AUTO_TUNE:
        selected_candidate, tune_trials = select_gp_candidate(
            train_df=residual_train_df,
            fan_center_xy=FAN_CENTER_XY,
        )
    else:
        selected_candidate = base_gp.GPTuneCandidate(
            name="manual",
            feature_mode=FEATURE_MODE,
            kernel_family=KERNEL_FAMILY,
            alpha_scale=float(ALPHA_SCALE),
        )
        tune_trials = []

    residual_gp_model = base_gp.fit_gp_model(
        train_df=residual_train_df,
        feature_mode=selected_candidate.feature_mode,
        kernel_family=selected_candidate.kernel_family,
        alpha_scale=selected_candidate.alpha_scale,
        fan_center_xy=FAN_CENTER_XY,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=RANDOM_STATE,
    )
    model = AnnularGPModelBundle(
        residual_gp_model=residual_gp_model,
        mean_model=mean_model,
    )

    pred_df = make_training_prediction_table(model=model, train_df=train_df)
    pred_df.to_csv(TRAIN_PRED_CSV_PATH, index=False)

    overall_metrics = base_gp.compute_regression_metrics(
        y_true=pred_df["w_obs_mps"].to_numpy(dtype=float),
        y_pred=pred_df["w_pred_mps"].to_numpy(dtype=float),
        sigma_mps=pred_df["sigma_mps"].to_numpy(dtype=float),
    )
    summary_metrics_df = build_summary_metrics_table(
        overall_metrics=overall_metrics,
        selected_candidate=selected_candidate,
        model=model,
    )
    per_sheet_df = base_gp.summarize_by_sheet(pred_df)
    analysis_df = base_gp.build_analysis_style_metrics_table(pred_df)
    hyper_df = build_hyperparameter_table(
        selected_candidate=selected_candidate,
        model=model,
    )
    autotune_cv_df = build_autotune_cv_table(
        tune_trials=tune_trials,
        selected_candidate=selected_candidate,
    )

    base_gp.write_tables_to_excel(
        SUMMARY_XLSX_PATH,
        {
            "overall_metrics": summary_metrics_df,
            "per_sheet_metrics": per_sheet_df,
            "hyperparameters": hyper_df,
            "autotune_cv": autotune_cv_df,
        },
    )
    base_gp.write_table_to_excel_no_index(
        SUMMARY_XLSX_PATH,
        analysis_df,
        ANALYSIS_SHEET_NAME,
    )
    base_gp.write_table_to_excel_no_index(
        ANALYSIS_XLSX_PATH,
        analysis_df,
        ANALYSIS_SHEET_NAME,
    )

    grid_tables = make_grid_prediction_tables(
        model=model,
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
        parse_sheet_height_m=base_gp.parse_sheet_height_m,
        read_slice_from_sheet=base_gp.read_slice_from_sheet,
        sheet_tag="annular_gp",
    )
    base_gp.write_tables_to_excel(GRID_PRED_XLSX_PATH, grid_tables)

    print("Annular-mean GP model fitted successfully.")
    print(f"Samples used: {int(train_df.shape[0])}")
    print(
        "Selected annular-GP config: "
        f"name={selected_candidate.name}, "
        f"feature_mode={selected_candidate.feature_mode}, "
        f"kernel_family={selected_candidate.kernel_family}, "
        f"alpha_scale={selected_candidate.alpha_scale:.2f}"
    )
    print(f"Mean function: {MEAN_FUNCTION_NAME}")
    print(f"Mean-parameter workbook: {MEAN_PARAMS_XLSX.resolve()}")
    print(f"Fitted residual kernel: {model.residual_gp_model.gp.kernel_}")
    print(
        "Overall metrics: "
        f"MAE={overall_metrics['mae_mps']:.4f} m/s, "
        f"RMSE={overall_metrics['rmse_mps']:.4f} m/s, "
        f"WRMSE={overall_metrics['wrmse_mps']:.4f} m/s, "
        f"R2={overall_metrics['r2']:.4f}"
    )
    print(f"Training predictions CSV: {TRAIN_PRED_CSV_PATH.resolve()}")
    print(f"Summary workbook: {SUMMARY_XLSX_PATH.resolve()}")
    print(f"Analysis workbook: {ANALYSIS_XLSX_PATH.resolve()}")
    print(f"Grid predictions workbook: {GRID_PRED_XLSX_PATH.resolve()}")


if __name__ == "__main__":
    main()
