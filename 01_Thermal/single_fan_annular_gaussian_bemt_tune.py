from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

import single_fan_annular_gaussian_bemt as base


TUNING_REPORT_XLSX = Path("B_results/single_annular_bemt_tuning_report.xlsx")
TUNING_SHEET_NAME = "tuning"


def build_tuning_table(
    selected_hyper: base.FitHyperParams,
    trials: List[base.HyperTuneTrial],
) -> pd.DataFrame:
    rows: List[Dict[str, float | str | bool]] = []
    for trial in trials:
        hyper = trial.hyper
        summary = trial.summary
        rows.append(
            {
                "name": hyper.name,
                "fourier_order": int(hyper.fourier_order),
                "robust_loss": str(hyper.robust_loss),
                "robust_f_scale": float(hyper.robust_f_scale),
                "harmonic_ridge_lambda": float(hyper.harmonic_ridge_lambda),
                "harmonic_rel_cap_lambda": float(
                    hyper.harmonic_rel_cap_lambda
                ),
                "harmonic_rel_max_to_a0": float(hyper.harmonic_rel_max_to_a0),
                "harmonic_order_weight_exp": float(
                    hyper.harmonic_order_weight_exp
                ),
                "mean_rmse": float(summary.mean_rmse),
                "mean_mae": float(summary.mean_mae),
                "max_rmse": float(summary.max_rmse),
                "total_samples": int(summary.total_samples),
                "is_selected": hyper.name == selected_hyper.name,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_rmse", "mean_mae"],
        ascending=[True, True],
    )


def main() -> None:
    selected_hyper, fit_results, trials = base.autotune_fit_hyper(base.SHEETS)

    base.save_azimuthal_fit_table(
        fit_results=fit_results,
        out_path=base.OUT_AZ_PARAMS_XLSX,
        sheet_name=base.OUT_AZ_PARAMS_SHEET,
        fourier_order=selected_hyper.fourier_order,
    )
    tuning_df = build_tuning_table(
        selected_hyper=selected_hyper,
        trials=trials,
    )
    TUNING_REPORT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        TUNING_REPORT_XLSX,
        engine="openpyxl",
        mode="w",
    ) as writer:
        tuning_df.to_excel(writer, index=False, sheet_name=TUNING_SHEET_NAME)

    summary = base.summarize_fit_results(fit_results)
    print("\nSelected single-fan harmonic annular hyper-parameters")
    print(
        f"name={selected_hyper.name}, "
        f"N={selected_hyper.fourier_order}, "
        f"loss={selected_hyper.robust_loss}, "
        f"f_scale={selected_hyper.robust_f_scale}, "
        f"ridge={selected_hyper.harmonic_ridge_lambda}, "
        f"relcap={selected_hyper.harmonic_rel_cap_lambda}, "
        f"relmax={selected_hyper.harmonic_rel_max_to_a0}, "
        f"order_exp={selected_hyper.harmonic_order_weight_exp}"
    )
    print(
        f"mean_RMSE={summary.mean_rmse:.6f}, "
        f"mean_MAE={summary.mean_mae:.6f}, "
        f"max_RMSE={summary.max_rmse:.6f}"
    )
    print(f"Saved fit parameters to: {base.OUT_AZ_PARAMS_XLSX.resolve()}")
    print(f"Saved tuning report to: {TUNING_REPORT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
