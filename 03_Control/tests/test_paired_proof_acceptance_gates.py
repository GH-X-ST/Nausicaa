from __future__ import annotations

import pandas as pd
import pytest

import aggregate_paired_w0_w1_archive as aggregation
import run_paired_w0_w1_archive_chunked as chunked
import run_paired_w0_w1_partitioned_planning as planning
from dense_archive_table_io import TableManifest


EXPECTED_RECOMMENDED_PAIRED_PROOF_COMMAND = (
    "python 03_Control/04_Scenarios/run_paired_w0_w1_partitioned_planning.py "
    "--run-id 13 --paired-scale-mode proof --proof-target-trials-per-environment 2500 "
    "--partition-rows 2500\n"
    "python 03_Control/04_Scenarios/run_paired_w0_w1_archive_chunked.py "
    "--run-id 14 --planning-run-id 13 --workers 8 --max-workers 8 --resume\n"
    "python 03_Control/04_Scenarios/aggregate_paired_w0_w1_archive.py "
    "--run-id 14 --planning-run-id 13 --expected-trials-per-environment 2500 "
    "--build-upload-package --build-governor-package"
)


def test_recommended_paired_proof_command_is_exact() -> None:
    assert (
        chunked.RECOMMENDED_PAIRED_PROOF_COMMAND
        == EXPECTED_RECOMMENDED_PAIRED_PROOF_COMMAND
    )


def test_proof_mode_keeps_production_floors_out_of_acceptance_gate() -> None:
    planning._validate_config(
        planning.PairedW0W1PartitionedPlanningConfig(
            run_id=20,
            paired_scale_mode="proof",
            proof_target_trials_per_environment=4,
            w1_floor_trials_per_branch=350000,
            w1_target_trials_per_branch=4,
            partition_rows=2,
        )
    )


def test_production_mode_requires_w1_branch_and_floor_gate() -> None:
    with pytest.raises(ValueError, match="must not exceed target"):
        planning._validate_config(
            planning.PairedW0W1PartitionedPlanningConfig(
                run_id=20,
                paired_scale_mode="production",
                w1_floor_trials_per_branch=6,
                w1_target_trials_per_branch=4,
                partition_rows=2,
            )
        )
    with pytest.raises(ValueError, match="requires both W1 branches active"):
        planning._validate_config(
            planning.PairedW0W1PartitionedPlanningConfig(
                run_id=20,
                paired_scale_mode="production",
                active_environment_modes=("W1_single_fan",),
                w1_floor_trials_per_branch=4,
                w1_target_trials_per_branch=4,
                partition_rows=2,
            )
        )


def test_default_proof_aggregation_manifest_keeps_no_overclaiming_fields() -> None:
    config = aggregation.PairedAggregationConfig()
    manifest = aggregation._manifest(
        config=config,
        outputs=aggregation._outputs(config),
        descriptors=pd.DataFrame(
            {
                "test_environment_mode": list(planning.PAIRED_ENVIRONMENT_MODES),
                "layout_branch_id": [
                    "single_fan_branch",
                    "single_fan_branch",
                    "four_fan_branch",
                    "four_fan_branch",
                ],
                "fan_layout": ["single_fan", "single_fan", "four_fan", "four_fan"],
                "paired_sample_key": [
                    "pair_single",
                    "pair_single",
                    "pair_four",
                    "pair_four",
                ],
                "family": ["mild_bank", "mild_bank", "mild_bank", "mild_bank"],
                "target_heading_deg": [30.0, 30.0, 30.0, 30.0],
                "direction_sign": [1, 1, 1, 1],
                "start_class": [
                    "favourable",
                    "favourable",
                    "favourable",
                    "favourable",
                ],
                "seed": [11, 11, 12, 12],
            }
        ),
        chunk_summary=pd.DataFrame(),
        table_manifest=TableManifest(
            run_id=14,
            root="root",
            storage_format="csv_gz",
            tables=(),
        ),
        paired_summary=pd.DataFrame(),
        w0_failed_w1_valid=pd.DataFrame(),
        w1_envelope=pd.DataFrame(),
        progress={},
        profile={},
    )

    assert manifest["paired_scale_mode"] == "proof"
    assert manifest["active_environment_modes"] == list(planning.PAIRED_ENVIRONMENT_MODES)
    assert manifest["w1_scheduled_independent_of_w0_success"] is True
    assert manifest["full_w1_production_claim"] is False
    assert manifest["w2_w3_w4_w5_performed"] is False
    assert manifest["hardware_or_mission_claim"] is False
    assert manifest["sim_to_real_transfer_claim"] is False
    assert manifest["governor_artifacts_scan_raw_tables"] is False
    assert manifest["governor_package_contains_w0_candidates"] is False
    assert manifest["governor_package_branch_local_only"] is True
