from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from calibration_profile import ACTIVE_CALIBRATION_PROFILE


FLIGHT_TEST_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = FLIGHT_TEST_ROOT / "01_Runtime"
CONTROLLER_ROOT = FLIGHT_TEST_ROOT / "02_Controller"
FROZEN_INPUT_ROOT = FLIGHT_TEST_ROOT / "03_Frozen_Inputs"
RESULT_ROOT = FLIGHT_TEST_ROOT / "05_Results"
if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from primitive_timing_contract import LAUNCH_HANDOFF_DURATION_S, LAUNCH_HANDOFF_POLICY_VERSION  # noqa: E402

OPERATIONAL_REGION_CENTER_M = (3.9, 2.2, 1.95)
DEFAULT_VICON_POSITION_OFFSET_M = ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m
DEFAULT_VICON_ATTITUDE_SIGNS = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs
DEFAULT_VICON_ATTITUDE_OFFSET_RAD = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
DEFAULT_REAL_FLIGHT_LIBRARY_TIER = "balanced_cluster"
REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON = (
    "balanced_cluster_selected_for_first_real_flight_from_e01_real_flight_aligned_validation;"
    "deployment_tier_prioritises_transition_diversity_and_defensible_high_energy_validation_after_real_flight_safety_updates;"
    "heavy_cluster_is_compact_runtime_fallback"
)


@dataclass(frozen=True)
class FlightRuntimeConfig:
    run_label: str
    library_tier: str = DEFAULT_REAL_FLIGHT_LIBRARY_TIER
    controller_mode: str = "closed_loop"
    experiment_case_id: str = ""
    experiment_case_name: str = ""
    experiment_memory_enabled: bool = False
    experiment_layout_id: str = ""
    throw_index: int = 0
    attempt_index: int = 0
    serial_port: str = "COM11"
    serial_baud: int = 1_000_000
    vicon_host: str = "192.168.0.100:801"
    vicon_subject_name: str = "Nausicaa"
    governor_period_s: float = 0.100
    serial_period_s: float = 0.020
    vicon_poll_period_s: float = 0.005
    launch_handoff_duration_s: float = LAUNCH_HANDOFF_DURATION_S
    launch_handoff_policy_version: str = LAUNCH_HANDOFF_POLICY_VERSION
    max_duration_s: float = 20.0
    launch_wait_timeout_s: float = 8.0
    launch_gate_required_consecutive_frames: int = 2
    post_exit_neutral_tail_s: float = 0.30
    retry_cooldown_s: float = 5.0
    stale_vicon_timeout_s: float = 0.120
    derivative_cutoff_hz: float = 8.0
    body_rate_limit_rad_s: float = 6.0
    body_rate_observer_window_frames: int = 7
    launch_gate_rate_confidence_min: float = 0.65
    launch_gate_body_rate_limits_rad_s: tuple[float, float, float] = (1.2, 1.2, 1.8)
    rejected_launch_attempt_min_speed_m_s: float = 2.0
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    vicon_position_offset_m: tuple[float, float, float] = DEFAULT_VICON_POSITION_OFFSET_M
    vicon_yaw_alignment_deg: float = ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg
    vicon_attitude_signs: tuple[float, float, float] = DEFAULT_VICON_ATTITUDE_SIGNS
    vicon_attitude_offset_rad: tuple[float, float, float] = DEFAULT_VICON_ATTITUDE_OFFSET_RAD
    vicon_frame_description: str = "full_xyz_position_and_roll_pitch_yaw_attitude_offset_correction"
    calibration_profile_id: str = ACTIVE_CALIBRATION_PROFILE.profile_id
    calibration_profile_hash: str = ACTIVE_CALIBRATION_PROFILE.profile_hash()
    vicon_calibration_source: str = "active_calibration_profile"
    deployment_evidence_manifest_path: Path = FROZEN_INPUT_ROOT / "deployment_evidence_manifest.json"
    deployment_evidence_required_for_armed_closed_loop: bool = True
    output_root: Path = RESULT_ROOT
    library_manifest_root: Path = FROZEN_INPUT_ROOT / "R8_library_size_study" / "E01" / "manifests"
    outcome_table_path: Path = FROZEN_INPUT_ROOT / "R8_outcome" / "E01" / "metrics" / "outcome_model_table.csv"
    controller_bundle_path: Path = (
        FROZEN_INPUT_ROOT / "R5_dense" / "E01" / "manifests" / "frozen_w01_controller_bundle.json"
    )
    governor_config_path: Path = (
        FROZEN_INPUT_ROOT / "R10_learn" / "E01" / "manifests" / "frozen_governor_config_for_r11.json"
    )

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / self.run_label

    @property
    def library_manifest_path(self) -> Path:
        return Path(self.library_manifest_root) / f"{self.library_tier}_primitive_library.json"


def default_run_label(prefix: str = "F") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"
