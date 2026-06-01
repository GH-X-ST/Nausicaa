from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


FLIGHT_TEST_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = FLIGHT_TEST_ROOT / "01_Runtime"
CONTROLLER_ROOT = FLIGHT_TEST_ROOT / "02_Controller"
FROZEN_INPUT_ROOT = FLIGHT_TEST_ROOT / "03_Frozen_Inputs"
RESULT_ROOT = FLIGHT_TEST_ROOT / "05_Results"


@dataclass(frozen=True)
class FlightRuntimeConfig:
    run_label: str
    library_tier: str = "balanced_cluster"
    serial_port: str = "COM11"
    serial_baud: int = 1_000_000
    vicon_host: str = "192.168.0.100:801"
    vicon_subject_name: str = "Nausicaa"
    governor_period_s: float = 0.100
    serial_period_s: float = 0.020
    max_duration_s: float = 20.0
    stale_vicon_timeout_s: float = 0.120
    derivative_cutoff_hz: float = 20.0
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    output_root: Path = RESULT_ROOT
    library_manifest_root: Path = FROZEN_INPUT_ROOT / "R8_library_size_study" / "B02" / "manifests"
    outcome_table_path: Path = FROZEN_INPUT_ROOT / "R8_outcome" / "B02" / "metrics" / "outcome_model_table.csv"
    controller_bundle_path: Path = (
        FROZEN_INPUT_ROOT / "R5_dense" / "B02" / "manifests" / "frozen_w01_controller_bundle.json"
    )
    governor_config_path: Path = (
        FROZEN_INPUT_ROOT / "R10_learn" / "D01" / "manifests" / "frozen_governor_config_for_r11.json"
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
