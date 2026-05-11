from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_latency_config_records_real_flight_servo_cap() -> None:
    text = (REPO_ROOT / "C_Overall_Latency" / "defaultOverallLatencyConfig.m").read_text(
        encoding="utf-8"
    )

    assert "cfg.surfaceRangeDeg = [30.0, 30.0, 30.0, 30.0];" in text
    assert "cfg.servoCommandLimitNorm = [0.70, 0.70, 0.70, 0.70];" in text
    assert "cfg.benchDeflectionCalibrationLimitNorm = [1.00, 1.00, 1.00, 1.00];" in text


def test_run_control_path_contains_servo_limit_validation_and_packet_clamp() -> None:
    text = (REPO_ROOT / "C_Overall_Latency" / "Run_Control_Path.m").read_text(
        encoding="utf-8"
    )

    assert "Run_Control_Path:InvalidServoCommandLimit" in text
    assert "servoCommandLimitNorm must contain values in the interval (0, 1]." in text
    assert "cmd.surfaceNorm = min(max(cmd.surfaceNorm, -servoLimit), servoLimit);" in text
    assert "surfaceNorm = min(max(surfaceNorm, -servoLimit), servoLimit);" in text
    assert "packetSurfaceNorm = min(max(config.servoSigns .* surfaceNorm, -1), 1);" in text


def test_servo_limit_clip_contract() -> None:
    surface_norm = np.asarray([1.0, -1.0, 0.8, -0.8], dtype=float)
    servo_limit = np.asarray([0.70, 0.70, 0.70, 0.70], dtype=float)

    clipped = np.minimum(np.maximum(surface_norm, -servo_limit), servo_limit)

    np.testing.assert_allclose(clipped, [0.70, -0.70, 0.70, -0.70])

