from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flight_config import CONTROLLER_ROOT

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconSample  # noqa: E402


@dataclass(frozen=True)
class ViconFrameStatus:
    valid: bool
    reason: str
    frame_number: int = -1
    vicon_latency_s: float = 0.0


class LiveNausicaaViconRigidBody:
    """Minimal Vicon rigid-body reader for one subject named Nausicaa."""

    def __init__(self, *, host: str, subject_name: str = "Nausicaa") -> None:
        self.host = str(host)
        self.subject_name = str(subject_name)
        self.client = None
        self.root_segment_name = ""

    def open(self) -> "LiveNausicaaViconRigidBody":
        try:
            from vicon_dssdk import ViconDataStream  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Vicon Python SDK is unavailable. The copied SDK should live under "
                "04_Flight_Test/01_Runtime/vicon_dssdk."
            ) from exc
        client = ViconDataStream.Client()
        client.Connect(self.host)
        client.SetBufferSize(1)
        client.EnableSegmentData()
        client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
        client.SetAxisMapping(
            ViconDataStream.Client.AxisMapping.EForward,
            ViconDataStream.Client.AxisMapping.ELeft,
            ViconDataStream.Client.AxisMapping.EUp,
        )
        root = client.GetSubjectRootSegmentName(self.subject_name)
        self.root_segment_name = str(root)
        self.client = client
        return self

    def read_latest(self) -> tuple[NausicaaViconSample | None, ViconFrameStatus]:
        if self.client is None:
            raise RuntimeError("Vicon reader is not open.")
        try:
            self.client.GetFrame()
            frame_number = int(self.client.GetFrameNumber())
            translation, translation_occluded = self.client.GetSegmentGlobalTranslation(
                self.subject_name,
                self.root_segment_name,
            )
            euler, euler_occluded = self.client.GetSegmentGlobalRotationEulerXYZ(
                self.subject_name,
                self.root_segment_name,
            )
            quaternion, quaternion_occluded = self.client.GetSegmentGlobalRotationQuaternion(
                self.subject_name,
                self.root_segment_name,
            )
            if bool(translation_occluded) or bool(euler_occluded) or bool(quaternion_occluded):
                return None, ViconFrameStatus(False, "vicon_subject_occluded", frame_number=frame_number)
            latency_s = float(self.client.GetLatencyTotal())
            timestamp_s = time.perf_counter() - latency_s
            sample = NausicaaViconSample(
                timestamp_s=timestamp_s,
                position_m=tuple(float(value) / 1000.0 for value in translation),
                euler_rad=tuple(float(value) for value in euler),
                quaternion_xyzw=tuple(float(value) for value in quaternion),
                vicon_latency_s=latency_s,
            )
            return sample, ViconFrameStatus(True, "ok", frame_number=frame_number, vicon_latency_s=latency_s)
        except Exception as exc:
            return None, ViconFrameStatus(False, f"vicon_read_failed:{type(exc).__name__}:{exc}")

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.Disconnect()
            except Exception:
                pass
            self.client = None


class ReplayNausicaaViconRigidBody:
    """Deterministic hardware-free source used by dry-run and tests."""

    def __init__(self, *, dt_s: float = 0.02, speed_m_s: float = 6.0) -> None:
        self.dt_s = float(dt_s)
        self.speed_m_s = float(speed_m_s)
        self.index = 0

    def open(self) -> "ReplayNausicaaViconRigidBody":
        return self

    def read_latest(self) -> tuple[NausicaaViconSample, ViconFrameStatus]:
        t = self.index * self.dt_s
        self.index += 1
        x = 1.2 + self.speed_m_s * t
        y = 2.2
        z = 1.7 + 0.02 * math.sin(2.0 * math.pi * 0.5 * t)
        sample = NausicaaViconSample(
            timestamp_s=t,
            position_m=(x, y, z),
            euler_rad=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            vicon_latency_s=0.0,
        )
        return sample, ViconFrameStatus(True, "replay", frame_number=self.index)

    def close(self) -> None:
        pass


def state_speed_m_s(state: np.ndarray) -> float:
    velocity = np.asarray(state, dtype=float)[6:9]
    return float(np.linalg.norm(velocity))
