from __future__ import annotations

import numpy as np

from command_contract import normalised_command_to_surface_rad
from real_flight_io import (
    NausicaaViconSample,
    NausicaaViconStateAdapter,
    aggregate_to_physical_surface_norm,
    encode_arduino_command_packet,
)
from state_contract import STATE_INDEX


def _code(value: float) -> int:
    return int(np.rint((float(value) + 1.0) * 0.5 * 65535.0))


def test_aggregate_command_expands_to_physical_surfaces() -> None:
    physical = aggregate_to_physical_surface_norm([0.21, -0.39, 0.62])

    np.testing.assert_allclose(physical, [0.2, -0.2, 0.6, -0.4])


def test_arduino_packet_matches_receiver_order_and_servo_signs() -> None:
    packet = encode_arduino_command_packet([0.21, -0.39, 0.62], sequence=42)

    assert len(packet.packet_bytes) == 15
    assert packet.packet_bytes[0] == ord("V")
    assert packet.packet_bytes[1] == 4
    assert int.from_bytes(packet.packet_bytes[3:7], byteorder="little") == 42
    np.testing.assert_allclose(packet.aggregate_command_norm, [0.2, -0.4, 0.6])
    np.testing.assert_allclose(packet.physical_surface_norm, [0.2, -0.2, 0.6, -0.4])
    np.testing.assert_allclose(packet.packet_surface_norm, [0.2, 0.2, 0.6, 0.4])
    assert packet.receiver_channel_codes == (
        _code(0.2),
        _code(0.2),
        _code(0.6),
        _code(0.4),
    )


def test_arduino_neutral_packet_uses_mid_codes() -> None:
    packet = encode_arduino_command_packet([0.0, 0.0, 0.0])

    assert packet.receiver_channel_codes == (32768, 32768, 32768, 32768)


def test_vicon_state_adapter_packs_canonical_state_and_derivatives() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        actuator_tau_s=(0.1, 0.1, 0.1),
    )
    first = NausicaaViconSample(
        timestamp_s=0.0,
        position_m=(1.0, 2.0, 1.5),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    second = NausicaaViconSample(
        timestamp_s=0.1,
        position_m=(1.1, 2.0, 1.5),
        euler_rad=(0.0, 0.0, 0.0),
    )

    adapter.update(first, command_norm=[0.0, 0.0, 0.0])
    state = adapter.update(second, command_norm=[1.0, 0.0, 0.0])

    assert state[STATE_INDEX["x_w"]] == 1.1
    assert np.isclose(state[STATE_INDEX["u"]], 1.0)
    assert state[STATE_INDEX["v"]] == 0.0
    assert state[STATE_INDEX["w"]] == 0.0
    assert state[STATE_INDEX["delta_a"]] > 0.0
    assert state[STATE_INDEX["delta_a"]] < normalised_command_to_surface_rad([1.0, 0.0, 0.0])[0]


def test_vicon_state_adapter_applies_measured_command_delay_before_surface_lag() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        actuator_tau_s=(0.01, 0.01, 0.01),
        command_delay_s=0.073,
    )
    adapter.update(
        NausicaaViconSample(0.0, (1.0, 2.0, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )
    delayed_state = adapter.update(
        NausicaaViconSample(0.05, (1.1, 2.0, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[1.0, 0.0, 0.0],
    )
    active_state = adapter.update(
        NausicaaViconSample(0.13, (1.2, 2.0, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[1.0, 0.0, 0.0],
    )

    assert delayed_state[STATE_INDEX["delta_a"]] == 0.0
    assert active_state[STATE_INDEX["delta_a"]] > 0.0
    assert adapter.estimator_status()["surface_command_delay_s"] == 0.073
