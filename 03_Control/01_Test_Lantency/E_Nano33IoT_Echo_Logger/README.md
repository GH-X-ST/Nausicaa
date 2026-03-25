# Nano 33 IoT Echo Logger

This folder contains the standalone Nano 33 IoT firmware used by [Arduino_Test.m](../Arduino_Test.m) and [Servo_Test.m](../Servo_Test.m) for wireless latency logging.

It is not compatible with MATLAB's `arduino()` / `servo()` support-package firmware. Uploading the sketch switches the board to the custom logger transport.

## Transport

- Protocol: one-way UDP datagrams
- Default board IP: `192.168.0.33`
- Default port: `9500`
- Firmware: `Nano33IoT_Echo_Logger_V4_UDP`

MATLAB sends command datagrams to the Nano. The Nano timestamps reception and application on-board, then emits telemetry datagrams back to MATLAB. The MATLAB test scripts save those telemetry packets as:

- `host_dispatch_log.csv`
- `host_sync_roundtrip.csv`
- `board_command_log.csv`
- `board_sync_log.csv`
- `arduino_echo_import.csv`

No request-reply loop or post-run board dump is required anymore.

Firmware `V4_UDP` is controller-oriented:

- MATLAB still sends one actuator-vector command per sample.
- The Nano drains queued UDP packets and applies only the newest valid vector command.
- Controller mode emits one compact `VECTOR_EVENT` per sample.
- Instrumentation mode emits richer per-surface `COMMAND_EVENT` telemetry.
- The binary vector command path avoids text parsing in the critical loop.

## Command Datagrams

Control datagrams remain ASCII. The low-overhead vector command uses a compact binary payload.

### `HELLO`

Outbound:

```text
HELLO
```

Telemetry:

```text
HELLO_EVENT,Nano33IoT_Echo_Logger_V4_UDP,<board_ip>,9500,<telemetry_mode>,<board_now_us>
```

### `STATUS`

Outbound:

```text
STATUS
```

Telemetry:

```text
STATUS_EVENT,telemetry_mode=<mode>,vector_event_count=<n>,command_event_count=<n>,sync_event_count=<n>,error_count=<n>,wifi_status=<code>
```

### `MODE`

Outbound:

```text
MODE,CONTROLLER
```

or

```text
MODE,INSTRUMENTATION
```

Telemetry:

```text
OK_EVENT,MODE_CONTROLLER
```

### `CLEAR_LOGS`

Outbound:

```text
CLEAR_LOGS
```

Telemetry:

```text
OK_EVENT,CLEAR_LOGS
```

### `SET_NEUTRAL`

Outbound:

```text
SET_NEUTRAL
```

Telemetry:

```text
OK_EVENT,SET_NEUTRAL
```

### `SYNC`

Outbound:

```text
SYNC,<sync_id>,<host_tx_us>
```

Telemetry:

```text
SYNC_EVENT,<sync_id>,<host_tx_us>,<board_rx_us>,<board_tx_us>
```

### `SET`

Outbound:

```text
SET,<surface_name>,<command_sequence>,<position_norm>
```

Telemetry:

```text
COMMAND_EVENT,<surface_name>,<command_sequence>,<board_rx_us>,<apply_us>,<applied_position>,<pulse_us>
```

### `SET_ALL`

Outbound:

```text
SET_ALL,<sample_sequence>,<surface_count>,<surface_name_1>,<command_sequence_1>,<position_norm_1>,...
```

This legacy ASCII vector command is still accepted, but `Arduino_Test.m` now defaults to the compact binary vector command below.

### Binary Vector Command

Outbound:

```text
byte 0   : 'V'
byte 1   : surface_count
byte 2   : active_surface_mask
byte 3-6 : sample_sequence (uint32, little-endian)
byte 7.. : per-surface position_code (uint16, little-endian, Q0.16 over [0, 1])
```

For the default four-surface configuration the packet length is 15 bytes.

### Controller-Mode Telemetry

Telemetry:

```text
VECTOR_EVENT,<sample_sequence>,<active_surface_mask>,<board_rx_us>,<apply_us_1>,<position_code_1>,<pulse_us_1>,...
```

One compact `VECTOR_EVENT` is emitted per sample after the full actuator vector has been applied.

### Instrumentation-Mode Telemetry

Telemetry:

```text
COMMAND_EVENT,<surface_name>,<command_sequence>,<board_rx_us>,<apply_us>,<applied_position>,<pulse_us>
```

One `COMMAND_EVENT` is emitted per active surface after the full vector has already been applied.

## Timing Interpretation

- `board_rx_us` is common to all surfaces in a vector command.
- `apply_us` is captured per surface immediately after each servo write.
- In controller mode, telemetry overhead is reduced to one packet per sample.
- In instrumentation mode, telemetry remains surface-wise for richer inspection.
- Small surface-to-surface differences can still remain because servo writes are executed sequentially on the MCU.

## Upload

1. Open `Nano33IoT_Echo_Logger/Nano33IoT_Echo_Logger.ino` in Arduino IDE.
2. Select `Arduino Nano 33 IoT`.
3. Upload the sketch.
4. Open Serial Monitor at `115200`.
5. Confirm:

```text
Requesting static IP: 192.168.0.33
Connected to WiFi, IP: 192.168.0.33
Nano33IoT UDP echo logger ready.
```

## Smoke Test

Run [Smoke_Test.m](./Smoke_Test.m). It sends a UDP `HELLO` datagram and prints the `HELLO_EVENT` returned by the board.
