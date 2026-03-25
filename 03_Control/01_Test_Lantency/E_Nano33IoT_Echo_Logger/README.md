# Nano 33 IoT Echo Logger

This folder contains the standalone Nano 33 IoT firmware used by [Arduino_Test.m](../Arduino_Test.m) and [Servo_Test.m](../Servo_Test.m) for wireless latency logging.

It is not compatible with MATLAB's `arduino()` / `servo()` support-package firmware. Uploading the sketch switches the board to the custom logger transport.

## Transport

- Protocol: one-way UDP datagrams
- Default board IP: `192.168.0.33`
- Default port: `9500`
- Firmware: `Nano33IoT_Echo_Logger_V3_UDP`

MATLAB sends command datagrams to the Nano. The Nano timestamps reception and application on-board, then emits telemetry datagrams back to MATLAB. The MATLAB test scripts save those telemetry packets as:

- `host_dispatch_log.csv`
- `host_sync_roundtrip.csv`
- `board_command_log.csv`
- `board_sync_log.csv`
- `arduino_echo_import.csv`

No request-reply loop or post-run board dump is required anymore.

For `SET_ALL`, firmware `V3_UDP` parses the whole actuator vector, applies all surfaces first, records the per-surface `apply_us` timestamps, and only then emits telemetry. This avoids inflating later-surface apply latency with UDP telemetry transmission time.

## Command Datagrams

Each UDP datagram carries one ASCII payload.

### `HELLO`

Outbound:

```text
HELLO
```

Telemetry:

```text
HELLO_EVENT,Nano33IoT_Echo_Logger_V3_UDP,<board_ip>,9500,<board_now_us>
```

### `STATUS`

Outbound:

```text
STATUS
```

Telemetry:

```text
STATUS_EVENT,command_log_count=<n>,command_log_overflow=<n>,sync_log_count=<n>,sync_log_overflow=<n>,wifi_status=<code>
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

Telemetry:

```text
COMMAND_EVENT,<surface_name>,<command_sequence>,<board_rx_us>,<apply_us>,<applied_position>,<pulse_us>
```

One `COMMAND_EVENT` is emitted per surface after the full vector has already been applied.

## Timing Interpretation

- `board_rx_us` is common to all surfaces in a `SET_ALL` datagram.
- `apply_us` is captured per surface immediately after each servo write.
- In `V3_UDP`, per-surface `apply_us` is no longer biased by telemetry output for earlier surfaces.
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
