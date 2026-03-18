# Nano 33 IoT Echo Logger

This folder adds a standalone latency-measurement path for the Nano 33 IoT.

It is not compatible with MATLAB's `arduino()` / `servo()` support-package firmware. If you upload this sketch, you must drive the board with a custom TCP client such as MATLAB `tcpclient`.

## Purpose

The board timestamps:

- when a complete command line has been received
- when the servo output has been applied

It stores those timestamps locally in ring buffers and only dumps them after the run, so the inner loop is not disturbed by per-command echo traffic.

## TCP Protocol

Default port: `9500`

Default network behavior: the sketch is configured to request static IP `192.168.0.33`.

Commands are ASCII lines terminated by `\n`.

### `HELLO`

Request:

```text
HELLO
```

Reply:

```text
HELLO_REPLY,Nano33IoT_Echo_Logger_V1,<board_ip>,9500,<board_now_us>
```

### `STATUS`

Request:

```text
STATUS
```

Reply:

```text
STATUS_REPLY,command_log_count=<n>,command_log_overflow=<n>,sync_log_count=<n>,sync_log_overflow=<n>,wifi_status=<code>
```

### `CLEAR_LOGS`

Request:

```text
CLEAR_LOGS
```

Reply:

```text
OK,CLEAR_LOGS
```

### `SET_NEUTRAL`

Request:

```text
SET_NEUTRAL
```

Reply:

```text
OK,SET_NEUTRAL
```

### `SYNC`

Use this to align the host clock with the board clock.

Request:

```text
SYNC,<sync_id>,<host_tx_us>
```

Reply:

```text
SYNC_REPLY,<sync_id>,<host_tx_us>,<board_rx_us>,<board_tx_us>
```

`host_tx_us` should be a host-relative monotonic timestamp, not wall clock time. Use one host timer reference for the whole session.

### `SET`

Applies one servo command and logs it locally.

Request:

```text
SET,<surface_name>,<command_sequence>,<position_norm>
```

Example:

```text
SET,Aileron_L,17,0.625000
```

`position_norm` is expected in `[0, 1]`.

By default the sketch does not reply to each `SET`, so the command loop stays one-way.

### `DUMP_COMMAND_LOG`

Request:

```text
DUMP_COMMAND_LOG
```

Reply:

```text
#COMMAND_LOG_BEGIN,V1
#overflow_count=<n>
surface_name,command_sequence,rx_us,apply_us,receive_to_apply_us,applied_position,pulse_us
Aileron_L,1,123456,123470,14,0.500000,1500
...
#COMMAND_LOG_END
```

### `DUMP_SYNC_LOG`

Request:

```text
DUMP_SYNC_LOG
```

Reply:

```text
#SYNC_LOG_BEGIN,V1
#overflow_count=<n>
sync_id,host_tx_us,board_rx_us,board_tx_us,board_turnaround_us
1,20000,8412,8420,8
...
#SYNC_LOG_END
```

## Offline Latency Estimation

The board alone cannot know one-way MATLAB-to-Arduino latency, because the host and board clocks are different clocks.

The intended workflow is:

1. Host sends repeated `SYNC` requests before and after the run.
2. Host records `host_tx_us` and `host_rx_us`.
3. Board stores `board_rx_us` and `board_tx_us`.
4. Offline, fit a clock map from board time to host time using sync midpoints.
5. Convert `rx_us` or `apply_us` from the command dump into host time.
6. Subtract the host command dispatch timestamp.

For short tests, a linear fit is usually sufficient:

```text
host_mid_us  = 0.5 * (host_tx_us + host_rx_us)
board_mid_us = 0.5 * (board_rx_us + board_tx_us)
host_us ~= a * board_us + b
```

Then:

```text
arduino_echo_host_us = a * apply_us + b
computer_to_arduino_latency_s = (arduino_echo_host_us - command_dispatch_us) / 1e6
```

## Integration With The MATLAB Importer

The updated `Arduino_Test.m` and `Servo_Test.m` can now import post-processed Arduino echo files. The expected import columns are:

```text
surface_name
command_sequence
arduino_echo_time_s
computer_to_arduino_latency_s
applied_position
applied_equivalent_deg
```

If you build a CSV with those columns, you can point the test config at it with:

```matlab
config.arduinoEchoImport.filePath = "C:\path\to\arduino_echo_import.csv";
config.arduinoEchoImport.surfaceColumn = "surface_name";
config.arduinoEchoImport.sequenceColumn = "command_sequence";
config.arduinoEchoImport.echoTimeColumn = "arduino_echo_time_s";
config.arduinoEchoImport.latencyColumn = "computer_to_arduino_latency_s";
config.arduinoEchoImport.appliedPositionColumn = "applied_position";
config.arduinoEchoImport.appliedEquivalentDegreesColumn = "applied_equivalent_deg";
```

## Included MATLAB Helpers

This folder also includes:

- `Nano33IoT_Echo_Client_Example.m`
  It connects with `tcpclient`, performs sync bursts, sends `SET` commands, and saves:
  `host_dispatch_log.csv`, `host_sync_roundtrip.csv`, `board_command_log.csv`, `board_sync_log.csv`
- `Build_Arduino_Echo_Import_From_Dump.m`
  It reads those files, estimates the board-to-host clock map, and produces:
  `arduino_echo_import.csv`

Typical sequence:

```matlab
artifacts = Nano33IoT_Echo_Client_Example();

echoImport = Build_Arduino_Echo_Import_From_Dump(struct( ...
    "hostDispatchCsvPath", artifacts.dispatchLogPath, ...
    "syncRoundTripCsvPath", artifacts.syncRoundTripPath, ...
    "boardCommandLogCsvPath", artifacts.commandDumpPath));
```

## Sketch Configuration

Edit these constants before uploading:

- `Config::kWifiSsid`
- `Config::kWifiPassword`
- `Config::kUseStaticIp`
- `Config::kStaticIp`
- `Config::kStaticGateway`
- `Config::kStaticSubnet`
- `Config::kStaticDns`
- `Config::kSurfaceNames`
- `Config::kServoPins`
- `Config::kMinPulseUs`
- `Config::kMaxPulseUs`
- `Config::kNeutralPositions`

The current defaults match the four-surface layout already used in this repo.
The current network defaults request `192.168.0.33/24` with gateway and DNS at `192.168.0.1`, so change those if your LAN uses different values.
