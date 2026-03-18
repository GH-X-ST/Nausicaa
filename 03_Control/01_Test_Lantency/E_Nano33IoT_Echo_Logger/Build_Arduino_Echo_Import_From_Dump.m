function importTable = Build_Arduino_Echo_Import_From_Dump(config)
%BUILD_ARDUINO_ECHO_IMPORT_FROM_DUMP Build a MATLAB-importable echo CSV.
%   This converts the standalone Nano logger outputs into the flat table
%   expected by Arduino_Test.m / Servo_Test.m via config.arduinoEchoImport.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);

hostDispatchLog = readtable(config.hostDispatchCsvPath);
syncRoundTripLog = readtable(config.syncRoundTripCsvPath);
boardCommandLog = readCommentCsv(config.boardCommandLogCsvPath);

validateHostDispatchLog(hostDispatchLog);
validateSyncRoundTripLog(syncRoundTripLog);
validateBoardCommandLog(boardCommandLog);

[clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog);
boardEchoUs = clockSlope .* boardCommandLog.(config.boardTimestampColumn) + clockIntercept;
boardReceiveUs = clockSlope .* boardCommandLog.rx_us + clockIntercept;

boardCommandLog.surface_name = reshape(string(boardCommandLog.surface_name), [], 1);
hostDispatchLog.surface_name = reshape(string(hostDispatchLog.surface_name), [], 1);
hostDispatchLog.command_sequence = reshape(double(hostDispatchLog.command_sequence), [], 1);
boardCommandLog.command_sequence = reshape(double(boardCommandLog.command_sequence), [], 1);

importTable = innerjoin( ...
    hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'}), ...
    boardCommandLog, ...
    'Keys', {'surface_name', 'command_sequence'});

matchedEchoUs = clockSlope .* importTable.(config.boardTimestampColumn) + clockIntercept;
matchedReceiveUs = clockSlope .* importTable.rx_us + clockIntercept;
latencyUs = matchedEchoUs - importTable.command_dispatch_us;
minimumLatencyUs = min(latencyUs);
if minimumLatencyUs < 0
    % Re-anchor the fitted clock so matched commands cannot precede dispatch.
    matchedEchoUs = matchedEchoUs - minimumLatencyUs;
    matchedReceiveUs = matchedReceiveUs - minimumLatencyUs;
    latencyUs = latencyUs - minimumLatencyUs;
end

importTable.arduino_echo_time_s = nan(height(importTable), 1);
importTable.arduino_receive_time_s = matchedReceiveUs ./ 1e6;
importTable.computer_to_arduino_latency_s = latencyUs ./ 1e6;
importTable.computer_to_arduino_receive_latency_s = ...
    (matchedReceiveUs - importTable.command_dispatch_us) ./ 1e6;
importTable.board_receive_to_apply_s = ...
    (double(importTable.apply_us) - double(importTable.rx_us)) ./ 1e6;

if ~ismember("applied_position", string(importTable.Properties.VariableNames))
    importTable.applied_position = nan(height(importTable), 1);
end

if ~ismember("applied_equivalent_deg", string(importTable.Properties.VariableNames))
    importTable.applied_equivalent_deg = nan(height(importTable), 1);
end

importTable = movevars(importTable, ...
    ["surface_name", "command_sequence", "arduino_echo_time_s", "computer_to_arduino_latency_s"], ...
    "Before", 1);

if strlength(config.outputCsvPath) > 0
    writetable(importTable, config.outputCsvPath);
end
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));

config.hostDispatchCsvPath = getTextScalarField( ...
    config, ...
    "hostDispatchCsvPath", ...
    fullfile(rootFolder, "output", "host_dispatch_log.csv"));
config.syncRoundTripCsvPath = getTextScalarField( ...
    config, ...
    "syncRoundTripCsvPath", ...
    fullfile(rootFolder, "output", "host_sync_roundtrip.csv"));
config.boardCommandLogCsvPath = getTextScalarField( ...
    config, ...
    "boardCommandLogCsvPath", ...
    fullfile(rootFolder, "output", "board_command_log.csv"));
config.outputCsvPath = getOptionalTextScalarField( ...
    config, ...
    "outputCsvPath", ...
    fullfile(rootFolder, "output", "arduino_echo_import.csv"));
config.boardTimestampColumn = getTextScalarField(config, "boardTimestampColumn", "apply_us");
end

function tableData = readCommentCsv(filePath)
opts = detectImportOptions(filePath, "FileType", "text", "CommentStyle", "#");
opts.CommentStyle = "#";
tableData = readtable(filePath, opts);
end

function [clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog)
% Calibrate against host send time and board receive time only.
% This avoids reply-side skew from delayed host reads.
hostTxUs = double(syncRoundTripLog.host_tx_us);
boardRxUs = double(syncRoundTripLog.board_rx_us);

if numel(hostTxUs) >= 2
    polynomialCoefficients = polyfit(boardRxUs, hostTxUs, 1);
    clockSlope = polynomialCoefficients(1);
    clockIntercept = polynomialCoefficients(2);
else
    clockSlope = 1.0;
    clockIntercept = hostTxUs(1) - boardRxUs(1);
end

syncForwardLatencyUs = clockSlope .* boardRxUs + clockIntercept - hostTxUs;
minimumForwardLatencyUs = min(syncForwardLatencyUs);
if minimumForwardLatencyUs < 0
    clockIntercept = clockIntercept - minimumForwardLatencyUs;
end
end

function validateHostDispatchLog(hostDispatchLog)
requiredColumns = ["surface_name", "command_sequence", "command_dispatch_us"];
assertHasColumns(hostDispatchLog, requiredColumns, "host dispatch log");
end

function validateSyncRoundTripLog(syncRoundTripLog)
requiredColumns = ["sync_id", "host_tx_us", "host_rx_us", "board_rx_us", "board_tx_us"];
assertHasColumns(syncRoundTripLog, requiredColumns, "sync round-trip log");
if height(syncRoundTripLog) < 1
    error("Build_Arduino_Echo_Import_From_Dump:EmptySyncLog", ...
        "At least one sync sample is required.");
end
end

function validateBoardCommandLog(boardCommandLog)
requiredColumns = ["surface_name", "command_sequence", "rx_us", "apply_us", "applied_position"];
assertHasColumns(boardCommandLog, requiredColumns, "board command log");
end

function assertHasColumns(tableData, requiredColumns, tableLabel)
variableNames = string(tableData.Properties.VariableNames);
missingColumns = requiredColumns(~ismember(requiredColumns, variableNames));
if ~isempty(missingColumns)
    error("Build_Arduino_Echo_Import_From_Dump:MissingColumns", ...
        "The %s is missing: %s", ...
        tableLabel, ...
        char(join(missingColumns, ", ")));
end
end

function defaultValue = getFieldOrDefault(config, fieldName, defaultValue)
if isfield(config, fieldName)
    defaultValue = config.(fieldName);
end
end

function value = getTextScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if ischar(value)
    value = string(value);
end

if ~(isstring(value) && isscalar(value))
    error("Build_Arduino_Echo_Import_From_Dump:InvalidConfigType", "%s must be a text scalar.", fieldName);
end
end

function value = getOptionalTextScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = defaultValue;
end

if ischar(value)
    value = string(value);
end

if ~(isstring(value) && isscalar(value))
    error("Build_Arduino_Echo_Import_From_Dump:InvalidConfigType", "%s must be a text scalar.", fieldName);
end
end
