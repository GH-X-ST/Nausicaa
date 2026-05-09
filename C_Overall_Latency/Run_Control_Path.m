function runData = Run_Control_Path(config, commandFcn)
%RUN_CONTROL_PATH Fixed MATLAB runtime command path for surface experiments.
arguments
    config (1,1) struct
    commandFcn (1,1) function_handle
end

config = normalizeRuntimeConfig(config);
runData = initializeRunData(config);

serialObj = [];
tracker = [];
serialRxBuffer = "";
caughtException = [];
profileState = struct();
commandRows = emptyCommandRowsStruct();
serialRows = emptySerialRowsStruct();
rawLogParts = {};
stateLogParts = {};
sampleIndex = 0;
runClock = tic;

try
    serialObj = openNanoSerial(config);
    [serialRows, serialRxBuffer] = sendAsciiCommand(serialObj, "HELLO", runClock, serialRows, serialRxBuffer, 0.75);
    [serialRows, serialRxBuffer] = sendAsciiCommand(serialObj, "SET_NEUTRAL", runClock, serialRows, serialRxBuffer, 0.25);

    tracker = Vicon(config.vicon);
    tracker.calibrateNeutral(runClock, config.neutralDurationSeconds);

    nextCommandTimeS = toc(runClock);

    while toc(runClock) < config.totalRunSeconds
        [rawRows, stateRows, viconSample] = tracker.readLatest(runClock);
        if ~isempty(rawRows)
            rawLogParts{end + 1} = rawRows; %#ok<AGROW>
        end
        if ~isempty(stateRows)
            stateLogParts{end + 1} = stateRows; %#ok<AGROW>
        end

        [serialRows, serialRxBuffer] = drainSerialTelemetry(serialObj, runClock, serialRows, serialRxBuffer);

        nowS = toc(runClock);
        if nowS >= nextCommandTimeS
            [cmd, profileState] = commandFcn(nowS, viconSample, config, profileState);
            cmd = normalizeCommand(cmd, sampleIndex, config);
            [packetBytes, packetSurfaceNorm, packetCodes, activeMask] = encodeSurfaceCommand(cmd, config);

            writeStartHostS = toc(runClock);
            write(serialObj, packetBytes, "uint8");
            writeStopHostS = toc(runClock);

            sampleIndex = sampleIndex + 1;
            commandRows(end + 1) = buildCommandRow( ... %#ok<AGROW>
                sampleIndex, cmd, nextCommandTimeS, writeStartHostS, writeStopHostS, ...
                packetSurfaceNorm, packetCodes, activeMask, config);

            nextCommandTimeS = nextCommandTimeS + config.commandDtSeconds;
            if nextCommandTimeS < nowS - config.commandDtSeconds
                missedPeriods = floor((nowS - nextCommandTimeS) / config.commandDtSeconds);
                nextCommandTimeS = nextCommandTimeS + missedPeriods * config.commandDtSeconds;
            end
        end

        pause(0.0005);
    end

    runData.runInfo.status = "completed";
    runData.runInfo.reason = "";
catch executionException
    caughtException = executionException;
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
end

try
    if ~isempty(serialObj)
        [commandRows, ~] = writeNeutralVector(serialObj, runClock, commandRows, sampleIndex, config);
        [serialRows, serialRxBuffer] = sendAsciiCommand(serialObj, "SET_NEUTRAL", runClock, serialRows, serialRxBuffer, 0.10);
    end
catch cleanupException
    runData.runInfo.cleanupWarning = string(cleanupException.message);
end

try
    if ~isempty(serialObj)
        [serialRows, ~] = drainSerialTelemetry(serialObj, runClock, serialRows, serialRxBuffer);
        clear serialObj
    end
catch
end

try
    if ~isempty(tracker)
        tracker.close();
    end
catch
end

runData.commandLog = commandRowsToTable(commandRows);
runData.viconRawLog = vertcatOrEmpty(rawLogParts, emptyViconRawTable());
runData.viconStateLog = vertcatOrEmpty(stateLogParts, emptyViconStateTable());
runData.serialTelemetryRaw = serialRowsToTable(serialRows);
runData.eventTable = extractEventTable(profileState);
runData.runInfo.stopTime = datetime("now", "TimeZone", "local");
runData.runInfo.elapsedHostSeconds = toc(runClock);
if isfield(profileState, "profileInfo")
    runData.runInfo.profileInfo = profileState.profileInfo;
end

runData = saveRunData(runData);

if ~isempty(caughtException)
    rethrow(caughtException);
end
end

function config = normalizeRuntimeConfig(config)
modeWasMissing = ~isfield(config, "mode") || strlength(string(config.mode)) == 0;
if modeWasMissing
    config.mode = "latency";
else
    config.mode = lower(string(config.mode));
end

activeWasSet = isfield(config, "activeCommandSeconds");
viconPortWasSet = isfield(config, "viconPort");
config.serialPort = getText(config, "serialPort", "COM11");
config.baudRate = getPositiveScalar(config, "baudRate", 1000000);
config.commandDtSeconds = getPositiveScalar(config, "commandDtSeconds", 0.02);
config.neutralLeadSeconds = getNonnegativeScalar(config, "neutralLeadSeconds", 5.0);
config.activeCommandSeconds = getPositiveScalar(config, "activeCommandSeconds", 90.0);
config.neutralTailSeconds = getNonnegativeScalar(config, "neutralTailSeconds", 5.0);
config.neutralDurationSeconds = getPositiveScalar(config, "neutralDurationSeconds", 5.0);
config.randomSeed = getScalar(config, "randomSeed", 2);
config.surfaceNames = getStringRow(config, "surfaceNames", ["Aileron_L", "Aileron_R", "Rudder", "Elevator"]);
config.surfaceOrder = getStringRow(config, "surfaceOrder", config.surfaceNames);
config.receiverChannelSurfaceOrder = getStringRow(config, "receiverChannelSurfaceOrder", ["Aileron_R", "Aileron_L", "Elevator", "Rudder"]);
config.surfaceEulerAxes = getStringRow(config, "surfaceEulerAxes", ["X", "X", "X", "X"]);
config.servoSigns = getNumericRow(config, "servoSigns", [1, 1, 1, 1]);
config.surfaceRangeDeg = getNumericRow(config, "surfaceRangeDeg", [30, 30, 30, 30]);
config.viconHostName = getText(config, "viconHostName", "localhost");
config.viconPort = getPositiveScalar(config, "viconPort", 801);
if any(char(config.viconHostName) == ':')
    hostParts = split(config.viconHostName, ":");
    config.viconHostName = hostParts(1);
    parsedPort = str2double(hostParts(end));
    if ~viconPortWasSet && isfinite(parsedPort) && parsedPort > 0
        config.viconPort = parsedPort;
    end
end
config.outputRoot = getText(config, "outputRoot", fullfile("C_Overall_Latency", "data", "raw"));
config.processedRoot = getText(config, "processedRoot", fullfile("C_Overall_Latency", "data", "processed"));
config.saveCsv = getLogical(config, "saveCsv", true);
config.makePlots = getLogical(config, "makePlots", false);
config.serialTimeoutSeconds = getPositiveScalar(config, "serialTimeoutSeconds", 0.05);
config.aeroCommandOrder = ["delta_a_cmd", "delta_e_cmd", "delta_r_cmd"];

surfaceCount = numel(config.surfaceOrder);
if surfaceCount ~= 4
    error("Run_Control_Path:InvalidSurfaceOrder", ...
        "surfaceOrder must contain exactly four physical surfaces for the Nano packet.");
end
config.surfaceEulerAxes = resizeRow(config.surfaceEulerAxes, surfaceCount, "X");
config.servoSigns = resizeRow(config.servoSigns, surfaceCount, 1);
config.surfaceRangeDeg = resizeRow(config.surfaceRangeDeg, surfaceCount, 30);
config.receiverChannelSurfaceOrder = validateReceiverChannelSurfaceOrder(config.receiverChannelSurfaceOrder, config.surfaceOrder);
config.receiverChannelSurfaceIndex = mapSurfaceNamesToIndices(config.receiverChannelSurfaceOrder, config.surfaceOrder);

if config.mode == "deflection" && ~activeWasSet
    holdSeconds = getPositiveScalar(config, "deflectionHoldSeconds", 0.75);
    enabledCount = numel(getFieldOrDefault(config, "enabledSurfaceIndices", 1:surfaceCount));
    config.activeCommandSeconds = max(config.activeCommandSeconds, enabledCount * 34 * holdSeconds);
end
if config.mode == "latency"
    config.eventHoldSeconds = getPositiveScalar(config, "eventHoldSeconds", 0.50);
end

config.totalRunSeconds = config.neutralLeadSeconds + config.activeCommandSeconds + config.neutralTailSeconds;
config.runLabel = getText(config, "runLabel", string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_" + config.mode);
config.outputFolder = getText(config, "outputFolder", fullfile(config.outputRoot, config.runLabel));

config.viconRawSubjectNames = getViconStringRow(config, "rawSubjectNames", config.surfaceOrder);
config.viconSurfaceSubjectNames = getViconStringRow(config, "surfaceSubjectNames", config.surfaceOrder);
config.bodySubjectName = getViconText(config, "bodySubjectName", "");
config.viconHostAndPort = config.viconHostName + ":" + string(round(config.viconPort));
config.vicon = buildViconConfig(config);
config.surfaceMapping = buildSurfaceMapping(config);

if modeWasMissing
    config.profileMode = "latency";
else
    config.profileMode = config.mode;
end
end

function viconConfig = buildViconConfig(config)
viconConfig = getFieldOrDefault(config, "vicon", struct());
viconConfig.hostName = getFieldOrDefault(viconConfig, "hostName", config.viconHostName);
viconConfig.port = getFieldOrDefault(viconConfig, "port", config.viconPort);
viconConfig.hostAndPort = getFieldOrDefault(viconConfig, "hostAndPort", config.viconHostAndPort);
viconConfig.rawSubjectNames = getFieldOrDefault(viconConfig, "rawSubjectNames", config.viconRawSubjectNames);
viconConfig.surfaceSubjectNames = getFieldOrDefault(viconConfig, "surfaceSubjectNames", config.viconSurfaceSubjectNames);
viconConfig.surfaceOrder = config.surfaceOrder;
viconConfig.surfaceEulerAxes = config.surfaceEulerAxes;
viconConfig.bodySubjectName = getFieldOrDefault(viconConfig, "bodySubjectName", config.bodySubjectName);
viconConfig.axisMapping = getFieldOrDefault(viconConfig, "axisMapping", "ZUp");
viconConfig.streamMode = getFieldOrDefault(viconConfig, "streamMode", "ServerPush");
viconConfig.connectTimeoutSeconds = getFieldOrDefault(viconConfig, "connectTimeoutSeconds", 5.0);
viconConfig.connectRetryPauseSeconds = getFieldOrDefault(viconConfig, "connectRetryPauseSeconds", 0.25);
viconConfig.maxConnectionAttempts = getFieldOrDefault(viconConfig, "maxConnectionAttempts", 3);
end

function mapping = buildSurfaceMapping(config)
mapping = struct();
mapping.physicalSurfaceOrder = config.surfaceOrder;
mapping.receiverChannelSurfaceOrder = config.receiverChannelSurfaceOrder;
mapping.receiverChannelSurfaceIndex = config.receiverChannelSurfaceIndex;
mapping.receiverChannels = 1:numel(config.receiverChannelSurfaceOrder);
mapping.aeroCommandOrder = config.aeroCommandOrder;
mapping.surfaceSignConvention = "servoSigns are applied in MATLAB before packet encoding; values are hardware sign-check assumptions.";
mapping.packetConvention = "Binary packet codes are written in receiverChannelSurfaceOrder; command vectors remain in physicalSurfaceOrder.";
mapping.physicalFromAeroMatrix = [ ...
    1, 0, 0; ...
   -1, 0, 0; ...
    0, 0, 1; ...
    0, 1, 0];
mapping.physicalFromAeroRows = config.surfaceOrder;
mapping.physicalFromAeroColumns = config.aeroCommandOrder;
end

function channelSurfaceOrder = validateReceiverChannelSurfaceOrder(channelSurfaceOrder, surfaceOrder)
channelSurfaceOrder = reshape(string(channelSurfaceOrder), 1, []);
surfaceOrder = reshape(string(surfaceOrder), 1, []);
if numel(channelSurfaceOrder) ~= numel(surfaceOrder)
    error("Run_Control_Path:InvalidReceiverChannelSurfaceOrder", ...
        "receiverChannelSurfaceOrder must contain exactly the same number of entries as surfaceOrder.");
end

missingSurfaces = setdiff(surfaceOrder, channelSurfaceOrder, "stable");
extraSurfaces = setdiff(channelSurfaceOrder, surfaceOrder, "stable");
if ~isempty(missingSurfaces) || ~isempty(extraSurfaces) || numel(unique(channelSurfaceOrder, "stable")) ~= numel(channelSurfaceOrder)
    error("Run_Control_Path:InvalidReceiverChannelSurfaceOrder", ...
        "receiverChannelSurfaceOrder must be an exact permutation of surfaceOrder.");
end
end

function surfaceIndices = mapSurfaceNamesToIndices(channelSurfaceOrder, surfaceOrder)
surfaceIndices = zeros(1, numel(channelSurfaceOrder));
for channelIndex = 1:numel(channelSurfaceOrder)
    matchIndex = find(surfaceOrder == channelSurfaceOrder(channelIndex), 1, "first");
    if isempty(matchIndex)
        error("Run_Control_Path:InvalidReceiverChannelSurfaceOrder", ...
            "Receiver channel surface '%s' is not present in surfaceOrder.", ...
            char(channelSurfaceOrder(channelIndex)));
    end
    surfaceIndices(channelIndex) = matchIndex;
end
end

function runData = initializeRunData(config)
runData = struct();
runData.config = config;
runData.commandLog = emptyCommandTable();
runData.viconRawLog = emptyViconRawTable();
runData.viconStateLog = emptyViconStateTable();
runData.serialTelemetryRaw = emptySerialTable();
runData.eventTable = table();
runData.outputFiles = struct();
runData.runInfo = struct( ...
    "status", "initialized", ...
    "reason", "", ...
    "startTime", datetime("now", "TimeZone", "local"), ...
    "stopTime", NaT, ...
    "elapsedHostSeconds", NaN, ...
    "cleanupWarning", "");
end

function serialObj = openNanoSerial(config)
serialObj = serialport(char(config.serialPort), config.baudRate, "Timeout", config.serialTimeoutSeconds);
configureTerminator(serialObj, "LF");
flush(serialObj);
end

function [serialRows, serialRxBuffer] = sendAsciiCommand(serialObj, commandText, runClock, serialRows, serialRxBuffer, waitSeconds)
writeline(serialObj, commandText);
waitStartS = toc(runClock);
while toc(runClock) - waitStartS < waitSeconds
    [serialRows, serialRxBuffer] = drainSerialTelemetry(serialObj, runClock, serialRows, serialRxBuffer);
    pause(0.005);
end
end

function [serialRows, serialRxBuffer] = drainSerialTelemetry(serialObj, runClock, serialRows, serialRxBuffer)
if isempty(serialObj) || serialObj.NumBytesAvailable <= 0
    return;
end

rawBytes = read(serialObj, serialObj.NumBytesAvailable, "uint8");
readTimeS = toc(runClock);
serialRxBuffer = serialRxBuffer + string(char(rawBytes(:).'));
serialRxBuffer = replace(serialRxBuffer, sprintf("\r"), "");
parts = split(serialRxBuffer, sprintf("\n"));
if strlength(serialRxBuffer) > 0 && ~endsWith(serialRxBuffer, sprintf("\n"))
    completeParts = parts(1:end - 1);
    serialRxBuffer = parts(end);
else
    completeParts = parts;
    serialRxBuffer = "";
end

for lineIndex = 1:numel(completeParts)
    lineText = strtrim(completeParts(lineIndex));
    if strlength(lineText) == 0
        continue;
    end
    serialRows(end + 1) = struct( ... %#ok<AGROW>
        "t_read_host_s", readTimeS, ...
        "line_text", lineText);
end
end

function [commandRows, sampleIndex] = writeNeutralVector(serialObj, runClock, commandRows, sampleIndex, config)
cmd = struct( ...
    "eventId", NaN, ...
    "sequence", sampleIndex, ...
    "surfaceNorm", zeros(1, numel(config.surfaceOrder)), ...
    "aeroCmdRad", NaN(1, 3), ...
    "activeSurfaceIndex", NaN, ...
    "activeSurfaceName", "", ...
    "commandLevelNorm", 0, ...
    "description", "cleanup_neutral");
[packetBytes, packetSurfaceNorm, packetCodes, activeMask] = encodeSurfaceCommand(cmd, config);
writeStartHostS = toc(runClock);
write(serialObj, packetBytes, "uint8");
writeStopHostS = toc(runClock);
sampleIndex = sampleIndex + 1;
commandRows(end + 1) = buildCommandRow( ...
    sampleIndex, cmd, writeStartHostS, writeStartHostS, writeStopHostS, ...
    packetSurfaceNorm, packetCodes, activeMask, config);
end

function cmd = normalizeCommand(cmd, fallbackSequence, config)
if ~isfield(cmd, "eventId")
    cmd.eventId = NaN;
end
if ~isfield(cmd, "sequence") || ~isfinite(double(cmd.sequence))
    cmd.sequence = fallbackSequence;
end
if ~isfield(cmd, "surfaceNorm")
    cmd.surfaceNorm = zeros(1, numel(config.surfaceOrder));
end
cmd.surfaceNorm = resizeRow(double(cmd.surfaceNorm), numel(config.surfaceOrder), 0);
cmd.surfaceNorm = min(max(cmd.surfaceNorm, -1), 1);
if ~isfield(cmd, "aeroCmdRad")
    cmd.aeroCmdRad = NaN(1, 3);
end
cmd.aeroCmdRad = resizeRow(double(cmd.aeroCmdRad), 3, NaN);
if ~isfield(cmd, "activeSurfaceIndex")
    cmd.activeSurfaceIndex = NaN;
end
if ~isfield(cmd, "activeSurfaceName")
    cmd.activeSurfaceName = "";
end
if ~isfield(cmd, "commandLevelNorm")
    cmd.commandLevelNorm = NaN;
end
if ~isfield(cmd, "description")
    cmd.description = "";
end
end

function [packetBytes, packetSurfaceNorm, packetCodes, activeMask] = encodeSurfaceCommand(cmd, config)
surfaceNorm = resizeRow(double(cmd.surfaceNorm), numel(config.surfaceOrder), 0);
packetSurfaceNorm = min(max(config.servoSigns .* surfaceNorm, -1), 1);
packetCodes = uint16(round((packetSurfaceNorm + 1) .* 0.5 .* 65535));
channelPacketCodes = packetCodes(config.receiverChannelSurfaceIndex);

activeMask = uint8(0);
if isfield(cmd, "activeSurfaceMask") && isfinite(double(cmd.activeSurfaceMask))
    activeMask = physicalMaskToReceiverChannelMask(uint8(cmd.activeSurfaceMask), config.receiverChannelSurfaceIndex);
elseif isfield(cmd, "activeSurfaceIndex") && isfinite(double(cmd.activeSurfaceIndex))
    activeIndex = round(double(cmd.activeSurfaceIndex));
    if activeIndex >= 1 && activeIndex <= numel(config.surfaceOrder)
        channelIndex = find(config.receiverChannelSurfaceIndex == activeIndex, 1, "first");
        activeMask = bitshift(uint8(1), channelIndex - 1);
    end
end

packetBytes = zeros(1, 15, "uint8");
packetBytes(1) = uint8('V');
packetBytes(2) = uint8(4);
packetBytes(3) = activeMask;
packetBytes(4:7) = encodeUint32LittleEndian(uint32(max(0, round(double(cmd.sequence)))));
writeIndex = 8;
for channelIndex = 1:4
    packetBytes(writeIndex:writeIndex + 1) = encodeUint16LittleEndian(channelPacketCodes(channelIndex));
    writeIndex = writeIndex + 2;
end
end

function channelMask = physicalMaskToReceiverChannelMask(physicalMask, receiverChannelSurfaceIndex)
channelMask = uint8(0);
for channelIndex = 1:numel(receiverChannelSurfaceIndex)
    physicalIndex = receiverChannelSurfaceIndex(channelIndex);
    if bitand(physicalMask, bitshift(uint8(1), physicalIndex - 1)) ~= 0
        channelMask = bitor(channelMask, bitshift(uint8(1), channelIndex - 1));
    end
end
end

function row = buildCommandRow(sampleIndex, cmd, scheduledTimeS, writeStartHostS, writeStopHostS, packetSurfaceNorm, packetCodes, activeMask, config)
surfaceNorm = resizeRow(double(cmd.surfaceNorm), 4, 0);
aeroCmdRad = resizeRow(double(cmd.aeroCmdRad), 3, NaN);
row = struct( ...
    "sample_index", sampleIndex, ...
    "sample_sequence", double(cmd.sequence), ...
    "event_id", double(cmd.eventId), ...
    "scheduled_time_s", scheduledTimeS, ...
    "write_start_host_s", writeStartHostS, ...
    "write_stop_host_s", writeStopHostS, ...
    "active_surface_index", double(cmd.activeSurfaceIndex), ...
    "active_surface_name", string(cmd.activeSurfaceName), ...
    "active_surface_mask", double(activeMask), ...
    "surface_norm_aileron_l", surfaceNorm(1), ...
    "surface_norm_aileron_r", surfaceNorm(2), ...
    "surface_norm_rudder", surfaceNorm(3), ...
    "surface_norm_elevator", surfaceNorm(4), ...
    "packet_surface_norm_aileron_l", packetSurfaceNorm(1), ...
    "packet_surface_norm_aileron_r", packetSurfaceNorm(2), ...
    "packet_surface_norm_rudder", packetSurfaceNorm(3), ...
    "packet_surface_norm_elevator", packetSurfaceNorm(4), ...
    "packet_code_aileron_l", double(packetCodes(1)), ...
    "packet_code_aileron_r", double(packetCodes(2)), ...
    "packet_code_rudder", double(packetCodes(3)), ...
    "packet_code_elevator", double(packetCodes(4)), ...
    "delta_a_cmd_rad", aeroCmdRad(1), ...
    "delta_e_cmd_rad", aeroCmdRad(2), ...
    "delta_r_cmd_rad", aeroCmdRad(3), ...
    "profile_mode", string(config.mode), ...
    "random_seed", double(config.randomSeed), ...
    "description", string(cmd.description));
end

function rows = emptyCommandRowsStruct()
rows = struct( ...
    "sample_index", {}, ...
    "sample_sequence", {}, ...
    "event_id", {}, ...
    "scheduled_time_s", {}, ...
    "write_start_host_s", {}, ...
    "write_stop_host_s", {}, ...
    "active_surface_index", {}, ...
    "active_surface_name", {}, ...
    "active_surface_mask", {}, ...
    "surface_norm_aileron_l", {}, ...
    "surface_norm_aileron_r", {}, ...
    "surface_norm_rudder", {}, ...
    "surface_norm_elevator", {}, ...
    "packet_surface_norm_aileron_l", {}, ...
    "packet_surface_norm_aileron_r", {}, ...
    "packet_surface_norm_rudder", {}, ...
    "packet_surface_norm_elevator", {}, ...
    "packet_code_aileron_l", {}, ...
    "packet_code_aileron_r", {}, ...
    "packet_code_rudder", {}, ...
    "packet_code_elevator", {}, ...
    "delta_a_cmd_rad", {}, ...
    "delta_e_cmd_rad", {}, ...
    "delta_r_cmd_rad", {}, ...
    "profile_mode", {}, ...
    "random_seed", {}, ...
    "description", {});
end

function rows = emptySerialRowsStruct()
rows = struct( ...
    "t_read_host_s", {}, ...
    "line_text", {});
end

function encodedBytes = encodeUint16LittleEndian(value)
value = uint16(value);
encodedBytes = uint8([bitand(value, uint16(255)), bitshift(value, -8)]);
end

function encodedBytes = encodeUint32LittleEndian(value)
value = uint32(value);
encodedBytes = uint8([ ...
    bitand(value, uint32(255)), ...
    bitand(bitshift(value, -8), uint32(255)), ...
    bitand(bitshift(value, -16), uint32(255)), ...
    bitshift(value, -24)]);
end

function eventTable = extractEventTable(profileState)
if isstruct(profileState) && isfield(profileState, "eventTable")
    eventTable = profileState.eventTable;
else
    eventTable = table();
end
end

function runData = saveRunData(runData)
ensureFolder(runData.config.outputFolder);
runData.outputFiles.runDataMat = string(fullfile(runData.config.outputFolder, "run_data.mat"));
save(runData.outputFiles.runDataMat, "runData");

if runData.config.saveCsv
    runData.outputFiles.commandLogCsv = string(fullfile(runData.config.outputFolder, "command_log.csv"));
    runData.outputFiles.viconRawLogCsv = string(fullfile(runData.config.outputFolder, "vicon_raw_log.csv"));
    runData.outputFiles.viconStateLogCsv = string(fullfile(runData.config.outputFolder, "vicon_state_log.csv"));
    runData.outputFiles.serialTelemetryCsv = string(fullfile(runData.config.outputFolder, "serial_telemetry_raw.csv"));
    runData.outputFiles.eventTableCsv = string(fullfile(runData.config.outputFolder, "event_table.csv"));
    writetable(runData.commandLog, runData.outputFiles.commandLogCsv);
    writetable(runData.viconRawLog, runData.outputFiles.viconRawLogCsv);
    writetable(runData.viconStateLog, runData.outputFiles.viconStateLogCsv);
    writetable(runData.serialTelemetryRaw, runData.outputFiles.serialTelemetryCsv);
    if ~isempty(runData.eventTable)
        writetable(runData.eventTable, runData.outputFiles.eventTableCsv);
    end
    save(runData.outputFiles.runDataMat, "runData");
end
end

function ensureFolder(folderPath)
if ~isfolder(folderPath)
    mkdir(folderPath);
end
end

function tableOut = commandRowsToTable(rows)
if isempty(rows)
    tableOut = emptyCommandTable();
else
    tableOut = struct2table(rows);
end
end

function tableOut = serialRowsToTable(rows)
if isempty(rows)
    tableOut = emptySerialTable();
else
    tableOut = struct2table(rows);
end
end

function tableOut = vertcatOrEmpty(parts, emptyTable)
if isempty(parts)
    tableOut = emptyTable;
else
    tableOut = vertcat(parts{:});
end
end

function tableOut = emptyCommandTable()
tableOut = table( ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), strings(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), zeros(0, 1), strings(0, 1), ...
    'VariableNames', { ...
    'sample_index', 'sample_sequence', 'event_id', 'scheduled_time_s', 'write_start_host_s', 'write_stop_host_s', ...
    'active_surface_index', 'active_surface_name', 'active_surface_mask', ...
    'surface_norm_aileron_l', 'surface_norm_aileron_r', 'surface_norm_rudder', 'surface_norm_elevator', ...
    'packet_surface_norm_aileron_l', 'packet_surface_norm_aileron_r', 'packet_surface_norm_rudder', 'packet_surface_norm_elevator', ...
    'packet_code_aileron_l', 'packet_code_aileron_r', 'packet_code_rudder', 'packet_code_elevator', ...
    'delta_a_cmd_rad', 'delta_e_cmd_rad', 'delta_r_cmd_rad', 'profile_mode', 'random_seed', 'description'});
end

function tableOut = emptyViconRawTable()
tableOut = table( ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), strings(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), false(0, 1), ...
    'VariableNames', {'frame_number', 't_read_host_s', 'vicon_latency_s', 't_capture_host_s', ...
    'subject_name', 'segment_name', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw', ...
    'roll_rad', 'pitch_rad', 'yaw_rad', 'is_occluded'});
end

function tableOut = emptyViconStateTable()
tableOut = table( ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), zeros(0, 1), strings(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), ...
    'VariableNames', {'frame_number', 't_capture_host_s', 'state_ready_host_s', 'subject_name', ...
    'surface_index', 'surface_name', 'surface_angle_rad', 'surface_angle_deg', ...
    'surface_rate_radps', 'surface_rate_degps', 'quality_flag'});
end

function tableOut = emptySerialTable()
tableOut = table(zeros(0, 1), strings(0, 1), 'VariableNames', {'t_read_host_s', 'line_text'});
end

function value = getFieldOrDefault(config, fieldName, defaultValue)
if isstruct(config) && isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
end

function value = getText(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if ischar(value)
    value = string(value);
end
if ~(isstring(value) && isscalar(value))
    error("Run_Control_Path:InvalidConfig", "%s must be a text scalar.", fieldName);
end
end

function value = getViconText(config, fieldName, defaultValue)
viconConfig = getFieldOrDefault(config, "vicon", struct());
value = getFieldOrDefault(config, fieldName, []);
if isempty(value)
    value = getFieldOrDefault(config, "vicon" + upper(extractBefore(fieldName, 2)) + extractAfter(fieldName, 1), []);
end
if isempty(value)
    value = getFieldOrDefault(viconConfig, fieldName, defaultValue);
end
if ischar(value)
    value = string(value);
end
if ~(isstring(value) && isscalar(value))
    value = string(defaultValue);
end
end

function value = getStringRow(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
value = string(value);
value = reshape(value, 1, []);
end

function value = getViconStringRow(config, fieldName, defaultValue)
viconConfig = getFieldOrDefault(config, "vicon", struct());
topFieldName = "vicon" + upper(extractBefore(fieldName, 2)) + extractAfter(fieldName, 1);
value = getFieldOrDefault(config, topFieldName, []);
if isempty(value)
    value = getFieldOrDefault(viconConfig, fieldName, defaultValue);
end
value = string(value);
value = reshape(value, 1, []);
end

function value = getNumericRow(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'vector'}, mfilename, fieldName);
value = reshape(double(value), 1, []);
end

function value = getPositiveScalar(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar', 'positive'}, mfilename, fieldName);
value = double(value);
end

function value = getNonnegativeScalar(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar', 'nonnegative'}, mfilename, fieldName);
value = double(value);
end

function value = getScalar(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar'}, mfilename, fieldName);
value = double(value);
end

function value = getLogical(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
value = logical(value);
end

function row = resizeRow(row, targetCount, fillValue)
row = reshape(row, 1, []);
if numel(row) < targetCount
    row = [row, repmat(fillValue, 1, targetCount - numel(row))];
elseif numel(row) > targetCount
    row = row(1:targetCount);
end
end
