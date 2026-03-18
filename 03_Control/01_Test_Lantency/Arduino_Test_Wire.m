function runData = Arduino_Test_Wire(config)
%ARDUINO_TEST_WIRE Execute an Arduino-only latency test over wired serial.

arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);
runData.runInfo.startTime = datetime("now");

assignin("base", "ArduinoTestWireRunData", runData);
assignin("base", "ArduinoTestWireLatestState", struct([]));

commandInterface = struct();
cleanupHandle = onCleanup(@() cleanupResources(commandInterface, config));

try
    [commandInterface, arduinoInfo] = connectToArduino(config);
    runData.connectionInfo.arduino = arduinoInfo;
    runData.config = config;
    assignin("base", "ArduinoTestWireRunData", runData);

    if ~arduinoInfo.isConnected
        runData.runInfo.status = "arduino_connection_failed";
        runData.runInfo.reason = arduinoInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ArduinoTestWireRunData", runData);
        printConnectionStatus(runData);
        return;
    end

    printConnectionStatus(runData);
    moveServosToNeutral(commandInterface);
    pause(config.neutralSettleSeconds);

    [storage, runInfo, config] = executeWireTest(commandInterface, config);
    runData.config = config;
    runData.runInfo = runInfo;
    runData.logs = buildLogTimetables(storage, config);
    runData.surfaceSummary = buildSurfaceSummary(storage, config);
    runData.artifacts = exportRunData(runData);

    assignin("base", "ArduinoTestWireRunData", runData);
    assignin("base", "ArduinoTestWireLatestState", buildLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupResources(commandInterface, config);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    assignin("base", "ArduinoTestWireRunData", runData);
    rethrow(executionException);
end
end

function loggerSession = startWireLoggerSession(serialObject, config, sampleCount, activeSurfaceCount)
dispatchCapacity = max(1, sampleCount .* activeSurfaceCount);
syncCapacity = max(1, config.arduinoTransport.syncCountBeforeRun + config.arduinoTransport.syncCountAfterRun + 8);
telemetryCapacity = dispatchCapacity + syncCapacity + 32;

loggerSession = struct( ...
    "hostTimer", tic, ...
    "testStartOffsetUs", uint32(0), ...
    "testStartOffsetSeconds", 0, ...
    "receiveBuffer", "", ...
    "dispatchCount", 0, ...
    "dispatchSurfaceName", strings(dispatchCapacity, 1), ...
    "dispatchCommandSequence", nan(dispatchCapacity, 1), ...
    "dispatchCommandUs", nan(dispatchCapacity, 1), ...
    "dispatchPosition", nan(dispatchCapacity, 1), ...
    "telemetryLineCount", 0, ...
    "telemetryLineText", strings(telemetryCapacity, 1), ...
    "telemetryHostRxUs", nan(telemetryCapacity, 1), ...
    "boardCommandCount", 0, ...
    "boardCommandSurfaceName", strings(dispatchCapacity, 1), ...
    "boardCommandSequence", nan(dispatchCapacity, 1), ...
    "boardCommandHostRxUs", nan(dispatchCapacity, 1), ...
    "boardCommandRxUs", nan(dispatchCapacity, 1), ...
    "boardCommandApplyUs", nan(dispatchCapacity, 1), ...
    "boardCommandAppliedPosition", nan(dispatchCapacity, 1), ...
    "boardCommandPulseUs", nan(dispatchCapacity, 1), ...
    "boardSyncCount", 0, ...
    "boardSyncId", nan(syncCapacity, 1), ...
    "boardSyncHostTxUs", nan(syncCapacity, 1), ...
    "boardSyncHostRxUs", nan(syncCapacity, 1), ...
    "boardSyncBoardRxUs", nan(syncCapacity, 1), ...
    "boardSyncBoardTxUs", nan(syncCapacity, 1));

if config.arduinoTransport.clearLogsBeforeRun
    sendControlBurst(serialObject, "CLEAR_LOGS", 3, 0.02);
end

for syncIndex = 1:config.arduinoTransport.syncCountBeforeRun
    writeline(serialObject, sprintf("SYNC,%u,%u", uint32(syncIndex), hostNowUs(loggerSession.hostTimer)));
    loggerSession = pauseAndDrain(serialObject, loggerSession, config.arduinoTransport.syncPauseSeconds, config.arduinoTransport.linePollPauseSeconds);
end

loggerSession.testStartOffsetUs = hostNowUs(loggerSession.hostTimer);
loggerSession.testStartOffsetSeconds = double(loggerSession.testStartOffsetUs) ./ 1e6;
end

function [dispatchTimesSeconds, writeStopSeconds, loggerSession] = writeServoPositions( ...
    serialObject, servoPositions, surfaceNames, activeSurfaceMask, commandSequenceNumbers, loggerSession)
dispatchTimesSeconds = nan(1, numel(surfaceNames));

for surfaceIndex = 1:numel(surfaceNames)
    if ~activeSurfaceMask(surfaceIndex)
        continue;
    end

    dispatchAbsoluteUs = hostNowUs(loggerSession.hostTimer);
    writeline(serialObject, sprintf("SET,%s,%u,%.6f", char(surfaceNames(surfaceIndex)), uint32(commandSequenceNumbers(surfaceIndex)), servoPositions(surfaceIndex)));

    loggerSession.dispatchCount = loggerSession.dispatchCount + 1;
    rowIndex = loggerSession.dispatchCount;
    loggerSession.dispatchSurfaceName(rowIndex) = surfaceNames(surfaceIndex);
    loggerSession.dispatchCommandSequence(rowIndex) = commandSequenceNumbers(surfaceIndex);
    loggerSession.dispatchCommandUs(rowIndex) = double(dispatchAbsoluteUs);
    loggerSession.dispatchPosition(rowIndex) = servoPositions(surfaceIndex);
    dispatchTimesSeconds(surfaceIndex) = max(0, (double(dispatchAbsoluteUs) - double(loggerSession.testStartOffsetUs)) ./ 1e6);
end

loggerSession = drainWireTelemetry(serialObject, loggerSession);
writeStopSeconds = max(0, toc(loggerSession.hostTimer) - loggerSession.testStartOffsetSeconds);
end

function [loggerSession, echoImportTable, config] = finalizeWireLoggerSession(serialObject, loggerSession, config)
if config.returnToNeutralOnExit
    sendControlBurst(serialObject, "SET_NEUTRAL", 2, 0.02);
end

loggerSession = pauseAndDrain(serialObject, loggerSession, config.arduinoTransport.postRunSettleSeconds, config.arduinoTransport.linePollPauseSeconds);

for syncIndex = 1:config.arduinoTransport.syncCountAfterRun
    syncId = uint32(config.arduinoTransport.syncCountBeforeRun + syncIndex);
    writeline(serialObject, sprintf("SYNC,%u,%u", syncId, hostNowUs(loggerSession.hostTimer)));
    loggerSession = pauseAndDrain(serialObject, loggerSession, config.arduinoTransport.syncPauseSeconds, config.arduinoTransport.linePollPauseSeconds);
end

loggerSession = collectRemainingTelemetry(serialObject, loggerSession, 2.0, config.arduinoTransport.lineIdleTimeoutSeconds);

hostDispatchLog = table( ...
    loggerSession.dispatchSurfaceName(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchCommandSequence(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchCommandUs(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchPosition(1:loggerSession.dispatchCount), ...
    'VariableNames', {'surface_name', 'command_sequence', 'command_dispatch_us', 'position_norm'});

boardCommandLog = table( ...
    loggerSession.boardCommandSurfaceName(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandSequence(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandHostRxUs(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandRxUs(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandApplyUs(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandApplyUs(1:loggerSession.boardCommandCount) - loggerSession.boardCommandRxUs(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandAppliedPosition(1:loggerSession.boardCommandCount), ...
    loggerSession.boardCommandPulseUs(1:loggerSession.boardCommandCount), ...
    'VariableNames', {'surface_name', 'command_sequence', 'host_rx_us', 'rx_us', 'apply_us', 'receive_to_apply_us', 'applied_position', 'pulse_us'});

boardSyncLog = table( ...
    loggerSession.boardSyncId(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncHostTxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncHostRxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardRxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardTxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardTxUs(1:loggerSession.boardSyncCount) - loggerSession.boardSyncBoardRxUs(1:loggerSession.boardSyncCount), ...
    'VariableNames', {'sync_id', 'host_tx_us', 'host_rx_us', 'board_rx_us', 'board_tx_us', 'board_turnaround_us'});

if isempty(boardSyncLog)
    error("Arduino_Test_Wire:MissingWireSyncTelemetry", "No wire SYNC_EVENT lines were received.");
end

syncRoundTripLog = boardSyncLog(:, {'sync_id', 'host_tx_us', 'host_rx_us', 'board_rx_us', 'board_tx_us'});
[clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog);

joinedTable = innerjoin( ...
    hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'}), ...
    boardCommandLog(:, {'surface_name', 'command_sequence', 'apply_us', 'applied_position'}), ...
    'Keys', {'surface_name', 'command_sequence'});

applyHostUs = clockSlope .* double(joinedTable.apply_us) + clockIntercept;
latencyUs = applyHostUs - double(joinedTable.command_dispatch_us);
if ~isempty(latencyUs)
    minimumLatencyUs = min(latencyUs);
    if minimumLatencyUs < 0
        applyHostUs = applyHostUs - minimumLatencyUs;
        latencyUs = latencyUs - minimumLatencyUs;
    end
end

echoImportTable = table( ...
    joinedTable.surface_name, ...
    joinedTable.command_sequence, ...
    (applyHostUs - double(loggerSession.testStartOffsetUs)) ./ 1e6, ...
    latencyUs ./ 1e6, ...
    joinedTable.applied_position, ...
    nan(height(joinedTable), 1), ...
    'VariableNames', {'surface_name', 'command_sequence', 'arduino_echo_time_s', 'computer_to_arduino_latency_s', 'applied_position', 'applied_equivalent_deg'});

loggerOutputFolder = config.arduinoTransport.loggerOutputFolder;
if ~isfolder(loggerOutputFolder)
    mkdir(loggerOutputFolder);
end

writetable(hostDispatchLog, fullfile(loggerOutputFolder, "host_dispatch_log.csv"));
writetable(syncRoundTripLog, fullfile(loggerOutputFolder, "host_sync_roundtrip.csv"));
writetable(boardCommandLog, fullfile(loggerOutputFolder, "board_command_log.csv"));
writetable(boardSyncLog, fullfile(loggerOutputFolder, "board_sync_log.csv"));
writetable(echoImportTable, fullfile(loggerOutputFolder, "arduino_echo_import.csv"));

if loggerSession.telemetryLineCount > 0
    writelines( ...
        compose("%.0f,%s", loggerSession.telemetryHostRxUs(1:loggerSession.telemetryLineCount), loggerSession.telemetryLineText(1:loggerSession.telemetryLineCount)), ...
        fullfile(loggerOutputFolder, "serial_telemetry_log.txt"));
end

config.arduinoEchoImport = struct( ...
    "filePath", string(fullfile(loggerOutputFolder, "arduino_echo_import.csv")), ...
    "loggerOutputFolder", string(loggerOutputFolder), ...
    "tableData", echoImportTable);
config.arduinoTransport.captureSucceeded = true;
config.arduinoTransport.captureMessage = "Captured " + height(boardCommandLog) + " command telemetry rows and " + height(boardSyncLog) + " sync telemetry rows.";
config.arduinoTransport.captureRowCount = height(echoImportTable);
end

function storage = applyEchoImport(storage, config, echoImportTable)
for sampleIndex = 1:storage.sampleCount
    for surfaceIndex = 1:numel(config.surfaceNames)
        commandSequence = storage.commandSequenceNumbers(sampleIndex, surfaceIndex);
        if ~isfinite(commandSequence)
            continue;
        end

        matchMask = ...
            echoImportTable.surface_name == config.surfaceNames(surfaceIndex) & ...
            echoImportTable.command_sequence == commandSequence;
        if ~any(matchMask)
            continue;
        end

        matchedRow = find(matchMask, 1, "first");
        storage.arduinoReadStartSeconds(sampleIndex) = storage.commandWriteStopSeconds(sampleIndex);
        storage.arduinoEchoSeconds(sampleIndex, surfaceIndex) = echoImportTable.arduino_echo_time_s(matchedRow);
        storage.appliedServoPositions(sampleIndex, surfaceIndex) = echoImportTable.applied_position(matchedRow);
        storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex) = ...
            (echoImportTable.applied_position(matchedRow) - config.servoNeutralPositions(surfaceIndex)) ./ ...
            config.servoUnitsPerDegree(surfaceIndex);
    end

    finiteEchoTimes = storage.arduinoEchoSeconds(sampleIndex, isfinite(storage.arduinoEchoSeconds(sampleIndex, :)));
    if ~isempty(finiteEchoTimes)
        storage.arduinoReadStopSeconds(sampleIndex) = max(finiteEchoTimes);
    end
end
end

function loggerSession = waitForScheduledTimeAndDrain(referenceTimer, targetTimeSeconds, serialObject, loggerSession, pauseSeconds)
while true
    loggerSession = drainWireTelemetry(serialObject, loggerSession);
    remainingSeconds = targetTimeSeconds - toc(referenceTimer);
    if remainingSeconds <= 0
        return;
    end
    pause(min(pauseSeconds, remainingSeconds));
end
end

function loggerSession = pauseAndDrain(serialObject, loggerSession, durationSeconds, pauseSeconds)
pauseTimer = tic;
while toc(pauseTimer) < durationSeconds
    loggerSession = drainWireTelemetry(serialObject, loggerSession);
    pause(min(pauseSeconds, durationSeconds - toc(pauseTimer)));
end
end

function loggerSession = collectRemainingTelemetry(serialObject, loggerSession, maxWaitSeconds, idleTimeoutSeconds)
collectStart = tic;
lastReceiveElapsedSeconds = 0;
while toc(collectStart) < maxWaitSeconds
    oldLineCount = loggerSession.telemetryLineCount;
    loggerSession = drainWireTelemetry(serialObject, loggerSession);
    if loggerSession.telemetryLineCount > oldLineCount
        lastReceiveElapsedSeconds = toc(collectStart);
        continue;
    end
    elapsedSeconds = toc(collectStart);
    if elapsedSeconds >= idleTimeoutSeconds && (elapsedSeconds - lastReceiveElapsedSeconds) >= idleTimeoutSeconds
        return;
    end
    pause(0.01);
end
end

function loggerSession = drainWireTelemetry(serialObject, loggerSession)
[receivedLines, receiveBuffer] = readWireLines(serialObject, loggerSession.receiveBuffer);
loggerSession.receiveBuffer = receiveBuffer;
if isempty(receivedLines)
    return;
end

hostRxUs = double(hostNowUs(loggerSession.hostTimer));
for lineIndex = 1:numel(receivedLines)
    loggerSession.telemetryLineCount = loggerSession.telemetryLineCount + 1;
    lineRow = loggerSession.telemetryLineCount;
    if lineRow <= numel(loggerSession.telemetryLineText)
        loggerSession.telemetryLineText(lineRow) = receivedLines(lineIndex);
        loggerSession.telemetryHostRxUs(lineRow) = hostRxUs;
    end

    telemetryParts = split(receivedLines(lineIndex), ",");
    if isempty(telemetryParts)
        continue;
    end

    switch telemetryParts(1)
        case "COMMAND_EVENT"
            if numel(telemetryParts) >= 7
                loggerSession.boardCommandCount = loggerSession.boardCommandCount + 1;
                rowIndex = loggerSession.boardCommandCount;
                loggerSession.boardCommandSurfaceName(rowIndex) = telemetryParts(2);
                loggerSession.boardCommandSequence(rowIndex) = str2double(telemetryParts(3));
                loggerSession.boardCommandHostRxUs(rowIndex) = hostRxUs;
                loggerSession.boardCommandRxUs(rowIndex) = str2double(telemetryParts(4));
                loggerSession.boardCommandApplyUs(rowIndex) = str2double(telemetryParts(5));
                loggerSession.boardCommandAppliedPosition(rowIndex) = str2double(telemetryParts(6));
                loggerSession.boardCommandPulseUs(rowIndex) = str2double(telemetryParts(7));
            end
        case "SYNC_EVENT"
            if numel(telemetryParts) >= 5
                loggerSession.boardSyncCount = loggerSession.boardSyncCount + 1;
                rowIndex = loggerSession.boardSyncCount;
                loggerSession.boardSyncId(rowIndex) = str2double(telemetryParts(2));
                loggerSession.boardSyncHostTxUs(rowIndex) = str2double(telemetryParts(3));
                loggerSession.boardSyncHostRxUs(rowIndex) = hostRxUs;
                loggerSession.boardSyncBoardRxUs(rowIndex) = str2double(telemetryParts(4));
                loggerSession.boardSyncBoardTxUs(rowIndex) = str2double(telemetryParts(5));
            end
    end
end
end

function sendControlBurst(serialObject, payloadText, repeatCount, pauseSeconds)
for repeatIndex = 1:repeatCount
    writeline(serialObject, payloadText);
    if repeatIndex < repeatCount
        pause(pauseSeconds);
    end
end
end

function moveServosToNeutral(commandInterface)
if isempty(fieldnames(commandInterface))
    return;
end
sendControlBurst(commandInterface.connection, "SET_NEUTRAL", 2, 0.02);
end

function [receivedLines, receiveBuffer] = readWireLines(serialObject, receiveBuffer)
receivedLines = strings(0, 1);
availableByteCount = double(serialObject.NumBytesAvailable);
if availableByteCount <= 0
    return;
end

payloadBytes = read(serialObject, availableByteCount, "uint8");
receiveBuffer = receiveBuffer + string(char(payloadBytes(:).'));
bufferText = char(receiveBuffer);
newlineIndices = find(bufferText == newline);
if isempty(newlineIndices)
    return;
end

receivedLines = strings(numel(newlineIndices), 1);
lineCount = 0;
startIndex = 1;

for newlineIndex = reshape(newlineIndices, 1, [])
    lineText = strtrim(string(bufferText(startIndex:newlineIndex - 1)));
    startIndex = newlineIndex + 1;
    if strlength(lineText) > 0
        lineCount = lineCount + 1;
        receivedLines(lineCount) = lineText;
    end
end

receivedLines = receivedLines(1:lineCount);
receiveBuffer = string(bufferText(startIndex:end));
end

function [clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog)
hostTxUs = double(syncRoundTripLog.host_tx_us);
boardRxUs = double(syncRoundTripLog.board_rx_us);
if numel(hostTxUs) >= 2
    fitCoefficients = polyfit(boardRxUs, hostTxUs, 1);
    clockSlope = fitCoefficients(1);
    clockIntercept = fitCoefficients(2);
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

function nowUs = hostNowUs(hostTimer)
nowUs = uint32(round(toc(hostTimer) .* 1e6));
end

function logs = buildLogTimetables(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceNames = config.surfaceNames;

commandLog = array2timetable( ...
    [storage.baseCommandDegrees(sampleIndices), storage.desiredDeflectionsDegrees(sampleIndices, :), storage.commandedServoPositions(sampleIndices, :), double(storage.commandSaturated(sampleIndices, :))], ...
    'RowTimes', seconds(storage.commandWriteStopSeconds(sampleIndices)), ...
    'VariableNames', cellstr(["base_command_deg", buildSurfaceVariableNames(surfaceNames, "desired_deg"), buildSurfaceVariableNames(surfaceNames, "command_position"), buildSurfaceVariableNames(surfaceNames, "command_saturated")]));
commandLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
commandLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
commandLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);

arduinoRowTimesSeconds = storage.commandWriteStopSeconds(sampleIndices);
validArduinoTimes = isfinite(storage.arduinoReadStopSeconds(sampleIndices));
arduinoRowTimesSeconds(validArduinoTimes) = storage.arduinoReadStopSeconds(sampleIndices(validArduinoTimes));

arduinoLog = array2timetable( ...
    [storage.commandSequenceNumbers(sampleIndices, :), storage.appliedServoPositions(sampleIndices, :), storage.appliedEquivalentDegrees(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr([buildSurfaceVariableNames(surfaceNames, "command_sequence"), buildSurfaceVariableNames(surfaceNames, "applied_position"), buildSurfaceVariableNames(surfaceNames, "applied_equivalent_deg")]));

latencyLog = array2timetable( ...
    [storage.commandSequenceNumbers(sampleIndices, :), storage.commandDispatchSeconds(sampleIndices, :), storage.arduinoEchoSeconds(sampleIndices, :), storage.arduinoEchoSeconds(sampleIndices, :) - storage.commandDispatchSeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr([buildSurfaceVariableNames(surfaceNames, "command_sequence"), buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), buildSurfaceVariableNames(surfaceNames, "arduino_echo_s"), buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_latency_s")]));

sampleSummary = table( ...
    storage.scheduledTimeSeconds(sampleIndices), ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.commandWriteStartSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices), ...
    storage.arduinoReadStartSeconds(sampleIndices), ...
    storage.arduinoReadStopSeconds(sampleIndices), ...
    any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2), ...
    'VariableNames', {'scheduled_time_s', 'base_command_deg', 'command_write_start_s', 'command_write_stop_s', 'arduino_read_start_s', 'arduino_read_stop_s', 'arduino_echo_available'});

logs = struct( ...
    "inputSignal", commandLog, ...
    "arduinoEcho", arduinoLog, ...
    "computerToArduinoLatency", latencyLog, ...
    "latencySummary", buildLatencySummary(storage, config), ...
    "sampleSummary", sampleSummary);
end

function latencySummary = buildLatencySummary(storage, config)
surfaceCount = numel(config.surfaceNames);
dispatchedCommandCount = zeros(surfaceCount, 1);
matchedCommandCount = zeros(surfaceCount, 1);
latencyMeanSeconds = nan(surfaceCount, 1);
latencyMedianSeconds = nan(surfaceCount, 1);
latencyP95Seconds = nan(surfaceCount, 1);
latencyMaxSeconds = nan(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    dispatchedCommandCount(surfaceIndex) = sum(isfinite(storage.commandSequenceNumbers(:, surfaceIndex)));
    latencyValuesSeconds = storage.arduinoEchoSeconds(:, surfaceIndex) - storage.commandDispatchSeconds(:, surfaceIndex);
    latencyValuesSeconds = latencyValuesSeconds(isfinite(latencyValuesSeconds));
    matchedCommandCount(surfaceIndex) = numel(latencyValuesSeconds);

    if isempty(latencyValuesSeconds)
        continue;
    end

    latencyMeanSeconds(surfaceIndex) = mean(latencyValuesSeconds);
    latencyMedianSeconds(surfaceIndex) = percentileValue(latencyValuesSeconds, 50);
    latencyP95Seconds(surfaceIndex) = percentileValue(latencyValuesSeconds, 95);
    latencyMaxSeconds(surfaceIndex) = max(latencyValuesSeconds);
end

latencySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    dispatchedCommandCount, ...
    matchedCommandCount, ...
    latencyMeanSeconds, ...
    latencyMedianSeconds, ...
    latencyP95Seconds, ...
    latencyMaxSeconds, ...
    'VariableNames', {'SurfaceName', 'IsActive', 'DispatchedCommandCount', 'MatchedCommandCount', 'ComputerToArduinoLatencyMean_s', 'ComputerToArduinoLatencyMedian_s', 'ComputerToArduinoLatencyP95_s', 'ComputerToArduinoLatencyMax_s'});
end

function surfaceSummary = buildSurfaceSummary(storage, config)
surfaceSummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    sum(isfinite(storage.appliedServoPositions), 1).', ...
    sum(storage.commandSaturated, 1).', ...
    'VariableNames', {'SurfaceName', 'IsActive', 'AppliedSampleCount', 'CommandSaturationCount'});
end

function latestState = buildLatestState(storage, config, sampleIndex)
surfaceCount = numel(config.surfaceNames);
surfaceStates = repmat(struct("name", "", "desired_deg", NaN, "command_position", NaN, "applied_position", NaN, "applied_equivalent_deg", NaN), surfaceCount, 1);
for surfaceIndex = 1:surfaceCount
    surfaceStates(surfaceIndex).name = config.surfaceNames(surfaceIndex);
    surfaceStates(surfaceIndex).desired_deg = storage.desiredDeflectionsDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).command_position = storage.commandedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_position = storage.appliedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_equivalent_deg = storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex);
end

latestState = struct( ...
    "sampleIndex", sampleIndex, ...
    "scheduledTimeSeconds", storage.scheduledTimeSeconds(sampleIndex), ...
    "commandWriteTimeSeconds", storage.commandWriteStopSeconds(sampleIndex), ...
    "arduinoReadTimeSeconds", storage.arduinoReadStopSeconds(sampleIndex), ...
    "surfaces", surfaceStates);
end

function artifacts = exportRunData(runData)
matFilePath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".mat");
workbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");
artifacts = struct("matFilePath", string(matFilePath), "workbookPath", string(workbookPath));

save(matFilePath, "runData", "-v7.3");
writetable(buildCriticalSettingsTable(runData), workbookPath, 'Sheet', 'CriticalSettings');
writetable(timetableToExportTable(runData.logs.inputSignal), workbookPath, 'Sheet', 'InputSignal');
writetable(timetableToExportTable(runData.logs.arduinoEcho), workbookPath, 'Sheet', 'ArduinoEcho');
writetable(timetableToExportTable(runData.logs.computerToArduinoLatency), workbookPath, 'Sheet', 'ComputerToArduinoLatency');
writetable(runData.logs.latencySummary, workbookPath, 'Sheet', 'LatencySummary');
writetable(runData.surfaceSetup, workbookPath, 'Sheet', 'SurfaceSetup');
writetable(runData.logs.sampleSummary, workbookPath, 'Sheet', 'SampleSummary');
writetable(runData.surfaceSummary, workbookPath, 'Sheet', 'SurfaceSummary');
end

function criticalSettingsTable = buildCriticalSettingsTable(runData)
config = runData.config;
criticalSettingsTable = table( ...
    ["Run"; "Run"; "Arduino"; "Arduino"; "Transport"; "Transport"], ...
    ["RunLabel"; "Status"; "SerialPort"; "BaudRate"; "CaptureSucceeded"; "CaptureMessage"], ...
    [string(config.runLabel); string(runData.runInfo.status); string(config.serialPort); string(config.arduinoTransport.baudRate); string(config.arduinoTransport.captureSucceeded); string(config.arduinoTransport.captureMessage)], ...
    'VariableNames', {'Category', 'Setting', 'Value'});
end

function exportTable = timetableToExportTable(timetableData)
exportTable = timetable2table(timetableData);
exportTable.Properties.VariableNames{1} = 'time_s';
exportTable.time_s = seconds(exportTable.time_s);
end

function cleanupResources(commandInterface, config)
if isempty(fieldnames(commandInterface))
    return;
end

if config.returnToNeutralOnExit
    try
        moveServosToNeutral(commandInterface);
    catch
    end
end

try
    delete(commandInterface.connection);
catch
end
end

function value = percentileValue(sampleValues, percentile)
sampleValues = sort(reshape(sampleValues, [], 1));
samplePosition = 1 + (numel(sampleValues) - 1) .* percentile ./ 100;
lowerIndex = floor(samplePosition);
upperIndex = ceil(samplePosition);
if lowerIndex == upperIndex
    value = sampleValues(lowerIndex);
else
    upperWeight = samplePosition - lowerIndex;
    value = (1 - upperWeight) .* sampleValues(lowerIndex) + upperWeight .* sampleValues(upperIndex);
end
end

function output = squareWave(phaseRadians)
output = ones(size(phaseRadians));
output(sin(phaseRadians) < 0) = -1;
end

function variableNames = buildSurfaceVariableNames(surfaceNames, suffix)
variableNames = strings(1, numel(surfaceNames));
for surfaceIndex = 1:numel(surfaceNames)
    variableNames(surfaceIndex) = matlab.lang.makeValidName(surfaceNames(surfaceIndex) + "_" + suffix);
end
end

function value = getField(config, fieldName, defaultValue)
if isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
end

function value = getText(config, fieldName, defaultValue)
value = getField(config, fieldName, defaultValue);
if ischar(value)
    value = string(value);
end
validateattributes(value, {"string"}, {"scalar"}, char(mfilename), char(fieldName));
end

function value = getStringArray(config, fieldName, defaultValue)
value = getField(config, fieldName, defaultValue);
if ischar(value)
    value = string({value});
elseif iscell(value)
    value = string(value);
end
value = reshape(string(value), [], 1);
end

function value = getLogical(config, fieldName, defaultValue)
value = logical(getField(config, fieldName, defaultValue));
end

function value = getPositiveScalar(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "positive"}, char(mfilename), char(fieldName));
end

function value = getPositiveInteger(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "positive"}, char(mfilename), char(fieldName));
end

function value = getNonnegativeScalar(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "nonnegative"}, char(mfilename), char(fieldName));
end

function value = getNonnegativeInteger(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "nonnegative"}, char(mfilename), char(fieldName));
end

function value = getScalar(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "finite", "scalar"}, char(mfilename), char(fieldName));
end

function value = getNumericColumn(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
validateattributes(value, {"numeric"}, {"real", "column"}, char(mfilename), char(fieldName));
end

function value = getNumericVector(config, fieldName, defaultValue)
value = double(getField(config, fieldName, defaultValue));
if isempty(value)
    value = [];
else
    validateattributes(value, {"numeric"}, {"real", "finite", "vector"}, char(mfilename), char(fieldName));
    value = reshape(value, [], 1);
end
end

function validateattributes(varargin)
validationInputs = varargin;
validationInputs{2} = normalizeValidationList(validationInputs{2});
validationInputs{3} = normalizeValidationList(validationInputs{3});

if numel(validationInputs) >= 4 && isstring(validationInputs{4})
    validationInputs{4} = char(validationInputs{4});
end

if numel(validationInputs) >= 5 && isstring(validationInputs{5})
    validationInputs{5} = char(validationInputs{5});
end

builtin('validateattributes', validationInputs{:});
end

function normalizedList = normalizeValidationList(validationList)
if isstring(validationList)
    normalizedList = cellstr(validationList);
    return;
end

normalizedList = validationList;
if ~iscell(normalizedList)
    return;
end

for elementIndex = 1:numel(normalizedList)
    if isstring(normalizedList{elementIndex})
        normalizedList{elementIndex} = char(normalizedList{elementIndex});
    end
end
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));
defaultSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];
defaultSurfacePins = ["D9"; "D10"; "D11"; "D12"];

config.serialPort = getText(config, "serialPort", "COM5");
config.surfaceNames = getStringArray(config, "surfaceNames", defaultSurfaceNames);
config.surfacePins = getStringArray(config, "surfacePins", defaultSurfacePins);
config.servoNeutralPositions = getNumericColumn(config, "servoNeutralPositions", 0.5 .* ones(numel(config.surfaceNames), 1));
config.servoUnitsPerDegree = getNumericColumn(config, "servoUnitsPerDegree", (1 ./ 180) .* ones(numel(config.surfaceNames), 1));
config.servoMinimumPositions = getNumericColumn(config, "servoMinimumPositions", zeros(numel(config.surfaceNames), 1));
config.servoMaximumPositions = getNumericColumn(config, "servoMaximumPositions", ones(numel(config.surfaceNames), 1));
config.commandDeflectionScales = getNumericColumn(config, "commandDeflectionScales", ones(numel(config.surfaceNames), 1));
config.commandDeflectionOffsetsDegrees = getNumericColumn(config, "commandDeflectionOffsetsDegrees", zeros(numel(config.surfaceNames), 1));
config.commandMode = getText(config, "commandMode", "all");
config.singleSurfaceName = getText(config, "singleSurfaceName", "Aileron_L");
config.commandProfile = normalizeCommandProfile(getField(config, "commandProfile", struct()));
config.neutralSettleSeconds = getNonnegativeScalar(config, "neutralSettleSeconds", 1.0);
config.returnToNeutralOnExit = getLogical(config, "returnToNeutralOnExit", true);
config.outputFolder = getText(config, "outputFolder", fullfile(rootFolder, "C_Arduino_Test"));
config.runLabel = getText(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    config.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_Wire";
end

transport = getField(config, "arduinoTransport", struct());
config.arduinoTransport = struct( ...
    "mode", "nano_logger_serial", ...
    "serialPort", getText(transport, "serialPort", config.serialPort), ...
    "baudRate", getPositiveInteger(transport, "baudRate", 115200), ...
    "serialResetSeconds", getNonnegativeScalar(transport, "serialResetSeconds", 4.0), ...
    "probeTimeoutSeconds", getPositiveScalar(transport, "probeTimeoutSeconds", 6.0), ...
    "helloRetrySeconds", getPositiveScalar(transport, "helloRetrySeconds", 0.25), ...
    "linePollPauseSeconds", getPositiveScalar(transport, "linePollPauseSeconds", 0.01), ...
    "lineIdleTimeoutSeconds", getPositiveScalar(transport, "lineIdleTimeoutSeconds", 0.25), ...
    "syncCountBeforeRun", getNonnegativeInteger(transport, "syncCountBeforeRun", 10), ...
    "syncCountAfterRun", getNonnegativeInteger(transport, "syncCountAfterRun", 10), ...
    "syncPauseSeconds", getNonnegativeScalar(transport, "syncPauseSeconds", 0.05), ...
    "postRunSettleSeconds", getNonnegativeScalar(transport, "postRunSettleSeconds", 0.25), ...
    "clearLogsBeforeRun", getLogical(transport, "clearLogsBeforeRun", true), ...
    "loggerOutputFolder", getText(transport, "loggerOutputFolder", fullfile(config.outputFolder, config.runLabel + "_ArduinoLogger")), ...
    "captureSucceeded", false, ...
    "captureMessage", "", ...
    "captureRowCount", 0);
config.serialPort = config.arduinoTransport.serialPort;

surfaceCount = numel(config.surfaceNames);
validateattributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoNeutralPositions');
validateattributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoUnitsPerDegree');
validateattributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMinimumPositions');
validateattributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMaximumPositions');

if ~any(config.commandMode == ["single", "all"])
    error("Arduino_Test_Wire:InvalidCommandMode", "commandMode must be 'single' or 'all'.");
end

activeSurfaceMask = false(surfaceCount, 1);
if config.commandMode == "single"
    activeSurfaceMask(config.surfaceNames == config.singleSurfaceName) = true;
else
    activeSurfaceMask(:) = true;
end

if ~any(activeSurfaceMask)
    error("Arduino_Test_Wire:InvalidSingleSurface", ...
        "singleSurfaceName must match one of: %s.", ...
        char(join(config.surfaceNames, ", ")));
end

config.activeSurfaceMask = activeSurfaceMask;
config.activeSurfaceNames = config.surfaceNames(activeSurfaceMask);
config.surfaceSetup = table( ...
    config.surfaceNames, ...
    config.surfacePins, ...
    config.activeSurfaceMask, ...
    config.servoNeutralPositions, ...
    config.servoUnitsPerDegree, ...
    config.servoMinimumPositions, ...
    config.servoMaximumPositions, ...
    config.commandDeflectionScales, ...
    config.commandDeflectionOffsetsDegrees, ...
    'VariableNames', { ...
        'SurfaceName', 'ArduinoPin', 'IsActive', 'ServoNeutralPosition', 'ServoUnitsPerDegree', ...
        'ServoMinimumPosition', 'ServoMaximumPosition', 'CommandScale', 'CommandOffsetDeg'});
end

function commandProfile = normalizeCommandProfile(commandProfileConfig)
commandProfile.type = getText(commandProfileConfig, "type", "sine");
commandProfile.sampleTimeSeconds = getPositiveScalar(commandProfileConfig, "sampleTimeSeconds", 0.05);
commandProfile.preCommandNeutralSeconds = getNonnegativeScalar(commandProfileConfig, "preCommandNeutralSeconds", 0.0);
commandProfile.postCommandNeutralSeconds = getNonnegativeScalar(commandProfileConfig, "postCommandNeutralSeconds", 1.0);
commandProfile.durationSeconds = getPositiveScalar(commandProfileConfig, "durationSeconds", 6.0);
commandProfile.amplitudeDegrees = getScalar(commandProfileConfig, "amplitudeDegrees", 45.0);
commandProfile.offsetDegrees = getScalar(commandProfileConfig, "offsetDegrees", 0.0);
commandProfile.frequencyHz = getPositiveScalar(commandProfileConfig, "frequencyHz", 0.5);
commandProfile.phaseDegrees = getScalar(commandProfileConfig, "phaseDegrees", 90.0);
commandProfile.doubletHoldSeconds = getPositiveScalar(commandProfileConfig, "doubletHoldSeconds", 0.5);
commandProfile.customTimeSeconds = getNumericVector(commandProfileConfig, "customTimeSeconds", []);
commandProfile.customDeflectionDegrees = getNumericVector(commandProfileConfig, "customDeflectionDegrees", []);
commandProfile.customInterpolationMethod = getText(commandProfileConfig, "customInterpolationMethod", "linear");
commandProfile.customFunction = getField(commandProfileConfig, "customFunction", []);

if ~any(commandProfile.type == ["sine", "square", "doublet", "custom", "function"])
    error("Arduino_Test_Wire:InvalidProfileType", ...
        "commandProfile.type must be 'sine', 'square', 'doublet', 'custom', or 'function'.");
end

if commandProfile.type == "doublet"
    commandProfile.durationSeconds = max(commandProfile.durationSeconds, 2 .* commandProfile.doubletHoldSeconds);
end
end

function runData = initializeRunData(config)
runData = struct( ...
    "config", config, ...
    "connectionInfo", struct("arduino", struct()), ...
    "runInfo", struct("status", "initialized", "reason", "", "startTime", NaT, "stopTime", NaT, "sampleCount", 0, "scheduledDurationSeconds", NaN), ...
    "surfaceSetup", config.surfaceSetup, ...
    "logs", struct(), ...
    "surfaceSummary", table(), ...
    "artifacts", struct("matFilePath", "", "workbookPath", ""));
end

function [commandInterface, connectionInfo] = connectToArduino(config)
commandInterface = struct();
connectionInfo = struct("port", config.serialPort, "baudRate", config.arduinoTransport.baudRate, "isConnected", false, "connectElapsedSeconds", NaN, "connectionMessage", "", "loggerFirmwareVersion", "");

connectStart = tic;
serialObject = [];

try
    serialObject = serialport(char(config.serialPort), config.arduinoTransport.baudRate);
    configureTerminator(serialObject, "LF");
    pause(config.arduinoTransport.serialResetSeconds);
    flush(serialObject);

    [helloLine, firmwareVersion] = probeWireLogger(serialObject, config.arduinoTransport);
    commandInterface = struct("transportMode", "nano_logger_serial", "connection", serialObject, "surfaceNames", config.surfaceNames);

    connectionInfo.isConnected = true;
    connectionInfo.connectionMessage = "Connected via USB serial echo logger.";
    connectionInfo.loggerFirmwareVersion = firmwareVersion;
    connectionInfo.loggerGreeting = helloLine;
catch connectionException
    if ~isempty(serialObject)
        try
            delete(serialObject);
        catch
        end
    end
    connectionInfo.connectionMessage = string(connectionException.message);
end

connectionInfo.connectElapsedSeconds = toc(connectStart);
end

function [helloLine, firmwareVersion] = probeWireLogger(serialObject, transportConfig)
helloLine = "";
firmwareVersion = "";
receiveBuffer = "";
probeStart = tic;
lastHelloSendSeconds = -inf;

while toc(probeStart) < transportConfig.probeTimeoutSeconds
    elapsedSeconds = toc(probeStart);
    if elapsedSeconds - lastHelloSendSeconds >= transportConfig.helloRetrySeconds
        writeline(serialObject, "HELLO");
        lastHelloSendSeconds = elapsedSeconds;
    end

    [receivedLines, receiveBuffer] = readWireLines(serialObject, receiveBuffer);
    for lineIndex = 1:numel(receivedLines)
        if startsWith(receivedLines(lineIndex), "HELLO_EVENT")
            helloLine = receivedLines(lineIndex);
            helloParts = split(helloLine, ",");
            if numel(helloParts) >= 2
                firmwareVersion = helloParts(2);
            end
            return;
        end
    end

    pause(transportConfig.linePollPauseSeconds);
end

error("Arduino_Test_Wire:NoHelloEvent", ...
    "No HELLO_EVENT was received from %s within %.1f s.", ...
    char(transportConfig.serialPort), ...
    transportConfig.probeTimeoutSeconds);
end

function printConnectionStatus(runData)
fprintf("\nArduino_Test_Wire connection summary\n");
fprintf("  Arduino (%s): %s\n", char(runData.config.serialPort), char(getStatusText(runData.connectionInfo.arduino)));
fprintf("  Active command surfaces: %s\n\n", char(join(runData.config.activeSurfaceNames, ", ")));
end

function statusText = getStatusText(connectionInfo)
if isempty(fieldnames(connectionInfo))
    statusText = "not attempted";
elseif connectionInfo.isConnected
    statusText = "connected";
else
    statusText = "not connected - " + string(connectionInfo.connectionMessage);
end
end

function [storage, runInfo, config] = executeWireTest(commandInterface, config)
[scheduledTimeSeconds, profileInfo] = buildCommandSchedule(config.commandProfile);
surfaceCount = numel(config.surfaceNames);
sampleCount = numel(scheduledTimeSeconds);
storage = initializeStorage(sampleCount, surfaceCount);
storage.scheduledTimeSeconds = scheduledTimeSeconds;
storage.profileInfo = profileInfo;

runInfo = struct("status", "running", "reason", "", "startTime", datetime("now"), "stopTime", NaT, "sampleCount", 0, "scheduledDurationSeconds", profileInfo.totalDurationSeconds);

loggerSession = startWireLoggerSession(commandInterface.connection, config, sampleCount, nnz(config.activeSurfaceMask));
surfaceCommandCounts = zeros(1, surfaceCount);

for sampleIndex = 1:sampleCount
    loggerSession = waitForScheduledTimeAndDrain( ...
        loggerSession.hostTimer, ...
        loggerSession.testStartOffsetSeconds + scheduledTimeSeconds(sampleIndex), ...
        commandInterface.connection, ...
        loggerSession, ...
        config.arduinoTransport.linePollPauseSeconds);

    baseCommandDegrees = evaluateBaseCommandDegrees(profileInfo, scheduledTimeSeconds(sampleIndex));
    desiredDeflectionsDegrees = zeros(1, surfaceCount);
    desiredDeflectionsDegrees(config.activeSurfaceMask.') = ...
        config.commandDeflectionScales(config.activeSurfaceMask).' .* baseCommandDegrees + ...
        config.commandDeflectionOffsetsDegrees(config.activeSurfaceMask).';

    rawServoPositions = config.servoNeutralPositions.' + desiredDeflectionsDegrees .* config.servoUnitsPerDegree.';
    commandedServoPositions = min(max(rawServoPositions, config.servoMinimumPositions.'), config.servoMaximumPositions.');
    saturatedMask = abs(commandedServoPositions - rawServoPositions) > 10 .* eps;
    nextCommandSequenceNumbers = nan(1, surfaceCount);
    nextCommandSequenceNumbers(config.activeSurfaceMask.') = surfaceCommandCounts(config.activeSurfaceMask.') + 1;

    storage.commandWriteStartSeconds(sampleIndex) = max(0, toc(loggerSession.hostTimer) - loggerSession.testStartOffsetSeconds);
    [commandDispatchSeconds, commandWriteStopSeconds, loggerSession] = writeServoPositions( ...
        commandInterface.connection, commandedServoPositions, config.surfaceNames, config.activeSurfaceMask.', nextCommandSequenceNumbers, loggerSession);

    surfaceCommandCounts(config.activeSurfaceMask.') = nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.sampleCount = sampleIndex;
    storage.baseCommandDegrees(sampleIndex) = baseCommandDegrees;
    storage.commandWriteStopSeconds(sampleIndex) = commandWriteStopSeconds;
    storage.commandDispatchSeconds(sampleIndex, :) = commandDispatchSeconds;
    storage.commandSequenceNumbers(sampleIndex, config.activeSurfaceMask.') = nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.desiredDeflectionsDegrees(sampleIndex, :) = desiredDeflectionsDegrees;
    storage.commandedServoPositions(sampleIndex, :) = commandedServoPositions;
    storage.commandSaturated(sampleIndex, :) = saturatedMask;
end

[loggerSession, echoImportTable, config] = finalizeWireLoggerSession(commandInterface.connection, loggerSession, config);
storage = applyEchoImport(storage, config, echoImportTable);

runInfo.status = "completed";
runInfo.stopTime = datetime("now");
runInfo.sampleCount = storage.sampleCount;
end

function [scheduledTimeSeconds, profileInfo] = buildCommandSchedule(commandProfile)
profileInfo = commandProfile;
profileInfo.commandStartSeconds = commandProfile.preCommandNeutralSeconds;
profileInfo.commandStopSeconds = commandProfile.preCommandNeutralSeconds + commandProfile.durationSeconds;
profileInfo.totalDurationSeconds = ...
    commandProfile.preCommandNeutralSeconds + commandProfile.durationSeconds + commandProfile.postCommandNeutralSeconds;

scheduledTimeSeconds = (0:commandProfile.sampleTimeSeconds:profileInfo.totalDurationSeconds).';
if scheduledTimeSeconds(end) < profileInfo.totalDurationSeconds
    scheduledTimeSeconds(end + 1, 1) = profileInfo.totalDurationSeconds;
end
end

function storage = initializeStorage(sampleCount, surfaceCount)
storage = struct( ...
    "sampleCount", 0, ...
    "scheduledTimeSeconds", nan(sampleCount, 1), ...
    "baseCommandDegrees", nan(sampleCount, 1), ...
    "commandWriteStartSeconds", nan(sampleCount, 1), ...
    "commandWriteStopSeconds", nan(sampleCount, 1), ...
    "commandDispatchSeconds", nan(sampleCount, surfaceCount), ...
    "commandSequenceNumbers", nan(sampleCount, surfaceCount), ...
    "arduinoReadStartSeconds", nan(sampleCount, 1), ...
    "arduinoReadStopSeconds", nan(sampleCount, 1), ...
    "arduinoEchoSeconds", nan(sampleCount, surfaceCount), ...
    "desiredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "commandedServoPositions", nan(sampleCount, surfaceCount), ...
    "commandSaturated", false(sampleCount, surfaceCount), ...
    "appliedServoPositions", nan(sampleCount, surfaceCount), ...
    "appliedEquivalentDegrees", nan(sampleCount, surfaceCount), ...
    "profileInfo", struct());
end

function commandDegrees = evaluateBaseCommandDegrees(profileInfo, elapsedTimeSeconds)
if elapsedTimeSeconds < profileInfo.commandStartSeconds || elapsedTimeSeconds > profileInfo.commandStopSeconds
    commandDegrees = 0;
    return;
end

profileTimeSeconds = elapsedTimeSeconds - profileInfo.commandStartSeconds;
phaseRadians = deg2rad(profileInfo.phaseDegrees);

switch profileInfo.type
    case "sine"
        commandDegrees = profileInfo.offsetDegrees + profileInfo.amplitudeDegrees .* sin(2 .* pi .* profileInfo.frequencyHz .* profileTimeSeconds + phaseRadians);
    case "square"
        commandDegrees = profileInfo.offsetDegrees + profileInfo.amplitudeDegrees .* squareWave(2 .* pi .* profileInfo.frequencyHz .* profileTimeSeconds + phaseRadians);
    case "doublet"
        if profileTimeSeconds <= profileInfo.doubletHoldSeconds
            commandDegrees = profileInfo.offsetDegrees + profileInfo.amplitudeDegrees;
        elseif profileTimeSeconds <= 2 .* profileInfo.doubletHoldSeconds
            commandDegrees = profileInfo.offsetDegrees - profileInfo.amplitudeDegrees;
        else
            commandDegrees = profileInfo.offsetDegrees;
        end
    case "custom"
        commandDegrees = interp1(profileInfo.customTimeSeconds, profileInfo.customDeflectionDegrees, profileTimeSeconds, char(profileInfo.customInterpolationMethod), 0);
    otherwise
        commandDegrees = double(profileInfo.customFunction(profileTimeSeconds));
end
end
