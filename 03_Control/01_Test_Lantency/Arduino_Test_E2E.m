function [runData, e2e] = Arduino_Test_E2E(config)
% ARDUINO_TEST_E2E Direct-servo E2E latency with logic-analyser output capture.
arguments
    config (1,1) struct = struct()
end

runHardware = getFieldLocal(config, "runHardware", true);
arduinoConfig = getFieldLocal(config, "arduinoConfig", struct());
loggerFolder = string(getFieldLocal(config, "loggerFolder", ""));
outputPrefix = string(getFieldLocal(config, "outputPrefix", "e2e_output"));
seed = double(getFieldLocal(config, "seed", 5));
sampleTimeSeconds = double(getFieldLocal(config, "sampleTimeSeconds", 0.02));
recordLeadSeconds = max(0, double(getFieldLocal(config, "recordLeadSeconds", 10.0)));
recordLagSeconds = max(0, double(getFieldLocal(config, "recordLagSeconds", 10.0)));
commandActiveSeconds = max(0, double(getFieldLocal(config, "commandActiveSeconds", 59.0)));
clockMapMode = lower(string(getFieldLocal(config, "clockMapMode", "command")));
matchingMode = lower(string(getFieldLocal(config, "matchingMode", "apply_anchored")));
maxOutputAssociationSeconds = double(getFieldLocal(config, "maxOutputAssociationSeconds", 0.05));
referenceAssociationWindowSeconds = double(getFieldLocal(config, "referenceAssociationWindowSeconds", 0.02));
transitionPulseThresholdUs = double(getFieldLocal(config, "transitionPulseThresholdUs", 4.0));
targetPulseToleranceUs = double(getFieldLocal(config, "targetPulseToleranceUs", 40.0));
previousPulseToleranceUs = double(getFieldLocal(config, "previousPulseToleranceUs", 25.0));
maximumApplyToOutputSeconds = double(getFieldLocal(config, "maximumApplyToOutputSeconds", 0.03));
logicAnalyzerConfig = getFieldLocal(config, "logicAnalyzer", struct());

if runHardware
    arduinoConfig = applyComparableProfileDefaultsLocal(arduinoConfig, seed, recordLeadSeconds, recordLagSeconds, commandActiveSeconds);
    loggerFolder = resolveLoggerFolderFromArduinoConfigLocal(arduinoConfig);
    logicAnalyzer = buildLogicAnalyzerConfigLocal(logicAnalyzerConfig, loggerFolder, string(arduinoConfig.runLabel));
    captureDurationSeconds = recordLeadSeconds + commandActiveSeconds + recordLagSeconds + 1.0;
    sigrokSession = struct("process", [], "isActive", false);
    cleanupHandle = onCleanup(@() cleanupSigrokSessionLocal(sigrokSession)); %#ok<NASGU>
    if logicAnalyzer.enabled && logicAnalyzer.mode == "sigrok_auto"
        sigrokSession = startSigrokCaptureLocal(logicAnalyzer, captureDurationSeconds);
        pause(logicAnalyzer.captureStartLeadSeconds);
    end
    runData = Arduino_Test(arduinoConfig);
    loggerFolder = string(runData.config.arduinoTransport.loggerOutputFolder);
    if logicAnalyzer.enabled && logicAnalyzer.mode == "sigrok_auto"
        waitForSigrokCaptureLocal(sigrokSession, logicAnalyzer);
    end
else
    runData = struct();
    if strlength(loggerFolder) == 0
        loggerFolder = resolveLatestArduinoLoggerFolderLocal(fullfile(fileparts(mfilename("fullpath")), "C_Arduino_Test"));
    end
    loggerFolder = string(loggerFolder);
    logicAnalyzer = buildLogicAnalyzerConfigLocal(logicAnalyzerConfig, loggerFolder, stripLoggerSuffixLocal(loggerFolder));
end

logs = loadArduinoLogsLocal(loggerFolder, sampleTimeSeconds);
[referenceCapture, outputCapture] = importAnalyzerCaptureLocal(logicAnalyzer, logs.surfaceNames);
logs.referenceCapture = referenceCapture;
logs.outputCapture = outputCapture;

e2eEvents = computeArduinoE2ELocal( ...
    logs, ...
    clockMapMode, ...
    matchingMode, ...
    maxOutputAssociationSeconds, ...
    referenceAssociationWindowSeconds, ...
    transitionPulseThresholdUs, ...
    targetPulseToleranceUs, ...
    previousPulseToleranceUs, ...
    maximumApplyToOutputSeconds);
[surfaceSummary, overallSummary, integritySummary] = buildSummariesLocal(e2eEvents, logs.surfaceNames);

e2e = struct();
e2e.clockMapMode = string(clockMapMode);
e2e.matchingMode = string(matchingMode);
e2e.inputSignal = logs.inputSignal;
e2e.profileEvents = logs.profileEvents;
e2e.referenceCapture = referenceCapture;
e2e.outputCapture = outputCapture;
e2e.eventLatency = e2eEvents;
e2e.surfaceSummary = surfaceSummary;
e2e.overallSummary = overallSummary;
e2e.integritySummary = integritySummary;
e2e.outputPaths = exportArduinoE2ELocal(e2e, loggerFolder, outputPrefix);

if runHardware
    runData.e2e = e2e;
end

fprintf("Arduino_Test_E2E summary\n");
fprintf("  Logger folder: %s\n", char(loggerFolder));
fprintf("  Matching mode: %s\n", char(e2e.matchingMode));
fprintf("  Transition commands: %d\n", sum(e2e.eventLatency.is_transition_command));
fprintf("  Matched output transitions: %d\n", sum(e2e.eventLatency.matched_output_transition));
fprintf("  Valid output events: %d\n", sum(e2e.eventLatency.is_valid_e2e));
end

function arduinoConfig = applyComparableProfileDefaultsLocal(arduinoConfig, seed, recordLeadSeconds, recordLagSeconds, commandActiveSeconds)
rootFolder = fileparts(mfilename("fullpath"));
if ~isfield(arduinoConfig, "outputFolder")
    arduinoConfig.outputFolder = fullfile(rootFolder, "C_Arduino_Test");
end
if ~isfield(arduinoConfig, "arduinoTransport") || isempty(arduinoConfig.arduinoTransport)
    arduinoConfig.arduinoTransport = struct();
end
arduinoConfig.arduinoTransport.mode = "nano_logger_udp";
arduinoConfig.arduinoTransport.operatingMode = "controller";
arduinoConfig.arduinoTransport.commandEncoding = "binary_vector";
arduinoConfig.commandMode = "all";
arduinoConfig.commandProfile.type = "latency_step_train";
arduinoConfig.commandProfile.sampleTimeSeconds = 0.02;
arduinoConfig.commandProfile.preCommandNeutralSeconds = recordLeadSeconds;
arduinoConfig.commandProfile.postCommandNeutralSeconds = recordLagSeconds;
arduinoConfig.commandProfile.durationSeconds = commandActiveSeconds;
arduinoConfig.commandProfile.amplitudeDegrees = 45.0;
arduinoConfig.commandProfile.offsetDegrees = 0.0;
arduinoConfig.commandProfile.eventHoldSeconds = 0.20;
arduinoConfig.commandProfile.eventNeutralHoldSeconds = 0.10;
arduinoConfig.commandProfile.eventDwellSeconds = 0.60;
arduinoConfig.commandProfile.eventRandomJitterSeconds = 0.05;
arduinoConfig.commandProfile.randomSeed = seed;
if ~isfield(arduinoConfig, "runLabel") || strlength(string(arduinoConfig.runLabel)) == 0
    arduinoConfig.runLabel = compose("Seed_%d_Controller", round(seed));
end
end

function loggerFolder = resolveLoggerFolderFromArduinoConfigLocal(arduinoConfig)
outputFolder = string(getFieldLocal(arduinoConfig, "outputFolder", fullfile(fileparts(mfilename("fullpath")), "C_Arduino_Test")));
runLabel = string(getFieldLocal(arduinoConfig, "runLabel", "ArduinoE2E"));
transportConfig = getFieldLocal(arduinoConfig, "arduinoTransport", struct());
loggerFolder = string(getFieldLocal(transportConfig, "loggerOutputFolder", ""));
if strlength(loggerFolder) == 0
    loggerFolder = fullfile(outputFolder, runLabel + "_ArduinoLogger");
end
end

function logicAnalyzer = buildLogicAnalyzerConfigLocal(config, loggerFolder, runLabel)
loggerFolder = string(loggerFolder);
if ~isfolder(loggerFolder)
    mkdir(loggerFolder);
end
artifactPrefix = fullfile(loggerFolder, runLabel + "_sigrok");
enabled = logical(getFieldLocal(config, "enabled", true));
modeDefault = "import_only";
if enabled
    modeDefault = "sigrok_auto";
end
referenceChannel = double(getFieldLocal(getFieldLocal(config, "channelRoleMap", struct()), "reference", NaN));
logicAnalyzer = struct();
logicAnalyzer.enabled = enabled;
logicAnalyzer.mode = lower(string(getFieldLocal(config, "mode", modeDefault)));
logicAnalyzer.sigrokCliPath = string(getFieldLocal(config, "sigrokCliPath", "C:\Program Files\sigrok\sigrok-cli\sigrok-cli.exe"));
logicAnalyzer.deviceDriver = string(getFieldLocal(config, "deviceDriver", "fx2lafw"));
logicAnalyzer.deviceId = string(getFieldLocal(config, "deviceId", ""));
logicAnalyzer.sampleRateHz = double(getFieldLocal(config, "sampleRateHz", 4000000));
logicAnalyzer.captureStartLeadSeconds = double(getFieldLocal(config, "captureStartLeadSeconds", 0.25));
logicAnalyzer.captureStopLagSeconds = double(getFieldLocal(config, "captureStopLagSeconds", 0.50));
logicAnalyzer.channels = reshape(double(getFieldLocal(config, "channels", [0 1 2 3 4])), 1, []);
logicAnalyzer.channelNames = reshape(string(getFieldLocal(config, "channelNames", ["D0" "D1" "D2" "D3" "D4"])), 1, []);
logicAnalyzer.channelRoleMap = struct("output", reshape(double(getFieldLocal(getFieldLocal(config, "channelRoleMap", struct()), "output", [0 1 2 3])), 1, []), "reference", referenceChannel);
logicAnalyzer.rawCapturePath = string(getFieldLocal(config, "rawCapturePath", artifactPrefix + "_raw.sr"));
logicAnalyzer.logicStateExportPath = string(getFieldLocal(config, "logicStateExportPath", artifactPrefix + "_logic_state.csv"));
logicAnalyzer.outputCapturePath = string(getFieldLocal(config, "outputCapturePath", fullfile(loggerFolder, "output_capture.csv")));
logicAnalyzer.referenceCapturePath = string(getFieldLocal(config, "referenceCapturePath", fullfile(loggerFolder, "reference_capture.csv")));
logicAnalyzer.storeStdoutPath = string(getFieldLocal(config, "storeStdoutPath", artifactPrefix + "_stdout.txt"));
logicAnalyzer.storeStderrPath = string(getFieldLocal(config, "storeStderrPath", artifactPrefix + "_stderr.txt"));
logicAnalyzer.referenceDebounceUs = double(getFieldLocal(config, "referenceDebounceUs", 100));
logicAnalyzer.minimumPulseUs = double(getFieldLocal(config, "minimumPulseUs", 800));
logicAnalyzer.maximumPulseUs = double(getFieldLocal(config, "maximumPulseUs", 2200));
end

function sigrokSession = startSigrokCaptureLocal(logicAnalyzer, captureDurationSeconds)
deleteFileIfPresentLocal(logicAnalyzer.rawCapturePath);
deleteFileIfPresentLocal(logicAnalyzer.logicStateExportPath);
deleteFileIfPresentLocal(logicAnalyzer.outputCapturePath);
deleteFileIfPresentLocal(logicAnalyzer.referenceCapturePath);
deleteFileIfPresentLocal(logicAnalyzer.storeStdoutPath);
deleteFileIfPresentLocal(logicAnalyzer.storeStderrPath);
sigrokCliPath = validateSigrokExecutableLocal(logicAnalyzer.sigrokCliPath);
driverSpec = string(logicAnalyzer.deviceDriver);
if strlength(logicAnalyzer.deviceId) > 0
    driverSpec = driverSpec + ":" + logicAnalyzer.deviceId;
end
channelAssignments = join(logicAnalyzer.channelNames + "=" + string(round(logicAnalyzer.channels)), ",");
captureDurationMs = max(1, ceil(1000 .* (captureDurationSeconds + logicAnalyzer.captureStartLeadSeconds + logicAnalyzer.captureStopLagSeconds)));
commandText = strjoin([ ...
    quoteWindowsArgumentLocal(sigrokCliPath), ...
    "--driver", quoteWindowsArgumentLocal(driverSpec), ...
    "--config", quoteWindowsArgumentLocal("samplerate=" + string(round(logicAnalyzer.sampleRateHz))), ...
    "--channels", quoteWindowsArgumentLocal(string(channelAssignments)), ...
    "--time", quoteWindowsArgumentLocal(string(captureDurationMs)), ...
    "--output-file", quoteWindowsArgumentLocal(logicAnalyzer.rawCapturePath)], " ");
processStartInfo = System.Diagnostics.ProcessStartInfo();
processStartInfo.FileName = 'cmd.exe';
processStartInfo.Arguments = char("/d /s /c " + quoteWindowsArgumentLocal(commandText + " 1>" + quoteWindowsArgumentLocal(logicAnalyzer.storeStdoutPath) + " 2>" + quoteWindowsArgumentLocal(logicAnalyzer.storeStderrPath)));
processStartInfo.UseShellExecute = false;
processStartInfo.CreateNoWindow = true;
processStartInfo.WorkingDirectory = char(fileparts(logicAnalyzer.rawCapturePath));
processHandle = System.Diagnostics.Process();
processHandle.StartInfo = processStartInfo;
if ~processHandle.Start()
    error("Arduino_Test_E2E:SigrokLaunchFailed", "Failed to launch sigrok-cli raw capture.");
end
sigrokSession = struct("process", processHandle, "isActive", true);
end

function waitForSigrokCaptureLocal(sigrokSession, logicAnalyzer)
if ~isfield(sigrokSession, "isActive") || ~sigrokSession.isActive
    return;
end
sigrokSession.process.WaitForExit();
if double(sigrokSession.process.ExitCode) ~= 0
    error("Arduino_Test_E2E:SigrokCaptureFailed", "sigrok-cli raw capture failed. See %s.", char(logicAnalyzer.storeStderrPath));
end
if ~isfile(logicAnalyzer.rawCapturePath)
    error("Arduino_Test_E2E:MissingSigrokCapture", "sigrok-cli finished without creating the raw capture file.");
end
end

function cleanupSigrokSessionLocal(sigrokSession)
if ~isstruct(sigrokSession) || ~isfield(sigrokSession, "process") || isempty(sigrokSession.process)
    return;
end
try
    if ~sigrokSession.process.HasExited
        sigrokSession.process.Kill();
    end
catch
end
end
function logs = loadArduinoLogsLocal(loggerFolder, sampleTimeSeconds)
logs = struct();
logs.hostDispatchLog = readtable(fullfile(loggerFolder, "host_dispatch_log.csv"));
logs.boardCommandLog = readtable(fullfile(loggerFolder, "board_command_log.csv"));
logs.hostSyncRoundTrip = readtable(fullfile(loggerFolder, "host_sync_roundtrip.csv"));
logs.hostDispatchLog.surface_name = string(logs.hostDispatchLog.surface_name);
logs.boardCommandLog.surface_name = string(logs.boardCommandLog.surface_name);
logs.surfaceNames = unique(logs.hostDispatchLog.surface_name, "stable");
logs.inputSignal = buildInputSignalLocal(logs.hostDispatchLog, logs.surfaceNames, sampleTimeSeconds);
logs.profileEvents = table();
wb = fullfile(fileparts(loggerFolder), stripLoggerSuffixLocal(loggerFolder) + ".xlsx");
if isfile(wb)
    try
        logs.profileEvents = readtable(wb, "Sheet", "ProfileEvents");
    catch
        logs.profileEvents = table();
    end
end
end

function inputSignal = buildInputSignalLocal(hostDispatchLog, surfaceNames, sampleTimeSeconds)
dispatch = sortrows(hostDispatchLog, {'command_sequence', 'surface_name'});
seq = unique(double(dispatch.command_sequence), 'stable');
rowCount = numel(seq);
inputSignal = table();
inputSignal.command_sequence = seq;
inputSignal.scheduled_time_s = (seq - min(seq)) .* sampleTimeSeconds;
dispatchBySeq = groupsummary(dispatch(:, {'command_sequence', 'command_dispatch_us'}), 'command_sequence', 'median', 'command_dispatch_us');
originUs = min(double(dispatchBySeq.median_command_dispatch_us));
inputSignal.command_dispatch_s = (double(dispatchBySeq.median_command_dispatch_us) - originUs) ./ 1e6;
inputSignal.time_s = inputSignal.command_dispatch_s;
inputSignal.command_write_start_s = inputSignal.command_dispatch_s;
inputSignal.command_write_stop_s = inputSignal.command_dispatch_s;
inputSignal.base_command_deg = nan(rowCount, 1);
for i = 1:rowCount
    rowSubset = dispatch(double(dispatch.command_sequence) == seq(i), :);
    basePosition = median(double(rowSubset.position_norm), 'omitnan');
    inputSignal.base_command_deg(i) = 180.0 .* (basePosition - 0.5);
    for s = 1:numel(surfaceNames)
        surfaceName = surfaceNames(s);
        v = rowSubset(rowSubset.surface_name == surfaceName, :);
        if isempty(v)
            pos = NaN;
        else
            pos = median(double(v.position_norm), 'omitnan');
        end
        inputSignal.(char(surfaceName + "_desired_deg"))(i,1) = 180.0 .* (pos - 0.5);
        inputSignal.(char(surfaceName + "_command_position"))(i,1) = pos;
        inputSignal.(char(surfaceName + "_command_saturated"))(i,1) = false;
    end
end
end

function [referenceCapture, outputCapture] = importAnalyzerCaptureLocal(logicAnalyzer, surfaceNames)
referenceCapture = buildEmptyReferenceCaptureTableLocal();
outputCapture = buildEmptyPulseCaptureTableLocal();
if ~logicAnalyzer.enabled
    return;
end
if isfile(logicAnalyzer.outputCapturePath)
    outputCapture = normalizeOutputCaptureTableLocal(readCaptureCsvLocal(logicAnalyzer.outputCapturePath), logicAnalyzer, surfaceNames);
end
if isfile(logicAnalyzer.referenceCapturePath)
    referenceCapture = normalizeReferenceCaptureTableLocal(readCaptureCsvLocal(logicAnalyzer.referenceCapturePath), logicAnalyzer.referenceDebounceUs);
end
if ~isempty(outputCapture)
    return;
end
if ~isfile(logicAnalyzer.logicStateExportPath)
    if ~isfile(logicAnalyzer.rawCapturePath)
        error("Arduino_Test_E2E:MissingAnalyzerCapture", "No analyser capture artifact was found.");
    end
    sigrokCliPath = validateSigrokExecutableLocal(logicAnalyzer.sigrokCliPath);
    commandText = strjoin([ ...
        quoteWindowsArgumentLocal(sigrokCliPath), ...
        "--input-file", quoteWindowsArgumentLocal(logicAnalyzer.rawCapturePath), ...
        "--output-file", quoteWindowsArgumentLocal(logicAnalyzer.logicStateExportPath), ...
        "--output-format", quoteWindowsArgumentLocal("csv:time=true")], " ");
    [status, outputText] = runWindowsCommandLocal(commandText + " 1>>" + quoteWindowsArgumentLocal(logicAnalyzer.storeStdoutPath) + " 2>>" + quoteWindowsArgumentLocal(logicAnalyzer.storeStderrPath));
    if status ~= 0
        error("Arduino_Test_E2E:SigrokExportFailed", "sigrok-cli export failed. %s", strtrim(outputText));
    end
end
logicState = readLogicStateLocal(logicAnalyzer, surfaceNames);
if isfinite(logicAnalyzer.channelRoleMap.reference)
    referenceCapture = extractReferenceCaptureLocal(logicState, logicAnalyzer);
end
outputCapture = extractOutputCaptureLocal(logicState, logicAnalyzer, surfaceNames);
if isempty(outputCapture)
    error("Arduino_Test_E2E:EmptyOutputCapture", "No valid PWM pulses were decoded from the output analyser channels.");
end
writetable(outputCapture, logicAnalyzer.outputCapturePath);
if ~isempty(referenceCapture)
    writetable(referenceCapture, logicAnalyzer.referenceCapturePath);
end
end

function logicState = readLogicStateLocal(logicAnalyzer, surfaceNames)
rawTable = readCaptureCsvLocal(logicAnalyzer.logicStateExportPath);
variableNames = string(rawTable.Properties.VariableNames);
canonicalNames = lower(regexprep(variableNames, '[^a-zA-Z0-9]', ''));
timeIndex = find(contains(canonicalNames, 'time'), 1, 'first');
if isempty(timeIndex)
    error("Arduino_Test_E2E:MissingLogicStateTimeColumn", "Logic-state export is missing a time column.");
end
rawTimeData = convertColumnToNumericLocal(rawTable.(char(variableNames(timeIndex))));
timeSeconds = normalizeTimeColumnLocal(rawTimeData, logicAnalyzer.sampleRateHz);
sampleIndexColumn = find(ismember(canonicalNames, ["sampleindex", "samplenum", "sample"]), 1, 'first');
if isempty(sampleIndexColumn)
    sampleIndex = round(timeSeconds .* double(logicAnalyzer.sampleRateHz));
else
    sampleIndex = round(convertColumnToNumericLocal(rawTable.(char(variableNames(sampleIndexColumn)))));
end
channelData = nan(height(rawTable), numel(logicAnalyzer.channels));
for k = 1:numel(logicAnalyzer.channels)
    targetOptions = lower(regexprep([logicAnalyzer.channelNames(k), "D" + string(logicAnalyzer.channels(k)), string(logicAnalyzer.channels(k))], '[^a-zA-Z0-9]', ''));
    idx = [];
    for t = 1:numel(targetOptions)
        idx = find(canonicalNames == targetOptions(t), 1, 'first');
        if ~isempty(idx)
            break;
        end
    end
    if isempty(idx)
        error("Arduino_Test_E2E:MissingLogicStateChannel", "Logic-state export is missing channel %s.", char(logicAnalyzer.channelNames(k)));
    end
    channelData(:, k) = convertColumnToNumericLocal(rawTable.(char(variableNames(idx))));
end
logicState = struct('sampleIndex', reshape(sampleIndex, [], 1), 'sampleRateHz', double(logicAnalyzer.sampleRateHz), 'channelNames', logicAnalyzer.channelNames, 'stateMatrix', channelData);
end

function events = computeArduinoE2ELocal(logs, clockMapMode, matchingMode, maxOutputAssociationSeconds, referenceAssociationWindowSeconds, transitionPulseThresholdUs, targetPulseToleranceUs, previousPulseToleranceUs, maximumApplyToOutputSeconds)
dispatch = sortrows(logs.hostDispatchLog, {'command_sequence', 'surface_name'});
board = sortrows(logs.boardCommandLog, {'command_sequence', 'surface_name'});
if ~ismember('receive_to_apply_us', string(board.Properties.VariableNames))
    board.receive_to_apply_us = double(board.apply_us) - double(board.rx_us);
end
joined = outerjoin(dispatch, board, 'Keys', {'surface_name', 'command_sequence'}, 'MergeKeys', true, 'Type', 'left');
joined.sample_index = joined.command_sequence;
joined.scheduled_time_s = (double(joined.command_sequence) - min(double(joined.command_sequence))) .* (logs.inputSignal.scheduled_time_s(2) - logs.inputSignal.scheduled_time_s(1));
originUs = min(double(joined.command_dispatch_us));
joined.command_dispatch_s = (double(joined.command_dispatch_us) - originUs) ./ 1e6;
joined.host_scheduling_delay_s = joined.command_dispatch_s - joined.scheduled_time_s;
[cSlope, cIntercept] = estimateClockMapLocal(logs.hostSyncRoundTrip, joined, clockMapMode);
joined.board_rx_s = (cSlope .* double(joined.rx_us) + cIntercept - originUs) ./ 1e6;
joined.board_apply_s = (cSlope .* double(joined.apply_us) + cIntercept - originUs) ./ 1e6;
joined.dispatch_to_rx_latency_s = joined.board_rx_s - joined.command_dispatch_s;
joined.dispatch_to_apply_latency_s = joined.board_apply_s - joined.command_dispatch_s;
joined.scheduled_to_rx_latency_s = joined.board_rx_s - joined.scheduled_time_s;
joined.scheduled_to_apply_latency_s = joined.board_apply_s - joined.scheduled_time_s;
joined.rx_to_apply_latency_s = double(joined.receive_to_apply_us) ./ 1e6;
joined.expected_pulse_us = double(joined.pulse_us);
joined.reference_time_s = nan(height(joined), 1);
joined.anchor_time_s = joined.board_apply_s;
joined.anchor_source = repmat("apply", height(joined), 1);
if ~isempty(logs.referenceCapture)
    [refTime, ~] = matchReferenceTimesLocal(joined.board_apply_s, logs.referenceCapture.time_s, referenceAssociationWindowSeconds);
    joined.reference_time_s = refTime;
    if matchingMode == "shared_clock"
        mask = isfinite(refTime);
        joined.anchor_time_s(mask) = refTime(mask);
        joined.anchor_source(mask) = "reference";
    end
end
joined.previous_expected_pulse_us = nan(height(joined), 1);
joined.is_transition_command = false(height(joined), 1);
for s = 1:numel(logs.surfaceNames)
    mask = joined.surface_name == logs.surfaceNames(s);
    rows = find(mask);
    if numel(rows) < 2
        continue;
    end
    prevPulse = double(joined.expected_pulse_us(rows(1:end-1)));
    currPulse = double(joined.expected_pulse_us(rows(2:end)));
    tmask = isfinite(prevPulse) & isfinite(currPulse) & abs(currPulse - prevPulse) >= transitionPulseThresholdUs;
    joined.previous_expected_pulse_us(rows(2:end)) = prevPulse;
    joined.is_transition_command(rows(2:end)) = tmask;
end
joined.output_time_s = nan(height(joined),1);
joined.output_pulse_us = nan(height(joined),1);
joined.matched_output_transition = false(height(joined),1);
joined.apply_to_output_latency_s = nan(height(joined),1);
joined.dispatch_to_output_latency_s = nan(height(joined),1);
joined.scheduled_to_output_latency_s = nan(height(joined),1);
joined.is_valid_e2e = false(height(joined),1);
joined.non_realistic_reason = repmat("", height(joined), 1);
for s = 1:numel(logs.surfaceNames)
    surfaceRows = find(joined.surface_name == logs.surfaceNames(s) & joined.is_transition_command);
    surfaceCapture = logs.outputCapture(logs.outputCapture.surface_name == logs.surfaceNames(s), :);
    transitionTable = buildOutputTransitionTableLocal(surfaceCapture, transitionPulseThresholdUs);
    searchIndex = 1;
    for i = 1:numel(surfaceRows)
        rowIndex = surfaceRows(i);
        [matchIndex, searchIndex] = findNextOutputTransitionLocal(transitionTable, searchIndex, double(joined.anchor_time_s(rowIndex)), double(joined.previous_expected_pulse_us(rowIndex)), double(joined.expected_pulse_us(rowIndex)), previousPulseToleranceUs, targetPulseToleranceUs, maxOutputAssociationSeconds);
        if ~isfinite(matchIndex)
            joined.non_realistic_reason(rowIndex) = "unmatched_output_transition";
            continue;
        end
        joined.output_time_s(rowIndex) = double(transitionTable.time_s(matchIndex));
        joined.output_pulse_us(rowIndex) = double(transitionTable.pulse_us(matchIndex));
        joined.matched_output_transition(rowIndex) = true;
        joined.apply_to_output_latency_s(rowIndex) = joined.output_time_s(rowIndex) - joined.board_apply_s(rowIndex);
        if isfinite(joined.apply_to_output_latency_s(rowIndex)) && joined.apply_to_output_latency_s(rowIndex) > -5e-4 && joined.apply_to_output_latency_s(rowIndex) < 0
            joined.apply_to_output_latency_s(rowIndex) = 0;
        end
        joined.dispatch_to_output_latency_s(rowIndex) = joined.output_time_s(rowIndex) - joined.command_dispatch_s(rowIndex);
        joined.scheduled_to_output_latency_s(rowIndex) = joined.output_time_s(rowIndex) - joined.scheduled_time_s(rowIndex);
        joined.is_valid_e2e(rowIndex) = isfinite(joined.apply_to_output_latency_s(rowIndex)) && joined.apply_to_output_latency_s(rowIndex) >= -1e-6 && joined.apply_to_output_latency_s(rowIndex) <= maximumApplyToOutputSeconds;
        if joined.is_valid_e2e(rowIndex)
            joined.non_realistic_reason(rowIndex) = "ok";
        else
            joined.non_realistic_reason(rowIndex) = "apply_to_output_out_of_range";
        end
    end
end
events = joined(joined.is_transition_command, :);
end
function [surfaceSummary, overallSummary, integritySummary] = buildSummariesLocal(events, surfaceNames)
metricMap = { ...
    'HostSchedulingDelay', 'host_scheduling_delay_s'; ...
    'ComputerToArduinoRxLatency', 'dispatch_to_rx_latency_s'; ...
    'ComputerToArduinoApplyLatency', 'dispatch_to_apply_latency_s'; ...
    'ScheduledToArduinoRxLatency', 'scheduled_to_rx_latency_s'; ...
    'ArduinoReceiveToApplyLatency', 'rx_to_apply_latency_s'; ...
    'ScheduledToApplyLatency', 'scheduled_to_apply_latency_s'; ...
    'ApplyToOutputLatency', 'apply_to_output_latency_s'; ...
    'DispatchToOutputLatency', 'dispatch_to_output_latency_s'; ...
    'ScheduledToOutputLatency', 'scheduled_to_output_latency_s'};
surfaceSummary = table();
for s = 1:numel(surfaceNames)
    group = events(events.surface_name == surfaceNames(s), :);
    transitionCount = sum(group.is_transition_command);
    matchedCount = sum(group.matched_output_transition);
    row = table(string(surfaceNames(s)), true, transitionCount, matchedCount, transitionCount - matchedCount, nan, ...
        'VariableNames', {'SurfaceName', 'IsActive', 'TransitionCommandCount', 'MatchedOutputTransitionCount', 'UnmatchedOutputTransitionCount', 'UnmatchedOutputTransitionFraction'});
    if transitionCount > 0
        row.UnmatchedOutputTransitionFraction = row.UnmatchedOutputTransitionCount ./ transitionCount;
    end
    for m = 1:size(metricMap,1)
        prefix = metricMap{m,1};
        stats = latencyStatsLocal(group.(metricMap{m,2}));
        f = fieldnames(stats);
        for k = 1:numel(f)
            row.(char(prefix + string(f{k}))) = stats.(f{k});
        end
    end
    surfaceSummary = [surfaceSummary; row]; %#ok<AGROW>
end
overallSummary = table(strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'metric', 'count', 'mean_s', 'median_s', 'p95_s', 'p99_s', 'max_s'});
for metricName = ["dispatch_to_rx_latency_s", "dispatch_to_apply_latency_s", "scheduled_to_rx_latency_s", "scheduled_to_apply_latency_s", "rx_to_apply_latency_s", "apply_to_output_latency_s", "dispatch_to_output_latency_s", "scheduled_to_output_latency_s"]
    values = double(events.(char(metricName)));
    values = values(isfinite(values));
    if isempty(values)
        continue;
    end
    overallSummary = [overallSummary; table(metricName, numel(values), mean(values), median(values), prctile(values,95), prctile(values,99), max(values), 'VariableNames', overallSummary.Properties.VariableNames)]; %#ok<AGROW>
end
integritySummary = table();
for s = 1:numel(surfaceNames)
    group = events(events.surface_name == surfaceNames(s), :);
    transitionCount = sum(group.is_transition_command);
    matchedCount = sum(group.matched_output_transition);
    validCount = sum(group.is_valid_e2e);
    row = table(string(surfaceNames(s)), true, transitionCount, matchedCount, validCount, transitionCount - matchedCount, nan, ...
        'VariableNames', {'SurfaceName', 'IsActive', 'TransitionCommandCount', 'MatchedOutputTransitionCount', 'ValidE2ECount', 'UnmatchedOutputTransitionCount', 'UnmatchedOutputTransitionFraction'});
    if transitionCount > 0
        row.UnmatchedOutputTransitionFraction = row.UnmatchedOutputTransitionCount ./ transitionCount;
    end
    integritySummary = [integritySummary; row]; %#ok<AGROW>
end
end

function outputPaths = exportArduinoE2ELocal(e2e, loggerFolder, outputPrefix)
loggerFolder = string(loggerFolder);
events = e2e.eventLatency;
prefix = fullfile(loggerFolder, outputPrefix);
writetable(events, prefix + "_events.csv");
writetable(e2e.surfaceSummary, prefix + "_surface_summary.csv");
writetable(e2e.overallSummary, prefix + "_overall_summary.csv");
writetable(e2e.integritySummary, prefix + "_integrity_summary.csv");
writetable(e2e.inputSignal, prefix + "_input_signal.csv");
if ~isempty(e2e.profileEvents)
    writetable(e2e.profileEvents, prefix + "_profile_events.csv");
end
if ~isempty(e2e.referenceCapture)
    writetable(e2e.referenceCapture, fullfile(loggerFolder, "reference_capture.csv"));
end
if ~isempty(e2e.outputCapture)
    writetable(e2e.outputCapture, fullfile(loggerFolder, "output_capture.csv"));
end
outputPaths = struct('eventPath', prefix + "_events.csv", 'surfaceSummaryPath', prefix + "_surface_summary.csv", 'overallSummaryPath', prefix + "_overall_summary.csv", 'integritySummaryPath', prefix + "_integrity_summary.csv");
end

function [clockSlope, clockIntercept] = estimateClockMapLocal(syncRoundTripLog, joined, clockMapMode)
hostTxUs = double(syncRoundTripLog.host_tx_us); hostRxUs = double(syncRoundTripLog.host_rx_us); boardRxUs = double(syncRoundTripLog.board_rx_us); boardTxUs = double(syncRoundTripLog.board_tx_us);
useSync = string(clockMapMode) == "sync";
if useSync
    mask = isfinite(hostTxUs) & isfinite(hostRxUs) & isfinite(boardRxUs) & isfinite(boardTxUs);
    if nnz(mask) >= 2
        c = polyfit(0.5 .* (boardRxUs(mask) + boardTxUs(mask)), 0.5 .* (hostTxUs(mask) + hostRxUs(mask)), 1);
        clockSlope = c(1); clockIntercept = c(2);
    else
        useSync = false;
    end
end
if ~useSync
    mask = isfinite(double(joined.rx_us)) & isfinite(double(joined.command_dispatch_us));
    if nnz(mask) < 2
        error("Arduino_Test_E2E:ClockMapFitFailed", "Unable to estimate the board-to-host clock map from the available command or sync timestamps.");
    end
    c = polyfit(double(joined.rx_us(mask)), double(joined.command_dispatch_us(mask)), 1);
    rtt = hostRxUs - hostTxUs; rtt = rtt(isfinite(rtt) & rtt >= 0);
    oneWayUs = 0.0; if ~isempty(rtt), oneWayUs = 0.5 .* median(rtt, 'omitnan'); end
    clockSlope = c(1); clockIntercept = c(2) + oneWayUs;
end
rxHostUs = clockSlope .* double(joined.rx_us) + clockIntercept;
applyHostUs = clockSlope .* double(joined.apply_us) + clockIntercept;
dispatchUs = double(joined.command_dispatch_us);
minimumLatencyUs = min([rxHostUs - dispatchUs; applyHostUs - dispatchUs], [], 'omitnan');
if isfinite(minimumLatencyUs) && minimumLatencyUs < 0
    clockIntercept = clockIntercept - minimumLatencyUs;
end
end

function [refTime, refIndex] = matchReferenceTimesLocal(anchorTimes, referenceTimes, windowSeconds)
refTime = nan(size(anchorTimes)); refIndex = nan(size(anchorTimes)); if isempty(referenceTimes), return; end
referenceTimes = reshape(double(referenceTimes), [], 1); nextIndex = 1;
for i = 1:numel(anchorTimes)
    t = double(anchorTimes(i)); if ~isfinite(t), continue; end
    while nextIndex <= numel(referenceTimes) && referenceTimes(nextIndex) < t - windowSeconds, nextIndex = nextIndex + 1; end
    cands = [nextIndex - 1, nextIndex]; cands = cands(cands >= 1 & cands <= numel(referenceTimes)); if isempty(cands), continue; end
    [d, idx] = min(abs(referenceTimes(cands) - t)); if d <= windowSeconds, refTime(i) = referenceTimes(cands(idx)); refIndex(i) = cands(idx); end
end
end

function transitionTable = buildOutputTransitionTableLocal(surfaceCapture, thresholdUs)
if isempty(surfaceCapture) || height(surfaceCapture) < 2
    transitionTable = table(zeros(0,1), zeros(0,1), zeros(0,1), 'VariableNames', {'time_s', 'previous_pulse_us', 'pulse_us'});
    return;
end
surfaceCapture = sortrows(surfaceCapture, 'time_s');
time_s = []; previous_pulse_us = []; pulse_us = [];
for i = 2:height(surfaceCapture)
    p0 = double(surfaceCapture.pulse_us(i-1)); p1 = double(surfaceCapture.pulse_us(i));
    if isfinite(p0) && isfinite(p1) && abs(p1 - p0) >= thresholdUs
        time_s(end+1,1) = double(surfaceCapture.time_s(i)); %#ok<AGROW>
        previous_pulse_us(end+1,1) = p0; %#ok<AGROW>
        pulse_us(end+1,1) = p1; %#ok<AGROW>
    end
end
transitionTable = table(time_s, previous_pulse_us, pulse_us);
end

function [matchIndex, nextSearchIndex] = findNextOutputTransitionLocal(transitionTable, searchIndex, anchorTime, previousPulseUs, targetPulseUs, previousToleranceUs, targetToleranceUs, maxWindowSeconds)
matchIndex = NaN; nextSearchIndex = searchIndex; if isempty(transitionTable) || ~isfinite(anchorTime), return; end
while nextSearchIndex <= height(transitionTable) && double(transitionTable.time_s(nextSearchIndex)) < anchorTime - 1e-6, nextSearchIndex = nextSearchIndex + 1; end
windowEnd = anchorTime + maxWindowSeconds;
for i = nextSearchIndex:height(transitionTable)
    t = double(transitionTable.time_s(i)); if t > windowEnd, break; end
    if abs(double(transitionTable.pulse_us(i)) - targetPulseUs) > targetToleranceUs, continue; end
    if isfinite(previousPulseUs) && abs(double(transitionTable.previous_pulse_us(i)) - previousPulseUs) > previousToleranceUs, continue; end
    matchIndex = i; nextSearchIndex = i + 1; return;
end
end

function outputCapture = extractOutputCaptureLocal(logicState, logicAnalyzer, surfaceNames)
outputCapture = buildEmptyPulseCaptureTableLocal(); blocks = cell(numel(surfaceNames), 1);
for s = 1:min(numel(surfaceNames), numel(logicAnalyzer.channelRoleMap.output))
    col = resolveLogicStateChannelColumnIndexLocal(logicState, logicAnalyzer, logicAnalyzer.channelRoleMap.output(s));
    channelStates = logicState.stateMatrix(:, col); rising = logicState.sampleIndex([false; diff(channelStates) > 0]); falling = logicState.sampleIndex([false; diff(channelStates) < 0]);
    [startSamples, sampleCounts] = pairEdgeSamplesLocal(rising, falling); pulseUs = 1e6 .* double(sampleCounts) ./ double(logicState.sampleRateHz); valid = pulseUs >= logicAnalyzer.minimumPulseUs & pulseUs <= logicAnalyzer.maximumPulseUs;
    startSamples = startSamples(valid); sampleCounts = sampleCounts(valid); pulseUs = pulseUs(valid);
    if isempty(startSamples), continue; end
    blocks{s} = table(repmat(surfaceNames(s), numel(startSamples), 1), double(startSamples) ./ double(logicState.sampleRateHz), pulseUs, double(startSamples), double(sampleCounts), repmat(double(logicState.sampleRateHz), numel(startSamples), 1), 'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz'});
end
blocks = blocks(~cellfun(@isempty, blocks)); if isempty(blocks), return; end
outputCapture = vertcat(blocks{:}); outputCapture = sortrows(outputCapture, {'surface_name', 'sample_index'});
end

function referenceCapture = extractReferenceCaptureLocal(logicState, logicAnalyzer)
referenceCapture = buildEmptyReferenceCaptureTableLocal(); if ~isfinite(logicAnalyzer.channelRoleMap.reference), return; end
col = resolveLogicStateChannelColumnIndexLocal(logicState, logicAnalyzer, logicAnalyzer.channelRoleMap.reference); states = logicState.stateMatrix(:, col); edgeSamples = logicState.sampleIndex([false; diff(states) > 0]);
debounceSamples = round(double(logicAnalyzer.referenceDebounceUs) .* double(logicState.sampleRateHz) ./ 1e6); edgeSamples = applySampleDebounceLocal(edgeSamples, debounceSamples);
referenceCapture = table(double(edgeSamples) ./ double(logicState.sampleRateHz), double(edgeSamples), repmat(double(logicState.sampleRateHz), numel(edgeSamples), 1), 'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'});
end

function stats = latencyStatsLocal(values)
values = double(values); values = values(isfinite(values));
if isempty(values)
    stats = struct('SampleCount', 0, 'Mean_s', NaN, 'Std_s', NaN, 'Median_s', NaN, 'P95_s', NaN, 'P99_s', NaN, 'Max_s', NaN); return;
end
stats = struct('SampleCount', numel(values), 'Mean_s', mean(values), 'Std_s', std(values,1), 'Median_s', median(values), 'P95_s', prctile(values,95), 'P99_s', prctile(values,99), 'Max_s', max(values));
end

function captureTable = readCaptureCsvLocal(filePath)
opts = detectImportOptions(filePath, 'FileType', 'text', 'CommentStyle', '#'); opts.CommentStyle = '#'; if isprop(opts, 'VariableNamingRule'), opts.VariableNamingRule = 'preserve'; end
captureTable = readtable(filePath, opts);
end

function pulseCapture = normalizeOutputCaptureTableLocal(rawTable, logicAnalyzer, surfaceNames)
pulseCapture = buildEmptyPulseCaptureTableLocal(); if isempty(rawTable), return; end
vn = string(rawTable.Properties.VariableNames); cvn = lower(regexprep(vn, '[^a-zA-Z0-9]', ''));
timeIdx = find(contains(cvn, 'time'), 1, 'first'); pulseIdx = find(cvn == 'pulseus' | cvn == 'pulsewidthus' | cvn == 'widthus', 1, 'first'); surfIdx = find(cvn == 'surfacename' | cvn == 'channelname', 1, 'first');
if isempty(timeIdx) || isempty(pulseIdx) || isempty(surfIdx), return; end
time_s = normalizeTimeColumnLocal(convertColumnToNumericLocal(rawTable.(char(vn(timeIdx)))), logicAnalyzer.sampleRateHz); pulse_us = convertColumnToNumericLocal(rawTable.(char(vn(pulseIdx)))); surface_name = string(rawTable.(char(vn(surfIdx))));
valid = isfinite(time_s) & isfinite(pulse_us) & pulse_us >= logicAnalyzer.minimumPulseUs & pulse_us <= logicAnalyzer.maximumPulseUs & ismember(surface_name, surfaceNames);
if ~any(valid), return; end
pulseCapture = table(surface_name(valid), time_s(valid), pulse_us(valid), nan(sum(valid),1), nan(sum(valid),1), nan(sum(valid),1), 'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz'}); pulseCapture = sortrows(pulseCapture, {'surface_name', 'time_s'});
end

function referenceCapture = normalizeReferenceCaptureTableLocal(rawTable, debounceUs)
referenceCapture = buildEmptyReferenceCaptureTableLocal(); if isempty(rawTable), return; end
vn = string(rawTable.Properties.VariableNames); cvn = lower(regexprep(vn, '[^a-zA-Z0-9]', '')); timeIdx = find(contains(cvn, 'time'), 1, 'first'); if isempty(timeIdx), return; end
time_s = normalizeTimeColumnLocal(convertColumnToNumericLocal(rawTable.(char(vn(timeIdx)))), 1.0); valid = isfinite(time_s); if ~any(valid), return; end
referenceCapture = table(time_s(valid), nan(sum(valid),1), nan(sum(valid),1), 'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'}); referenceCapture = sortrows(referenceCapture, 'time_s');
if height(referenceCapture) >= 2, keep = [true; diff(double(referenceCapture.time_s)) > debounceUs ./ 1e6]; referenceCapture = referenceCapture(keep, :); end
end

function idx = resolveLogicStateChannelColumnIndexLocal(logicState, logicAnalyzer, roleChannel)
confIdx = find(double(logicAnalyzer.channels) == double(roleChannel), 1, 'first'); if isempty(confIdx), error('Arduino_Test_E2E:MissingLogicStateChannelRole', 'Configured logic channel %d was not found.', roleChannel); end
configuredName = string(logicAnalyzer.channelNames(confIdx)); idx = find(logicState.channelNames == configuredName, 1, 'first'); if isempty(idx), error('Arduino_Test_E2E:MissingLogicStateChannel', 'Logic-state export is missing configured channel %s.', char(configuredName)); end
end

function data = convertColumnToNumericLocal(columnData)
if isnumeric(columnData), data = double(columnData); return; end
textData = string(columnData); textData = replace(strtrim(textData), ["true", "false"], ["1", "0"]); data = str2double(textData);
end

function timeSeconds = normalizeTimeColumnLocal(rawTimeData, sampleRateHz)
rawTimeData = reshape(double(rawTimeData), [], 1); timeSeconds = rawTimeData; finiteValues = rawTimeData(isfinite(rawTimeData)); if isempty(finiteValues), return; end
if max(abs(finiteValues)) > 1e4, timeSeconds = rawTimeData ./ double(sampleRateHz); end
end

function debouncedSamples = applySampleDebounceLocal(edgeSamples, debounceSamples)
if isempty(edgeSamples), debouncedSamples = edgeSamples; return; end
debouncedSamples = edgeSamples(1); for i = 2:numel(edgeSamples), if edgeSamples(i) - debouncedSamples(end) > debounceSamples, debouncedSamples(end+1,1) = edgeSamples(i); end, end %#ok<AGROW>
end

function [pulseStartSamples, pulseSampleCounts] = pairEdgeSamplesLocal(risingSamples, fallingSamples)
pulseStartSamples = nan(0,1); pulseSampleCounts = nan(0,1); if isempty(risingSamples) || isempty(fallingSamples), return; end
fallIndex = 1; for i = 1:numel(risingSamples), rise = risingSamples(i); while fallIndex <= numel(fallingSamples) && fallingSamples(fallIndex) <= rise, fallIndex = fallIndex + 1; end, if fallIndex > numel(fallingSamples), break; end, pulseStartSamples(end+1,1) = rise; pulseSampleCounts(end+1,1) = fallingSamples(fallIndex) - rise; fallIndex = fallIndex + 1; end %#ok<AGROW>
end

function pulseCapture = buildEmptyPulseCaptureTableLocal()
pulseCapture = table(strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), 'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz'});
end

function referenceCapture = buildEmptyReferenceCaptureTableLocal()
referenceCapture = table(zeros(0,1), zeros(0,1), zeros(0,1), 'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'});
end

function loggerFolder = resolveLatestArduinoLoggerFolderLocal(rootFolder)
folderInfo = dir(fullfile(rootFolder, '*_ArduinoLogger')); if isempty(folderInfo), error('Arduino_Test_E2E:MissingLoggerFolder', 'No *_ArduinoLogger folder found in %s.', char(rootFolder)); end
[~, idx] = max([folderInfo.datenum]); loggerFolder = string(fullfile(folderInfo(idx).folder, folderInfo(idx).name));
end

function runLabel = stripLoggerSuffixLocal(loggerFolder)
[~, folderName] = fileparts(char(loggerFolder)); runLabel = string(regexprep(folderName, '_ArduinoLogger$', ''));
end

function deleteFileIfPresentLocal(filePath)
filePath = string(filePath); if strlength(filePath) > 0 && isfile(filePath), delete(filePath); end
end

function sigrokCliPath = validateSigrokExecutableLocal(sigrokCliPath)
sigrokCliPath = string(sigrokCliPath); if isfile(sigrokCliPath), return; end
[status, whereOutput] = system('where.exe sigrok-cli'); if status ~= 0, error('Arduino_Test_E2E:MissingSigrokCli', 'Unable to locate sigrok-cli.'); end
whereLines = splitlines(string(whereOutput)); whereLines = strtrim(whereLines(strlength(strtrim(whereLines)) > 0)); if isempty(whereLines), error('Arduino_Test_E2E:MissingSigrokCli', 'Unable to locate sigrok-cli.'); end
sigrokCliPath = whereLines(1);
end

function quotedText = quoteWindowsArgumentLocal(argumentText)
argumentText = replace(string(argumentText), '"', '\"'); quotedText = '"' + argumentText + '"';
end

function [status, outputText] = runWindowsCommandLocal(commandText)
[status, outputText] = system(char(commandText)); outputText = string(outputText);
end

function value = getFieldLocal(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName), value = structValue.(fieldName); else, value = defaultValue; end
end
