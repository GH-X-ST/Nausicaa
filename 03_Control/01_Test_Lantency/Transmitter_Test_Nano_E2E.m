function [runData, e2e] = Transmitter_Test_Nano_E2E(config)
% TRANSMITTER_TEST_NANO_E2E Run/load Nano 33 IoT transmitter data and use the Python post processor.
%   This wrapper mirrors Transmitter_Test_E2E, but keeps all Nano-specific
%   defaults and logger-folder resolution local to this file.
arguments
    config (1,1) struct = struct()
end

runHardware = logical(getFieldLocal(config, "runHardware", true));
runPostProcessor = logical(getFieldLocal(config, "runPostProcessor", true));
forceReprocess = logical(getFieldLocal(config, "forceReprocess", true));
printSummary = logical(getFieldLocal(config, "printSummary", false));
outputPrefix = normalizeTextScalarLocal(getFieldLocal(config, "outputPrefix", "post_transition_e2e"));
seed = double(getFieldLocal(config, "seed", 1));
enforceArduinoStepTrainDefaults = logical(getFieldLocal(config, "enforceArduinoStepTrainDefaults", true));
recordLeadSeconds = max(0, double(getFieldLocal(config, "recordLeadSeconds", 10.0)));
recordLagSeconds = max(0, double(getFieldLocal(config, "recordLagSeconds", 10.0)));
commandActiveSeconds = max(0, double(getFieldLocal(config, "commandActiveSeconds", 59.0)));
pythonExecutable = normalizeTextScalarLocal(getFieldLocal(config, "pythonExecutable", "python"));
pythonScriptPath = normalizeTextScalarLocal(getFieldLocal(config, "pythonScriptPath", fullfile(fileparts(mfilename("fullpath")), "Transmitter_Test_E2E_Post.py")));
postConfig = getFieldLocal(config, "postConfig", struct());

transmitterConfig = getFieldLocal(config, "transmitterConfig", struct());
transmitterConfig = applyNanoTransmitterDefaultsLocal(transmitterConfig, seed);
rootFolder = normalizeTextScalarLocal(getFieldLocal(config, "rootFolder", transmitterConfig.outputFolder));
loggerFolder = normalizeTextScalarLocal(getFieldLocal(config, "loggerFolder", ""));
if strlength(loggerFolder) == 0
    loggerFolder = resolveLoggerFolderFromTransmitterConfigLocal(transmitterConfig);
end

if runHardware
    transmitterConfig = applyComparableProfileDefaultsLocal( ...
        transmitterConfig, ...
        seed, ...
        enforceArduinoStepTrainDefaults, ...
        recordLeadSeconds, ...
        recordLagSeconds, ...
        commandActiveSeconds);
    runData = Transmitter_Test(transmitterConfig);
    loggerFolder = normalizeTextScalarLocal(getFieldLocal(runData.artifacts, "loggerFolderPath", ""));
    if strlength(loggerFolder) == 0
        loggerFolder = resolveLoggerFolderFromTransmitterConfigLocal(runData.config);
    end
else
    runData = struct();
    if strlength(loggerFolder) == 0
        loggerFolder = resolveLatestLoggerFolderLocal(rootFolder);
    end
end

if runHardware
    validateNanoHardwareRunSucceededLocal(runData, loggerFolder);
end

ensureFolderExistsLocal(loggerFolder);

clockRepair = repairNanoLoggerClockOfflineLocal(loggerFolder);

outputPaths = buildOutputPathsLocal(loggerFolder, outputPrefix);
if runPostProcessor && (forceReprocess || clockRepair.rewroteLogs || ~isfile(outputPaths.eventPath))
    runPythonPostProcessorLocal(loggerFolder, outputPrefix, pythonExecutable, pythonScriptPath, rootFolder, postConfig);
end

if ~isfile(outputPaths.eventPath)
    error("Transmitter_Test_Nano_E2E:MissingPostOutput", ...
        "Expected post-processed event file was not found: %s", ...
        char(outputPaths.eventPath));
end

e2e = loadPostOutputsLocal(loggerFolder, outputPrefix);
e2e.nanoClockRepair = clockRepair;
if runHardware
    runData.e2e = e2e;
    runData.nanoClockRepair = clockRepair;
    if clockRepair.rewroteLogs && isfield(runData, "logs")
        runData.logs.boardRxLog = clockRepair.boardRxLog;
        runData.logs.boardCommitLog = clockRepair.boardCommitLog;
    end
end

if printSummary
    fprintf("Transmitter_Test_Nano_E2E summary\n");
    fprintf("  Method: %s\n", char(e2e.method));
    fprintf("  Logger folder: %s\n", char(loggerFolder));
    if clockRepair.applied
        fprintf("  Nano clock repair: %s (min RX-after-dispatch %.3f ms, median %.3f ms)\n", ...
            char(clockRepair.method), ...
            1e-3 .* clockRepair.minRxAfterDispatchUs, ...
            1e-3 .* clockRepair.medianRxAfterDispatchUs);
    end
    fprintf("  Event rows: %d\n", height(e2e.eventLatency));
    if ismember("trainer_transition_found", string(e2e.eventLatency.Properties.VariableNames))
        fprintf("  Trainer matched: %d / %d\n", sum(double(e2e.eventLatency.trainer_transition_found) ~= 0), height(e2e.eventLatency));
    end
    if ismember("receiver_transition_found", string(e2e.eventLatency.Properties.VariableNames))
        fprintf("  Receiver matched: %d / %d\n", sum(double(e2e.eventLatency.receiver_transition_found) ~= 0), height(e2e.eventLatency));
    end
    if ismember("scheduled_to_receiver_latency_s", string(e2e.eventLatency.Properties.VariableNames))
        values = double(e2e.eventLatency.scheduled_to_receiver_latency_s);
        values = values(isfinite(values));
        if ~isempty(values)
            fprintf("  Scheduled-to-receiver latency: median %.3f ms, p95 %.3f ms, max %.3f ms\n", ...
                1e3 .* median(values), 1e3 .* prctile(values, 95), 1e3 .* max(values));
        end
    end
end
end

function transmitterConfig = applyNanoTransmitterDefaultsLocal(transmitterConfig, seed)
transmitterConfig.arduinoBoard = normalizeTextScalarLocal(getFieldLocal(transmitterConfig, "arduinoBoard", "Nano33IoT"));

trainerPpm = getFieldLocal(transmitterConfig, "trainerPpm", struct());
trainerPpm.outputPin = normalizeTextScalarLocal(getFieldLocal(trainerPpm, "outputPin", "D3"));
trainerPpm.referencePin = normalizeTextScalarLocal(getFieldLocal(trainerPpm, "referencePin", "D2"));
transmitterConfig.trainerPpm = trainerPpm;

if ~isfield(transmitterConfig, "runLabel") || strlength(normalizeTextScalarLocal(transmitterConfig.runLabel)) == 0
    if isfinite(seed)
        transmitterConfig.runLabel = "Seed_" + string(round(seed)) + "_Nano_Transmitter";
    else
        transmitterConfig.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_Nano_Transmitter";
    end
end
transmitterConfig.runLabel = normalizeTextScalarLocal(transmitterConfig.runLabel);

if ~isfield(transmitterConfig, "outputFolder") || strlength(normalizeTextScalarLocal(transmitterConfig.outputFolder)) == 0
    transmitterConfig.outputFolder = fullfile(fileparts(mfilename("fullpath")), "D_Transmitter_Test");
end
transmitterConfig.outputFolder = normalizeTextScalarLocal(transmitterConfig.outputFolder);

transportConfig = getFieldLocal(transmitterConfig, "arduinoTransport", struct());
transportConfig.serialPort = normalizeTextScalarLocal(getFieldLocal(transportConfig, "serialPort", "COM11"));
transportConfig.loggerOutputFolder = normalizeTextScalarLocal(getFieldLocal( ...
    transportConfig, ...
    "loggerOutputFolder", ...
    fullfile(char(transmitterConfig.outputFolder), char(transmitterConfig.runLabel + "_TransmitterLogger"))));
transmitterConfig.arduinoTransport = transportConfig;
end

function loggerFolder = resolveLoggerFolderFromTransmitterConfigLocal(transmitterConfig)
loggerFolder = "";
if ~isstruct(transmitterConfig)
    return;
end

transportConfig = getFieldLocal(transmitterConfig, "arduinoTransport", struct());
loggerFolder = normalizeTextScalarLocal(getFieldLocal(transportConfig, "loggerOutputFolder", ""));
if strlength(loggerFolder) > 0
    return;
end

outputFolder = normalizeTextScalarLocal(getFieldLocal(transmitterConfig, "outputFolder", ""));
runLabel = normalizeTextScalarLocal(getFieldLocal(transmitterConfig, "runLabel", ""));
if strlength(outputFolder) > 0 && strlength(runLabel) > 0
    loggerFolder = normalizeTextScalarLocal(fullfile(char(outputFolder), char(runLabel + "_TransmitterLogger")));
end
end

function e2e = loadPostOutputsLocal(loggerFolder, outputPrefix)
paths = buildOutputPathsLocal(loggerFolder, outputPrefix);

e2e = struct();
e2e.method = "python_post_transition";
e2e.outputPrefix = string(outputPrefix);
e2e.loggerFolder = string(loggerFolder);
e2e.eventLatency = readtable(paths.eventPath);
e2e.trueE2EEvents = e2e.eventLatency;
e2e.surfaceSummary = readtable(paths.surfaceSummaryPath);
e2e.overallSummary = readtable(paths.overallSummaryPath);
e2e.alignmentDiagnostics = readtable(paths.alignmentPath);
e2e.stateCenters = readtable(paths.stateCentersPath);
e2e.anchorTable = table();
e2e.qualitySummary = table();
e2e.profileEvents = readProfileEventsLocal(loggerFolder);
e2e.hostDispatchLog = readOptionalTableLocal(fullfile(loggerFolder, "host_dispatch_log.csv"));
e2e.boardRxLog = readOptionalTableLocal(fullfile(loggerFolder, "board_rx_log.csv"));
e2e.boardCommitLog = readOptionalTableLocal(fullfile(loggerFolder, "board_commit_log.csv"));
e2e.referenceCapture = readOptionalTableLocal(fullfile(loggerFolder, "reference_capture.csv"));
e2e.trainerPpmCapture = readOptionalTableLocal(fullfile(loggerFolder, "trainer_ppm_capture.csv"));
e2e.receiverCapture = readOptionalTableLocal(fullfile(loggerFolder, "receiver_capture.csv"));
e2e.outputPaths = paths;
end

function outputPaths = buildOutputPathsLocal(loggerFolder, outputPrefix)
prefix = fullfile(loggerFolder, outputPrefix);
outputPaths = struct();
outputPaths.eventPath = string(prefix + "_events.csv");
outputPaths.surfaceSummaryPath = string(prefix + "_surface_summary.csv");
outputPaths.overallSummaryPath = string(prefix + "_overall_summary.csv");
outputPaths.alignmentPath = string(prefix + "_alignment.csv");
outputPaths.stateCentersPath = string(prefix + "_state_centers.csv");
end

function outputPaths = buildNanoPhysicalPathsLocal(loggerFolder, outputPrefix)
prefix = fullfile(loggerFolder, outputPrefix + "_physical");
outputPaths = struct();
outputPaths.eventPath = string(prefix + "_events.csv");
outputPaths.surfaceSummaryPath = string(prefix + "_surface_summary.csv");
outputPaths.overallSummaryPath = string(prefix + "_overall_summary.csv");
outputPaths.chainSummaryPath = string(fullfile(loggerFolder, outputPrefix + "_chain_summary.csv"));
end

function runPythonPostProcessorLocal(loggerFolder, outputPrefix, pythonExecutable, pythonScriptPath, rootFolder, postConfig)
if ~isfile(pythonScriptPath)
    error("Transmitter_Test_Nano_E2E:MissingPostProcessor", ...
        "Python post-processing script not found: %s", ...
        char(pythonScriptPath));
end

commandParts = [ ...
    quoteWindowsArgumentLocal(resolvePythonExecutableLocal(pythonExecutable)), ...
    quoteWindowsArgumentLocal(pythonScriptPath), ...
    "--logger-folder", quoteWindowsArgumentLocal(loggerFolder), ...
    "--output-prefix", quoteWindowsArgumentLocal(outputPrefix), ...
    "--root", quoteWindowsArgumentLocal(rootFolder)];

commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "offsetSearchMinMs", "--offset-search-min-ms");
commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "offsetSearchMaxMs", "--offset-search-max-ms");
commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "offsetBinMs", "--offset-bin-ms");
commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "offsetRefineMs", "--offset-refine-ms");
commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "trainerMatchWindowMs", "--trainer-match-window-ms");
commandParts = appendOptionalNumericArgLocal(commandParts, postConfig, "receiverMatchWindowMs", "--receiver-match-window-ms");
commandParts = appendOptionalIntegerArgLocal(commandParts, postConfig, "minStablePulses", "--min-stable-pulses");

commandText = strjoin(commandParts, " ");
[status, outputText] = system(char(commandText));
if status ~= 0
    error("Transmitter_Test_Nano_E2E:PostProcessorFailed", ...
        "Python post-processing failed with exit code %d.\nCommand: %s\nOutput:\n%s", ...
        status, char(commandText), char(string(outputText)));
end
end

function commandParts = appendOptionalNumericArgLocal(commandParts, cfg, fieldName, cliName)
if ~isstruct(cfg) || ~isfield(cfg, fieldName)
    return;
end
value = double(cfg.(fieldName));
if ~isfinite(value)
    return;
end
commandParts = [commandParts, string(cliName), string(sprintf("%.9g", value))]; %#ok<AGROW>
end

function commandParts = appendOptionalIntegerArgLocal(commandParts, cfg, fieldName, cliName)
if ~isstruct(cfg) || ~isfield(cfg, fieldName)
    return;
end
value = double(cfg.(fieldName));
if ~isfinite(value)
    return;
end
commandParts = [commandParts, string(cliName), string(round(value))]; %#ok<AGROW>
end

function profileEvents = readProfileEventsLocal(loggerFolder)
profileEvents = table();
workbookPath = fullfile(fileparts(loggerFolder), stripLoggerSuffixLocal(loggerFolder) + ".xlsx");
if ~isfile(workbookPath)
    return;
end
try
    profileEvents = readtable(workbookPath, "Sheet", "ProfileEvents");
catch
    profileEvents = table();
end
end

function tableData = readOptionalTableLocal(filePath)
if isfile(filePath)
    tableData = readtable(filePath);
else
    tableData = table();
end
end

function transmitterConfig = applyComparableProfileDefaultsLocal( ...
    transmitterConfig, ...
    seed, ...
    enforceArduinoStepTrainDefaults, ...
    recordLeadSeconds, ...
    recordLagSeconds, ...
    commandActiveSeconds)
if ~isfield(transmitterConfig, "commandProfile") || isempty(transmitterConfig.commandProfile)
    transmitterConfig.commandProfile = struct();
end

if enforceArduinoStepTrainDefaults
    transmitterConfig.commandProfile.type = "latency_step_train";
    transmitterConfig.commandProfile.sampleTimeSeconds = 0.02;
    transmitterConfig.commandProfile.preCommandNeutralSeconds = recordLeadSeconds;
    transmitterConfig.commandProfile.postCommandNeutralSeconds = recordLagSeconds;
    transmitterConfig.commandProfile.durationSeconds = commandActiveSeconds;
    transmitterConfig.commandProfile.amplitudeDegrees = 45.0;
    transmitterConfig.commandProfile.offsetDegrees = 0.0;
    transmitterConfig.commandProfile.frequencyHz = 0.5;
    transmitterConfig.commandProfile.phaseDegrees = 90.0;
    transmitterConfig.commandProfile.doubletHoldSeconds = 0.5;
    transmitterConfig.commandProfile.eventHoldSeconds = 0.20;
    transmitterConfig.commandProfile.eventNeutralHoldSeconds = 0.10;
    transmitterConfig.commandProfile.eventDwellSeconds = 0.60;
    transmitterConfig.commandProfile.eventRandomJitterSeconds = 0.05;
    if isfinite(seed)
        transmitterConfig.commandProfile.randomSeed = seed;
    else
        transmitterConfig.commandProfile.randomSeed = 5;
    end
    return;
end

if ~isfield(transmitterConfig.commandProfile, "type")
    transmitterConfig.commandProfile.type = "latency_step_train";
end
if ~isfield(transmitterConfig.commandProfile, "sampleTimeSeconds")
    transmitterConfig.commandProfile.sampleTimeSeconds = 0.02;
end
if ~isfield(transmitterConfig.commandProfile, "randomSeed")
    if isfinite(seed)
        transmitterConfig.commandProfile.randomSeed = seed;
    else
        transmitterConfig.commandProfile.randomSeed = 5;
    end
end
end

function loggerFolder = resolveLatestLoggerFolderLocal(rootFolder)
folderInfo = dir(fullfile(rootFolder, "*_TransmitterLogger"));
if isempty(folderInfo)
    error("Transmitter_Test_Nano_E2E:MissingLoggerFolder", ...
        "No *_TransmitterLogger folder found in %s.", ...
        char(rootFolder));
end
[~, idx] = max([folderInfo.datenum]);
loggerFolder = string(fullfile(folderInfo(idx).folder, folderInfo(idx).name));
end

function runLabel = stripLoggerSuffixLocal(loggerFolder)
[~, folderName] = fileparts(char(loggerFolder));
runLabel = string(regexprep(folderName, "_TransmitterLogger$", ""));
end

function ensureFolderExistsLocal(folderPath)
if ~isfolder(folderPath)
    error("Transmitter_Test_Nano_E2E:MissingLoggerFolder", ...
        "Logger folder not found: %s", ...
        char(folderPath));
end
end

function outputPaths = buildNanoPhysicalOutputsLocal(loggerFolder, outputPrefix)
outputPaths = buildNanoPhysicalPathsLocal(loggerFolder, outputPrefix);
directLatencyPath = fullfile(loggerFolder, "direct_latency_events.csv");
if ~isfile(directLatencyPath)
    return;
end

directEvents = readtable(directLatencyPath);
physicalEvents = convertDirectLatencyToPhysicalEventsLocal(loggerFolder, directEvents);
if isempty(physicalEvents)
    return;
end

writetable(physicalEvents, char(outputPaths.eventPath));
writetable(buildNanoPhysicalSurfaceSummaryLocal(physicalEvents), char(outputPaths.surfaceSummaryPath));
writetable(buildNanoPhysicalOverallSummaryLocal(physicalEvents), char(outputPaths.overallSummaryPath));
end

function physicalEvents = convertDirectLatencyToPhysicalEventsLocal(loggerFolder, directEvents)
requiredColumns = [ ...
    "surface_name", ...
    "sample_sequence", ...
    "command_sequence", ...
    "scheduled_time_s", ...
    "command_dispatch_s", ...
    "receiver_transition_s", ...
    "delta_ppm_us"];
if ~hasColumnsLocal(directEvents, requiredColumns)
    physicalEvents = table();
    return;
end

hostDispatchLog = readOptionalTableLocal(fullfile(loggerFolder, "host_dispatch_log.csv"));
boardRxLog = readOptionalTableLocal(fullfile(loggerFolder, "board_rx_log.csv"));
boardCommitLog = readOptionalTableLocal(fullfile(loggerFolder, "board_commit_log.csv"));

directEvents.surface_name = normalizeTextColumnLocal(directEvents.surface_name);
deltaPpmUs = double(directEvents.delta_ppm_us);
transitionMask = ...
    isfinite(deltaPpmUs) & ...
    abs(deltaPpmUs) > 0 & ...
    isfinite(directEvents.receiver_transition_s);
if ~any(transitionMask)
    physicalEvents = table();
    return;
end

physicalEvents = directEvents(transitionMask, :);
physicalEvents = sortrows(physicalEvents, {'surface_name', 'command_sequence'});
physicalEvents = unique( ...
    physicalEvents(:, intersect( ...
    {'surface_name', 'sample_sequence', 'command_sequence', 'scheduled_time_s', 'command_dispatch_s', ...
    'reference_strobe_s', 'receiver_transition_s', 'delta_ppm_us', 'anchor_source'}, ...
    physicalEvents.Properties.VariableNames, ...
    'stable')), ...
    'rows', ...
    'stable');

if ~isempty(hostDispatchLog) && hasColumnsLocal(hostDispatchLog, ["surface_name", "sample_sequence", "command_sequence", "scheduled_time_s", "command_dispatch_s"])
    hostDispatchLog.surface_name = normalizeTextColumnLocal(hostDispatchLog.surface_name);
    hostSubset = unique( ...
        hostDispatchLog(:, {'surface_name', 'sample_sequence', 'command_sequence', 'scheduled_time_s', 'command_dispatch_s'}), ...
        'rows', ...
        'stable');
    hostSubset = renamevars(hostSubset, {'scheduled_time_s', 'command_dispatch_s'}, {'scheduled_time_host_s', 'command_dispatch_host_s'});
    physicalEvents = outerjoin(physicalEvents, hostSubset, ...
        'Keys', {'surface_name', 'sample_sequence', 'command_sequence'}, ...
        'MergeKeys', true, ...
        'Type', 'left');
end

if ~isempty(boardRxLog) && hasColumnsLocal(boardRxLog, ["surface_name", "sample_sequence", "command_sequence", "rx_time_s"])
    boardRxLog.surface_name = normalizeTextColumnLocal(boardRxLog.surface_name);
    rxSubset = unique( ...
        boardRxLog(:, {'surface_name', 'sample_sequence', 'command_sequence', 'rx_time_s'}), ...
        'rows', ...
        'stable');
    rxSubset = renamevars(rxSubset, 'rx_time_s', 'board_rx_repaired_s');
    physicalEvents = outerjoin(physicalEvents, rxSubset, ...
        'Keys', {'surface_name', 'sample_sequence', 'command_sequence'}, ...
        'MergeKeys', true, ...
        'Type', 'left');
end

if ~isempty(boardCommitLog) && ismember("sample_sequence", string(boardCommitLog.Properties.VariableNames))
    if ~ismember("strobe_time_s", string(boardCommitLog.Properties.VariableNames)) && ...
            ismember("commit_time_s", string(boardCommitLog.Properties.VariableNames))
        boardCommitLog.strobe_time_s = double(boardCommitLog.commit_time_s);
    end
    commitKeep = intersect({'sample_sequence', 'commit_time_s', 'strobe_time_s'}, boardCommitLog.Properties.VariableNames, 'stable');
    commitSubset = unique(boardCommitLog(:, commitKeep), 'rows', 'stable');
    renameSource = intersect({'commit_time_s', 'strobe_time_s'}, commitSubset.Properties.VariableNames, 'stable');
    renameTarget = strings(0, 1);
    for renameIndex = 1:numel(renameSource)
        if renameSource(renameIndex) == "commit_time_s"
            renameTarget(end + 1, 1) = "board_commit_repaired_s"; %#ok<AGROW>
        else
            renameTarget(end + 1, 1) = "reference_time_repaired_s"; %#ok<AGROW>
        end
    end
    if ~isempty(renameSource)
        commitSubset = renamevars(commitSubset, cellstr(renameSource), cellstr(renameTarget));
    end
    physicalEvents = outerjoin(physicalEvents, commitSubset, ...
        'Keys', {'sample_sequence'}, ...
        'MergeKeys', true, ...
        'Type', 'left');
end

physicalEvents.scheduled_time_s = preferNumericColumnsLocal(physicalEvents, ["scheduled_time_host_s", "scheduled_time_s"], strings(0, 1));
physicalEvents.command_dispatch_s = preferNumericColumnsLocal(physicalEvents, ["command_dispatch_host_s", "command_dispatch_s"], strings(0, 1));
physicalEvents.board_rx_s = preferNumericColumnsLocal(physicalEvents, ["board_rx_repaired_s", "board_rx_s"], strings(0, 1));
physicalEvents.board_commit_s = preferNumericColumnsLocal(physicalEvents, ["board_commit_repaired_s", "board_commit_s"], strings(0, 1));
physicalEvents.reference_time_s = preferNumericColumnsLocal(physicalEvents, ["reference_strobe_s", "anchor_s", "reference_time_repaired_s"], strings(0, 1));
physicalEvents.receiver_transition_s = double(physicalEvents.receiver_transition_s);

physicalEvents.host_scheduling_delay_s = sanitizeLatencySegmentLocal(double(physicalEvents.command_dispatch_s) - double(physicalEvents.scheduled_time_s), 5e-4);
physicalEvents.dispatch_to_rx_latency_s = sanitizeLatencySegmentLocal(double(physicalEvents.board_rx_s) - double(physicalEvents.command_dispatch_s), 5e-4);
physicalEvents.rx_to_commit_latency_s = sanitizeLatencySegmentLocal(double(physicalEvents.board_commit_s) - double(physicalEvents.board_rx_s), 5e-4);
physicalEvents.commit_to_reference_latency_s = sanitizeLatencySegmentLocal(double(physicalEvents.reference_time_s) - double(physicalEvents.board_commit_s), 5e-4);
physicalEvents.reference_to_receiver_latency_s = sanitizeLatencySegmentLocal(double(physicalEvents.receiver_transition_s) - double(physicalEvents.reference_time_s), 5e-4);

validMask = isfinite(physicalEvents.host_scheduling_delay_s) & ...
    isfinite(physicalEvents.dispatch_to_rx_latency_s) & ...
    isfinite(physicalEvents.rx_to_commit_latency_s) & ...
    isfinite(physicalEvents.commit_to_reference_latency_s) & ...
    isfinite(physicalEvents.reference_to_receiver_latency_s);
physicalEvents = physicalEvents(validMask, :);
if isempty(physicalEvents)
    return;
end

physicalEvents.commit_to_receiver_latency_s = ...
    double(physicalEvents.commit_to_reference_latency_s) + ...
    double(physicalEvents.reference_to_receiver_latency_s);
physicalEvents.dispatch_to_receiver_latency_s = ...
    double(physicalEvents.dispatch_to_rx_latency_s) + ...
    double(physicalEvents.rx_to_commit_latency_s) + ...
    double(physicalEvents.commit_to_receiver_latency_s);
physicalEvents.scheduled_to_receiver_latency_s = ...
    double(physicalEvents.host_scheduling_delay_s) + ...
    double(physicalEvents.dispatch_to_receiver_latency_s);
physicalEvents.delta_ppm_us = double(physicalEvents.delta_ppm_us);
end

function surfaceSummary = buildNanoPhysicalSurfaceSummaryLocal(physicalEvents)
surfaceNames = unique(reshape(string(physicalEvents.surface_name), [], 1), 'stable');
rowTable = table();
for surfaceIndex = 1:numel(surfaceNames)
    surfaceName = surfaceNames(surfaceIndex);
    surfaceRows = physicalEvents(physicalEvents.surface_name == surfaceName, :);
    rowData = struct();
    rowData.SurfaceName = surfaceName;
    rowData.IsActive = true;
    rowData = addLatencyStatsToStructLocal(rowData, "HostSchedulingDelay", double(surfaceRows.host_scheduling_delay_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ComputerToArduinoRxLatency", double(surfaceRows.dispatch_to_rx_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ArduinoReceiveToPpmCommitLatency", double(surfaceRows.rx_to_commit_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "CommitToReferenceLatency", double(surfaceRows.commit_to_reference_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ReferenceToReceiver", double(surfaceRows.reference_to_receiver_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ScheduledToReceiver", double(surfaceRows.scheduled_to_receiver_latency_s));
    rowTable = [rowTable; struct2table(rowData, 'AsArray', true)]; %#ok<AGROW>
end
surfaceSummary = rowTable;
end

function overallSummary = buildNanoPhysicalOverallSummaryLocal(physicalEvents)
metricNames = [ ...
    "host_scheduling_delay_s", ...
    "dispatch_to_rx_latency_s", ...
    "rx_to_commit_latency_s", ...
    "commit_to_reference_latency_s", ...
    "reference_to_receiver_latency_s", ...
    "scheduled_to_receiver_latency_s"];
rows = table();
for metricIndex = 1:numel(metricNames)
    metricName = metricNames(metricIndex);
    stats = computeLatencyStatsLocal(double(physicalEvents.(metricName)));
    row = table( ...
        metricName, ...
        stats.sampleCount, ...
        stats.meanValue, ...
        stats.medianValue, ...
        stats.p95Value, ...
        stats.p99Value, ...
        stats.maxValue, ...
        'VariableNames', {'metric', 'count', 'mean_s', 'median_s', 'p95_s', 'p99_s', 'max_s'});
    rows = [rows; row]; %#ok<AGROW>
end
overallSummary = rows;
end

function rowData = addLatencyStatsToStructLocal(rowData, prefix, values)
stats = computeLatencyStatsLocal(values);
rowData.(char(prefix + "SampleCount")) = stats.sampleCount;
rowData.(char(prefix + "Mean_s")) = stats.meanValue;
rowData.(char(prefix + "Median_s")) = stats.medianValue;
rowData.(char(prefix + "P95_s")) = stats.p95Value;
rowData.(char(prefix + "P99_s")) = stats.p99Value;
rowData.(char(prefix + "Max_s")) = stats.maxValue;
end

function stats = computeLatencyStatsLocal(values)
values = reshape(double(values), [], 1);
values = values(isfinite(values));
if isempty(values)
    stats = struct( ...
        "sampleCount", 0, ...
        "meanValue", NaN, ...
        "medianValue", NaN, ...
        "p95Value", NaN, ...
        "p99Value", NaN, ...
        "maxValue", NaN);
    return;
end

stats = struct( ...
    "sampleCount", numel(values), ...
    "meanValue", mean(values, 'omitnan'), ...
    "medianValue", median(values, 'omitnan'), ...
    "p95Value", prctile(values, 95), ...
    "p99Value", prctile(values, 99), ...
    "maxValue", max(values));
end

function printNanoPhysicalSummaryLocal(overallSummary)
metricNames = string(overallSummary.metric);
scheduleMask = metricNames == "scheduled_to_receiver_latency_s";
if any(scheduleMask)
    row = overallSummary(find(scheduleMask, 1, 'first'), :);
    fprintf("  Trusted scheduled-to-receiver latency: median %.3f ms, p95 %.3f ms, max %.3f ms\n", ...
        1e3 .* double(row.median_s), ...
        1e3 .* double(row.p95_s), ...
        1e3 .* double(row.max_s));
end
end

function integrateNanoPhysicalSummaryLocal(loggerFolder, outputPrefix, physicalPaths)
eventPath = fullfile(loggerFolder, outputPrefix + "_events.csv");
overallPath = fullfile(loggerFolder, outputPrefix + "_overall_summary.csv");
surfacePath = fullfile(loggerFolder, outputPrefix + "_surface_summary.csv");
chainPath = char(physicalPaths.chainSummaryPath);

postEvents = readOptionalTableLocal(eventPath);
physicalOverall = readOptionalTableLocal(char(physicalPaths.overallSummaryPath));
physicalSurface = readOptionalTableLocal(char(physicalPaths.surfaceSummaryPath));
physicalEvents = readOptionalTableLocal(char(physicalPaths.eventPath));

if ~isempty(postEvents) && ~isempty(physicalEvents)
    postEvents = applyTrustedChainToPostEventsLocal(postEvents, physicalEvents);
    writetable(postEvents, eventPath);
end

postOverall = buildNanoTrustedOverallSummaryLocal(postEvents);
postSurface = buildNanoTrustedSurfaceSummaryLocal(postEvents);

if ~isempty(postOverall)
    writetable(postOverall, overallPath);
end

if ~isempty(postSurface)
    writetable(postSurface, surfacePath);
end

chainSummary = buildNanoChainSummaryLocal(postOverall, physicalOverall);
if ~isempty(chainSummary)
    writetable(chainSummary, chainPath);
end
end

function trustedEvents = applyTrustedChainToPostEventsLocal(postEvents, physicalEvents)
trustedEvents = postEvents;
if isempty(postEvents) || isempty(physicalEvents)
    return;
end

postEvents.surface_name = normalizeTextColumnLocal(postEvents.surface_name);
physicalEvents.surface_name = normalizeTextColumnLocal(physicalEvents.surface_name);
postEvents.row_index = transpose((1:height(postEvents)));
joinKeys = {'surface_name', 'sample_sequence', 'command_sequence'};
rightVariables = { ...
    'board_rx_s', ...
    'board_commit_s', ...
    'reference_time_s', ...
    'host_scheduling_delay_s', ...
    'dispatch_to_rx_latency_s', ...
    'rx_to_commit_latency_s', ...
    'commit_to_reference_latency_s', ...
    'reference_to_receiver_latency_s', ...
    'commit_to_receiver_latency_s', ...
    'dispatch_to_receiver_latency_s', ...
    'scheduled_to_receiver_latency_s'};
trustedEvents = outerjoin( ...
    postEvents, ...
    physicalEvents(:, [joinKeys, rightVariables]), ...
    'Keys', joinKeys, ...
    'MergeKeys', true, ...
    'Type', 'left', ...
    'LeftVariables', postEvents.Properties.VariableNames, ...
    'RightVariables', rightVariables);
trustedEvents = sortrows(trustedEvents, 'row_index');
trustedEvents.row_index = [];

trustedEvents.board_rx_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["board_rx_s_physicalEvents", "board_rx_s"], ...
    strings(0, 1));
trustedEvents.commit_time_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["board_commit_s_physicalEvents", "commit_time_s"], ...
    ["board_commit_s_postEvents"]);
trustedEvents.reference_time_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["reference_time_s_physicalEvents", "reference_time_s"], ...
    ["reference_time_s_postEvents"]);
trustedEvents.ppm_time_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["reference_time_s_physicalEvents", "ppm_time_s"], ...
    ["ppm_time_s_postEvents", "reference_time_s_postEvents"]);
trustedEvents.host_scheduling_delay_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["host_scheduling_delay_s_physicalEvents", "host_scheduling_delay_s"], ...
    ["host_scheduling_delay_s_postEvents"]);
trustedEvents.dispatch_to_rx_latency_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["dispatch_to_rx_latency_s_physicalEvents", "dispatch_to_rx_latency_s"], ...
    ["dispatch_to_rx_latency_s_postEvents"]);
trustedEvents.rx_to_commit_latency_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["rx_to_commit_latency_s_physicalEvents", "rx_to_commit_latency_s"], ...
    ["rx_to_commit_latency_s_postEvents"]);
trustedEvents.commit_to_reference_latency_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["commit_to_reference_latency_s_physicalEvents", "commit_to_reference_latency_s"], ...
    ["commit_to_reference_latency_s_postEvents"]);
trustedEvents.reference_to_receiver_latency_s = preferNumericColumnsLocal( ...
    trustedEvents, ...
    ["reference_to_receiver_latency_s_physicalEvents", "reference_to_receiver_latency_s"], ...
    ["reference_to_receiver_latency_s_postEvents"]);
trustedEvents.commit_to_receiver_latency_s = ...
    trustedEvents.commit_to_reference_latency_s + trustedEvents.reference_to_receiver_latency_s;
invalidCommitMask = ~isfinite(trustedEvents.commit_to_reference_latency_s) | ~isfinite(trustedEvents.reference_to_receiver_latency_s);
trustedEvents.commit_to_receiver_latency_s(invalidCommitMask) = nan;
trustedEvents.dispatch_to_receiver_latency_s = ...
    trustedEvents.dispatch_to_rx_latency_s + trustedEvents.rx_to_commit_latency_s + trustedEvents.commit_to_receiver_latency_s;
invalidDispatchMask = ...
    ~isfinite(trustedEvents.dispatch_to_rx_latency_s) | ...
    ~isfinite(trustedEvents.rx_to_commit_latency_s) | ...
    ~isfinite(trustedEvents.commit_to_receiver_latency_s);
trustedEvents.dispatch_to_receiver_latency_s(invalidDispatchMask) = nan;
trustedEvents.scheduled_to_receiver_latency_s = ...
    trustedEvents.host_scheduling_delay_s + trustedEvents.dispatch_to_receiver_latency_s;
invalidScheduleMask = ...
    ~isfinite(trustedEvents.host_scheduling_delay_s) | ...
    ~isfinite(trustedEvents.dispatch_to_receiver_latency_s);
trustedEvents.scheduled_to_receiver_latency_s(invalidScheduleMask) = nan;
trustedEvents.true_dispatch_to_receiver_latency_s = trustedEvents.dispatch_to_receiver_latency_s;
trustedEvents.true_scheduled_to_receiver_latency_s = trustedEvents.scheduled_to_receiver_latency_s;
trustedEvents.anchor_to_receiver_latency_s = trustedEvents.reference_to_receiver_latency_s;
suffixMask = endsWith(string(trustedEvents.Properties.VariableNames), ["_postEvents", "_physicalEvents"]);
trustedEvents(:, suffixMask) = [];
end

function values = preferNumericColumnsLocal(tableData, preferredNames, fallbackNames)
values = nan(height(tableData), 1);
preferredNames = reshape(string(preferredNames), 1, []);
fallbackNames = reshape(string(fallbackNames), 1, []);

for preferredIndex = 1:numel(preferredNames)
    preferredName = preferredNames(preferredIndex);
    if ismember(preferredName, string(tableData.Properties.VariableNames))
        candidateValues = double(tableData.(char(preferredName)));
        replaceMask = ~isfinite(values) & isfinite(candidateValues);
        values(replaceMask) = candidateValues(replaceMask);
    end
end

for fallbackIndex = 1:numel(fallbackNames)
    fallbackName = fallbackNames(fallbackIndex);
    if ismember(fallbackName, string(tableData.Properties.VariableNames))
        replacementValues = double(tableData.(char(fallbackName)));
        replaceMask = ~isfinite(values) & isfinite(replacementValues);
        values(replaceMask) = replacementValues(replaceMask);
    end
end
end

function values = sanitizeLatencySegmentLocal(values, toleranceSeconds)
values = reshape(double(values), [], 1);
smallNegativeMask = isfinite(values) & values < 0 & values >= -abs(double(toleranceSeconds));
values(smallNegativeMask) = 0.0;
invalidNegativeMask = isfinite(values) & values < -abs(double(toleranceSeconds));
values(invalidNegativeMask) = nan;
end

function overallSummary = buildNanoTrustedOverallSummaryLocal(trustedEvents)
overallSummary = table();
if isempty(trustedEvents)
    return;
end
metricNames = [ ...
    "host_scheduling_delay_s", ...
    "dispatch_to_rx_latency_s", ...
    "rx_to_commit_latency_s", ...
    "commit_to_reference_latency_s", ...
    "reference_to_receiver_latency_s", ...
    "scheduled_to_receiver_latency_s"];
for metricIndex = 1:numel(metricNames)
    metricName = metricNames(metricIndex);
    if ~ismember(metricName, string(trustedEvents.Properties.VariableNames))
        continue;
    end
    stats = computeLatencyStatsLocal(double(trustedEvents.(metricName)));
    row = table( ...
        metricName, ...
        stats.sampleCount, ...
        stats.meanValue, ...
        stats.medianValue, ...
        stats.p95Value, ...
        stats.p99Value, ...
        stats.maxValue, ...
        'VariableNames', {'metric', 'count', 'mean_s', 'median_s', 'p95_s', 'p99_s', 'max_s'});
    overallSummary = [overallSummary; row]; %#ok<AGROW>
end
end

function surfaceSummary = buildNanoTrustedSurfaceSummaryLocal(trustedEvents)
surfaceSummary = table();
if isempty(trustedEvents) || ~ismember("surface_name", string(trustedEvents.Properties.VariableNames))
    return;
end
surfaceNames = unique(reshape(string(trustedEvents.surface_name), [], 1), 'stable');
for surfaceIndex = 1:numel(surfaceNames)
    surfaceName = surfaceNames(surfaceIndex);
    surfaceRows = trustedEvents(trustedEvents.surface_name == surfaceName, :);
    rowData = struct();
    rowData.SurfaceName = surfaceName;
    rowData.IsActive = true;
    rowData = addLatencyStatsToStructLocal(rowData, "HostSchedulingDelay", double(surfaceRows.host_scheduling_delay_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ComputerToArduinoRxLatency", double(surfaceRows.dispatch_to_rx_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ArduinoReceiveToPpmCommitLatency", double(surfaceRows.rx_to_commit_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "CommitToReferenceLatency", double(surfaceRows.commit_to_reference_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ReferenceToReceiverLatency", double(surfaceRows.reference_to_receiver_latency_s));
    rowData = addLatencyStatsToStructLocal(rowData, "ScheduledToReceiverLatency", double(surfaceRows.scheduled_to_receiver_latency_s));
    surfaceSummary = [surfaceSummary; struct2table(rowData, 'AsArray', true)]; %#ok<AGROW>
end
end

function overallSummary = mergeOverallSummaryTablesLocal(postOverall, physicalOverall)
overallSummary = postOverall;
if ~ismember("metric", string(physicalOverall.Properties.VariableNames))
    return;
end

existingMetricNames = strings(0, 1);
if ismember("metric", string(overallSummary.Properties.VariableNames))
    existingMetricNames = reshape(string(overallSummary.metric), [], 1);
end

for rowIndex = 1:height(physicalOverall)
    metricName = string(physicalOverall.metric(rowIndex));
    if any(existingMetricNames == metricName)
        overallSummary(existingMetricNames == metricName, :) = physicalOverall(rowIndex, :);
    else
        overallSummary = [overallSummary; physicalOverall(rowIndex, :)]; %#ok<AGROW>
        existingMetricNames = [existingMetricNames; metricName]; %#ok<AGROW>
    end
end
end

function surfaceSummary = mergeSurfaceSummaryTablesLocal(postSurface, physicalSurface)
surfaceSummary = postSurface;
if ~ismember("SurfaceName", string(postSurface.Properties.VariableNames)) || ...
        ~ismember("SurfaceName", string(physicalSurface.Properties.VariableNames))
    return;
end

physicalSurface = renamePhysicalSurfaceColumnsLocal(physicalSurface);
existingNames = string(surfaceSummary.SurfaceName);

for rowIndex = 1:height(physicalSurface)
    surfaceName = string(physicalSurface.SurfaceName(rowIndex));
    targetMask = existingNames == surfaceName;
    if ~any(targetMask)
        surfaceSummary = outerjoin(surfaceSummary, physicalSurface(rowIndex, :), ...
            'Keys', "SurfaceName", ...
            'MergeKeys', true, ...
            'Type', 'full');
        existingNames = string(surfaceSummary.SurfaceName);
        continue;
    end

    for variableIndex = 1:numel(physicalSurface.Properties.VariableNames)
        variableName = string(physicalSurface.Properties.VariableNames{variableIndex});
        if variableName == "SurfaceName" || variableName == "IsActive"
            continue;
        end
        if ~ismember(variableName, string(surfaceSummary.Properties.VariableNames))
            surfaceSummary.(char(variableName)) = nan(height(surfaceSummary), 1);
        end
        surfaceSummary.(char(variableName))(targetMask) = physicalSurface.(char(variableName))(rowIndex);
    end
end
end

function physicalSurface = renamePhysicalSurfaceColumnsLocal(physicalSurface)
renameMap = { ...
    "ReferenceToTrainerSampleCount", "AnchorToPpmLatencySampleCount"; ...
    "ReferenceToTrainerMean_s", "AnchorToPpmLatencyMean_s"; ...
    "ReferenceToTrainerMedian_s", "AnchorToPpmLatencyMedian_s"; ...
    "ReferenceToTrainerP95_s", "AnchorToPpmLatencyP95_s"; ...
    "ReferenceToTrainerP99_s", "AnchorToPpmLatencyP99_s"; ...
    "ReferenceToTrainerMax_s", "AnchorToPpmLatencyMax_s"; ...
    "ReferenceToReceiverSampleCount", "AnchorToReceiverLatencySampleCount"; ...
    "ReferenceToReceiverMean_s", "AnchorToReceiverLatencyMean_s"; ...
    "ReferenceToReceiverMedian_s", "AnchorToReceiverLatencyMedian_s"; ...
    "ReferenceToReceiverP95_s", "AnchorToReceiverLatencyP95_s"; ...
    "ReferenceToReceiverP99_s", "AnchorToReceiverLatencyP99_s"; ...
    "ReferenceToReceiverMax_s", "AnchorToReceiverLatencyMax_s"};

for rowIndex = 1:size(renameMap, 1)
    sourceName = string(renameMap{rowIndex, 1});
    targetName = string(renameMap{rowIndex, 2});
    if ismember(sourceName, string(physicalSurface.Properties.VariableNames))
        physicalSurface = renamevars(physicalSurface, char(sourceName), char(targetName));
    end
end
end

function chainSummary = buildNanoChainSummaryLocal(postOverall, physicalOverall)
chainSummary = table();
orderedMetrics = [ ...
    "host_scheduling_delay_s", ...
    "dispatch_to_rx_latency_s", ...
    "rx_to_commit_latency_s", ...
    "commit_to_reference_latency_s", ...
    "reference_to_receiver_latency_s", ...
    "scheduled_to_receiver_latency_s"];
sourceLabels = [ ...
    "host", ...
    "repaired_host_board", ...
    "repaired_host_board", ...
    "physical_capture", ...
    "physical_capture", ...
    "trusted_full_system"];
descriptionLabels = [ ...
    "Host scheduled to host dispatch", ...
    "Host dispatch to Nano receive", ...
    "Nano receive to commit", ...
    "Nano commit to reference strobe", ...
    "Reference strobe to receiver transition", ...
    "Scheduled command time to receiver transition"];

combinedSummary = postOverall;
if isempty(combinedSummary)
    combinedSummary = physicalOverall;
end
if isempty(combinedSummary) || ~ismember("metric", string(combinedSummary.Properties.VariableNames))
    return;
end

metricNames = reshape(string(combinedSummary.metric), [], 1);
for metricIndex = 1:numel(orderedMetrics)
    metricName = orderedMetrics(metricIndex);
    rowMask = metricNames == metricName;
    if ~any(rowMask)
        continue;
    end
    sourceRow = combinedSummary(find(rowMask, 1, 'first'), :);
    chainRow = table( ...
        metricName, ...
        sourceLabels(metricIndex), ...
        descriptionLabels(metricIndex), ...
        double(sourceRow.count), ...
        double(sourceRow.mean_s), ...
        double(sourceRow.median_s), ...
        double(sourceRow.p95_s), ...
        double(sourceRow.p99_s), ...
        double(sourceRow.max_s), ...
        'VariableNames', {'metric', 'source', 'description', 'count', 'mean_s', 'median_s', 'p95_s', 'p99_s', 'max_s'});
    chainSummary = [chainSummary; chainRow]; %#ok<AGROW>
end
end

function printNanoTrustedChainSummaryLocal(chainSummary)
metricNames = reshape(string(chainSummary.metric), [], 1);
orderedMetrics = [ ...
    "host_scheduling_delay_s", ...
    "dispatch_to_rx_latency_s", ...
    "rx_to_commit_latency_s", ...
    "commit_to_reference_latency_s", ...
    "reference_to_receiver_latency_s", ...
    "scheduled_to_receiver_latency_s"];
labels = [ ...
    "Host scheduling delay", ...
    "Dispatch-to-RX latency", ...
    "RX-to-commit latency", ...
    "Commit-to-reference latency", ...
    "Reference-to-receiver latency", ...
    "Trusted scheduled-to-receiver latency"];
for metricIndex = 1:numel(orderedMetrics)
    rowMask = metricNames == orderedMetrics(metricIndex);
    if ~any(rowMask)
        continue;
    end
    row = chainSummary(find(rowMask, 1, 'first'), :);
    fprintf("  %s: median %.3f ms, p95 %.3f ms, max %.3f ms\n", ...
        char(labels(metricIndex)), ...
        1e3 .* double(row.median_s), ...
        1e3 .* double(row.p95_s), ...
        1e3 .* double(row.max_s));
end
end

function repairResult = repairNanoLoggerClockOfflineLocal(loggerFolder)
repairResult = struct( ...
    "applied", false, ...
    "rewroteLogs", false, ...
    "method", "skipped", ...
    "clockSlope", NaN, ...
    "clockInterceptUs", NaN, ...
    "oneWayBaselineUs", NaN, ...
    "testStartOffsetUs", NaN, ...
    "minRxAfterDispatchUs", NaN, ...
    "medianRxAfterDispatchUs", NaN, ...
    "p95RxAfterDispatchUs", NaN, ...
    "boardRxLog", table(), ...
    "boardCommitLog", table());

hostDispatchLog = readOptionalTableLocal(fullfile(loggerFolder, "host_dispatch_log.csv"));
boardRxLog = readOptionalTableLocal(fullfile(loggerFolder, "board_rx_log.csv"));
boardCommitLog = readOptionalTableLocal(fullfile(loggerFolder, "board_commit_log.csv"));
boardSyncLog = readOptionalTableLocal(fullfile(loggerFolder, "board_sync_log.csv"));

if isempty(hostDispatchLog) || isempty(boardRxLog) || isempty(boardCommitLog)
    return;
end

if ~hasColumnsLocal(hostDispatchLog, ["surface_name", "command_sequence", "command_dispatch_us", "command_dispatch_s"]) || ...
        ~hasColumnsLocal(boardRxLog, ["surface_name", "command_sequence", "rx_us"]) || ...
        ~hasColumnsLocal(boardCommitLog, ["board_commit_us"]) || ...
        ~hasColumnsLocal(boardSyncLog, ["host_tx_us", "host_rx_us", "board_rx_us", "board_tx_us"])
    return;
end

hostDispatchLog.surface_name = normalizeTextColumnLocal(hostDispatchLog.surface_name);
boardRxLog.surface_name = normalizeTextColumnLocal(boardRxLog.surface_name);

[testStartOffsetUs, ~] = estimateHostTimeOriginUsLocal(hostDispatchLog);
[clockSlope, clockInterceptUs, clockMethod, oneWayBaselineUs] = ...
    estimateNanoClockMapOfflineLocal(hostDispatchLog, boardRxLog, boardSyncLog);

rxHostUs = clockSlope .* double(boardRxLog.rx_us) + clockInterceptUs;
commitHostUs = clockSlope .* double(boardCommitLog.board_commit_us) + clockInterceptUs;
strobeHostUs = nan(height(boardCommitLog), 1);
if ismember("strobe_us", string(boardCommitLog.Properties.VariableNames))
    strobeMask = isfinite(boardCommitLog.strobe_us) & boardCommitLog.strobe_us > 0;
    strobeHostUs(strobeMask) = clockSlope .* double(boardCommitLog.strobe_us(strobeMask)) + clockInterceptUs;
else
    strobeMask = false(height(boardCommitLog), 1);
end

boardRxLog.rx_time_s = (rxHostUs - testStartOffsetUs) ./ 1e6;
boardCommitLog.commit_time_s = (commitHostUs - testStartOffsetUs) ./ 1e6;
boardCommitLog.strobe_time_s = nan(height(boardCommitLog), 1);
boardCommitLog.strobe_time_s(strobeMask) = (strobeHostUs(strobeMask) - testStartOffsetUs) ./ 1e6;

writetable(boardRxLog, fullfile(loggerFolder, "board_rx_log.csv"));
writetable(boardCommitLog, fullfile(loggerFolder, "board_commit_log.csv"));

rxAfterDispatchUs = computeRxAfterDispatchUsLocal(hostDispatchLog, boardRxLog, rxHostUs);
diagnosticTable = table( ...
    string(clockMethod), ...
    clockSlope, ...
    clockInterceptUs, ...
    oneWayBaselineUs, ...
    testStartOffsetUs, ...
    summarizeResidualLocal(rxAfterDispatchUs, "min"), ...
    summarizeResidualLocal(rxAfterDispatchUs, "median"), ...
    summarizeResidualLocal(rxAfterDispatchUs, "p95"), ...
    'VariableNames', { ...
        'method', ...
        'clock_slope', ...
        'clock_intercept_us', ...
        'one_way_baseline_us', ...
        'test_start_offset_us', ...
        'min_rx_after_dispatch_us', ...
        'median_rx_after_dispatch_us', ...
        'p95_rx_after_dispatch_us'});
writetable(diagnosticTable, fullfile(loggerFolder, "nano_offline_clock_repair.csv"));

repairResult.applied = true;
repairResult.rewroteLogs = true;
repairResult.method = string(clockMethod);
repairResult.clockSlope = clockSlope;
repairResult.clockInterceptUs = clockInterceptUs;
repairResult.oneWayBaselineUs = oneWayBaselineUs;
repairResult.testStartOffsetUs = testStartOffsetUs;
repairResult.minRxAfterDispatchUs = summarizeResidualLocal(rxAfterDispatchUs, "min");
repairResult.medianRxAfterDispatchUs = summarizeResidualLocal(rxAfterDispatchUs, "median");
repairResult.p95RxAfterDispatchUs = summarizeResidualLocal(rxAfterDispatchUs, "p95");
repairResult.boardRxLog = boardRxLog;
repairResult.boardCommitLog = boardCommitLog;
end

function [clockSlope, clockInterceptUs, clockMethod, oneWayBaselineUs] = estimateNanoClockMapOfflineLocal( ...
    hostDispatchLog, ...
    boardRxLog, ...
    boardSyncLog)
clockSlope = 1.0;
clockMethod = "sync_midpoint_median";
oneWayBaselineUs = estimateNanoOneWayBaselineUsLocal(boardSyncLog);

hostReferenceUs = 0.5 .* (double(boardSyncLog.host_tx_us) + double(boardSyncLog.host_rx_us));
boardReferenceUs = 0.5 .* (double(boardSyncLog.board_rx_us) + double(boardSyncLog.board_tx_us));
validSyncMask = ...
    isfinite(hostReferenceUs) & ...
    isfinite(boardReferenceUs);

if any(validSyncMask)
    clockInterceptUs = median(hostReferenceUs(validSyncMask) - boardReferenceUs(validSyncMask), "omitnan");
else
    joinedTable = buildRxDispatchJoinLocal(hostDispatchLog, boardRxLog);
    if isempty(joinedTable)
        error("Transmitter_Test_Nano_E2E:MissingNanoClockReference", ...
            "Unable to reconstruct Nano timestamps because neither sync telemetry nor matched dispatch/RX rows were available.");
    end
    dispatchUs = double(joinedTable.command_dispatch_us);
    rxUs = double(joinedTable.rx_us);
    validJoinMask = isfinite(dispatchUs) & isfinite(rxUs);
    clockInterceptUs = median(dispatchUs(validJoinMask) - rxUs(validJoinMask), "omitnan");
    clockMethod = "dispatch_median";
end

if ~isfinite(clockInterceptUs)
    error("Transmitter_Test_Nano_E2E:InvalidNanoClockMap", ...
        "Unable to estimate a valid Nano offline clock intercept.");
end

joinedTable = buildRxDispatchJoinLocal(hostDispatchLog, boardRxLog);
if isempty(joinedTable)
    return;
end

dispatchUs = double(joinedTable.command_dispatch_us);
rxHostUs = clockSlope .* double(joinedTable.rx_us) + clockInterceptUs;
rxAfterDispatchUs = rxHostUs - dispatchUs;
validResidualMask = isfinite(rxAfterDispatchUs);
if any(validResidualMask)
    minimumResidualUs = min(rxAfterDispatchUs(validResidualMask));
    if minimumResidualUs < 0
        clockInterceptUs = clockInterceptUs - minimumResidualUs;
        clockMethod = clockMethod + "_causal_reanchor";
    end
end
end

function joinedTable = buildRxDispatchJoinLocal(hostDispatchLog, boardRxLog)
dispatchColumns = hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'});
dispatchColumns = unique(dispatchColumns, 'rows', 'stable');
rxColumns = boardRxLog(:, {'surface_name', 'command_sequence', 'rx_us'});
rxColumns = unique(rxColumns, 'rows', 'stable');
joinedTable = innerjoin(dispatchColumns, rxColumns, 'Keys', {'surface_name', 'command_sequence'});
end

function [hostTimeOriginUs, sourceName] = estimateHostTimeOriginUsLocal(hostDispatchLog)
dispatchUs = double(hostDispatchLog.command_dispatch_us);
dispatchSeconds = double(hostDispatchLog.command_dispatch_s);
validMask = isfinite(dispatchUs) & isfinite(dispatchSeconds);
if any(validMask)
    hostTimeOriginUs = median(dispatchUs(validMask) - 1e6 .* dispatchSeconds(validMask), "omitnan");
    sourceName = "dispatch_log";
else
    hostTimeOriginUs = 0.0;
    sourceName = "default_zero";
end
end

function oneWayBaselineUs = estimateNanoOneWayBaselineUsLocal(boardSyncLog)
hostTxUs = double(boardSyncLog.host_tx_us);
hostRxUs = double(boardSyncLog.host_rx_us);
validRoundTripMask = isfinite(hostTxUs) & isfinite(hostRxUs) & hostRxUs >= hostTxUs;
if ~any(validRoundTripMask)
    oneWayBaselineUs = NaN;
    return;
end
roundTripUs = hostRxUs(validRoundTripMask) - hostTxUs(validRoundTripMask);
oneWayBaselineUs = 0.5 .* median(roundTripUs, "omitnan");
end

function rxAfterDispatchUs = computeRxAfterDispatchUsLocal(hostDispatchLog, boardRxLog, rxHostUs)
rxProjection = boardRxLog(:, {'surface_name', 'command_sequence'});
rxProjection.rx_host_us = rxHostUs;
rxProjection = unique(rxProjection, 'rows', 'stable');
dispatchProjection = hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'});
dispatchProjection = unique(dispatchProjection, 'rows', 'stable');
joinedTable = innerjoin(dispatchProjection, rxProjection, 'Keys', {'surface_name', 'command_sequence'});
rxAfterDispatchUs = double(joinedTable.rx_host_us) - double(joinedTable.command_dispatch_us);
end

function value = summarizeResidualLocal(residualUs, modeName)
validValues = residualUs(isfinite(residualUs));
if isempty(validValues)
    value = NaN;
    return;
end

switch string(modeName)
    case "min"
        value = min(validValues);
    case "median"
        value = median(validValues, "omitnan");
    case "p95"
        value = prctile(validValues, 95);
    otherwise
        error("Transmitter_Test_Nano_E2E:UnknownResidualSummary", ...
            "Unsupported residual summary mode: %s", ...
            char(string(modeName)));
end
end

function tf = hasColumnsLocal(tableData, columnNames)
tf = all(ismember(string(columnNames), string(tableData.Properties.VariableNames)));
end

function textColumn = normalizeTextColumnLocal(textColumn)
textColumn = reshape(string(textColumn), [], 1);
textColumn(ismissing(textColumn)) = "";
end

function validateNanoHardwareRunSucceededLocal(runData, loggerFolder)
runInfo = getFieldLocal(runData, "runInfo", struct());
statusText = normalizeTextScalarLocal(getFieldLocal(runInfo, "status", ""));
if statusText == "arduino_connection_failed"
    reasonText = normalizeTextScalarLocal(getFieldLocal(runInfo, "reason", ""));
    config = getFieldLocal(runData, "config", struct());
    transportConfig = getFieldLocal(config, "arduinoTransport", struct());
    serialPort = normalizeTextScalarLocal(getFieldLocal(transportConfig, "serialPort", ""));
    if strlength(serialPort) == 0
        serialPort = "unknown";
    end
    error("Transmitter_Test_Nano_E2E:ArduinoConnectionFailed", ...
        "Hardware run did not connect to the Nano transmitter on %s, so the logger folder was never created.\nReason: %s\nExpected logger folder: %s", ...
        char(serialPort), ...
        char(reasonText), ...
        char(loggerFolder));
end
end

function pythonExecutable = resolvePythonExecutableLocal(pythonExecutable)
pythonExecutable = string(pythonExecutable);
if isfile(pythonExecutable)
    if pythonSupportsPostProcessorLocal(pythonExecutable)
        return;
    end
    error("Transmitter_Test_Nano_E2E:InvalidPythonEnvironment", ...
        "The configured Python interpreter does not provide the required modules (pandas, numpy): %s", ...
        char(pythonExecutable));
end

candidates = strings(0, 1);
whereLines = strings(0, 1);
[status, whereOutput] = system("where.exe python");
if status == 0
    whereLines = splitlines(string(whereOutput));
    whereLines = strtrim(whereLines(strlength(strtrim(whereLines)) > 0));
    candidates = [candidates; whereLines]; %#ok<AGROW>
end

knownCandidates = [ ...
    "C:\ProgramData\miniforge3\python.exe"; ...
    "C:\Python312\python.exe"; ...
    "C:\Python311\python.exe"];
candidates = [candidates; knownCandidates]; %#ok<AGROW>
candidates = unique(candidates, "stable");

for i = 1:numel(candidates)
    candidate = candidates(i);
    if ~isfile(candidate)
        continue;
    end
    if contains(lower(candidate), "windowsapps")
        continue;
    end
    if pythonSupportsPostProcessorLocal(candidate)
        pythonExecutable = candidate;
        return;
    end
end

if strlength(pythonExecutable) > 0 && ~contains(lower(pythonExecutable), "windowsapps")
    if pythonSupportsPostProcessorLocal(pythonExecutable)
        return;
    end
end

if ~isempty(whereLines)
    fallback = whereLines(1);
else
    fallback = pythonExecutable;
end
pythonExecutable = fallback;
end

function tf = pythonSupportsPostProcessorLocal(pythonExecutable)
checkCommand = quoteWindowsArgumentLocal(pythonExecutable) + ...
    " -c " + quoteWindowsArgumentLocal("import pandas, numpy");
[status, ~] = system(char(checkCommand));
tf = status == 0;
end

function quotedText = quoteWindowsArgumentLocal(argumentText)
argumentText = replace(string(argumentText), '"', '\"');
quotedText = '"' + argumentText + '"';
end

function textValue = normalizeTextScalarLocal(value)
if isstring(value)
    value = reshape(value, 1, []);
    if isempty(value)
        textValue = "";
        return;
    end
    value = value(1);
    if ismissing(value)
        textValue = "";
    else
        textValue = string(value);
    end
    return;
end

if ischar(value)
    textValue = string(value);
    return;
end

if isempty(value)
    textValue = "";
    return;
end

textValue = string(value);
if isempty(textValue) || any(ismissing(textValue))
    textValue = "";
else
    textValue = textValue(1);
end
end

function value = getFieldLocal(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName)
    value = structValue.(fieldName);
else
    value = defaultValue;
end
end
