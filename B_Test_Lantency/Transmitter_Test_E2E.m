function [runData, e2e] = Transmitter_Test_E2E(config)
% TRANSMITTER_TEST_E2E Run/load transmitter data and use the Python post processor.
%   This wrapper keeps the MATLAB entry point, but the E2E method is now the
%   validated transition-based post-processing implemented in
%   Transmitter_Test_E2E_Post.py.
%
%% =============================================================================
% SECTION MAP
% =============================================================================
% 1) Public Entry Point
% 2) Post-Output Loading and Python Post-Processing
% 3) Comparable Transmitter Profile Defaults
% 4) Path, Python, and Config Utilities
% =============================================================================
%
%% =============================================================================
% 1) Public Entry Point
% =============================================================================
arguments
    config (1,1) struct = struct()
end

runHardware = logical(getFieldLocal(config, "runHardware", true));
runPostProcessor = logical(getFieldLocal(config, "runPostProcessor", true));
forceReprocess = logical(getFieldLocal(config, "forceReprocess", true));
transmitterConfig = getFieldLocal(config, "transmitterConfig", struct());
loggerFolder = string(getFieldLocal(config, "loggerFolder", ""));
outputPrefix = string(getFieldLocal(config, "outputPrefix", "post_transition_e2e"));
seed = double(getFieldLocal(config, "seed", 5));
enforceArduinoStepTrainDefaults = logical(getFieldLocal(config, "enforceArduinoStepTrainDefaults", true));
% Neutral lead/lag intervals give the post-processor stable pre/post PPM
% states, so startup and shutdown transients are not treated as events.
recordLeadSeconds = max(0, double(getFieldLocal(config, "recordLeadSeconds", 10.0)));
recordLagSeconds = max(0, double(getFieldLocal(config, "recordLagSeconds", 10.0)));
commandActiveSeconds = max(0, double(getFieldLocal(config, "commandActiveSeconds", 59.0)));
pythonExecutable = string(getFieldLocal(config, "pythonExecutable", "python"));
pythonScriptPath = string(getFieldLocal(config, "pythonScriptPath", fullfile(fileparts(mfilename("fullpath")), "Transmitter_Test_E2E_Post.py")));
rootFolder = string(getFieldLocal(config, "rootFolder", fullfile(fileparts(mfilename("fullpath")), "D_Transmitter_Test")));
postConfig = getFieldLocal(config, "postConfig", struct());

if runHardware
    transmitterConfig = applyComparableProfileDefaultsLocal( ...
        transmitterConfig, ...
        seed, ...
        enforceArduinoStepTrainDefaults, ...
        recordLeadSeconds, ...
        recordLagSeconds, ...
        commandActiveSeconds);
    runData = Transmitter_Test(transmitterConfig);
    loggerFolder = string(runData.artifacts.loggerFolderPath);
else
    runData = struct();
    if strlength(loggerFolder) == 0
        loggerFolder = resolveLatestLoggerFolderLocal(rootFolder);
    end
end

loggerFolder = string(loggerFolder);
ensureFolderExistsLocal(loggerFolder);

outputPaths = buildOutputPathsLocal(loggerFolder, outputPrefix);
if runPostProcessor && (forceReprocess || ~isfile(outputPaths.eventPath))
    % Transition matching lives in Python because the trainer/receiver
    % capture parsing and offset search are shared with offline re-runs.
    runPythonPostProcessorLocal(loggerFolder, outputPrefix, pythonExecutable, pythonScriptPath, rootFolder, postConfig);
end

if ~isfile(outputPaths.eventPath)
    error("Transmitter_Test_E2E:MissingPostOutput", "Expected post-processed event file was not found: %s", char(outputPaths.eventPath));
end

e2e = loadPostOutputsLocal(loggerFolder, outputPrefix);
if runHardware
    runData.e2e = e2e;
end

fprintf("Transmitter_Test_E2E summary\n");
fprintf("  Method: %s\n", char(e2e.method));
fprintf("  Logger folder: %s\n", char(loggerFolder));
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

%% =============================================================================
% 2) Post-Output Loading and Python Post-Processing
% =============================================================================
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
% Raw logs are optional provenance: older hardware-only folders may not have
% every capture, but retaining what exists makes later audit/replot possible.
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

function runPythonPostProcessorLocal(loggerFolder, outputPrefix, pythonExecutable, pythonScriptPath, rootFolder, postConfig)
if ~isfile(pythonScriptPath)
    error("Transmitter_Test_E2E:MissingPostProcessor", "Python post-processing script not found: %s", char(pythonScriptPath));
end

commandParts = [ ...
    quoteWindowsArgumentLocal(resolvePythonExecutableLocal(pythonExecutable)), ...
    quoteWindowsArgumentLocal(pythonScriptPath), ...
    "--logger-folder", quoteWindowsArgumentLocal(loggerFolder), ...
    "--output-prefix", quoteWindowsArgumentLocal(outputPrefix), ...
    "--root", quoteWindowsArgumentLocal(rootFolder)];

% Matching windows are passed in milliseconds to match the Python CLI and
% make sensitivity sweeps possible without changing the post-processor.
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
    error("Transmitter_Test_E2E:PostProcessorFailed", ...
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

%% =============================================================================
% 3) Comparable Transmitter Profile Defaults
% =============================================================================
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
    % This profile mirrors the Arduino E2E seed runs: 20 ms command cadence
    % and neutral padding around a transition-rich active window.
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

%% =============================================================================
% 4) Path, Python, and Config Utilities
% =============================================================================
function loggerFolder = resolveLatestLoggerFolderLocal(rootFolder)
folderInfo = dir(fullfile(rootFolder, "*_TransmitterLogger"));
if isempty(folderInfo)
    error("Transmitter_Test_E2E:MissingLoggerFolder", "No *_TransmitterLogger folder found in %s.", char(rootFolder));
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
    error("Transmitter_Test_E2E:MissingLoggerFolder", "Logger folder not found: %s", char(folderPath));
end
end

function pythonExecutable = resolvePythonExecutableLocal(pythonExecutable)
pythonExecutable = string(pythonExecutable);
if isfile(pythonExecutable)
    return;
end
[status, whereOutput] = system("where.exe python");
if status == 0
    whereLines = splitlines(string(whereOutput));
    whereLines = strtrim(whereLines(strlength(strtrim(whereLines)) > 0));
    if ~isempty(whereLines)
        pythonExecutable = whereLines(1);
        return;
    end
end
end

function quotedText = quoteWindowsArgumentLocal(argumentText)
argumentText = replace(string(argumentText), '"', '\"');
quotedText = '"' + argumentText + '"';
end

function value = getFieldLocal(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName)
    value = structValue.(fieldName);
else
    value = defaultValue;
end
end
