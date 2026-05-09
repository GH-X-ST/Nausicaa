function result = Post_Analysis(mode, varargin)
%POST_ANALYSIS Offline surface calibration and latency analysis.
mode = lower(string(mode));
[args, analysisConfig] = parseAnalysisConfig(varargin);

switch mode
    case "deflection"
        if numel(args) < 1
            error("Post_Analysis:MissingInput", "Deflection mode requires rawRunFile.");
        end
        result = analyzeDeflection(string(args{1}), analysisConfig);
    case "latency"
        if numel(args) < 1
            error("Post_Analysis:MissingInput", "Latency mode requires rawRunFile.");
        end
        calibrationFile = "";
        if numel(args) >= 2
            calibrationFile = string(args{2});
        end
        result = analyzeLatency(string(args{1}), calibrationFile, analysisConfig);
    case "all"
        if numel(args) < 2
            error("Post_Analysis:MissingInput", "All mode requires deflectionRunFile and latencyRunFile.");
        end
        deflection = analyzeDeflection(string(args{1}), analysisConfig);
        latency = analyzeLatency(string(args{2}), string(deflection.outputFiles.calibrationMat), analysisConfig);
        result = struct("mode", "all", "deflection", deflection, "latency", latency);
    otherwise
        error("Post_Analysis:InvalidMode", "mode must be 'deflection', 'latency', or 'all'.");
end
end

function result = analyzeDeflection(rawRunFile, analysisConfig)
runData = loadRunData(rawRunFile);
events = runData.eventTable;
states = runData.viconStateLog;
commandLog = runData.commandLog;

if isempty(events) || isempty(states)
    error("Post_Analysis:MissingDeflectionLogs", "Deflection analysis requires eventTable and viconStateLog.");
end

outputFolder = resolveProcessedFolder(runData, "deflection");
ensureFolder(outputFolder);

holdRows = table();
for eventIndex = 1:height(events)
    eventRow = events(eventIndex, :);
    surfaceIndex = double(eventRow.surface_index);
    sweepPolarity = tableStringScalar(eventRow, "sweep_polarity", "");
    sweepPhase = tableStringScalar(eventRow, "sweep_phase", "");
    sweepStepIndex = tableDoubleScalar(eventRow, "sweep_step_index", NaN);
    sweepStepCount = tableDoubleScalar(eventRow, "sweep_step_count", NaN);
    holdWindowS = chooseSettledWindow(eventRow);
    stateMask = double(states.surface_index) == surfaceIndex & ...
        states.t_capture_host_s >= holdWindowS(1) & ...
        states.t_capture_host_s <= holdWindowS(2);
    windowRows = states(stateMask, :);

    settledDeg = median(double(windowRows.surface_angle_deg), "omitnan");
    instabilityDeg = std(double(windowRows.surface_angle_deg), 0, "omitnan");
    qualityText = join(unique(string(windowRows.quality_flag)), "|");
    if isempty(qualityText)
        qualityText = "missing_data";
    end

    rejectionReason = "";
    valid = true;
    if isempty(windowRows)
        valid = false;
        rejectionReason = "missing_vicon_window";
    elseif any(string(windowRows.quality_flag) ~= "ok")
        valid = false;
        rejectionReason = "vicon_quality_" + qualityText;
    elseif isfinite(instabilityDeg) && instabilityDeg > analysisConfig.deflectionUnstableStdDeg
        valid = false;
        rejectionReason = "unstable_final_window";
    end

    holdRows = [holdRows; table( ... %#ok<AGROW>
        double(eventRow.event_id), surfaceIndex, string(eventRow.surface_name), ...
        double(eventRow.command_level_norm), string(eventRow.direction_group), ...
        sweepPolarity, sweepPhase, sweepStepIndex, sweepStepCount, ...
        double(eventRow.scheduled_start_s), double(eventRow.scheduled_end_s), ...
        settledDeg, instabilityDeg, valid, string(rejectionReason), ...
        'VariableNames', {'event_id', 'surface_index', 'surface_name', 'command_level_norm', ...
        'direction_group', 'sweep_polarity', 'sweep_phase', 'sweep_step_index', 'sweep_step_count', ...
        'scheduled_start_s', 'scheduled_end_s', 'settled_deflection_deg', 'settled_std_deg', ...
        'valid', 'rejection_reason'})];
end

[surfaceTable, implementationError] = summarizeDeflection(holdRows, states, runData.config);
calibration = struct( ...
    "surfaceTable", surfaceTable, ...
    "holdTable", holdRows, ...
    "implementationError", implementationError, ...
    "commandLog", commandLog, ...
    "config", runData.config);

result = struct();
result.mode = "deflection";
result.calibration = calibration;
result.summaryTable = surfaceTable;
result.outputFiles = struct( ...
    "calibrationMat", string(fullfile(outputFolder, "surface_calibration.mat")), ...
    "calibrationCsv", string(fullfile(outputFolder, "surface_calibration.csv")), ...
    "implementationErrorCsv", string(fullfile(outputFolder, "implementation_error.csv")));

save(result.outputFiles.calibrationMat, "calibration", "result");
writetable(surfaceTable, result.outputFiles.calibrationCsv);
writetable(implementationError, result.outputFiles.implementationErrorCsv);

if analysisConfig.makePlots
    warning("Post_Analysis:PlotsSkipped", "Diagnostic plotting is not implemented in this compact analysis.");
end
end

function [surfaceTable, implementationError] = summarizeDeflection(holdRows, states, config)
surfaceNames = reshape(string(config.surfaceOrder), 1, []);
surfaceTable = table();
implementationError = table();

for surfaceIndex = 1:numel(surfaceNames)
    surfaceRows = holdRows(double(holdRows.surface_index) == surfaceIndex, :);
    if isempty(surfaceRows)
        continue;
    end

    neutralRows = surfaceRows(abs(double(surfaceRows.command_level_norm)) < 10 * eps & surfaceRows.valid, :);
    neutralDeg = median(double(neutralRows.settled_deflection_deg), "omitnan");
    if ~isfinite(neutralDeg)
        neutralDeg = 0;
    end

    neutralStateMask = double(states.surface_index) == surfaceIndex & absStateAroundNeutral(states, surfaceRows);
    sigmaNeutralDeg = std(double(states.surface_angle_deg(neutralStateMask)), 0, "omitnan");
    if ~isfinite(sigmaNeutralDeg)
        sigmaNeutralDeg = std(double(neutralRows.settled_deflection_deg), 0, "omitnan");
    end
    if ~isfinite(sigmaNeutralDeg)
        sigmaNeutralDeg = 0;
    end
    detectionThresholdDeg = max(3 * sigmaNeutralDeg, 1.0);

    surfaceRows.delta_rel_deg = double(surfaceRows.settled_deflection_deg) - neutralDeg;
    validRows = surfaceRows(surfaceRows.valid & isfinite(surfaceRows.delta_rel_deg), :);
    positiveRows = validRows(double(validRows.command_level_norm) > 0, :);
    negativeRows = validRows(double(validRows.command_level_norm) < 0, :);

    deadbandPositive = firstDetectedDeadband(positiveRows, detectionThresholdDeg, 1);
    deadbandNegative = firstDetectedDeadband(negativeRows, detectionThresholdDeg, -1);
    gainPositive = fitGain(positiveRows, deadbandPositive, 1);
    gainNegative = fitGain(negativeRows, deadbandNegative, -1);

    fittedDeg = fittedDeflection(surfaceRows.command_level_norm, deadbandPositive, deadbandNegative, gainPositive, gainNegative);
    if any(isfinite(surfaceRows.delta_rel_deg))
        fittedDeg = min(max(fittedDeg, min(surfaceRows.delta_rel_deg, [], "omitnan")), max(surfaceRows.delta_rel_deg, [], "omitnan"));
    end
    errorDeg = surfaceRows.delta_rel_deg - fittedDeg;

    implementationError = [implementationError; table( ... %#ok<AGROW>
        surfaceRows.event_id, surfaceRows.surface_index, surfaceRows.surface_name, ...
        surfaceRows.command_level_norm, surfaceRows.direction_group, surfaceRows.sweep_polarity, ...
        surfaceRows.sweep_phase, surfaceRows.sweep_step_index, surfaceRows.sweep_step_count, ...
        surfaceRows.delta_rel_deg, fittedDeg, errorDeg, ...
        surfaceRows.valid, surfaceRows.rejection_reason, ...
        'VariableNames', {'event_id', 'surface_index', 'surface_name', 'command_level_norm', ...
        'direction_group', 'sweep_polarity', 'sweep_phase', 'sweep_step_index', 'sweep_step_count', ...
        'settled_deflection_deg', 'fitted_deflection_deg', 'implementation_error_deg', ...
        'valid', 'rejection_reason'})];

    minDeflectionDeg = minFinite(validRows.delta_rel_deg);
    maxDeflectionDeg = maxFinite(validRows.delta_rel_deg);
    positiveRangeDeg = maxDeflectionDeg;
    negativeRangeDeg = abs(minDeflectionDeg);
    positiveFullRows = validRows(abs(double(validRows.command_level_norm) - 1) < 10 * eps, :);
    negativeFullRows = validRows(abs(double(validRows.command_level_norm) + 1) < 10 * eps, :);

    surfaceTable = [surfaceTable; table( ... %#ok<AGROW>
        surfaceIndex, surfaceNames(surfaceIndex), neutralDeg, ...
        minDeflectionDeg, maxDeflectionDeg, positiveRangeDeg, negativeRangeDeg, ...
        any(abs(double(surfaceRows.command_level_norm) - 1) < 10 * eps), ...
        any(abs(double(surfaceRows.command_level_norm) + 1) < 10 * eps), ...
        ~isempty(positiveFullRows), ~isempty(negativeFullRows), ...
        median(double(positiveFullRows.delta_rel_deg), "omitnan"), ...
        median(double(negativeFullRows.delta_rel_deg), "omitnan"), ...
        deadbandPositive, deadbandNegative, gainPositive, gainNegative, ...
        estimateHysteresis(validRows), estimateReturnHysteresis(validRows), estimateRepeatability(validRows), ...
        sqrt(mean(errorDeg(surfaceRows.valid).^2, "omitnan")), ...
        string(mat2str(double(surfaceRows.command_level_norm).')), ...
        string(mat2str(double(surfaceRows.delta_rel_deg).')), ...
        'VariableNames', {'surface_index', 'surface_name', 'neutral_deg', 'min_deflection_deg', ...
        'max_deflection_deg', 'positive_range_deg', 'negative_range_deg', ...
        'positive_full_commanded', 'negative_full_commanded', ...
        'positive_full_valid', 'negative_full_valid', ...
        'positive_full_deflection_deg', 'negative_full_deflection_deg', ...
        'deadband_positive_norm', 'deadband_negative_norm', ...
        'gain_positive_deg_per_norm', 'gain_negative_deg_per_norm', ...
        'hysteresis_deg', 'return_hysteresis_deg', 'repeatability_std_deg', 'fit_rmse_deg', ...
        'command_levels_norm', 'settled_deflection_deg'})];
end
end

function result = analyzeLatency(rawRunFile, calibrationFile, analysisConfig)
runData = loadRunData(rawRunFile);
events = runData.eventTable;
states = runData.viconStateLog;
commands = runData.commandLog;
calibration = loadCalibrationOptional(calibrationFile);

if isempty(events) || isempty(states) || isempty(commands)
    error("Post_Analysis:MissingLatencyLogs", "Latency analysis requires eventTable, commandLog, and viconStateLog.");
end

outputFolder = resolveProcessedFolder(runData, "latency");
ensureFolder(outputFolder);

latencyEvents = table();
rejectedEvents = table();
for eventIndex = 1:height(events)
    eventRow = events(eventIndex, :);
    eventId = double(eventRow.event_id);
    commandRows = commands(double(commands.event_id) == eventId, :);
    if isempty(commandRows)
        row = rejectedLatencyRow(eventRow, "missing_command_sample");
        rejectedEvents = [rejectedEvents; row]; %#ok<AGROW>
        latencyEvents = [latencyEvents; row]; %#ok<AGROW>
        continue;
    end

    t0 = double(commandRows.write_start_host_s(1));
    surfaceIndex = double(eventRow.surface_index);
    surfaceRows = states(double(states.surface_index) == surfaceIndex, :);
    analysisEndS = double(eventRow.scheduled_end_s);

    [eventResult, rejectionReason] = analyzeOneLatencyEvent(eventRow, surfaceRows, t0, analysisEndS, calibration, analysisConfig);
    latencyEvents = [latencyEvents; eventResult]; %#ok<AGROW>
    if strlength(rejectionReason) > 0
        rejectedEvents = [rejectedEvents; eventResult]; %#ok<AGROW>
    end
end

summaryTable = summarizeLatencyEvents(latencyEvents, runData.config);

result = struct();
result.mode = "latency";
result.latencyEvents = latencyEvents;
result.summaryTable = summaryTable;
result.rejectedEvents = rejectedEvents;
result.outputFiles = struct( ...
    "latencyMat", string(fullfile(outputFolder, "latency_result.mat")), ...
    "latencyEventsCsv", string(fullfile(outputFolder, "latency_events.csv")), ...
    "latencySummaryCsv", string(fullfile(outputFolder, "latency_summary.csv")), ...
    "rejectedEventsCsv", string(fullfile(outputFolder, "rejected_latency_events.csv")));

save(result.outputFiles.latencyMat, "result");
writetable(latencyEvents, result.outputFiles.latencyEventsCsv);
writetable(summaryTable, result.outputFiles.latencySummaryCsv);
writetable(rejectedEvents, result.outputFiles.rejectedEventsCsv);
end

function [eventResult, rejectionReason] = analyzeOneLatencyEvent(eventRow, surfaceRows, t0, analysisEndS, calibration, analysisConfig)
surfaceIndex = double(eventRow.surface_index);
surfaceName = string(eventRow.surface_name);
beforeCommand = tableVectorValue(eventRow.command_before_norm);
afterCommand = tableVectorValue(eventRow.command_after_norm);
commandStepNorm = afterCommand(surfaceIndex) - beforeCommand(surfaceIndex);
rejectionReason = "";

baseMask = surfaceRows.t_capture_host_s >= t0 - 0.15 & surfaceRows.t_capture_host_s <= t0 - 0.02;
responseMask = surfaceRows.t_capture_host_s >= t0 & surfaceRows.t_capture_host_s <= analysisEndS;
finalMask = surfaceRows.t_capture_host_s >= max(t0, analysisEndS - 0.10) & surfaceRows.t_capture_host_s <= analysisEndS;

baseRows = surfaceRows(baseMask, :);
responseRows = surfaceRows(responseMask, :);
finalRows = surfaceRows(finalMask, :);

deltaBaseDeg = median(double(baseRows.surface_angle_deg), "omitnan");
sigmaBaseDeg = std(double(baseRows.surface_angle_deg), 0, "omitnan");
deltaFinalDeg = median(double(finalRows.surface_angle_deg), "omitnan");
responseAmplitudeDeg = deltaFinalDeg - deltaBaseDeg;

valid = true;
if isempty(baseRows) || isempty(responseRows) || isempty(finalRows)
    valid = false;
    rejectionReason = appendReason(rejectionReason, "missing_vicon_window");
end
if any(string(responseRows.quality_flag) ~= "ok")
    valid = false;
    rejectionReason = appendReason(rejectionReason, "vicon_quality");
end
if ~isfinite(responseAmplitudeDeg) || abs(responseAmplitudeDeg) < analysisConfig.minLatencyResponseDeg
    valid = false;
    rejectionReason = appendReason(rejectionReason, "response_amplitude_too_small");
end
if signNonzero(commandStepNorm) ~= 0 && signNonzero(responseAmplitudeDeg) ~= 0 && ...
        signNonzero(commandStepNorm) ~= signNonzero(responseAmplitudeDeg)
    valid = false;
    rejectionReason = appendReason(rejectionReason, "response_direction_wrong");
end

deadbandThreshold = extractDeadbandThreshold(calibration, surfaceIndex);
if isfinite(deadbandThreshold) && abs(commandStepNorm) < deadbandThreshold
    valid = false;
    rejectionReason = appendReason(rejectionReason, "command_step_below_calibrated_deadband");
end

level10 = deltaBaseDeg + 0.10 * responseAmplitudeDeg;
level50 = deltaBaseDeg + 0.50 * responseAmplitudeDeg;
level90 = deltaBaseDeg + 0.90 * responseAmplitudeDeg;
direction = signNonzero(responseAmplitudeDeg);

[t10, reached10] = findCrossing(responseRows.t_capture_host_s, responseRows.surface_angle_deg, level10, direction);
[t50, reached50] = findCrossing(responseRows.t_capture_host_s, responseRows.surface_angle_deg, level50, direction);
[t90, reached90] = findCrossing(responseRows.t_capture_host_s, responseRows.surface_angle_deg, level90, direction);

if ~reached10 || ~reached50
    valid = false;
    rejectionReason = appendReason(rejectionReason, "t10_or_t50_not_detected");
end

t90Valid = reached90 && t90 <= analysisEndS;
if ~t90Valid
    t90 = NaN;
end

eventResult = table( ...
    double(eventRow.event_id), surfaceIndex, surfaceName, t0, analysisEndS, ...
    commandStepNorm, deltaBaseDeg, sigmaBaseDeg, deltaFinalDeg, responseAmplitudeDeg, ...
    level10, level50, level90, ...
    t10, t50, t90, ...
    t10 - t0, t50 - t0, t90 - t0, ...
    valid, t90Valid, string(rejectionReason), ...
    'VariableNames', {'event_id', 'surface_index', 'surface_name', 't0_write_start_host_s', ...
    'response_window_end_s', 'command_step_norm', 'delta_base_deg', 'sigma_base_deg', ...
    'delta_final_deg', 'response_amplitude_deg', 'level_10_deg', 'level_50_deg', 'level_90_deg', ...
    't10_capture_s', 't50_capture_s', 't90_capture_s', ...
    't10_latency_s', 't50_latency_s', 't90_latency_s', 'valid', 't90_valid', 'rejection_reason'});
end

function summaryTable = summarizeLatencyEvents(latencyEvents, config)
validEvents = latencyEvents(latencyEvents.valid & isfinite(latencyEvents.t50_latency_s), :);
surfaceNames = ["all", reshape(string(config.surfaceOrder), 1, [])];
surfaceIndices = [0, 1:numel(config.surfaceOrder)];
summaryTable = table();

for rowIndex = 1:numel(surfaceNames)
    if surfaceIndices(rowIndex) == 0
        rows = validEvents;
        rejectedCount = height(latencyEvents) - height(validEvents);
    else
        rows = validEvents(double(validEvents.surface_index) == surfaceIndices(rowIndex), :);
        allRows = latencyEvents(double(latencyEvents.surface_index) == surfaceIndices(rowIndex), :);
        rejectedCount = height(allRows) - height(rows);
    end

    t10 = double(rows.t10_latency_s);
    t50 = double(rows.t50_latency_s);
    t90 = double(rows.t90_latency_s);
    summaryTable = [summaryTable; table( ... %#ok<AGROW>
        surfaceIndices(rowIndex), surfaceNames(rowIndex), height(rows), rejectedCount, ...
        msMedian(t10), msPrc(t10, 95), msMedian(t50), msPrc(t50, 95), ...
        msMedian(t90), msPrc(t90, 95), msMin(t50), msMax(t50), ...
        prcSeconds(t50, 5), medianFinite(t50), prcSeconds(t50, 95), minFinite(t50), maxFinite(t50), ...
        'VariableNames', {'surface_index', 'surface_name', 'valid_event_count', 'rejected_event_count', ...
        'median_t10_ms', 'p95_t10_ms', 'median_t50_ms', 'p95_t50_ms', ...
        'median_t90_ms', 'p95_t90_ms', 'min_t50_ms', 'max_t50_ms', ...
        'tau_min_s', 'tau_nom_s', 'tau_max_s', 'tau_min_conservative_s', 'tau_max_conservative_s'})];
end
end

function [args, analysisConfig] = parseAnalysisConfig(args)
analysisConfig = struct( ...
    "makePlots", false, ...
    "deflectionUnstableStdDeg", 2.0, ...
    "minLatencyResponseDeg", 1.0);
if ~isempty(args) && isstruct(args{end})
    userConfig = args{end};
    args = args(1:end - 1);
    fields = fieldnames(userConfig);
    for fieldIndex = 1:numel(fields)
        analysisConfig.(fields{fieldIndex}) = userConfig.(fields{fieldIndex});
    end
end
analysisConfig.makePlots = logical(analysisConfig.makePlots);
end

function runData = loadRunData(rawRunFile)
loadedData = load(rawRunFile);
if isfield(loadedData, "runData")
    runData = loadedData.runData;
else
    error("Post_Analysis:MissingRunData", "File does not contain runData: %s", rawRunFile);
end
end

function outputFolder = resolveProcessedFolder(runData, suffix)
processedRoot = fullfile("C_Overall_Latency", "data", "processed");
if isfield(runData, "config") && isfield(runData.config, "processedRoot")
    processedRoot = runData.config.processedRoot;
end
runLabel = "run";
if isfield(runData.config, "runLabel")
    runLabel = runData.config.runLabel;
end
outputFolder = fullfile(processedRoot, string(runLabel) + "_" + suffix);
end

function ensureFolder(folderPath)
if ~isfolder(folderPath)
    mkdir(folderPath);
end
end

function windowS = chooseSettledWindow(eventRow)
holdSeconds = double(eventRow.scheduled_end_s) - double(eventRow.scheduled_start_s);
if holdSeconds < 0.30
    windowLength = max(0.05, 0.5 * holdSeconds);
else
    windowLength = 0.20;
end
windowS = [double(eventRow.scheduled_end_s) - windowLength, double(eventRow.scheduled_end_s)];
end

function value = tableStringScalar(row, variableName, defaultValue)
if ~ismember(variableName, string(row.Properties.VariableNames))
    value = string(defaultValue);
    return;
end
value = string(row.(variableName));
if isempty(value) || ismissing(value(1))
    value = string(defaultValue);
else
    value = value(1);
end
end

function value = tableDoubleScalar(row, variableName, defaultValue)
if ~ismember(variableName, string(row.Properties.VariableNames))
    value = defaultValue;
    return;
end
value = double(row.(variableName));
if isempty(value) || ~isfinite(value(1))
    value = defaultValue;
else
    value = value(1);
end
end

function mask = absStateAroundNeutral(states, surfaceRows)
mask = false(height(states), 1);
neutralRows = surfaceRows(abs(double(surfaceRows.command_level_norm)) < 10 * eps, :);
for rowIndex = 1:height(neutralRows)
    mask = mask | (states.t_capture_host_s >= neutralRows.scheduled_start_s(rowIndex) & ...
        states.t_capture_host_s <= neutralRows.scheduled_end_s(rowIndex));
end
end

function deadband = firstDetectedDeadband(rows, thresholdDeg, signDirection)
deadband = NaN;
if isempty(rows)
    return;
end
[~, order] = sort(abs(double(rows.command_level_norm)));
rows = rows(order, :);
for rowIndex = 1:height(rows)
    if signDirection * double(rows.delta_rel_deg(rowIndex)) > thresholdDeg
        deadband = abs(double(rows.command_level_norm(rowIndex)));
        return;
    end
end
end

function gain = fitGain(rows, deadband, signDirection)
gain = NaN;
if isempty(rows) || ~isfinite(deadband)
    return;
end
u = double(rows.command_level_norm);
y = double(rows.delta_rel_deg);
if signDirection > 0
    x = u - deadband;
else
    x = u + deadband;
end
valid = isfinite(x) & isfinite(y) & abs(x) > 10 * eps;
if any(valid)
    gain = sum(x(valid) .* y(valid)) ./ sum(x(valid) .^ 2);
end
end

function y = fittedDeflection(u, dbPlus, dbMinus, gainPlus, gainMinus)
u = double(u);
y = zeros(size(u));
if isfinite(dbPlus) && isfinite(gainPlus)
    positiveMask = u > dbPlus;
    y(positiveMask) = gainPlus .* (u(positiveMask) - dbPlus);
end
if isfinite(dbMinus) && isfinite(gainMinus)
    negativeMask = u < -dbMinus;
    y(negativeMask) = gainMinus .* (u(negativeMask) + dbMinus);
end
end

function value = estimateHysteresis(rows)
value = NaN;
if isempty(rows) || ~ismember("direction_group", string(rows.Properties.VariableNames))
    return;
end
levels = unique(double(rows.command_level_norm));
diffs = nan(numel(levels), 1);
for i = 1:numel(levels)
    levelRows = rows(abs(double(rows.command_level_norm) - levels(i)) < 10 * eps, :);
    posRows = levelRows(string(levelRows.direction_group) == "positive_first", :);
    negRows = levelRows(string(levelRows.direction_group) == "negative_first", :);
    if ~isempty(posRows) && ~isempty(negRows)
        diffs(i) = abs(median(double(posRows.delta_rel_deg), "omitnan") - median(double(negRows.delta_rel_deg), "omitnan"));
    end
end
value = median(diffs, "omitnan");
end

function value = estimateReturnHysteresis(rows)
value = NaN;
requiredVariables = ["sweep_phase", "direction_group", "command_level_norm", "delta_rel_deg"];
if isempty(rows) || any(~ismember(requiredVariables, string(rows.Properties.VariableNames)))
    return;
end

directionGroups = unique(string(rows.direction_group));
levels = unique(double(rows.command_level_norm));
diffs = nan(numel(directionGroups) * numel(levels), 1);
diffIndex = 0;
for groupIndex = 1:numel(directionGroups)
    groupRows = rows(string(rows.direction_group) == directionGroups(groupIndex), :);
    for levelIndex = 1:numel(levels)
        levelRows = groupRows(abs(double(groupRows.command_level_norm) - levels(levelIndex)) < 10 * eps, :);
        outboundRows = levelRows(string(levelRows.sweep_phase) == "outbound", :);
        returnRows = levelRows(string(levelRows.sweep_phase) == "return", :);
        if ~isempty(outboundRows) && ~isempty(returnRows)
            diffIndex = diffIndex + 1;
            diffs(diffIndex) = abs(median(double(outboundRows.delta_rel_deg), "omitnan") - ...
                median(double(returnRows.delta_rel_deg), "omitnan"));
        end
    end
end
value = median(diffs(1:diffIndex), "omitnan");
end

function value = estimateRepeatability(rows)
levels = unique(double(rows.command_level_norm));
stdValues = nan(numel(levels), 1);
for i = 1:numel(levels)
    levelRows = rows(abs(double(rows.command_level_norm) - levels(i)) < 10 * eps, :);
    stdValues(i) = std(double(levelRows.delta_rel_deg), 0, "omitnan");
end
value = median(stdValues, "omitnan");
end

function calibration = loadCalibrationOptional(calibrationFile)
calibration = [];
if strlength(calibrationFile) == 0 || ~isfile(calibrationFile)
    return;
end
loadedData = load(calibrationFile);
if isfield(loadedData, "calibration")
    calibration = loadedData.calibration;
elseif isfield(loadedData, "result") && isfield(loadedData.result, "calibration")
    calibration = loadedData.result.calibration;
end
end

function threshold = extractDeadbandThreshold(calibration, surfaceIndex)
threshold = NaN;
if isempty(calibration)
    return;
end
surfaceTable = [];
if isstruct(calibration) && isfield(calibration, "surfaceTable")
    surfaceTable = calibration.surfaceTable;
elseif istable(calibration)
    surfaceTable = calibration;
end
if isempty(surfaceTable)
    return;
end
row = surfaceTable(double(surfaceTable.surface_index) == surfaceIndex, :);
if isempty(row)
    return;
end
threshold = max(abs(double(row.deadband_positive_norm)), abs(double(row.deadband_negative_norm)));
end

function row = rejectedLatencyRow(eventRow, reason)
row = table( ...
    double(eventRow.event_id), double(eventRow.surface_index), string(eventRow.surface_name), ...
    NaN, double(eventRow.scheduled_end_s), NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
    NaN, NaN, NaN, NaN, NaN, NaN, false, false, string(reason), ...
    'VariableNames', {'event_id', 'surface_index', 'surface_name', 't0_write_start_host_s', ...
    'response_window_end_s', 'command_step_norm', 'delta_base_deg', 'sigma_base_deg', ...
    'delta_final_deg', 'response_amplitude_deg', 'level_10_deg', 'level_50_deg', 'level_90_deg', ...
    't10_capture_s', 't50_capture_s', 't90_capture_s', ...
    't10_latency_s', 't50_latency_s', 't90_latency_s', 'valid', 't90_valid', 'rejection_reason'});
end

function reason = appendReason(reason, newReason)
if strlength(reason) == 0
    reason = string(newReason);
else
    reason = reason + ";" + string(newReason);
end
end

function signValue = signNonzero(value)
if ~isfinite(value) || abs(value) < 10 * eps
    signValue = 0;
else
    signValue = sign(value);
end
end

function [crossTime, found] = findCrossing(times, values, level, direction)
times = double(times);
values = double(values);
valid = isfinite(times) & isfinite(values);
times = times(valid);
values = values(valid);
crossTime = NaN;
found = false;
if numel(times) < 2 || direction == 0 || ~isfinite(level)
    return;
end

for index = 2:numel(times)
    prevValue = values(index - 1);
    currValue = values(index);
    if direction > 0
        crossed = prevValue < level && currValue >= level;
    else
        crossed = prevValue > level && currValue <= level;
    end
    if crossed
        fraction = (level - prevValue) ./ (currValue - prevValue);
        crossTime = times(index - 1) + fraction .* (times(index) - times(index - 1));
        found = true;
        return;
    end
end
end

function value = medianFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = median(values);
end
end

function value = minFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = min(values);
end
end

function value = maxFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end

function value = prcSeconds(values, percentile)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = prctile(values, percentile);
end
end

function value = msMedian(values)
value = 1000 * medianFinite(values);
end

function value = msPrc(values, percentile)
value = 1000 * prcSeconds(values, percentile);
end

function value = msMin(values)
value = 1000 * minFinite(values);
end

function value = msMax(values)
value = 1000 * maxFinite(values);
end

function value = tableVectorValue(tableValue)
if iscell(tableValue)
    value = double(tableValue{1});
elseif isstring(tableValue) || ischar(tableValue)
    value = parseNumericTextVector(tableValue);
else
    value = double(tableValue);
end
value = reshape(value, 1, []);
end

function value = parseNumericTextVector(textValue)
cleanText = erase(string(textValue), ["[", "]", ";", ","]);
value = sscanf(char(cleanText), "%f").';
end
