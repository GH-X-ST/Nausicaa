function [cmd, profileState] = Test_Latency_Profile(tNowS, viconSample, config, profileState)
%TEST_LATENCY_PROFILE Seeded sparse one-surface step command provider.
if nargin < 4 || isempty(profileState) || ~isfield(profileState, "isInitialized")
    profileState = initializeProfile(config);
end

eventIndex = find(tNowS >= profileState.eventTable.scheduled_start_s & ...
    tNowS < profileState.eventTable.scheduled_end_s, 1, "last");

surfaceNorm = zeros(1, numel(config.surfaceOrder));
eventId = NaN;
activeSurfaceIndex = NaN;
activeSurfaceName = "";
commandLevelNorm = 0.0;
description = "neutral";

if ~isempty(eventIndex)
    eventRow = profileState.eventTable(eventIndex, :);
    surfaceNorm = tableVectorValue(eventRow.command_after_norm);
    eventId = eventRow.event_id;
    activeSurfaceIndex = eventRow.surface_index;
    activeSurfaceName = string(eventRow.surface_name);
    commandLevelNorm = surfaceNorm(activeSurfaceIndex);
    description = "latency_step_" + activeSurfaceName + "_" + string(commandLevelNorm);
end

cmd = struct( ...
    "eventId", eventId, ...
    "sequence", profileState.sequence, ...
    "surfaceNorm", surfaceNorm, ...
    "aeroCmdRad", NaN(1, 3), ...
    "activeSurfaceIndex", activeSurfaceIndex, ...
    "activeSurfaceName", activeSurfaceName, ...
    "commandLevelNorm", commandLevelNorm, ...
    "description", description);

profileState.sequence = profileState.sequence + 1;
profileState.lastViconSample = viconSample;
end

function profileState = initializeProfile(config)
surfaceOrder = reshape(string(config.surfaceOrder), 1, []);
surfaceCount = numel(surfaceOrder);
surfaceIndices = getFieldOrDefault(config, "enabledSurfaceIndices", 1:surfaceCount);
surfaceIndices = reshape(double(surfaceIndices), 1, []);
surfaceIndices = surfaceIndices(surfaceIndices >= 1 & surfaceIndices <= surfaceCount);
if isempty(surfaceIndices)
    surfaceIndices = 1:surfaceCount;
end

eventHoldSeconds = getFieldOrDefault(config, "eventHoldSeconds", 0.50);
activeStartS = double(config.neutralLeadSeconds);
activeStopS = activeStartS + double(config.activeCommandSeconds);
randomSeed = double(config.randomSeed);
amplitudeSet = reshape(double(getFieldOrDefault( ...
    config, "latencyAmplitudeSetNorm", [-1.0, -0.7, -0.5, 0.5, 0.7, 1.0])), 1, []);

[amplitudesBySurface, profileInfo] = resolveAmplitudeSet(config, surfaceCount, amplitudeSet);
rng(randomSeed, "twister");

eventRows = struct( ...
    "event_id", {}, ...
    "surface_index", {}, ...
    "surface_name", {}, ...
    "command_before_norm", {}, ...
    "command_after_norm", {}, ...
    "scheduled_start_s", {}, ...
    "scheduled_end_s", {}, ...
    "random_seed", {});

currentCommand = zeros(1, surfaceCount);
nextStartS = activeStartS;
eventId = 1;
while nextStartS < activeStopS
    surfaceIndex = surfaceIndices(randi(numel(surfaceIndices)));
    usableAmplitudes = amplitudesBySurface{surfaceIndex};
    if isempty(usableAmplitudes)
        usableAmplitudes = [-1, 1];
    end

    previousValue = currentCommand(surfaceIndex);
    candidateAmplitudes = usableAmplitudes(abs(usableAmplitudes - previousValue) > 10 * eps);
    if isempty(candidateAmplitudes)
        candidateAmplitudes = usableAmplitudes;
    end

    nextValue = candidateAmplitudes(randi(numel(candidateAmplitudes)));
    commandBefore = currentCommand;
    currentCommand(surfaceIndex) = nextValue;
    commandAfter = currentCommand;

    eventRows(end + 1) = struct( ... %#ok<AGROW>
        "event_id", eventId, ...
        "surface_index", surfaceIndex, ...
        "surface_name", surfaceOrder(surfaceIndex), ...
        "command_before_norm", {commandBefore}, ...
        "command_after_norm", {commandAfter}, ...
        "scheduled_start_s", nextStartS, ...
        "scheduled_end_s", min(nextStartS + eventHoldSeconds, activeStopS), ...
        "random_seed", randomSeed);

    nextStartS = nextStartS + eventHoldSeconds;
    eventId = eventId + 1;
end

profileState = struct();
profileState.isInitialized = true;
profileState.sequence = 0;
profileState.eventTable = struct2table(eventRows);
profileState.profileInfo = profileInfo;
profileState.profileInfo.profileMode = "latency";
profileState.profileInfo.eventHoldSeconds = eventHoldSeconds;
profileState.profileInfo.randomSeed = randomSeed;
profileState.profileInfo.enabledSurfaceIndices = surfaceIndices;
end

function [amplitudesBySurface, profileInfo] = resolveAmplitudeSet(config, surfaceCount, amplitudeSet)
amplitudesBySurface = repmat({amplitudeSet(abs(amplitudeSet) >= 0.20)}, surfaceCount, 1);
profileInfo = struct( ...
    "calibrationUsed", false, ...
    "deadbandAssumption", "not supplied", ...
    "deadbandThresholdNorm", nan(1, surfaceCount), ...
    "fallbackAmplitudeUsed", false);

if ~isfield(config, "calibrationFile") || strlength(string(config.calibrationFile)) == 0
    return;
end

calibrationFile = string(config.calibrationFile);
if ~isfile(calibrationFile)
    profileInfo.deadbandAssumption = "calibration file not found";
    return;
end

calibration = loadCalibration(calibrationFile);
if isempty(calibration)
    profileInfo.deadbandAssumption = "calibration file unreadable";
    return;
end

profileInfo.calibrationUsed = true;
profileInfo.deadbandAssumption = "from calibration";

for surfaceIndex = 1:surfaceCount
    deadband = extractSurfaceDeadband(calibration, surfaceIndex);
    threshold = max(3 * deadband, 0.20);
    profileInfo.deadbandThresholdNorm(surfaceIndex) = threshold;
    usable = amplitudeSet(abs(amplitudeSet) >= threshold);
    if isempty(usable)
        usable = [-1, 1];
        profileInfo.fallbackAmplitudeUsed = true;
    end
    amplitudesBySurface{surfaceIndex} = usable;
end
end

function calibration = loadCalibration(calibrationFile)
calibration = [];
try
    loadedData = load(calibrationFile);
catch
    return;
end

if isfield(loadedData, "calibration")
    calibration = loadedData.calibration;
elseif isfield(loadedData, "result") && isfield(loadedData.result, "calibration")
    calibration = loadedData.result.calibration;
elseif isfield(loadedData, "surfaceCalibration")
    calibration = loadedData.surfaceCalibration;
end
end

function deadband = extractSurfaceDeadband(calibration, surfaceIndex)
deadband = NaN;
if istable(calibration)
    rowMask = true(height(calibration), 1);
    if ismember("surface_index", string(calibration.Properties.VariableNames))
        rowMask = double(calibration.surface_index) == surfaceIndex;
    end
    rowIndex = find(rowMask, 1, "first");
    if ~isempty(rowIndex)
        deadband = max(abs(readTableValue(calibration, rowIndex, "deadband_positive_norm")), ...
            abs(readTableValue(calibration, rowIndex, "deadband_negative_norm")));
    end
elseif isstruct(calibration) && isfield(calibration, "surfaceTable")
    deadband = extractSurfaceDeadband(calibration.surfaceTable, surfaceIndex);
elseif isstruct(calibration) && isfield(calibration, "summaryTable")
    deadband = extractSurfaceDeadband(calibration.summaryTable, surfaceIndex);
end

if ~isfinite(deadband)
    deadband = 0.20;
end
end

function value = readTableValue(tableData, rowIndex, variableName)
if ismember(variableName, string(tableData.Properties.VariableNames))
    value = double(tableData.(char(variableName))(rowIndex));
else
    value = NaN;
end
end

function value = getFieldOrDefault(config, fieldName, defaultValue)
if isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
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
