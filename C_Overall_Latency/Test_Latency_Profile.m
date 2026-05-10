function [cmd, profileState] = Test_Latency_Profile(tNowS, viconSample, config, profileState)
%TEST_LATENCY_PROFILE Bang-bang one-surface latency command provider.
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
    description = string(eventRow.step_label);
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

eventHoldSeconds = double(getFieldOrDefault(config, "eventHoldSeconds", 0.60));
profileStartHostS = double(getFieldOrDefault(config, "profileStartHostS", 0));
activeStartS = profileStartHostS + double(config.neutralLeadSeconds);
activeStopS = activeStartS + double(config.activeCommandSeconds);
randomSeed = double(config.randomSeed);
amplitudeBySurface = resolveAmplitudeBySurface(config, surfaceCount);
repetitionsPerSurface = max(1, round(double(getFieldOrDefault(config, "latencyRepetitionsPerSurface", 4))));

rng(randomSeed, "twister");

eventRows = emptyEventRows();
currentCommand = zeros(1, surfaceCount);
nextStartS = activeStartS;
eventId = 1;
stopBuilding = false;

for repetitionIndex = 1:repetitionsPerSurface
    if stopBuilding
        break;
    end

    shuffledSurfaceIndices = surfaceIndices(randperm(numel(surfaceIndices)));
    if mod(repetitionIndex, 2) == 0
        signedDirections = [-1, 1];
    else
        signedDirections = [1, -1];
    end

    for surfaceIndex = shuffledSurfaceIndices
        for signedDirection = signedDirections
            targetCommand = signedDirection * amplitudeBySurface(surfaceIndex);
            [eventRows, currentCommand, nextStartS, eventId, stopBuilding] = appendLatencyEvent( ...
                eventRows, currentCommand, nextStartS, eventHoldSeconds, activeStopS, ...
                eventId, surfaceIndex, surfaceOrder(surfaceIndex), targetCommand, ...
                randomSeed, repetitionIndex, "step_to_command");
            if stopBuilding
                break;
            end

            [eventRows, currentCommand, nextStartS, eventId, stopBuilding] = appendLatencyEvent( ...
                eventRows, currentCommand, nextStartS, eventHoldSeconds, activeStopS, ...
                eventId, surfaceIndex, surfaceOrder(surfaceIndex), 0.0, ...
                randomSeed, repetitionIndex, "return_to_neutral");
            if stopBuilding
                break;
            end
        end
        if stopBuilding
            break;
        end
    end
end

profileState = struct();
profileState.isInitialized = true;
profileState.sequence = 0;
profileState.eventTable = struct2table(eventRows);
profileState.profileInfo = struct( ...
    "profileMode", "latency_bang_bang_measured_fraction", ...
    "eventHoldSeconds", eventHoldSeconds, ...
    "profileStartHostS", profileStartHostS, ...
    "activeStartHostS", activeStartS, ...
    "activeStopHostS", activeStopS, ...
    "randomSeed", randomSeed, ...
    "enabledSurfaceIndices", surfaceIndices, ...
    "latencyAmplitudeNorm", amplitudeBySurface, ...
    "latencyRepetitionsPerSurface", repetitionsPerSurface, ...
    "calibrationUsed", false, ...
    "deadbandAssumption", "not used for bang-bang latency", ...
    "latencyTimingObservable", "Vicon measured response fraction");
end

function [eventRows, currentCommand, nextStartS, eventId, stopBuilding] = appendLatencyEvent( ...
    eventRows, currentCommand, nextStartS, eventHoldSeconds, activeStopS, ...
    eventId, surfaceIndex, surfaceName, targetCommand, randomSeed, repetitionIndex, stepKind)

stopBuilding = false;
scheduledEndS = nextStartS + eventHoldSeconds;
if scheduledEndS > activeStopS + 10 * eps
    stopBuilding = true;
    return;
end

commandBefore = currentCommand;
currentCommand(surfaceIndex) = targetCommand;
commandAfter = currentCommand;
commandStepNorm = commandAfter(surfaceIndex) - commandBefore(surfaceIndex);
stepLabel = sprintf("latency_bang_bang_%s_%s_%+.3f", ...
    char(surfaceName), char(stepKind), commandStepNorm);

eventRows(end + 1) = struct( ... %#ok<AGROW>
    "event_id", eventId, ...
    "surface_index", surfaceIndex, ...
    "surface_name", surfaceName, ...
    "command_before_norm", {commandBefore}, ...
    "command_after_norm", {commandAfter}, ...
    "scheduled_start_s", nextStartS, ...
    "scheduled_end_s", scheduledEndS, ...
    "random_seed", randomSeed, ...
    "repetition_index", repetitionIndex, ...
    "step_kind", string(stepKind), ...
    "step_label", string(stepLabel), ...
    "latency_method", "bang_bang_measured_response_fraction");

nextStartS = scheduledEndS;
eventId = eventId + 1;
end

function eventRows = emptyEventRows()
eventRows = struct( ...
    "event_id", {}, ...
    "surface_index", {}, ...
    "surface_name", {}, ...
    "command_before_norm", {}, ...
    "command_after_norm", {}, ...
    "scheduled_start_s", {}, ...
    "scheduled_end_s", {}, ...
    "random_seed", {}, ...
    "repetition_index", {}, ...
    "step_kind", {}, ...
    "step_label", {}, ...
    "latency_method", {});
end

function amplitudeBySurface = resolveAmplitudeBySurface(config, surfaceCount)
amplitude = getFieldOrDefault(config, "latencyBangBangAmplitudeNorm", 0.70);
amplitude = reshape(double(amplitude), 1, []);
if isempty(amplitude)
    amplitude = 0.70;
end
amplitude = min(max(abs(amplitude), 0.20), 1.0);
if numel(amplitude) == 1
    amplitudeBySurface = repmat(amplitude, 1, surfaceCount);
else
    amplitudeBySurface = resizeRow(amplitude, surfaceCount, amplitude(end));
end
end

function value = getFieldOrDefault(config, fieldName, defaultValue)
if isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
end

function row = resizeRow(row, targetCount, fillValue)
row = reshape(row, 1, []);
if numel(row) < targetCount
    row = [row, repmat(fillValue, 1, targetCount - numel(row))];
elseif numel(row) > targetCount
    row = row(1:targetCount);
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
