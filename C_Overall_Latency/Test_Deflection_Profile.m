function [cmd, profileState] = Test_Deflection_Profile(tNowS, viconSample, config, profileState)
%TEST_DEFLECTION_PROFILE Static one-surface sweep command provider.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Public command-provider entry point
% 2) Event-table construction
% 3) Ramp-level and configuration helpers
% ==========================================================================
%% =========================================================================
% 1) Public command-provider entry point
% ==========================================================================
if nargin < 4 || isempty(profileState) || ~isfield(profileState, "isInitialized")
    profileState = initializeProfile(config);
end

eventIndex = find(tNowS >= profileState.eventTable.scheduled_start_s & ...
    tNowS < profileState.eventTable.scheduled_end_s, 1, "first");

surfaceNorm = zeros(1, numel(config.surfaceOrder));
eventId = NaN;
activeSurfaceIndex = NaN;
activeSurfaceName = "";
commandLevelNorm = 0.0;
description = "neutral";

if ~isempty(eventIndex)
    eventRow = profileState.eventTable(eventIndex, :);
    eventId = eventRow.event_id;
    activeSurfaceIndex = eventRow.surface_index;
    activeSurfaceName = string(eventRow.surface_name);
    commandLevelNorm = eventRow.command_level_norm;
    surfaceNorm(activeSurfaceIndex) = commandLevelNorm;
    description = "deflection_" + activeSurfaceName + "_" + string(commandLevelNorm);
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

%% =========================================================================
% 2) Event-table construction
% ==========================================================================
function profileState = initializeProfile(config)
surfaceOrder = reshape(string(config.surfaceOrder), 1, []);
surfaceIndices = getFieldOrDefault(config, "enabledSurfaceIndices", 1:numel(surfaceOrder));
surfaceIndices = reshape(double(surfaceIndices), 1, []);
surfaceIndices = surfaceIndices(surfaceIndices >= 1 & surfaceIndices <= numel(surfaceOrder));
if isempty(surfaceIndices)
    surfaceIndices = 1:numel(surfaceOrder);
end

holdSeconds = getFieldOrDefault(config, "deflectionHoldSeconds", 0.75);
% Ramp levels are normalized surface commands, not degrees; Vicon measures the achieved deflection.
rampLevels = normalizeRampLevels(getFieldOrDefault( ...
    config, "deflectionRampLevels", [0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]));
[positiveLevels, positivePhases] = makeSignedRamp(rampLevels, 1);
[negativeLevels, negativePhases] = makeSignedRamp(rampLevels, -1);

eventRows = struct( ...
    "event_id", {}, ...
    "surface_index", {}, ...
    "surface_name", {}, ...
    "command_level_norm", {}, ...
    "scheduled_start_s", {}, ...
    "scheduled_end_s", {}, ...
    "direction_group", {}, ...
    "sweep_polarity", {}, ...
    "sweep_phase", {}, ...
    "sweep_step_index", {}, ...
    "sweep_step_count", {});

profileStartHostS = double(getFieldOrDefault(config, "profileStartHostS", 0));
% Profile time is offset by Run_Control_Path after serial setup and Vicon neutral calibration.
nextStartS = profileStartHostS + double(config.neutralLeadSeconds);
eventId = 1;
for surfaceIndex = surfaceIndices
    [eventRows, nextStartS, eventId] = appendSweepRows( ...
        eventRows, eventId, surfaceIndex, surfaceOrder(surfaceIndex), ...
        positiveLevels, positivePhases, holdSeconds, nextStartS, "positive_first", "positive");
    [eventRows, nextStartS, eventId] = appendSweepRows( ...
        eventRows, eventId, surfaceIndex, surfaceOrder(surfaceIndex), ...
        negativeLevels, negativePhases, holdSeconds, nextStartS, "positive_first", "negative");
    [eventRows, nextStartS, eventId] = appendSweepRows( ...
        eventRows, eventId, surfaceIndex, surfaceOrder(surfaceIndex), ...
        negativeLevels, negativePhases, holdSeconds, nextStartS, "negative_first", "negative");
    [eventRows, nextStartS, eventId] = appendSweepRows( ...
        eventRows, eventId, surfaceIndex, surfaceOrder(surfaceIndex), ...
        positiveLevels, positivePhases, holdSeconds, nextStartS, "negative_first", "positive");
end

profileState = struct();
profileState.isInitialized = true;
profileState.sequence = 0;
profileState.eventTable = struct2table(eventRows);
eventsPerSurface = height(profileState.eventTable) ./ max(1, numel(surfaceIndices));
profileState.profileInfo = struct( ...
    "profileMode", "deflection", ...
    "deflectionHoldSeconds", holdSeconds, ...
    "deflectionRampLevels", rampLevels, ...
    "eventsPerSurface", eventsPerSurface, ...
    "enabledSurfaceIndices", surfaceIndices, ...
    "requiredActiveSeconds", max(0, nextStartS - profileStartHostS - double(config.neutralLeadSeconds)));
end

function [eventRows, nextStartS, eventId] = appendSweepRows( ...
    eventRows, eventId, surfaceIndex, surfaceName, levels, phaseLabels, holdSeconds, nextStartS, directionGroup, polarity)
stepCount = numel(levels);
for levelIndex = 1:numel(levels)
    eventRows(end + 1) = struct( ... %#ok<AGROW>
        "event_id", eventId, ...
        "surface_index", surfaceIndex, ...
        "surface_name", surfaceName, ...
        "command_level_norm", levels(levelIndex), ...
        "scheduled_start_s", nextStartS, ...
        "scheduled_end_s", nextStartS + holdSeconds, ...
        "direction_group", directionGroup, ...
        "sweep_polarity", polarity, ...
        "sweep_phase", phaseLabels(levelIndex), ...
        "sweep_step_index", levelIndex, ...
        "sweep_step_count", stepCount);
    nextStartS = nextStartS + holdSeconds;
    eventId = eventId + 1;
end
end

%% =========================================================================
% 3) Ramp-level and configuration helpers
% ==========================================================================
function levels = normalizeRampLevels(levels)
levels = reshape(abs(double(levels)), 1, []);
levels = levels(isfinite(levels));
levels = min(max(levels, 0), 1);
levels = unique(levels, "stable");
levels = sort(levels);
if isempty(levels) || levels(1) ~= 0
    levels = [0, levels];
end
if levels(end) ~= 1
    levels = [levels, 1];
end
end

function [levels, phaseLabels] = makeSignedRamp(baseLevels, polaritySign)
% Outbound and return phases keep hysteresis separate from one-way gain estimates.
outboundLevels = baseLevels;
returnLevels = fliplr(baseLevels(1:end - 1));
levels = polaritySign .* [outboundLevels, returnLevels];
phaseLabels = [ ...
    repmat("outbound", 1, numel(outboundLevels)), ...
    repmat("return", 1, numel(returnLevels))];
end

function value = getFieldOrDefault(config, fieldName, defaultValue)
if isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
end
