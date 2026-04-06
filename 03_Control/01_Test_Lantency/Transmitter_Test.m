function runData = Transmitter_Test(config)
% TRANSMITTER_TEST Execute a serial-to-PPM transmitter latency test.
%   runData = Transmitter_Test(config) commands the Uno transmitter
%   firmware, captures reference, trainer, and receiver waveforms, and
%   exports matched latency results and diagnostic summaries.
arguments
    config (1,1) struct = struct()
end

config = normalizeTransmitterConfig(config);
runPlan = buildTransmitterRunPlan(config);
runData = initializeTransmitterRunData(config);
runData.runInfo.startTime = datetime("now");
runData.runInfo.scheduledDurationSeconds = runPlan.scheduledDurationSeconds;

commandInterface = struct();
sigrokSession = createEmptySigrokSession();
cleanupHandle = onCleanup(@() cleanupTransmitterResources(commandInterface, config, sigrokSession));

try
    [commandInterface, arduinoInfo] = connectToTransmitter(config);
    runData.config = config;
    runData.connectionInfo.arduino = arduinoInfo;

    if ~arduinoInfo.isConnected
        runData.runInfo.status = "arduino_connection_failed";
        runData.runInfo.reason = arduinoInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        printTransmitterConnectionStatus(runData);
        return;
    end

    printTransmitterConnectionStatus(runData);

    moveTransmitterToNeutral(commandInterface);
    pause(config.neutralSettleSeconds);

    if isSigrokAutoMode(config)
        [sigrokSession, config] = startSigrokSession(config, runPlan);
        pause(config.logicAnalyzer.captureStartLeadSeconds);
    end

    [storage, runInfo, config] = executeTransmitterRun(commandInterface, config, runPlan);
    loggerData = storage.rawLogs;

    if isSigrokAutoMode(config)
        sigrokSession = waitForSigrokSession(sigrokSession, config);
        config = finalizeSigrokSession(sigrokSession, config);
    end

    [referenceCapture, trainerPpmCapture, receiverCapture] = importTransmitterCaptureData(config);
    loggerData.referenceCapture = referenceCapture;
    loggerData.trainerPpmCapture = trainerPpmCapture;
    loggerData.receiverCapture = receiverCapture;
    storage = finalizeTransmitterStorage(storage, config, loggerData);

    runData.config = config;
    runData.runInfo = runInfo;
    runData.logs = buildTransmitterLogs(storage, config);
    runData.surfaceSummary = buildTransmitterSurfaceSummaryFromLogs(runData.logs);
    runData.artifacts = exportTransmitterRunData(runData);

    clear cleanupHandle
    cleanupTransmitterResources(commandInterface, config, sigrokSession);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    rethrow(executionException);
end
end

function commandProfile = normalizeCommandProfile(commandProfileConfig)
if isempty(commandProfileConfig)
    commandProfileConfig = struct();
end

commandProfile.type = getTextScalarField(commandProfileConfig, "type", "latency_step_train");
commandProfile.sampleTimeSeconds = getPositiveScalarField(commandProfileConfig, "sampleTimeSeconds", 0.02);
commandProfile.preCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "preCommandNeutralSeconds", 5.0);
commandProfile.postCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "postCommandNeutralSeconds", 1.0);
commandProfile.durationSeconds = getPositiveScalarField(commandProfileConfig, "durationSeconds", 59.0);
commandProfile.amplitudeDegrees = getScalarField(commandProfileConfig, "amplitudeDegrees", 45.0);
commandProfile.offsetDegrees = getScalarField(commandProfileConfig, "offsetDegrees", 0.0);
commandProfile.frequencyHz = getPositiveScalarField(commandProfileConfig, "frequencyHz", 0.5);
commandProfile.phaseDegrees = getScalarField(commandProfileConfig, "phaseDegrees", 90.0);
commandProfile.doubletHoldSeconds = getPositiveScalarField(commandProfileConfig, "doubletHoldSeconds", 0.5);
commandProfile.eventHoldSeconds = getPositiveScalarField(commandProfileConfig, "eventHoldSeconds", 0.20);
commandProfile.eventNeutralHoldSeconds = getNonnegativeScalarField(commandProfileConfig, "eventNeutralHoldSeconds", 0.10);
commandProfile.eventDwellSeconds = getNonnegativeScalarField(commandProfileConfig, "eventDwellSeconds", 0.60);
commandProfile.eventRandomJitterSeconds = getNonnegativeScalarField(commandProfileConfig, "eventRandomJitterSeconds", 0.05);
commandProfile.randomSeed = getScalarField(commandProfileConfig, "randomSeed", 5);
commandProfile.customTimeSeconds = getNumericVectorField(commandProfileConfig, "customTimeSeconds", []);
commandProfile.customDeflectionDegrees = getNumericVectorField(commandProfileConfig, "customDeflectionDegrees", []);
commandProfile.customInterpolationMethod = getTextScalarField(commandProfileConfig, "customInterpolationMethod", "linear");
commandProfile.customFunction = getFieldOrDefault(commandProfileConfig, "customFunction", []);
commandProfile.effectiveRandomSeed = nan;
commandProfile.profileEvents = table();
commandProfile.precomputedBaseCommandDegrees = [];
commandProfile.precomputedCommandVectorDegrees = [];
commandProfile.vectorSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];

validProfileTypes = ["latency_step_train", "latency_vector_step_train", "latency_isolated_step", "sine", "square", "doublet", "custom", "function"];
if ~any(commandProfile.type == validProfileTypes)
    error("Transmitter_Test:InvalidProfileType", ...
        "commandProfile.type must be one of: %s.", ...
        char(join(validProfileTypes, ", ")));
end

if ~isempty(commandProfile.customFunction) && ~isa(commandProfile.customFunction, "function_handle")
    error("Transmitter_Test:InvalidCustomFunction", "commandProfile.customFunction must be a function handle when provided.");
end

if commandProfile.type == "doublet"
    commandProfile.durationSeconds = max(commandProfile.durationSeconds, 2 .* commandProfile.doubletHoldSeconds);
end

if commandProfile.type == "custom"
    validateInputAttributes(commandProfile.customTimeSeconds, {"numeric"}, {"real", "finite", "vector", "nonempty"}, char(mfilename), 'commandProfile.customTimeSeconds');
    validateInputAttributes(commandProfile.customDeflectionDegrees, {"numeric"}, {"real", "finite", "vector", "numel", numel(commandProfile.customTimeSeconds)}, char(mfilename), 'commandProfile.customDeflectionDegrees');

    commandProfile.customTimeSeconds = reshape(double(commandProfile.customTimeSeconds), [], 1);
    commandProfile.customDeflectionDegrees = reshape(double(commandProfile.customDeflectionDegrees), [], 1);

    if any(diff(commandProfile.customTimeSeconds) <= 0)
        error("Transmitter_Test:InvalidCustomTimeVector", "commandProfile.customTimeSeconds must be strictly increasing.");
    end

    commandProfile.durationSeconds = commandProfile.customTimeSeconds(end);
end

if commandProfile.type == "function" && isempty(commandProfile.customFunction)
    error("Transmitter_Test:MissingCustomFunction", "commandProfile.customFunction is required when type is 'function'.");
end

if isfinite(commandProfile.randomSeed)
    validateInputAttributes(commandProfile.randomSeed, {"numeric"}, {"real", "finite", "scalar"}, char(mfilename), 'commandProfile.randomSeed');
end
end

function commandEncoding = canonicalizeNanoLoggerCommandEncoding(commandEncoding)
commandEncoding = lower(string(commandEncoding));
if ~any(commandEncoding == ["binary_vector", "text_set_all"])
    error("Transmitter_Test:InvalidNanoLoggerCommandEncoding", ...
        "arduinoTransport.commandEncoding must be 'binary_vector' or 'text_set_all'.");
end
end

function rxTimestampSource = canonicalizeRxTimestampSource(rxTimestampSource)
rxTimestampSource = lower(string(rxTimestampSource));
if ~any(rxTimestampSource == ["host_rx_us", "board_rx_us"])
    error("Transmitter_Test:InvalidRxTimestampSource", ...
        "arduinoTransport.rxTimestampSource must be 'host_rx_us' or 'board_rx_us'.");
end
end

function statusText = getStatusText(connectionInfo)
if isempty(fieldnames(connectionInfo))
    statusText = "not attempted";
elseif isfield(connectionInfo, "isConnected") && connectionInfo.isConnected
    statusText = "connected";
else
    if isfield(connectionInfo, "connectionMessage")
        statusText = "not connected - " + string(connectionInfo.connectionMessage);
    else
        statusText = "not connected";
    end
end
end

function [scheduledTimeSeconds, profileInfo] = buildCommandSchedule(commandProfile)
totalDurationSeconds = ...
    commandProfile.preCommandNeutralSeconds + ...
    commandProfile.durationSeconds + ...
    commandProfile.postCommandNeutralSeconds;

scheduledTimeSeconds = (0:commandProfile.sampleTimeSeconds:totalDurationSeconds).';
if scheduledTimeSeconds(end) < totalDurationSeconds
    scheduledTimeSeconds(end + 1, 1) = totalDurationSeconds;
end

profileInfo = commandProfile;
profileInfo.commandStartSeconds = commandProfile.preCommandNeutralSeconds;
profileInfo.commandStopSeconds = commandProfile.preCommandNeutralSeconds + commandProfile.durationSeconds;
profileInfo.totalDurationSeconds = totalDurationSeconds;
profileInfo.precomputedBaseCommandDegrees = zeros(size(scheduledTimeSeconds));
profileInfo.profileEvents = table( ...
    zeros(0, 1), ...
    strings(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', { ...
        'EventIndex', ...
        'EventLabel', ...
        'TargetDeflection_deg', ...
        'StartTime_s', ...
        'StopTime_s', ...
        'DwellJitter_s'});

switch profileInfo.type
    case "latency_step_train"
        [baseCommandDegrees, profileEvents, effectiveRandomSeed] = buildLatencyStepTrainSchedule( ...
            profileInfo, ...
            scheduledTimeSeconds);
        profileInfo.precomputedBaseCommandDegrees = baseCommandDegrees;
        profileInfo.profileEvents = profileEvents;
        profileInfo.effectiveRandomSeed = effectiveRandomSeed;

    case "latency_vector_step_train"
        [baseCommandDegrees, commandVectorDegrees, profileEvents, effectiveRandomSeed] = buildLatencyVectorStepTrainSchedule( ...
            profileInfo, ...
            scheduledTimeSeconds);
        profileInfo.precomputedBaseCommandDegrees = baseCommandDegrees;
        profileInfo.precomputedCommandVectorDegrees = commandVectorDegrees;
        profileInfo.profileEvents = profileEvents;
        profileInfo.effectiveRandomSeed = effectiveRandomSeed;

    case "latency_isolated_step"
        [baseCommandDegrees, profileEvents] = buildIsolatedLatencyStepSchedule( ...
            profileInfo, ...
            scheduledTimeSeconds);
        profileInfo.precomputedBaseCommandDegrees = baseCommandDegrees;
        profileInfo.profileEvents = profileEvents;
        profileInfo.effectiveRandomSeed = NaN;

    otherwise
        for sampleIndex = 1:numel(scheduledTimeSeconds)
            profileInfo.precomputedBaseCommandDegrees(sampleIndex) = ...
                evaluateBaseCommandDegrees(profileInfo, scheduledTimeSeconds(sampleIndex), NaN);
        end
end
end

function commandDegrees = evaluateBaseCommandDegrees(profileInfo, elapsedTimeSeconds, sampleIndex)
if nargin >= 3 && isfinite(sampleIndex) && ...
        isfield(profileInfo, "precomputedBaseCommandDegrees") && ...
        numel(profileInfo.precomputedBaseCommandDegrees) >= sampleIndex
    commandDegrees = profileInfo.precomputedBaseCommandDegrees(sampleIndex);
    return;
end

if elapsedTimeSeconds < profileInfo.commandStartSeconds || elapsedTimeSeconds > profileInfo.commandStopSeconds
    commandDegrees = 0;
    return;
end

profileTimeSeconds = elapsedTimeSeconds - profileInfo.commandStartSeconds;
phaseRadians = deg2rad(profileInfo.phaseDegrees);

switch profileInfo.type
    case "latency_step_train"
        commandDegrees = evaluateLatencyStepTrainAtTime(profileInfo.profileEvents, profileTimeSeconds);
    case "latency_isolated_step"
        commandDegrees = evaluateLatencyStepTrainAtTime(profileInfo.profileEvents, profileTimeSeconds);
    case "sine"
        commandDegrees = profileInfo.offsetDegrees + ...
            profileInfo.amplitudeDegrees .* sin(2 .* pi .* profileInfo.frequencyHz .* profileTimeSeconds + phaseRadians);
    case "square"
        commandDegrees = profileInfo.offsetDegrees + ...
            profileInfo.amplitudeDegrees .* squareWave(2 .* pi .* profileInfo.frequencyHz .* profileTimeSeconds + phaseRadians);
    case "doublet"
        if profileTimeSeconds <= profileInfo.doubletHoldSeconds
            commandDegrees = profileInfo.offsetDegrees + profileInfo.amplitudeDegrees;
        elseif profileTimeSeconds <= 2 .* profileInfo.doubletHoldSeconds
            commandDegrees = profileInfo.offsetDegrees - profileInfo.amplitudeDegrees;
        else
            commandDegrees = profileInfo.offsetDegrees;
        end
    case "custom"
        commandDegrees = interp1( ...
            profileInfo.customTimeSeconds, ...
            profileInfo.customDeflectionDegrees, ...
            profileTimeSeconds, ...
            char(profileInfo.customInterpolationMethod), ...
            0);
    otherwise
        commandDegrees = double(profileInfo.customFunction(profileTimeSeconds));
end
end

function commandVectorDegrees = evaluateCommandVectorDegrees(profileInfo, surfaceNames, sampleIndex)
commandVectorDegrees = nan(1, numel(surfaceNames));
if nargin < 3 || ~isfinite(sampleIndex)
    return;
end
if ~isstruct(profileInfo) || ...
        ~isfield(profileInfo, "precomputedCommandVectorDegrees") || ...
        isempty(profileInfo.precomputedCommandVectorDegrees) || ...
        size(profileInfo.precomputedCommandVectorDegrees, 1) < sampleIndex || ...
        ~isfield(profileInfo, "vectorSurfaceNames")
    return;
end

vectorSurfaceNames = reshape(string(profileInfo.vectorSurfaceNames), 1, []);
vectorRow = double(profileInfo.precomputedCommandVectorDegrees(sampleIndex, :));
if numel(vectorRow) ~= numel(vectorSurfaceNames)
    return;
end

commandVectorDegrees(:) = 0.0;
for surfaceIndex = 1:numel(surfaceNames)
    mappedIndex = find(vectorSurfaceNames == string(surfaceNames(surfaceIndex)), 1, "first");
    if isempty(mappedIndex)
        continue;
    end
    commandVectorDegrees(surfaceIndex) = vectorRow(mappedIndex);
end
end

function nowUs = hostNowUs(hostTimer)
nowUs = uint32(round(toc(hostTimer) .* 1e6));
end

function tableData = readCommentCsv(filePath)
options = detectImportOptions(filePath, "FileType", "text", "CommentStyle", "#");
options.CommentStyle = "#";
tableData = readtable(filePath, options);
end

function [clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog)
hostTxUs = double(syncRoundTripLog.host_tx_us);
hostRxUs = double(syncRoundTripLog.host_rx_us);
boardRxUs = double(syncRoundTripLog.board_rx_us);
boardTxUs = double(syncRoundTripLog.board_tx_us);

midpointMask = ...
    isfinite(hostTxUs) & ...
    isfinite(hostRxUs) & ...
    isfinite(boardRxUs) & ...
    isfinite(boardTxUs);
if any(midpointMask)
    % Use host and board midpoints when direct round-trip timestamps exist.
    hostReferenceUs = 0.5 .* (hostTxUs(midpointMask) + hostRxUs(midpointMask));
    boardReferenceUs = 0.5 .* (boardRxUs(midpointMask) + boardTxUs(midpointMask));
else
    % Fallback for legacy captures without immediate SYNC_EVENT round trips.
    hostReferenceUs = hostTxUs(isfinite(hostTxUs) & isfinite(boardRxUs));
    boardReferenceUs = boardRxUs(isfinite(hostTxUs) & isfinite(boardRxUs));
end

if isempty(hostReferenceUs)
    error("Transmitter_Test:MissingNanoLoggerSyncTelemetry", ...
        "No valid Nano logger sync timestamps were available for clock mapping.");
end

if numel(hostReferenceUs) >= 2
    fitCoefficients = polyfit(boardReferenceUs, hostReferenceUs, 1);
    clockSlope = fitCoefficients(1);
    clockIntercept = fitCoefficients(2);
else
    clockSlope = 1.0;
    clockIntercept = hostReferenceUs(1) - boardReferenceUs(1);
end

syncForwardLatencyUs = clockSlope .* boardReferenceUs + clockIntercept - hostReferenceUs;
minimumForwardLatencyUs = min(syncForwardLatencyUs);
if minimumForwardLatencyUs < 0
    clockIntercept = clockIntercept - minimumForwardLatencyUs;
end
end

function rowTimesSeconds = chooseFiniteRowTimes(primaryTimesSeconds, fallbackTimesSeconds)
rowTimesSeconds = primaryTimesSeconds;
missingMask = ~isfinite(rowTimesSeconds);
rowTimesSeconds(missingMask) = fallbackTimesSeconds(missingMask);
end

function variableName = resolveTableVariableName(tableData, requestedName, isRequired)
if nargin < 3
    isRequired = true;
end

if strlength(requestedName) == 0
    variableName = "";
    return;
end

variableNames = string(tableData.Properties.VariableNames);
exactMask = variableNames == requestedName;
if any(exactMask)
    variableName = variableNames(find(exactMask, 1, "first"));
    return;
end

caseInsensitiveMask = strcmpi(variableNames, requestedName);
if any(caseInsensitiveMask)
    variableName = variableNames(find(caseInsensitiveMask, 1, "first"));
    return;
end

if isRequired
    error("Transmitter_Test:MissingArduinoEchoColumn", ...
        "The Arduino echo import is missing column '%s'.", ...
        char(requestedName));
end

variableName = "";
end

function stats = computeLatencyStats(values)
stats = struct( ...
    "sampleCount", 0, ...
    "meanValue", nan, ...
    "stdValue", nan, ...
    "medianValue", nan, ...
    "p95Value", nan, ...
    "p99Value", nan, ...
    "maxValue", nan);

values = reshape(double(values), [], 1);
values = values(isfinite(values));

if isempty(values)
    return;
end

stats.sampleCount = numel(values);
stats.meanValue = mean(values);
if numel(values) >= 2
    stats.stdValue = std(values, 0);
end
stats.medianValue = computeSamplePercentile(values, 50);
stats.p95Value = computeSamplePercentile(values, 95);
stats.p99Value = computeSamplePercentile(values, 99);
stats.maxValue = max(values);
end

function percentileValue = computeSamplePercentile(sampleValues, percentile)
sampleValues = sort(reshape(double(sampleValues), [], 1));
sampleCount = numel(sampleValues);

if sampleCount == 0
    percentileValue = NaN;
    return;
end

if sampleCount == 1
    percentileValue = sampleValues(1);
    return;
end

clampedPercentile = min(max(double(percentile), 0), 100) ./ 100;
samplePosition = 1 + (sampleCount - 1) .* clampedPercentile;
lowerIndex = floor(samplePosition);
upperIndex = ceil(samplePosition);

if lowerIndex == upperIndex
    percentileValue = sampleValues(lowerIndex);
    return;
end

upperWeight = samplePosition - lowerIndex;
lowerWeight = 1 - upperWeight;
percentileValue = ...
    lowerWeight .* sampleValues(lowerIndex) + ...
    upperWeight .* sampleValues(upperIndex);
end

function settingValue = formatSettingValue(value)
if isa(value, "datetime")
    if isnat(value)
        settingValue = "NaT";
    else
        settingValue = string(value);
    end
elseif isstring(value)
    settingValue = join(reshape(value, 1, []), ", ");
elseif ischar(value)
    settingValue = string(value);
elseif islogical(value)
    settingValue = join(string(reshape(value, 1, [])), ", ");
elseif isnumeric(value)
    if isempty(value)
        settingValue = "";
    elseif isscalar(value)
        settingValue = string(value);
    else
        settingValue = join(compose("%.9g", reshape(value, 1, [])), ", ");
    end
elseif isa(value, "function_handle")
    settingValue = string(func2str(value));
else
    settingValue = string(value);
end
end

function exportTable = timetableToExportTable(timetableData)
exportTable = timetable2table(timetableData);
exportTable.Properties.VariableNames{1} = 'time_s';
exportTable.time_s = seconds(exportTable.time_s);
end

function output = squareWave(phaseRadians)
output = ones(size(phaseRadians));
output(sin(phaseRadians) < 0) = -1;
end

function [baseCommandDegrees, profileEvents, effectiveRandomSeed] = buildLatencyStepTrainSchedule(profileInfo, scheduledTimeSeconds)
baseCommandDegrees = zeros(size(scheduledTimeSeconds));
profileTimeSeconds = scheduledTimeSeconds - profileInfo.commandStartSeconds;
profileEventRows = cell(0, 1);
currentTimeSeconds = 0.0;
eventIndex = 0;
[randomStream, effectiveRandomSeed] = createLocalRandomStream(profileInfo.randomSeed);

while currentTimeSeconds < profileInfo.durationSeconds
    positiveDwellJitterSeconds = drawEventJitter(randomStream, profileInfo.eventRandomJitterSeconds);
    positiveBlockDurationSeconds = ...
        profileInfo.eventNeutralHoldSeconds + ...
        profileInfo.eventHoldSeconds + ...
        profileInfo.eventNeutralHoldSeconds + ...
        profileInfo.eventDwellSeconds + positiveDwellJitterSeconds;
    if currentTimeSeconds + positiveBlockDurationSeconds > profileInfo.durationSeconds
        break;
    end

    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_before_positive", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventNeutralHoldSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "positive_step", profileInfo.offsetDegrees + profileInfo.amplitudeDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventHoldSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_after_positive", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventNeutralHoldSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "positive_dwell", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventDwellSeconds + positiveDwellJitterSeconds, positiveDwellJitterSeconds);

    negativeDwellJitterSeconds = drawEventJitter(randomStream, profileInfo.eventRandomJitterSeconds);
    negativeBlockDurationSeconds = ...
        profileInfo.eventHoldSeconds + ...
        profileInfo.eventNeutralHoldSeconds + ...
        profileInfo.eventDwellSeconds + negativeDwellJitterSeconds;
    if currentTimeSeconds + negativeBlockDurationSeconds > profileInfo.durationSeconds
        break;
    end

    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "negative_step", profileInfo.offsetDegrees - profileInfo.amplitudeDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventHoldSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_after_negative", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventNeutralHoldSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "negative_dwell", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + profileInfo.eventDwellSeconds + negativeDwellJitterSeconds, negativeDwellJitterSeconds);
end

if currentTimeSeconds < profileInfo.durationSeconds
    [profileEventRows, eventIndex] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_tail", profileInfo.offsetDegrees, currentTimeSeconds, profileInfo.durationSeconds, 0.0);
end

profileEvents = profileEventRowsToTable(profileEventRows);
if isempty(profileEvents)
    return;
end

for eventRowIndex = 1:height(profileEvents)
    eventStartSeconds = profileEvents.StartTime_s(eventRowIndex);
    eventStopSeconds = profileEvents.StopTime_s(eventRowIndex);
    eventTargetDegrees = profileEvents.TargetDeflection_deg(eventRowIndex);
    if eventStopSeconds <= eventStartSeconds
        continue;
    end

    if eventRowIndex < height(profileEvents)
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds < eventStopSeconds;
    else
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds <= eventStopSeconds;
    end
    baseCommandDegrees(eventMask) = eventTargetDegrees;
end
end

function [baseCommandDegrees, commandVectorDegrees, profileEvents, effectiveRandomSeed] = buildLatencyVectorStepTrainSchedule(profileInfo, scheduledTimeSeconds)
surfaceNames = reshape(string(profileInfo.vectorSurfaceNames), 1, []);
surfaceCount = numel(surfaceNames);
sampleCount = numel(scheduledTimeSeconds);
baseCommandDegrees = nan(sampleCount, 1);
commandVectorDegrees = zeros(sampleCount, surfaceCount);
profileTimeSeconds = scheduledTimeSeconds - profileInfo.commandStartSeconds;

profileEventRows = cell(0, 1);
eventVectors = zeros(0, surfaceCount);
eventAxes = strings(0, 1);
currentTimeSeconds = 0.0;
eventIndex = 0;
currentTargetVectorDegrees = zeros(1, surfaceCount);
[randomStream, effectiveRandomSeed] = createLocalRandomStream(profileInfo.randomSeed);

while currentTimeSeconds < profileInfo.durationSeconds
    dwellJitterSeconds = drawEventJitter(randomStream, profileInfo.eventRandomJitterSeconds);
    segmentDurationSeconds = max( ...
        profileInfo.sampleTimeSeconds, ...
        profileInfo.eventHoldSeconds + profileInfo.eventDwellSeconds + dwellJitterSeconds);
    stopTimeSeconds = min(profileInfo.durationSeconds, currentTimeSeconds + segmentDurationSeconds);
    [targetVectorDegrees, axisLabel] = proposeLatencyVectorStepTargetLocal(profileInfo, currentTargetVectorDegrees, randomStream);

    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, ...
        eventIndex, ...
        axisLabel, ...
        max(abs(targetVectorDegrees)), ...
        currentTimeSeconds, ...
        stopTimeSeconds, ...
        dwellJitterSeconds);
    eventVectors(end + 1, :) = targetVectorDegrees; %#ok<AGROW>
    eventAxes(end + 1, 1) = string(axisLabel); %#ok<AGROW>
    currentTargetVectorDegrees = targetVectorDegrees;
end

profileEvents = profileEventRowsToTable(profileEventRows);
profileEvents = addLatencyVectorEventColumnsLocal(profileEvents, eventVectors, eventAxes, surfaceNames);
if isempty(profileEvents)
    return;
end

for eventRowIndex = 1:height(profileEvents)
    eventStartSeconds = profileEvents.StartTime_s(eventRowIndex);
    eventStopSeconds = profileEvents.StopTime_s(eventRowIndex);
    if eventStopSeconds <= eventStartSeconds
        continue;
    end

    if eventRowIndex < height(profileEvents)
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds < eventStopSeconds;
    else
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds <= eventStopSeconds;
    end
    commandVectorDegrees(eventMask, :) = repmat(eventVectors(eventRowIndex, :), sum(eventMask), 1);
end
end

function commandDegrees = evaluateLatencyStepTrainAtTime(profileEvents, profileTimeSeconds)
commandDegrees = 0.0;
if isempty(profileEvents)
    return;
end

matchedEventIndex = find( ...
    profileTimeSeconds >= profileEvents.StartTime_s & ...
    profileTimeSeconds < profileEvents.StopTime_s, ...
    1, ...
    "first");
if isempty(matchedEventIndex) && profileTimeSeconds == profileEvents.StopTime_s(end)
    matchedEventIndex = height(profileEvents);
end
if ~isempty(matchedEventIndex)
    commandDegrees = profileEvents.TargetDeflection_deg(matchedEventIndex);
end
end

function [profileEventRows, eventIndex, updatedTimeSeconds] = appendProfileEvent( ...
    profileEventRows, eventIndex, eventLabel, targetDeflectionDegrees, startTimeSeconds, stopTimeSeconds, dwellJitterSeconds)
updatedTimeSeconds = stopTimeSeconds;
if stopTimeSeconds <= startTimeSeconds
    return;
end

eventIndex = eventIndex + 1;
profileEventRows{end + 1, 1} = { ...
    double(eventIndex), ...
    string(eventLabel), ...
    double(targetDeflectionDegrees), ...
    double(startTimeSeconds), ...
    double(stopTimeSeconds), ...
    double(dwellJitterSeconds)};
end

function profileEvents = profileEventRowsToTable(profileEventRows)
if isempty(profileEventRows)
    profileEvents = table( ...
        zeros(0, 1), ...
        strings(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'EventIndex', ...
            'EventLabel', ...
            'TargetDeflection_deg', ...
            'StartTime_s', ...
            'StopTime_s', ...
            'DwellJitter_s'});
    return;
end

eventMatrix = vertcat(profileEventRows{:});
profileEvents = table( ...
    cell2mat(eventMatrix(:, 1)), ...
    string(eventMatrix(:, 2)), ...
    cell2mat(eventMatrix(:, 3)), ...
    cell2mat(eventMatrix(:, 4)), ...
    cell2mat(eventMatrix(:, 5)), ...
    cell2mat(eventMatrix(:, 6)), ...
    'VariableNames', { ...
        'EventIndex', ...
        'EventLabel', ...
        'TargetDeflection_deg', ...
        'StartTime_s', ...
        'StopTime_s', ...
        'DwellJitter_s'});
end

function [randomStream, effectiveRandomSeed] = createLocalRandomStream(requestedSeed)
maxSeed = double(intmax("uint32")) - 1.0;
if isfinite(requestedSeed)
    effectiveRandomSeed = floor(double(requestedSeed));
    if effectiveRandomSeed < 1.0 || effectiveRandomSeed > maxSeed
        effectiveRandomSeed = mod(effectiveRandomSeed - 1.0, maxSeed) + 1.0;
    end
else
    effectiveRandomSeed = mod(floor(posixtime(datetime("now")) .* 1e6), maxSeed) + 1.0;
end

randomStream = RandStream("mt19937ar", "Seed", effectiveRandomSeed);
end

function runLabel = formatSeedRunLabel(randomSeed)
if abs(randomSeed - round(randomSeed)) <= 10.0 .* eps(max(1.0, abs(randomSeed)))
    seedText = string(round(randomSeed));
else
    seedText = replace(compose("%.9g", double(randomSeed)), ".", "p");
end

runLabel = "Seed_" + seedText;
end

function dwellJitterSeconds = drawEventJitter(randomStream, randomJitterSeconds)
if randomJitterSeconds <= 0
    dwellJitterSeconds = 0.0;
    return;
end

dwellJitterSeconds = (2.0 .* rand(randomStream, 1, 1) - 1.0) .* randomJitterSeconds;
end

function [targetVectorDegrees, axisLabel] = proposeLatencyVectorStepTargetLocal(profileInfo, currentVectorDegrees, randomStream)
surfaceNames = reshape(string(profileInfo.vectorSurfaceNames), 1, []);
surfaceCount = numel(surfaceNames);
targetVectorDegrees = reshape(double(currentVectorDegrees), 1, []);
if numel(targetVectorDegrees) ~= surfaceCount
    targetVectorDegrees = zeros(1, surfaceCount);
end

maximumStepDegrees = max(2.0, abs(double(profileInfo.amplitudeDegrees)));
stepMagnitudes = maximumStepDegrees .* [0.25, 0.5, 1.0];
magnitudeWeights = [0.55, 0.30, 0.15];
selectedMagnitudeDegrees = stepMagnitudes(drawWeightedIndexLocal(randomStream, magnitudeWeights));
stepSign = chooseSignedStepLocal(randomStream);

axisModes = ["roll", "pitch", "yaw", "roll_yaw", "pitch_roll", "neutral"];
axisWeights = [0.40, 0.28, 0.14, 0.10, 0.05, 0.03];
axisLabel = axisModes(drawWeightedIndexLocal(randomStream, axisWeights));

rollValueDegrees = stepSign .* selectedMagnitudeDegrees;
pitchValueDegrees = chooseSignedStepLocal(randomStream) .* selectedMagnitudeDegrees;
yawValueDegrees = chooseSignedStepLocal(randomStream) .* max(1.5, 0.65 .* selectedMagnitudeDegrees);

switch axisLabel
    case "roll"
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_L", rollValueDegrees);
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_R", -rollValueDegrees);
    case "pitch"
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Elevator", pitchValueDegrees);
    case "yaw"
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Rudder", yawValueDegrees);
    case "roll_yaw"
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_L", rollValueDegrees);
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_R", -rollValueDegrees);
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Rudder", 0.35 .* rollValueDegrees);
    case "pitch_roll"
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Elevator", pitchValueDegrees);
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_L", 0.35 .* rollValueDegrees);
        targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, "Aileron_R", -0.35 .* rollValueDegrees);
    otherwise
        targetVectorDegrees(:) = 0.0;
        axisLabel = "neutral";
end
end

function profileEvents = addLatencyVectorEventColumnsLocal(profileEvents, eventVectors, eventAxes, surfaceNames)
profileEvents.AxisLabel = reshape(string(eventAxes), [], 1);
profileEvents.ActiveSurfaceCount = sum(abs(eventVectors) > 10 .* eps, 2);
profileEvents.VectorNorm_deg = sqrt(sum(eventVectors .^ 2, 2));

for surfaceIndex = 1:numel(surfaceNames)
    variableName = matlab.lang.makeValidName(surfaceNames(surfaceIndex) + "_Target_deg");
    profileEvents.(char(variableName)) = eventVectors(:, surfaceIndex);
end
end

function selectedIndex = drawWeightedIndexLocal(randomStream, weights)
weights = reshape(double(weights), 1, []);
weights(~isfinite(weights) | weights < 0) = 0;
if ~any(weights > 0)
    weights = ones(size(weights));
end

normalizedWeights = weights ./ sum(weights);
drawValue = rand(randomStream, 1, 1);
cumulativeWeights = cumsum(normalizedWeights);
selectedIndex = find(drawValue <= cumulativeWeights, 1, "first");
if isempty(selectedIndex)
    selectedIndex = numel(weights);
end
end

function signedValue = chooseSignedStepLocal(randomStream)
if rand(randomStream, 1, 1) < 0.5
    signedValue = -1.0;
else
    signedValue = 1.0;
end
end

function targetVectorDegrees = assignLatencyVectorSurfaceLocal(targetVectorDegrees, surfaceNames, surfaceName, surfaceValueDegrees)
surfaceIndex = find(surfaceNames == string(surfaceName), 1, "first");
if isempty(surfaceIndex)
    return;
end
targetVectorDegrees(surfaceIndex) = double(surfaceValueDegrees);
end

function uniqueEchoImportTable = deduplicateEchoImportTable(echoImportTable)
if isempty(echoImportTable)
    uniqueEchoImportTable = echoImportTable;
    return;
end

[~, firstOccurrenceIndex] = unique( ...
    table(echoImportTable.surface_name, echoImportTable.command_sequence), ...
    "rows", ...
    "stable");
uniqueEchoImportTable = echoImportTable(sort(firstOccurrenceIndex), :);
end

function commandKeys = buildSurfaceSequenceKeys(surfaceNames, commandSequences)
surfaceNames = reshape(string(surfaceNames), [], 1);
commandSequences = reshape(double(commandSequences), [], 1);
commandKeys = surfaceNames + "|" + compose("%.0f", commandSequences);
end

function [sampleCount, meanValue, stdValue, medianValue, p95Value, p99Value, maxValue] = unpackLatencyStats(stats)
sampleCount = stats.sampleCount;
meanValue = stats.meanValue;
stdValue = stats.stdValue;
medianValue = stats.medianValue;
p95Value = stats.p95Value;
p99Value = stats.p99Value;
maxValue = stats.maxValue;
end

function variableNames = buildSurfaceVariableNames(surfaceNames, suffix)
surfaceCount = numel(surfaceNames);
variableNames = strings(1, surfaceCount);

for surfaceIndex = 1:surfaceCount
    variableNames(surfaceIndex) = matlab.lang.makeValidName(surfaceNames(surfaceIndex) + "_" + suffix);
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
    error("Transmitter_Test:InvalidConfigType", "%s must be a text scalar.", fieldName);
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
    error("Transmitter_Test:InvalidConfigType", "%s must be a text scalar.", fieldName);
end
end

function value = getStringArrayField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if ischar(value)
    value = string({value});
elseif iscell(value)
    value = string(value);
elseif isstring(value)
    value = string(value);
elseif isempty(value)
    value = strings(0, 1);
else
    error("Transmitter_Test:InvalidConfigType", ...
        "%s must be text, a string array, or a cell array of character vectors.", ...
        fieldName);
end

value = reshape(value, [], 1);
end

function value = getLogicalField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"logical", "numeric"}, {"scalar"}, char(mfilename), char(fieldName));
value = logical(value);
end

function value = getPositiveScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "finite", "scalar", "positive"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getPositiveIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "positive"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNonnegativeIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "nonnegative"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNonnegativeScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "finite", "scalar", "nonnegative"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getOptionalScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = defaultValue;
end

validateInputAttributes(value, {"numeric"}, {"real", "scalar"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "finite", "scalar"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNumericColumnField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateInputAttributes(value, {"numeric"}, {"real", "column"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNumericVectorField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = [];
    return;
end

validateInputAttributes(value, {"numeric"}, {"real", "finite", "vector"}, char(mfilename), char(fieldName));
value = reshape(double(value), [], 1);
end

function [baseCommandDegrees, profileEvents] = buildIsolatedLatencyStepSchedule(profileInfo, scheduledTimeSeconds)
baseCommandDegrees = zeros(size(scheduledTimeSeconds));
profileTimeSeconds = scheduledTimeSeconds - profileInfo.commandStartSeconds;
profileEventRows = cell(0, 1);
currentTimeSeconds = 0.0;
eventIndex = 0;

isolatedNeutralSeconds = max(profileInfo.eventNeutralHoldSeconds, 2.0 .* profileInfo.sampleTimeSeconds);
isolatedStepSeconds = max(profileInfo.eventHoldSeconds, profileInfo.sampleTimeSeconds);
postStepNeutralSeconds = isolatedNeutralSeconds + profileInfo.eventDwellSeconds;

while currentTimeSeconds < profileInfo.durationSeconds
    positiveBlockDurationSeconds = ...
        isolatedNeutralSeconds + ...
        isolatedStepSeconds + ...
        postStepNeutralSeconds;
    if currentTimeSeconds + positiveBlockDurationSeconds > profileInfo.durationSeconds
        break;
    end

    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_before_positive", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + isolatedNeutralSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "positive_step", profileInfo.offsetDegrees + profileInfo.amplitudeDegrees, currentTimeSeconds, ...
        currentTimeSeconds + isolatedStepSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_after_positive", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + postStepNeutralSeconds, 0.0);

    negativeBlockDurationSeconds = isolatedStepSeconds + postStepNeutralSeconds;
    if currentTimeSeconds + negativeBlockDurationSeconds > profileInfo.durationSeconds
        break;
    end

    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "negative_step", profileInfo.offsetDegrees - profileInfo.amplitudeDegrees, currentTimeSeconds, ...
        currentTimeSeconds + isolatedStepSeconds, 0.0);
    [profileEventRows, eventIndex, currentTimeSeconds] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_after_negative", profileInfo.offsetDegrees, currentTimeSeconds, ...
        currentTimeSeconds + postStepNeutralSeconds, 0.0);
end

if currentTimeSeconds < profileInfo.durationSeconds
    [profileEventRows, eventIndex] = appendProfileEvent( ...
        profileEventRows, eventIndex, "neutral_tail", profileInfo.offsetDegrees, currentTimeSeconds, profileInfo.durationSeconds, 0.0);
end

profileEvents = profileEventRowsToTable(profileEventRows);
if isempty(profileEvents)
    return;
end

for eventRowIndex = 1:height(profileEvents)
    eventStartSeconds = profileEvents.StartTime_s(eventRowIndex);
    eventStopSeconds = profileEvents.StopTime_s(eventRowIndex);
    eventTargetDegrees = profileEvents.TargetDeflection_deg(eventRowIndex);
    if eventStopSeconds <= eventStartSeconds
        continue;
    end

    if eventRowIndex < height(profileEvents)
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds < eventStopSeconds;
    else
        eventMask = ...
            profileTimeSeconds >= eventStartSeconds & ...
            profileTimeSeconds <= eventStopSeconds;
    end
    baseCommandDegrees(eventMask) = eventTargetDegrees;
end
end

function validateInputAttributes(varargin)
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

function commandLine = buildNanoLoggerSetAllCommand(surfaceNames, commandSequenceNumbers, servoPositions, activeSurfaceMask)
activeIndices = find(activeSurfaceMask);
surfaceCount = numel(activeIndices);

if surfaceCount == 0
    error("Transmitter_Test:NoActiveSurfaces", "SET_ALL requires at least one active surface.");
end

sampleSequence = commandSequenceNumbers(activeIndices(1));
if ~all(commandSequenceNumbers(activeIndices) == sampleSequence)
    % This should never happen with the current sample-wise counter update,
    % but guard it explicitly.
    error("Transmitter_Test:InconsistentSetAllSequence", ...
        "All active surfaces must share the same sample sequence for SET_ALL.");
end

payloadParts = strings(1, 3 + 3 * surfaceCount);
payloadParts(1) = "SET_ALL";
payloadParts(2) = string(uint32(sampleSequence));
payloadParts(3) = string(surfaceCount);

writeIndex = 4;
for k = 1:surfaceCount
    surfaceIndex = activeIndices(k);
    payloadParts(writeIndex) = surfaceNames(surfaceIndex);
    payloadParts(writeIndex + 1) = string(uint32(commandSequenceNumbers(surfaceIndex)));
    payloadParts(writeIndex + 2) = compose("%.6f", servoPositions(surfaceIndex));
    writeIndex = writeIndex + 3;
end

commandLine = join(payloadParts, ",");
end

function payloadBytes = buildNanoLoggerBinaryVectorCommand(commandSequenceNumbers, servoPositions, activeSurfaceMask)
activeIndices = find(activeSurfaceMask);
surfaceCount = numel(activeSurfaceMask);

if isempty(activeIndices)
    error("Transmitter_Test:NoActiveSurfaces", "Binary vector command requires at least one active surface.");
end

sampleSequence = commandSequenceNumbers(activeIndices(1));
if ~all(commandSequenceNumbers(activeIndices) == sampleSequence)
    error("Transmitter_Test:InconsistentBinaryVectorSequence", ...
        "All active surfaces must share the same sample sequence.");
end

surfaceMask = uint8(0);
for surfaceIndex = activeIndices(:).'
    surfaceMask = bitor(surfaceMask, bitshift(uint8(1), surfaceIndex - 1));
end

positionCodes = uint16(round(min(max(reshape(servoPositions, 1, []), 0.0), 1.0) .* 65535.0));

payloadBytes = zeros(1, 7 + 2 * surfaceCount, "uint8");
payloadBytes(1) = uint8('V');
payloadBytes(2) = uint8(surfaceCount);
payloadBytes(3) = surfaceMask;
payloadBytes(4:7) = encodeUint32LittleEndian(uint32(sampleSequence));

writeIndex = 8;
for surfaceIndex = 1:surfaceCount
    payloadBytes(writeIndex:writeIndex + 1) = ...
        encodeUint16LittleEndian(positionCodes(surfaceIndex));
    writeIndex = writeIndex + 2;
end
end

function encodedBytes = encodeUint16LittleEndian(value)
value = uint16(value);
encodedBytes = uint8([ ...
    bitand(value, uint16(255)), ...
    bitshift(value, -8)]);
end

function encodedBytes = encodeUint32LittleEndian(value)
value = uint32(value);
encodedBytes = uint8([ ...
    bitand(value, uint32(255)), ...
    bitand(bitshift(value, -8), uint32(255)), ...
    bitand(bitshift(value, -16), uint32(255)), ...
    bitshift(value, -24)]);
end

function config = normalizeTransmitterConfig(config)
rootFolder = fileparts(mfilename("fullpath"));
defaultSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];
defaultTrainerChannelMap = [defaultSurfaceNames; ""; ""; ""; ""];
defaultLogicAnalyzerChannels = [0 1 2 3 4 5];
defaultLogicAnalyzerNames = ["RX_CH1"; "RX_CH2"; "RX_CH3"; "RX_CH4"; "REF_D2"; "TRAINER_PPM_D3"];
commandProfileConfig = getFieldOrDefault(config, "commandProfile", struct());
transportConfig = getFieldOrDefault(config, "arduinoTransport", struct());
trainerPpmConfig = getFieldOrDefault(config, "trainerPpm", struct());
referenceStrobeConfig = getFieldOrDefault(config, "referenceStrobe", struct());
logicAnalyzerConfig = getFieldOrDefault(config, "logicAnalyzer", struct());
matchingConfig = getFieldOrDefault(config, "matching", struct());
analysisConfig = getFieldOrDefault(config, "analysis", struct());
debugConfig = getFieldOrDefault(config, "debug", struct());

config.arduinoBoard = getTextScalarField(config, "arduinoBoard", "Uno");
config.surfaceNames = getStringArrayField(config, "surfaceNames", defaultSurfaceNames);
config.servoNeutralPositions = getNumericColumnField( ...
    config, ...
    "servoNeutralPositions", ...
    0.5 .* ones(numel(config.surfaceNames), 1));
config.servoUnitsPerDegree = getNumericColumnField( ...
    config, ...
    "servoUnitsPerDegree", ...
    (1 ./ 180) .* ones(numel(config.surfaceNames), 1));
config.servoMinimumPositions = getNumericColumnField( ...
    config, ...
    "servoMinimumPositions", ...
    zeros(numel(config.surfaceNames), 1));
config.servoMaximumPositions = getNumericColumnField( ...
    config, ...
    "servoMaximumPositions", ...
    ones(numel(config.surfaceNames), 1));
config.commandDeflectionScales = getNumericColumnField( ...
    config, ...
    "commandDeflectionScales", ...
    ones(numel(config.surfaceNames), 1));
config.commandDeflectionOffsetsDegrees = getNumericColumnField( ...
    config, ...
    "commandDeflectionOffsetsDegrees", ...
    zeros(numel(config.surfaceNames), 1));
config.commandMode = getTextScalarField(config, "commandMode", "all");
config.singleSurfaceName = getTextScalarField(config, "singleSurfaceName", "Aileron_L");
config.commandProfile = normalizeCommandProfile(commandProfileConfig);
config.neutralSettleSeconds = getNonnegativeScalarField(config, "neutralSettleSeconds", 1.0);
config.returnToNeutralOnExit = getLogicalField(config, "returnToNeutralOnExit", true);
config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "D_Transmitter_Test"));
config.runLabel = getTextScalarField(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    if isfinite(config.commandProfile.randomSeed)
        config.runLabel = formatSeedRunLabel(config.commandProfile.randomSeed) + "_Transmitter";
    else
        config.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_Transmitter";
    end
end

config.arduinoTransport = struct( ...
    "mode", "uno_transmitter_serial", ...
    "serialPort", getTextScalarField(transportConfig, "serialPort", "COM10"), ...
    "baudRate", getPositiveIntegerField(transportConfig, "baudRate", 1000000), ...
    "commandEncoding", canonicalizeNanoLoggerCommandEncoding( ...
        getTextScalarField(transportConfig, "commandEncoding", "binary_vector")), ...
    "rxTimestampSource", canonicalizeRxTimestampSource( ...
        getTextScalarField(transportConfig, "rxTimestampSource", "board_rx_us")), ...
    "serialResetSeconds", getNonnegativeScalarField(transportConfig, "serialResetSeconds", 4.0), ...
    "probeTimeoutSeconds", getPositiveScalarField(transportConfig, "probeTimeoutSeconds", 6.0), ...
    "helloRetrySeconds", getPositiveScalarField(transportConfig, "helloRetrySeconds", 0.25), ...
    "linePollPauseSeconds", getPositiveScalarField(transportConfig, "linePollPauseSeconds", 0.001), ...
    "lineIdleTimeoutSeconds", getPositiveScalarField(transportConfig, "lineIdleTimeoutSeconds", 0.25), ...
    "syncReplyTimeoutSeconds", getPositiveScalarField(transportConfig, "syncReplyTimeoutSeconds", 0.15), ...
    "syncCountBeforeRun", getNonnegativeIntegerField(transportConfig, "syncCountBeforeRun", 10), ...
    "syncCountAfterRun", getNonnegativeIntegerField(transportConfig, "syncCountAfterRun", 10), ...
    "syncPauseSeconds", getNonnegativeScalarField(transportConfig, "syncPauseSeconds", 0.05), ...
    "postRunSettleSeconds", getNonnegativeScalarField(transportConfig, "postRunSettleSeconds", 0.25), ...
    "clearLogsBeforeRun", getLogicalField(transportConfig, "clearLogsBeforeRun", true), ...
    "loggerOutputFolder", getOptionalTextScalarField( ...
        transportConfig, ...
        "loggerOutputFolder", ...
        fullfile(config.outputFolder, config.runLabel + "_TransmitterLogger")), ...
    "captureSucceeded", false, ...
    "captureMessage", "", ...
    "captureRowCount", 0);

config.trainerPpm = struct( ...
    "frameLengthUs", getPositiveIntegerField(trainerPpmConfig, "frameLengthUs", 20000), ...
    "markWidthUs", getPositiveIntegerField(trainerPpmConfig, "markWidthUs", 300), ...
    "channelCount", getPositiveIntegerField(trainerPpmConfig, "channelCount", 8), ...
    "idleHigh", getLogicalField(trainerPpmConfig, "idleHigh", false), ...
    "latchMode", getTextScalarField(trainerPpmConfig, "latchMode", "frame_boundary"), ...
    "commandTimeoutSeconds", getPositiveScalarField(trainerPpmConfig, "commandTimeoutSeconds", 0.25), ...
    "outputPin", getTextScalarField(trainerPpmConfig, "outputPin", "D3"), ...
    "referencePin", getTextScalarField(trainerPpmConfig, "referencePin", "D2"), ...
    "minimumPulseUs", getPositiveIntegerField(trainerPpmConfig, "minimumPulseUs", 1000), ...
    "maximumPulseUs", getPositiveIntegerField(trainerPpmConfig, "maximumPulseUs", 2000), ...
    "neutralPulseUs", getPositiveIntegerField(trainerPpmConfig, "neutralPulseUs", 1500), ...
    "channelSurfaceMap", getStringArrayField(trainerPpmConfig, "channelSurfaceMap", defaultTrainerChannelMap));

if ~isfield(commandProfileConfig, "type")
    % Keep parity with Arduino_Test default profile so Seed_N references are
    % directly comparable between wired and transmitter paths.
    config.commandProfile.type = "latency_step_train";
end
% Preserve 20 ms command cadence by default to match trainer frame timing.
if ~isfield(commandProfileConfig, "randomSeed")
    config.commandProfile.randomSeed = 5;
end
if ~isfield(commandProfileConfig, "sampleTimeSeconds")
    % Standard operating path: issue one MATLAB command per trainer frame.
    config.commandProfile.sampleTimeSeconds = double(config.trainerPpm.frameLengthUs) ./ 1e6;
end

if config.commandProfile.type == "latency_step_train"
    if ~isfield(commandProfileConfig, "eventHoldSeconds")
        config.commandProfile.eventHoldSeconds = 0.20;
    end
    if ~isfield(commandProfileConfig, "eventNeutralHoldSeconds")
        config.commandProfile.eventNeutralHoldSeconds = 0.10;
    end
    if ~isfield(commandProfileConfig, "eventDwellSeconds")
        config.commandProfile.eventDwellSeconds = 0.60;
    end
    if ~isfield(commandProfileConfig, "eventRandomJitterSeconds")
        config.commandProfile.eventRandomJitterSeconds = 0.05;
    end
elseif config.commandProfile.type == "latency_vector_step_train"
    if ~isfield(commandProfileConfig, "amplitudeDegrees")
        config.commandProfile.amplitudeDegrees = 12.0;
    end
    if ~isfield(commandProfileConfig, "eventHoldSeconds")
        config.commandProfile.eventHoldSeconds = max(0.04, 2.0 .* config.commandProfile.sampleTimeSeconds);
    end
    if ~isfield(commandProfileConfig, "eventNeutralHoldSeconds")
        config.commandProfile.eventNeutralHoldSeconds = 0.0;
    end
    if ~isfield(commandProfileConfig, "eventDwellSeconds")
        config.commandProfile.eventDwellSeconds = max(0.02, config.commandProfile.sampleTimeSeconds);
    end
    if ~isfield(commandProfileConfig, "eventRandomJitterSeconds")
        config.commandProfile.eventRandomJitterSeconds = 0.02;
    end
elseif config.commandProfile.type == "latency_isolated_step"
    if ~isfield(commandProfileConfig, "eventHoldSeconds")
        config.commandProfile.eventHoldSeconds = max(0.10, 2.0 .* config.commandProfile.sampleTimeSeconds);
    end
    if ~isfield(commandProfileConfig, "eventNeutralHoldSeconds")
        config.commandProfile.eventNeutralHoldSeconds = max(0.25, 4.0 .* config.commandProfile.sampleTimeSeconds);
    end
    if ~isfield(commandProfileConfig, "eventDwellSeconds")
        config.commandProfile.eventDwellSeconds = 0.0;
    end
    if ~isfield(commandProfileConfig, "eventRandomJitterSeconds")
        config.commandProfile.eventRandomJitterSeconds = 0.0;
    end
end

config.referenceStrobe = struct( ...
    "enabled", getLogicalField(referenceStrobeConfig, "enabled", true), ...
    "mode", canonicalizeReferenceStrobeMode( ...
        getTextScalarField(referenceStrobeConfig, "mode", "commit_only")), ...
    "pulseWidthUs", getPositiveIntegerField(referenceStrobeConfig, "pulseWidthUs", 50));

logicAnalyzerEnabled = getLogicalField(logicAnalyzerConfig, "enabled", true);
logicAnalyzerModeDefault = "import_only";
if logicAnalyzerEnabled
    logicAnalyzerModeDefault = "sigrok_auto";
end

logicAnalyzerMode = canonicalizeLogicAnalyzerMode( ...
    getTextScalarField(logicAnalyzerConfig, "mode", logicAnalyzerModeDefault));
automationModeDefault = "import_only";
if logicAnalyzerMode == "sigrok_auto"
    automationModeDefault = "capture_decode_import";
end

roleMapConfig = getFieldOrDefault(logicAnalyzerConfig, "channelRoleMap", struct());
channelRoleMap = struct( ...
    "receiver", getNumericVectorField(roleMapConfig, "receiver", [0 1 2 3]), ...
    "reference", getNonnegativeIntegerField(roleMapConfig, "reference", 4), ...
    "trainerPpm", getNonnegativeIntegerField(roleMapConfig, "trainerPpm", 5));
logicAnalyzerArtifactPrefix = fullfile( ...
    config.arduinoTransport.loggerOutputFolder, ...
    config.runLabel + "_sigrok");

config.logicAnalyzer = struct( ...
    "enabled", logicAnalyzerEnabled, ...
    "mode", logicAnalyzerMode, ...
    "automationMode", getTextScalarField(logicAnalyzerConfig, "automationMode", automationModeDefault), ...
    "sigrokCliPath", getOptionalTextScalarField(logicAnalyzerConfig, "sigrokCliPath", "C:\Program Files\sigrok\sigrok-cli\sigrok-cli.exe"), ...
    "deviceDriver", getTextScalarField(logicAnalyzerConfig, "deviceDriver", "fx2lafw"), ...
    "deviceId", getOptionalTextScalarField(logicAnalyzerConfig, "deviceId", ""), ...
    "sampleRateHz", getPositiveIntegerField(logicAnalyzerConfig, "sampleRateHz", 4000000), ...
    "maximumCaptureSamples", getNonnegativeScalarField(logicAnalyzerConfig, "maximumCaptureSamples", 0), ...
    "captureDurationSeconds", getOptionalScalarField(logicAnalyzerConfig, "captureDurationSeconds", NaN), ...
    "requestedCaptureDurationSeconds", NaN, ...
    "captureStartLeadSeconds", getNonnegativeScalarField(logicAnalyzerConfig, "captureStartLeadSeconds", 0.25), ...
    "captureStopLagSeconds", getNonnegativeScalarField(logicAnalyzerConfig, "captureStopLagSeconds", 0.50), ...
    "channels", getNumericVectorField(logicAnalyzerConfig, "channels", defaultLogicAnalyzerChannels), ...
    "channelNames", getStringArrayField(logicAnalyzerConfig, "channelNames", defaultLogicAnalyzerNames), ...
    "channelRoleMap", channelRoleMap, ...
    "outputFormat", getTextScalarField(logicAnalyzerConfig, "outputFormat", "logic_state_csv"), ...
    "rawCapturePath", getOptionalTextScalarField(logicAnalyzerConfig, "rawCapturePath", logicAnalyzerArtifactPrefix + "_raw.sr"), ...
    "logicStateExportPath", getOptionalTextScalarField(logicAnalyzerConfig, "logicStateExportPath", logicAnalyzerArtifactPrefix + "_logic_state.csv"), ...
    "maximumLogicStateExportBytes", getPositiveIntegerField(logicAnalyzerConfig, "maximumLogicStateExportBytes", 250000000), ...
    "decodedReferencePath", getOptionalTextScalarField(logicAnalyzerConfig, "decodedReferencePath", logicAnalyzerArtifactPrefix + "_reference.csv"), ...
    "decodedTrainerPpmPath", getOptionalTextScalarField(logicAnalyzerConfig, "decodedTrainerPpmPath", logicAnalyzerArtifactPrefix + "_trainer.csv"), ...
    "decodedReceiverPath", getOptionalTextScalarField(logicAnalyzerConfig, "decodedReceiverPath", logicAnalyzerArtifactPrefix + "_receiver.csv"), ...
    "storeStdoutPath", getOptionalTextScalarField(logicAnalyzerConfig, "storeStdoutPath", logicAnalyzerArtifactPrefix + "_stdout.txt"), ...
    "storeStderrPath", getOptionalTextScalarField(logicAnalyzerConfig, "storeStderrPath", logicAnalyzerArtifactPrefix + "_stderr.txt"), ...
    "deleteRawAfterDecode", getLogicalField(logicAnalyzerConfig, "deleteRawAfterDecode", false), ...
    "useTrigger", getLogicalField(logicAnalyzerConfig, "useTrigger", false), ...
    "triggerChannel", getOptionalTextScalarField(logicAnalyzerConfig, "triggerChannel", ""), ...
    "triggerType", getOptionalTextScalarField(logicAnalyzerConfig, "triggerType", ""), ...
    "referenceCapturePath", getOptionalTextScalarField(logicAnalyzerConfig, "referenceCapturePath", logicAnalyzerArtifactPrefix + "_reference.csv"), ...
    "trainerPpmCapturePath", getOptionalTextScalarField(logicAnalyzerConfig, "trainerPpmCapturePath", logicAnalyzerArtifactPrefix + "_trainer.csv"), ...
    "receiverCapturePath", getOptionalTextScalarField(logicAnalyzerConfig, "receiverCapturePath", logicAnalyzerArtifactPrefix + "_receiver.csv"));

config.matching = struct( ...
    "mode", getTextScalarField(matchingConfig, "mode", "shared_clock_minimal"), ...
    "anchorPriority", getTextScalarField(matchingConfig, "anchorPriority", "D4_then_D5"), ...
    "referenceDebounceUs", getPositiveIntegerField(matchingConfig, "referenceDebounceUs", 100), ...
    "enableDownstreamMatching", getLogicalField(matchingConfig, "enableDownstreamMatching", true), ...
    "ppmChangeThresholdUs", getPositiveIntegerField(matchingConfig, "ppmChangeThresholdUs", 4), ...
    "receiverChangeThresholdUs", getPositiveIntegerField(matchingConfig, "receiverChangeThresholdUs", 8), ...
    "transitionLeadSeconds", getNonnegativeScalarField(matchingConfig, "transitionLeadSeconds", 0.005), ...
    "stateToleranceUs", getPositiveIntegerField(matchingConfig, "stateToleranceUs", 80), ...
    "stableStatePulseCount", getPositiveIntegerField(matchingConfig, "stableStatePulseCount", 2), ...
    "transitionTargetToleranceUs", getPositiveIntegerField(matchingConfig, "transitionTargetToleranceUs", 80), ...
    "transitionPreviousToleranceUs", getPositiveIntegerField(matchingConfig, "transitionPreviousToleranceUs", 25), ...
    "useEstimatedLagForSearch", getLogicalField(matchingConfig, "useEstimatedLagForSearch", false), ...
    "requireTrainerBeforeReceiver", getLogicalField(matchingConfig, "requireTrainerBeforeReceiver", false), ...
    "composeReceiverLatencyFromChain", getLogicalField(matchingConfig, "composeReceiverLatencyFromChain", true), ...
    "downstreamFrameWindowSeconds", getPositiveScalarField( ...
        matchingConfig, ...
        "downstreamFrameWindowSeconds", ...
        max(0.02, 1.25 .* double(config.trainerPpm.frameLengthUs) ./ 1e6)), ...
    "referenceAssociationWindowSeconds", getPositiveScalarField( ...
        matchingConfig, ...
        "referenceAssociationWindowSeconds", ...
        max(0.02, 1.25 .* double(config.trainerPpm.frameLengthUs) ./ 1e6)), ...
    "maxCommitAssociationSeconds", getPositiveScalarField( ...
        matchingConfig, ...
        "maxCommitAssociationSeconds", ...
        0.25), ...
    "maxResponseWindowSeconds", getPositiveScalarField(matchingConfig, "maxResponseWindowSeconds", 0.12));
config.matching.truthSubsetMode = canonicalizeTruthSubsetMode( ...
    getTextScalarField(matchingConfig, "truthSubsetMode", "off"));
config.matching.mode = lower(string(config.matching.mode));
config.matching.anchorPriority = string(config.matching.anchorPriority);
config.analysis = struct( ...
    "composeReceiverLatencyFromChain", getLogicalField( ...
        analysisConfig, ...
        "composeReceiverLatencyFromChain", ...
        config.matching.composeReceiverLatencyFromChain));
config.debug = struct( ...
    "strictDownstreamMatching", getLogicalField(debugConfig, "strictDownstreamMatching", false));

surfaceCount = numel(config.surfaceNames);
if surfaceCount ~= 4
    error("Transmitter_Test:InvalidSurfaceCount", "Transmitter_Test requires exactly four logical surfaces.");
end
if config.trainerPpm.channelCount ~= 8
    error("Transmitter_Test:InvalidPpmChannelCount", "trainerPpm.channelCount must be 8.");
end
if numel(config.trainerPpm.channelSurfaceMap) ~= config.trainerPpm.channelCount
    error("Transmitter_Test:InvalidChannelSurfaceMap", "trainerPpm.channelSurfaceMap must contain one entry per PPM channel.");
end
if numel(config.logicAnalyzer.channels) ~= numel(config.logicAnalyzer.channelNames)
    error("Transmitter_Test:InvalidLogicAnalyzerChannels", "logicAnalyzer.channels and logicAnalyzer.channelNames must have matching lengths.");
end
if numel(config.logicAnalyzer.channelRoleMap.receiver) ~= surfaceCount
    error("Transmitter_Test:InvalidLogicAnalyzerReceiverRoleMap", "logicAnalyzer.channelRoleMap.receiver must contain one channel per logical surface.");
end
validateInputAttributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoNeutralPositions');
validateInputAttributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoUnitsPerDegree');
validateInputAttributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMinimumPositions');
validateInputAttributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMaximumPositions');
validateInputAttributes(config.commandDeflectionScales, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionScales');
validateInputAttributes(config.commandDeflectionOffsetsDegrees, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionOffsetsDegrees');

if any(abs(config.servoUnitsPerDegree) <= eps)
    error("Transmitter_Test:InvalidServoScale", "servoUnitsPerDegree must be nonzero for every surface.");
end

if any(config.servoMinimumPositions > config.servoMaximumPositions)
    error("Transmitter_Test:InvalidServoLimits", "servoMinimumPositions must not exceed servoMaximumPositions.");
end

if ~any(config.commandMode == ["single", "all"])
    error("Transmitter_Test:InvalidCommandMode", "commandMode must be 'single' or 'all'.");
end
if config.commandMode == "single" && ~any(config.surfaceNames == config.singleSurfaceName)
    error("Transmitter_Test:InvalidSingleSurface", "singleSurfaceName must match one configured surface.");
end

activeSurfaceMask = false(surfaceCount, 1);
if config.commandMode == "single"
    activeSurfaceMask(config.surfaceNames == config.singleSurfaceName) = true;
else
    activeSurfaceMask(:) = true;
end

config.activeSurfaceMask = activeSurfaceMask;
config.activeSurfaceNames = config.surfaceNames(activeSurfaceMask);
config.surfaceSetup = buildTransmitterSurfaceSetupTable(config);
end

function logicAnalyzerMode = canonicalizeLogicAnalyzerMode(logicAnalyzerMode)
logicAnalyzerMode = lower(string(logicAnalyzerMode));
validModes = ["import_only", "sigrok_auto"];
if ~any(logicAnalyzerMode == validModes)
    error( ...
        "Transmitter_Test:InvalidLogicAnalyzerMode", ...
        "logicAnalyzer.mode must be one of: %s.", ...
        char(join(validModes, ", ")));
end
end

function isEnabled = isSigrokAutoMode(config)
isEnabled = ...
    isfield(config, "logicAnalyzer") && ...
    config.logicAnalyzer.enabled && ...
    config.logicAnalyzer.mode == "sigrok_auto";
end

function sigrokSession = createEmptySigrokSession()
sigrokSession = struct( ...
    "process", [], ...
    "processId", NaN, ...
    "isActive", false, ...
    "rawCaptureCommand", "", ...
    "logicStateExportCommand", "", ...
    "rawCapturePath", "", ...
    "logicStateExportPath", "", ...
    "decodedReferencePath", "", ...
    "decodedTrainerPpmPath", "", ...
    "decodedReceiverPath", "", ...
    "stdoutPath", "", ...
    "stderrPath", "", ...
    "sampleRateHz", NaN, ...
    "channelRoleMap", struct());
end

function captureDurationSeconds = computeSigrokCaptureDurationSeconds(config, runPlan)
if isfinite(config.logicAnalyzer.captureDurationSeconds)
    captureDurationSeconds = config.logicAnalyzer.captureDurationSeconds;
    return;
end

statusAllowanceSeconds = 0.05;
drainAllowanceSeconds = 2.0 + config.arduinoTransport.lineIdleTimeoutSeconds;
syncAfterRunSeconds = ...
    double(config.arduinoTransport.syncCountAfterRun) .* ...
    config.arduinoTransport.syncPauseSeconds;
postRunSerialCollectionSeconds = ...
    config.arduinoTransport.postRunSettleSeconds + ...
    syncAfterRunSeconds + ...
    statusAllowanceSeconds + ...
    drainAllowanceSeconds;
captureDurationSeconds = ...
    config.logicAnalyzer.captureStartLeadSeconds + ...
    runPlan.scheduledDurationSeconds + ...
    postRunSerialCollectionSeconds + ...
    config.logicAnalyzer.captureStopLagSeconds + ...
    0.25;
end

function referenceStrobeMode = canonicalizeReferenceStrobeMode(referenceStrobeMode)
referenceStrobeMode = lower(string(referenceStrobeMode));
validModes = ["commit_only", "every_frame"];
if ~any(referenceStrobeMode == validModes)
    error( ...
        "Transmitter_Test:InvalidReferenceStrobeMode", ...
        "referenceStrobe.mode must be one of: %s.", ...
        char(join(validModes, ", ")));
end
end

function runData = initializeTransmitterRunData(config)
runData = struct( ...
    "config", config, ...
    "connectionInfo", struct("arduino", struct()), ...
    "runInfo", struct( ...
        "status", "initialized", ...
        "reason", "", ...
        "operatingMode", config.arduinoTransport.mode, ...
        "startTime", NaT, ...
        "stopTime", NaT, ...
        "sampleCount", 0, ...
        "scheduledDurationSeconds", NaN), ...
    "surfaceSetup", config.surfaceSetup, ...
    "logs", struct(), ...
    "surfaceSummary", table(), ...
    "artifacts", struct( ...
        "matFilePath", "", ...
        "workbookPath", "", ...
        "loggerFolderPath", "", ...
        "rawCapturePath", "", ...
        "logicStateExportPath", "", ...
        "decodedReferencePath", "", ...
        "decodedTrainerPpmPath", "", ...
        "decodedReceiverPath", "", ...
        "sigrokStdoutPath", "", ...
        "sigrokStderrPath", ""));
end

function [commandInterface, connectionInfo] = connectToTransmitter(config)
commandInterface = struct();
connectionInfo = struct( ...
    "boardName", config.arduinoBoard, ...
    "serialPort", config.arduinoTransport.serialPort, ...
    "baudRate", config.arduinoTransport.baudRate, ...
    "isConnected", false, ...
    "connectElapsedSeconds", NaN, ...
    "connectionMessage", "", ...
    "loggerGreeting", "", ...
    "loggerFirmwareVersion", "", ...
    "transportResolvedMode", config.arduinoTransport.mode, ...
    "transportDiagnostics", strings(0, 1));

connectTimer = tic;
serialObject = [];
try
    serialObject = serialport( ...
        char(config.arduinoTransport.serialPort), ...
        config.arduinoTransport.baudRate, ...
        "Timeout", ...
        config.arduinoTransport.probeTimeoutSeconds);
    configureTerminator(serialObject, "LF");
    pause(config.arduinoTransport.serialResetSeconds);
    flush(serialObject);

    [greeting, firmwareVersion] = probeTransmitterConnection(serialObject, config);
    commandInterface = struct( ...
        "transportMode", "uno_transmitter_serial", ...
        "connection", serialObject, ...
        "commandEncoding", config.arduinoTransport.commandEncoding, ...
        "surfaceNames", config.surfaceNames);

    connectionInfo.isConnected = true;
    connectionInfo.loggerGreeting = greeting;
    connectionInfo.loggerFirmwareVersion = firmwareVersion;
    connectionInfo.connectionMessage = "Connected to Uno transmitter firmware over serial and received HELLO_EVENT.";
catch connectionException
    connectionInfo.connectionMessage = string(connectionException.message);
    connectionInfo.transportDiagnostics = string(connectionException.message);
    if ~isempty(serialObject)
        try
            clear serialObject
        catch
        end
    end
end

connectionInfo.connectElapsedSeconds = toc(connectTimer);
end

function [greeting, firmwareVersion] = probeTransmitterConnection(serialObject, config)
greeting = "";
firmwareVersion = "";
receiveBuffer = "";
probeTimer = tic;
lastHelloSeconds = -inf;

while toc(probeTimer) < config.arduinoTransport.probeTimeoutSeconds
    if toc(probeTimer) - lastHelloSeconds >= config.arduinoTransport.helloRetrySeconds
        writeline(serialObject, "HELLO");
        lastHelloSeconds = toc(probeTimer);
    end

    [receivedLines, receiveBuffer] = readTransmitterLines(serialObject, receiveBuffer);
    if isempty(receivedLines)
        pause(min(0.02, config.arduinoTransport.helloRetrySeconds));
        continue;
    end

    for lineIndex = 1:numel(receivedLines)
        telemetryParts = split(receivedLines(lineIndex), ",");
        if isempty(telemetryParts) || telemetryParts(1) ~= "HELLO_EVENT"
            continue;
        end

        greeting = receivedLines(lineIndex);
        if numel(telemetryParts) >= 2
            firmwareVersion = telemetryParts(2);
        else
            firmwareVersion = "unknown";
        end
        return;
    end
end

error("Transmitter_Test:MissingHelloEvent", "No HELLO_EVENT was received from the transmitter firmware.");
end

function printTransmitterConnectionStatus(runData)
fprintf("\nTransmitter_Test connection summary\n");
fprintf("  Serial port: %s\n", char(runData.config.arduinoTransport.serialPort));
fprintf("  Baud rate: %u\n", uint32(runData.config.arduinoTransport.baudRate));
fprintf("  Arduino: %s\n", char(getStatusText(runData.connectionInfo.arduino)));
if isfield(runData.connectionInfo.arduino, "loggerFirmwareVersion") && ...
        strlength(string(runData.connectionInfo.arduino.loggerFirmwareVersion)) > 0
    fprintf("  Firmware: %s\n", char(runData.connectionInfo.arduino.loggerFirmwareVersion));
end
fprintf("  Active surfaces: %s\n\n", char(join(runData.config.activeSurfaceNames, ", ")));
end

function cleanupTransmitterResources(commandInterface, config, sigrokSession)
if nargin >= 3
    cleanupSigrokSession(sigrokSession);
end

if isempty(fieldnames(commandInterface))
    return;
end

if config.returnToNeutralOnExit
    try
        moveTransmitterToNeutral(commandInterface);
    catch
    end
end

try
    delete(commandInterface.connection);
catch
end
end

function moveTransmitterToNeutral(commandInterface)
if isempty(fieldnames(commandInterface))
    return;
end

sendTransmitterControlBurst(commandInterface.connection, "SET_NEUTRAL", 2, 0.02);
end

function sendTransmitterControlBurst(serialObject, payloadText, repeatCount, pauseSeconds)
for repeatIndex = 1:repeatCount
    writeline(serialObject, char(payloadText));
    if repeatIndex < repeatCount && pauseSeconds > 0
        pause(pauseSeconds);
    end
end
end

function runPlan = buildTransmitterRunPlan(config)
[scheduledTimeSeconds, profileInfo] = buildCommandSchedule(config.commandProfile);
runPlan = struct( ...
    "scheduledTimeSeconds", scheduledTimeSeconds, ...
    "profileInfo", profileInfo, ...
    "sampleCount", numel(scheduledTimeSeconds), ...
    "scheduledDurationSeconds", profileInfo.totalDurationSeconds);
end

function [storage, runInfo, config] = executeTransmitterRun(commandInterface, config, runPlan)
surfaceCount = numel(config.surfaceNames);
sampleCount = runPlan.sampleCount;
scheduledTimeSeconds = runPlan.scheduledTimeSeconds;
profileInfo = runPlan.profileInfo;

storage = initializeTransmitterStorage(sampleCount, surfaceCount);
storage.scheduledTimeSeconds = scheduledTimeSeconds;
storage.profileInfo = profileInfo;

runInfo = struct( ...
    "status", "running", ...
    "reason", "", ...
    "operatingMode", config.arduinoTransport.mode, ...
    "startTime", datetime("now"), ...
    "stopTime", NaT, ...
    "sampleCount", 0, ...
    "scheduledDurationSeconds", runPlan.scheduledDurationSeconds);

serialObject = commandInterface.connection;
loggerSession = startTransmitterLoggerSession(serialObject, config, sampleCount, nnz(config.activeSurfaceMask));
testStart = loggerSession.hostTimer;
surfaceCommandCounts = zeros(1, surfaceCount);

for sampleIndex = 1:sampleCount
    scheduledSampleTimeSeconds = scheduledTimeSeconds(sampleIndex);
    loggerSession = waitForScheduledTimeAndDrainTransmitter( ...
        testStart, ...
        loggerSession.testStartOffsetSeconds + scheduledSampleTimeSeconds, ...
        serialObject, ...
        loggerSession, ...
        config.arduinoTransport.linePollPauseSeconds);

    baseCommandDegrees = evaluateBaseCommandDegrees(profileInfo, scheduledSampleTimeSeconds, sampleIndex);
    directCommandVectorDegrees = evaluateCommandVectorDegrees(profileInfo, config.surfaceNames, sampleIndex);
    desiredDeflectionsDegrees = zeros(1, surfaceCount);
    if any(isfinite(directCommandVectorDegrees))
        desiredDeflectionsDegrees(config.activeSurfaceMask.') = ...
            config.commandDeflectionScales(config.activeSurfaceMask).' .* directCommandVectorDegrees(config.activeSurfaceMask.') + ...
            config.commandDeflectionOffsetsDegrees(config.activeSurfaceMask).';
    else
        desiredDeflectionsDegrees(config.activeSurfaceMask.') = ...
            config.commandDeflectionScales(config.activeSurfaceMask).' .* baseCommandDegrees + ...
            config.commandDeflectionOffsetsDegrees(config.activeSurfaceMask).';
    end

    commandedServoPositions = ...
        config.servoNeutralPositions.' + ...
        desiredDeflectionsDegrees .* config.servoUnitsPerDegree.';
    commandedServoPositionsClipped = min( ...
        max(commandedServoPositions, config.servoMinimumPositions.'), ...
        config.servoMaximumPositions.');
    saturatedMask = abs(commandedServoPositionsClipped - commandedServoPositions) > 10 .* eps;

    nextCommandSequenceNumbers = nan(1, surfaceCount);
    nextCommandSequenceNumbers(config.activeSurfaceMask.') = ...
        surfaceCommandCounts(config.activeSurfaceMask.') + 1;

    storage.commandWriteStartSeconds(sampleIndex) = ...
        max(0, toc(testStart) - loggerSession.testStartOffsetSeconds);
    [commandDispatchSeconds, commandWriteStopSeconds, loggerSession] = writeTransmitterVector( ...
        serialObject, ...
        commandedServoPositionsClipped, ...
        config.surfaceNames, ...
        config.activeSurfaceMask.', ...
        nextCommandSequenceNumbers, ...
        loggerSession, ...
        sampleIndex, ...
        scheduledSampleTimeSeconds);

    storage.commandDispatchSeconds(sampleIndex, :) = commandDispatchSeconds;
    storage.commandWriteStopSeconds(sampleIndex) = commandWriteStopSeconds;
    storage.hostSchedulingDelaySeconds(sampleIndex, :) = ...
        commandDispatchSeconds - scheduledSampleTimeSeconds;
    storage.commandSequenceNumbers(sampleIndex, config.activeSurfaceMask.') = ...
        nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.baseCommandDegrees(sampleIndex) = baseCommandDegrees;
    storage.desiredDeflectionsDegrees(sampleIndex, :) = desiredDeflectionsDegrees;
    storage.commandedServoPositions(sampleIndex, :) = commandedServoPositionsClipped;
    storage.commandSaturated(sampleIndex, :) = saturatedMask;
    storage.expectedPpmUs(sampleIndex, :) = ...
        buildExpectedPpmWidthsUs(commandedServoPositionsClipped, config);

    surfaceCommandCounts(config.activeSurfaceMask.') = nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.sampleCount = sampleIndex;
end

[loggerData, config] = finalizeTransmitterLoggerSession(serialObject, loggerSession, config);
storage.rawLogs = loggerData;

runInfo.status = "completed";
runInfo.stopTime = datetime("now");
runInfo.sampleCount = storage.sampleCount;

if config.returnToNeutralOnExit
    moveTransmitterToNeutral(commandInterface);
end
end

function storage = initializeTransmitterStorage(sampleCount, surfaceCount)
storage = struct( ...
    "sampleCount", 0, ...
    "scheduledTimeSeconds", nan(sampleCount, 1), ...
    "baseCommandDegrees", nan(sampleCount, 1), ...
    "commandWriteStartSeconds", nan(sampleCount, 1), ...
    "commandWriteStopSeconds", nan(sampleCount, 1), ...
    "commandDispatchSeconds", nan(sampleCount, surfaceCount), ...
    "hostSchedulingDelaySeconds", nan(sampleCount, surfaceCount), ...
    "commandSequenceNumbers", nan(sampleCount, surfaceCount), ...
    "desiredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "commandedServoPositions", nan(sampleCount, surfaceCount), ...
    "commandSaturated", false(sampleCount, surfaceCount), ...
    "expectedPpmUs", nan(sampleCount, 8), ...
    "boardReadStartSeconds", nan(sampleCount, 1), ...
    "boardReadStopSeconds", nan(sampleCount, 1), ...
    "boardRxSeconds", nan(sampleCount, surfaceCount), ...
    "boardCommitSeconds", nan(sampleCount, surfaceCount), ...
    "anchorSeconds", nan(sampleCount, surfaceCount), ...
    "referenceStrobeSeconds", nan(sampleCount, surfaceCount), ...
    "trainerPpmSeconds", nan(sampleCount, surfaceCount), ...
    "receiverResponseSeconds", nan(sampleCount, surfaceCount), ...
    "anchorToPpmLatencySeconds", nan(sampleCount, surfaceCount), ...
    "anchorToReceiverLatencySeconds", nan(sampleCount, surfaceCount), ...
    "computerToArduinoRxLatencySeconds", nan(sampleCount, surfaceCount), ...
    "arduinoReceiveToPpmCommitLatencySeconds", nan(sampleCount, surfaceCount), ...
    "ppmToReceiverLatencySeconds", nan(sampleCount, surfaceCount), ...
    "computerToReceiverLatencySeconds", nan(sampleCount, surfaceCount), ...
    "scheduledToReceiverLatencySeconds", nan(sampleCount, surfaceCount), ...
    "receivedPositionNorm", nan(sampleCount, surfaceCount), ...
    "committedPpmUs", nan(sampleCount, 8), ...
    "profileInfo", struct(), ...
    "rawLogs", struct(), ...
    "matchedEvents", table(), ...
    "integritySummary", table());
end

function loggerSession = startTransmitterLoggerSession(serialObject, config, sampleCount, activeSurfaceCount)
dispatchCapacity = max(32, sampleCount .* max(1, activeSurfaceCount));
rxCapacity = max(64, sampleCount + 32);
commitCapacity = max(64, sampleCount + 32);
syncCapacity = max(16, config.arduinoTransport.syncCountBeforeRun + config.arduinoTransport.syncCountAfterRun + 8);
telemetryCapacity = dispatchCapacity + rxCapacity + commitCapacity + syncCapacity + 128;

loggerSession = struct( ...
    "hostTimer", tic, ...
    "commandEncoding", config.arduinoTransport.commandEncoding, ...
    "testStartOffsetUs", uint32(0), ...
    "testStartOffsetSeconds", 0, ...
    "receiveBuffer", "", ...
    "dispatchCount", 0, ...
    "dispatchSurfaceName", strings(dispatchCapacity, 1), ...
    "dispatchSampleIndex", nan(dispatchCapacity, 1), ...
    "dispatchCommandSequence", nan(dispatchCapacity, 1), ...
    "dispatchSampleSequence", nan(dispatchCapacity, 1), ...
    "dispatchScheduledTimeSeconds", nan(dispatchCapacity, 1), ...
    "dispatchCommandUs", nan(dispatchCapacity, 1), ...
    "dispatchPosition", nan(dispatchCapacity, 1), ...
    "telemetryLineCount", 0, ...
    "telemetryLineText", strings(telemetryCapacity, 1), ...
    "telemetryHostRxUs", nan(telemetryCapacity, 1), ...
    "rxEventCount", 0, ...
    "rxHostRxUs", nan(rxCapacity, 1), ...
    "rxSampleSequence", nan(rxCapacity, 1), ...
    "rxActiveMask", nan(rxCapacity, 1), ...
    "rxBoardRxUs", nan(rxCapacity, 1), ...
    "rxPositionCode", nan(rxCapacity, 4), ...
    "commitEventCount", 0, ...
    "commitHostRxUs", nan(commitCapacity, 1), ...
    "commitSampleSequence", nan(commitCapacity, 1), ...
    "commitActiveMask", nan(commitCapacity, 1), ...
    "commitBoardRxUs", nan(commitCapacity, 1), ...
    "commitBoardCommitUs", nan(commitCapacity, 1), ...
    "commitReceiveToCommitUs", nan(commitCapacity, 1), ...
    "commitStrobeUs", nan(commitCapacity, 1), ...
    "commitFrameIndex", nan(commitCapacity, 1), ...
    "commitPpmUs", nan(commitCapacity, 8), ...
    "boardSyncCount", 0, ...
    "boardSyncId", nan(syncCapacity, 1), ...
    "boardSyncHostTxUs", nan(syncCapacity, 1), ...
    "boardSyncHostRxUs", nan(syncCapacity, 1), ...
    "boardSyncBoardRxUs", nan(syncCapacity, 1), ...
    "boardSyncBoardTxUs", nan(syncCapacity, 1));

if config.arduinoTransport.clearLogsBeforeRun
    sendTransmitterControlBurst(serialObject, "CLEAR_LOGS", 3, 0.02);
end

for syncIndex = 1:config.arduinoTransport.syncCountBeforeRun
    hostTxUs = hostNowUs(loggerSession.hostTimer);
    writeline(serialObject, sprintf("SYNC,%u,%u", uint32(syncIndex), hostTxUs));
    [loggerSession, isMatched] = waitForMatchingTransmitterSyncEvent( ...
        serialObject, ...
        loggerSession, ...
        uint32(syncIndex), ...
        double(hostTxUs), ...
        config.arduinoTransport.syncReplyTimeoutSeconds, ...
        config.arduinoTransport.linePollPauseSeconds);
    if ~isMatched
        loggerSession = pauseAndDrainTransmitterTelemetry( ...
            serialObject, ...
            loggerSession, ...
            config.arduinoTransport.syncPauseSeconds, ...
            config.arduinoTransport.linePollPauseSeconds);
    end
end

loggerSession.testStartOffsetUs = hostNowUs(loggerSession.hostTimer);
loggerSession.testStartOffsetSeconds = double(loggerSession.testStartOffsetUs) ./ 1e6;
end

function loggerSession = waitForScheduledTimeAndDrainTransmitter( ...
    referenceTimer, ...
    targetTimeSeconds, ...
    serialObject, ...
    loggerSession, ...
    pauseSeconds)
coarsePauseSeconds = max(0.001, min(pauseSeconds, 0.002));
finePauseSeconds = min(0.0005, coarsePauseSeconds);
drainCutoffSeconds = max(0.002, 4.0 .* finePauseSeconds);
while true
    remainingSeconds = targetTimeSeconds - toc(referenceTimer);
    if remainingSeconds <= 0
        return;
    end

    if remainingSeconds > drainCutoffSeconds
        loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
        remainingSeconds = targetTimeSeconds - toc(referenceTimer);
        if remainingSeconds <= 0
            return;
        end

        pauseSecondsThisIteration = min(coarsePauseSeconds, max(0, remainingSeconds - drainCutoffSeconds));
        if pauseSecondsThisIteration > 0
            pause(pauseSecondsThisIteration);
        end
        continue;
    end

    if remainingSeconds > finePauseSeconds
        pause(min(finePauseSeconds, remainingSeconds));
    end
end
end

function [dispatchTimesSeconds, writeStopSeconds, loggerSession] = writeTransmitterVector( ...
    serialObject, ...
    servoPositions, ...
    surfaceNames, ...
    activeSurfaceMask, ...
    commandSequenceNumbers, ...
    loggerSession, ...
    sampleIndex, ...
    scheduledSampleTimeSeconds)
dispatchTimesSeconds = nan(1, numel(surfaceNames));
activeIndices = find(activeSurfaceMask);

if isempty(activeIndices)
    writeStopSeconds = max(0, toc(loggerSession.hostTimer) - loggerSession.testStartOffsetSeconds);
    return;
end

dispatchAbsoluteUs = hostNowUs(loggerSession.hostTimer);
if numel(unique(commandSequenceNumbers(activeIndices))) ~= 1
    error("Transmitter_Test:InconsistentSampleSequence", ...
        "All active surfaces must share the same sample sequence.");
end
sampleSequence = commandSequenceNumbers(activeIndices(1));
if ~isfinite(sampleSequence) || sampleSequence < 0
    error("Transmitter_Test:InvalidSampleSequence", "Sample sequence must be finite and nonnegative.");
end

if commandSequenceNumbers(activeIndices(1)) >= 0
    if loggerSession.dispatchCount + numel(activeIndices) > numel(loggerSession.dispatchCommandUs)
        error("Transmitter_Test:DispatchCapacityExceeded", "Dispatch log capacity was exceeded.");
    end
end

if commandSequenceNumbers(activeIndices(1)) >= 0
    if configFromLoggerSessionCommandEncoding(loggerSession) == "binary_vector"
        payloadBytes = buildNanoLoggerBinaryVectorCommand( ...
            commandSequenceNumbers, ...
            servoPositions, ...
            activeSurfaceMask);
        write(serialObject, payloadBytes, "uint8");
    else
        writeline(serialObject, char(buildNanoLoggerSetAllCommand( ...
            surfaceNames, ...
            commandSequenceNumbers, ...
            servoPositions, ...
            activeSurfaceMask)));
    end
end

for activeIndex = 1:numel(activeIndices)
    surfaceIndex = activeIndices(activeIndex);
    loggerSession.dispatchCount = loggerSession.dispatchCount + 1;
    rowIndex = loggerSession.dispatchCount;
    loggerSession.dispatchSurfaceName(rowIndex) = surfaceNames(surfaceIndex);
    loggerSession.dispatchSampleIndex(rowIndex) = sampleIndex;
    loggerSession.dispatchCommandSequence(rowIndex) = commandSequenceNumbers(surfaceIndex);
    loggerSession.dispatchSampleSequence(rowIndex) = sampleSequence;
    loggerSession.dispatchScheduledTimeSeconds(rowIndex) = scheduledSampleTimeSeconds;
    loggerSession.dispatchCommandUs(rowIndex) = double(dispatchAbsoluteUs);
    loggerSession.dispatchPosition(rowIndex) = servoPositions(surfaceIndex);
    dispatchTimesSeconds(surfaceIndex) = ...
        max(0, (double(dispatchAbsoluteUs) - double(loggerSession.testStartOffsetUs)) ./ 1e6);
end

writeStopSeconds = max(0, (double(dispatchAbsoluteUs) - double(loggerSession.testStartOffsetUs)) ./ 1e6);
end

function commandEncoding = configFromLoggerSessionCommandEncoding(loggerSession)
commandEncoding = "binary_vector";
if isfield(loggerSession, "commandEncoding")
    commandEncoding = loggerSession.commandEncoding;
end
end

function loggerSession = pauseAndDrainTransmitterTelemetry(serialObject, loggerSession, durationSeconds, pauseSeconds)
pauseTimer = tic;
while toc(pauseTimer) < durationSeconds
    loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
    pause(min(pauseSeconds, durationSeconds - toc(pauseTimer)));
end
end

function [loggerSession, isMatched] = waitForMatchingTransmitterSyncEvent( ...
    serialObject, ...
    loggerSession, ...
    expectedSyncId, ...
    expectedHostTxUs, ...
    timeoutSeconds, ...
    pauseSeconds)
isMatched = false;
waitTimer = tic;
while toc(waitTimer) < timeoutSeconds
    [receivedLines, receiveBuffer] = readTransmitterLines(serialObject, loggerSession.receiveBuffer);
    loggerSession.receiveBuffer = receiveBuffer;
    if isempty(receivedLines)
        pause(min(pauseSeconds, timeoutSeconds - toc(waitTimer)));
        continue;
    end

    for lineIndex = 1:numel(receivedLines)
        hostRxUs = double(hostNowUs(loggerSession.hostTimer));
        receivedLine = receivedLines(lineIndex);
        loggerSession = appendTransmitterTelemetryLine(loggerSession, receivedLine, hostRxUs);

        telemetryParts = split(receivedLine, ",");
        if numel(telemetryParts) < 5 || telemetryParts(1) ~= "SYNC_EVENT"
            continue;
        end

        syncIdValue = str2double(telemetryParts(2));
        hostTxValue = str2double(telemetryParts(3));
        if ~isfinite(syncIdValue) || ~isfinite(hostTxValue)
            continue;
        end

        if uint32(syncIdValue) == uint32(expectedSyncId) && double(hostTxValue) == expectedHostTxUs
            isMatched = true;
            return;
        end
    end
end
end

function loggerSession = collectRemainingTransmitterTelemetry(serialObject, loggerSession, maxWaitSeconds, idleTimeoutSeconds)
collectTimer = tic;
lastReceiveSeconds = 0;
while toc(collectTimer) < maxWaitSeconds
    oldLineCount = loggerSession.telemetryLineCount;
    loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
    if loggerSession.telemetryLineCount > oldLineCount
        lastReceiveSeconds = toc(collectTimer);
        continue;
    end

    elapsedSeconds = toc(collectTimer);
    if elapsedSeconds >= idleTimeoutSeconds && (elapsedSeconds - lastReceiveSeconds) >= idleTimeoutSeconds
        return;
    end
    pause(0.01);
end
end

function loggerSession = drainTransmitterTelemetry(serialObject, loggerSession)
[receivedLines, receiveBuffer] = readTransmitterLines(serialObject, loggerSession.receiveBuffer);
loggerSession.receiveBuffer = receiveBuffer;
if isempty(receivedLines)
    return;
end

hostRxUs = double(hostNowUs(loggerSession.hostTimer));
for lineIndex = 1:numel(receivedLines)
    loggerSession = appendTransmitterTelemetryLine(loggerSession, receivedLines(lineIndex), hostRxUs);
end
end

function loggerSession = appendTransmitterTelemetryLine(loggerSession, receivedLine, hostRxUs)
if loggerSession.telemetryLineCount + 1 > numel(loggerSession.telemetryLineText)
    error("Transmitter_Test:TelemetryCapacityExceeded", "Serial telemetry log capacity was exceeded.");
end

loggerSession.telemetryLineCount = loggerSession.telemetryLineCount + 1;
telemetryRow = loggerSession.telemetryLineCount;
loggerSession.telemetryLineText(telemetryRow) = receivedLine;
loggerSession.telemetryHostRxUs(telemetryRow) = hostRxUs;
end

function parsedLoggerSession = parseStoredTransmitterTelemetry(loggerSession)
parsedLoggerSession = loggerSession;
parsedLoggerSession.rxEventCount = 0;
parsedLoggerSession.commitEventCount = 0;
parsedLoggerSession.boardSyncCount = 0;
parsedLoggerSession.rxHostRxUs(:) = nan;
parsedLoggerSession.rxSampleSequence(:) = nan;
parsedLoggerSession.rxActiveMask(:) = nan;
parsedLoggerSession.rxBoardRxUs(:) = nan;
parsedLoggerSession.rxPositionCode(:) = nan;
parsedLoggerSession.commitHostRxUs(:) = nan;
parsedLoggerSession.commitSampleSequence(:) = nan;
parsedLoggerSession.commitActiveMask(:) = nan;
parsedLoggerSession.commitBoardRxUs(:) = nan;
parsedLoggerSession.commitBoardCommitUs(:) = nan;
parsedLoggerSession.commitReceiveToCommitUs(:) = nan;
parsedLoggerSession.commitStrobeUs(:) = nan;
parsedLoggerSession.commitFrameIndex(:) = nan;
parsedLoggerSession.commitPpmUs(:) = nan;
parsedLoggerSession.boardSyncId(:) = nan;
parsedLoggerSession.boardSyncHostTxUs(:) = nan;
parsedLoggerSession.boardSyncHostRxUs(:) = nan;
parsedLoggerSession.boardSyncBoardRxUs(:) = nan;
parsedLoggerSession.boardSyncBoardTxUs(:) = nan;

for lineIndex = 1:loggerSession.telemetryLineCount
    parsedLoggerSession = appendParsedTransmitterTelemetryLine( ...
        parsedLoggerSession, ...
        loggerSession.telemetryLineText(lineIndex), ...
        loggerSession.telemetryHostRxUs(lineIndex));
end
end

function loggerSession = appendParsedTransmitterTelemetryLine(loggerSession, receivedLine, hostRxUs)
telemetryParts = split(receivedLine, ",");
if isempty(telemetryParts)
    return;
end

switch telemetryParts(1)
    case "RX_EVENT"
        if numel(telemetryParts) < 8
            return;
        end

        if loggerSession.rxEventCount + 1 > numel(loggerSession.rxHostRxUs)
            error("Transmitter_Test:RxCapacityExceeded", "RX_EVENT capacity was exceeded.");
        end

        loggerSession.rxEventCount = loggerSession.rxEventCount + 1;
        rowIndex = loggerSession.rxEventCount;
        loggerSession.rxHostRxUs(rowIndex) = hostRxUs;
        loggerSession.rxSampleSequence(rowIndex) = str2double(telemetryParts(2));
        loggerSession.rxActiveMask(rowIndex) = str2double(telemetryParts(3));
        loggerSession.rxBoardRxUs(rowIndex) = str2double(telemetryParts(4));
        loggerSession.rxPositionCode(rowIndex, :) = ...
            double(str2double(telemetryParts(5:8))).';

    case "COMMIT_EVENT"
        if numel(telemetryParts) < 15
            return;
        end

        if loggerSession.commitEventCount + 1 > numel(loggerSession.commitHostRxUs)
            error("Transmitter_Test:CommitCapacityExceeded", "COMMIT_EVENT capacity was exceeded.");
        end

        loggerSession.commitEventCount = loggerSession.commitEventCount + 1;
        rowIndex = loggerSession.commitEventCount;
        loggerSession.commitHostRxUs(rowIndex) = hostRxUs;
        loggerSession.commitSampleSequence(rowIndex) = str2double(telemetryParts(2));
        loggerSession.commitActiveMask(rowIndex) = str2double(telemetryParts(3));
        loggerSession.commitBoardRxUs(rowIndex) = str2double(telemetryParts(4));
        loggerSession.commitBoardCommitUs(rowIndex) = str2double(telemetryParts(5));
        if numel(telemetryParts) >= 16
            loggerSession.commitReceiveToCommitUs(rowIndex) = str2double(telemetryParts(16));
        end
        loggerSession.commitStrobeUs(rowIndex) = str2double(telemetryParts(6));
        loggerSession.commitFrameIndex(rowIndex) = str2double(telemetryParts(7));
        loggerSession.commitPpmUs(rowIndex, :) = ...
            double(str2double(telemetryParts(8:15))).';

    case "SYNC_EVENT"
        if numel(telemetryParts) < 5
            return;
        end

        if loggerSession.boardSyncCount + 1 > numel(loggerSession.boardSyncId)
            error("Transmitter_Test:SyncCapacityExceeded", "SYNC_EVENT capacity was exceeded.");
        end

        loggerSession.boardSyncCount = loggerSession.boardSyncCount + 1;
        rowIndex = loggerSession.boardSyncCount;
        loggerSession.boardSyncId(rowIndex) = str2double(telemetryParts(2));
        loggerSession.boardSyncHostTxUs(rowIndex) = str2double(telemetryParts(3));
        loggerSession.boardSyncHostRxUs(rowIndex) = hostRxUs;
        loggerSession.boardSyncBoardRxUs(rowIndex) = str2double(telemetryParts(4));
        loggerSession.boardSyncBoardTxUs(rowIndex) = str2double(telemetryParts(5));
end
end

function [receivedLines, receiveBuffer] = readTransmitterLines(serialObject, receiveBuffer)
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

function [loggerData, config] = finalizeTransmitterLoggerSession(serialObject, loggerSession, config)
if config.returnToNeutralOnExit
    sendTransmitterControlBurst(serialObject, "SET_NEUTRAL", 2, 0.02);
end

loggerSession = pauseAndDrainTransmitterTelemetry( ...
    serialObject, ...
    loggerSession, ...
    config.arduinoTransport.postRunSettleSeconds, ...
    config.arduinoTransport.linePollPauseSeconds);

for syncIndex = 1:config.arduinoTransport.syncCountAfterRun
    syncId = uint32(config.arduinoTransport.syncCountBeforeRun + syncIndex);
    hostTxUs = hostNowUs(loggerSession.hostTimer);
    writeline(serialObject, sprintf("SYNC,%u,%u", syncId, hostTxUs));
    [loggerSession, isMatched] = waitForMatchingTransmitterSyncEvent( ...
        serialObject, ...
        loggerSession, ...
        syncId, ...
        double(hostTxUs), ...
        config.arduinoTransport.syncReplyTimeoutSeconds, ...
        config.arduinoTransport.linePollPauseSeconds);
    if ~isMatched
        loggerSession = pauseAndDrainTransmitterTelemetry( ...
            serialObject, ...
            loggerSession, ...
            config.arduinoTransport.syncPauseSeconds, ...
            config.arduinoTransport.linePollPauseSeconds);
    end
end

writeline(serialObject, "STATUS");
loggerSession = pauseAndDrainTransmitterTelemetry( ...
    serialObject, ...
    loggerSession, ...
    0.05, ...
    config.arduinoTransport.linePollPauseSeconds);

loggerSession = collectRemainingTransmitterTelemetry( ...
    serialObject, ...
    loggerSession, ...
    2.0, ...
    config.arduinoTransport.lineIdleTimeoutSeconds);

parsedLoggerSession = parseStoredTransmitterTelemetry(loggerSession);
if parsedLoggerSession.boardSyncCount == 0
    error("Transmitter_Test:MissingSyncTelemetry", "No SYNC_EVENT lines were received from the transmitter.");
end

loggerData = struct( ...
    "testStartOffsetUs", double(loggerSession.testStartOffsetUs), ...
    "testStartOffsetSeconds", loggerSession.testStartOffsetSeconds, ...
    "hostDispatchLog", buildHostDispatchLogFromSession(loggerSession), ...
    "boardRxLog", buildBoardRxLogFromSession(parsedLoggerSession, config.surfaceNames), ...
    "boardCommitLog", buildBoardCommitLogFromSession(parsedLoggerSession), ...
    "boardSyncLog", buildBoardSyncLogFromSession(parsedLoggerSession), ...
    "serialTelemetryLog", compose( ...
        "%.0f,%s", ...
        loggerSession.telemetryHostRxUs(1:loggerSession.telemetryLineCount), ...
        loggerSession.telemetryLineText(1:loggerSession.telemetryLineCount)));

config.arduinoTransport.captureSucceeded = true;
config.arduinoTransport.captureRowCount = height(loggerData.boardCommitLog);
config.arduinoTransport.captureMessage = ...
    "Captured " + ...
    height(loggerData.boardRxLog) + " RX rows, " + ...
    height(loggerData.boardCommitLog) + " COMMIT rows, and " + ...
    height(loggerData.boardSyncLog) + " SYNC rows.";
end

function [sigrokSession, config] = startSigrokSession(config, runPlan)
if ~isSigrokAutoMode(config)
    sigrokSession = createEmptySigrokSession();
    return;
end

createFolderIfMissing(config.arduinoTransport.loggerOutputFolder);
deleteFileIfPresent(config.logicAnalyzer.rawCapturePath);
deleteFileIfPresent(config.logicAnalyzer.logicStateExportPath);
deleteFileIfPresent(config.logicAnalyzer.decodedReferencePath);
deleteFileIfPresent(config.logicAnalyzer.decodedTrainerPpmPath);
deleteFileIfPresent(config.logicAnalyzer.decodedReceiverPath);
deleteFileIfPresent(config.logicAnalyzer.storeStdoutPath);
deleteFileIfPresent(config.logicAnalyzer.storeStderrPath);

config.logicAnalyzer.sigrokCliPath = validateSigrokExecutable(config.logicAnalyzer.sigrokCliPath);
validateSigrokDevice(config);

captureDurationSeconds = computeSigrokCaptureDurationSeconds(config, runPlan);
config = adjustSigrokSampleRateForCaptureDuration(config, captureDurationSeconds);
config.logicAnalyzer.requestedCaptureDurationSeconds = captureDurationSeconds;
printSigrokAnalyzerConfiguration(config, captureDurationSeconds);
rawCaptureCommand = buildSigrokRawCaptureCommand(config, captureDurationSeconds);
processArguments = buildCmdExeArguments( ...
    rawCaptureCommand + ...
    " 1>" + quoteWindowsArgument(config.logicAnalyzer.storeStdoutPath) + ...
    " 2>" + quoteWindowsArgument(config.logicAnalyzer.storeStderrPath));

processStartInfo = System.Diagnostics.ProcessStartInfo();
processStartInfo.FileName = 'cmd.exe';
processStartInfo.Arguments = char(processArguments);
processStartInfo.UseShellExecute = false;
processStartInfo.CreateNoWindow = true;
processStartInfo.WorkingDirectory = char(config.outputFolder);

processHandle = System.Diagnostics.Process();
processHandle.StartInfo = processStartInfo;
if ~processHandle.Start()
    error("Transmitter_Test:SigrokLaunchFailed", "Failed to launch sigrok-cli raw capture.");
end

sigrokSession = struct( ...
    "process", processHandle, ...
    "processId", double(processHandle.Id), ...
    "isActive", true, ...
    "rawCaptureCommand", rawCaptureCommand, ...
    "logicStateExportCommand", "", ...
    "rawCapturePath", config.logicAnalyzer.rawCapturePath, ...
    "logicStateExportPath", config.logicAnalyzer.logicStateExportPath, ...
    "decodedReferencePath", config.logicAnalyzer.decodedReferencePath, ...
    "decodedTrainerPpmPath", config.logicAnalyzer.decodedTrainerPpmPath, ...
    "decodedReceiverPath", config.logicAnalyzer.decodedReceiverPath, ...
    "stdoutPath", config.logicAnalyzer.storeStdoutPath, ...
    "stderrPath", config.logicAnalyzer.storeStderrPath, ...
    "sampleRateHz", config.logicAnalyzer.sampleRateHz, ...
    "channelRoleMap", config.logicAnalyzer.channelRoleMap);
end

function sigrokSession = waitForSigrokSession(sigrokSession, config)
if ~sigrokSession.isActive
    return;
end

sigrokSession.process.WaitForExit();
exitCode = double(sigrokSession.process.ExitCode);
sigrokSession.isActive = false;
if exitCode ~= 0
    error( ...
        "Transmitter_Test:SigrokCaptureFailed", ...
        "sigrok-cli raw capture failed with exit code %d. See %s.", ...
        exitCode, ...
        char(config.logicAnalyzer.storeStderrPath));
end
if ~isfile(sigrokSession.rawCapturePath)
    error("Transmitter_Test:MissingSigrokCapture", "sigrok-cli finished without creating the raw capture file.");
end
end

function config = finalizeSigrokSession(sigrokSession, config)
[logicStateTable, decodedSampleRateHz, usedRawCaptureParse] = tryBuildLogicStateTableFromSigrokRawCapture(config);
if usedRawCaptureParse
    config.logicAnalyzer.sampleRateHz = decodedSampleRateHz;
    writetable(logicStateTable, config.logicAnalyzer.logicStateExportPath);
else
    logicStateExportCommand = buildSigrokLogicStateExportCommand(config);
    sigrokSession.logicStateExportCommand = logicStateExportCommand;
    [status, outputText] = runWindowsCommand( ...
        logicStateExportCommand + ...
        " 1>>" + quoteWindowsArgument(sigrokSession.stdoutPath) + ...
        " 2>>" + quoteWindowsArgument(sigrokSession.stderrPath));
    if status ~= 0
        error( ...
            "Transmitter_Test:SigrokExportFailed", ...
            "sigrok-cli logicStateExportPath export failed. %s", ...
            strtrim(outputText));
    end

    logicStateTable = readLogicStateExportCsv(config.logicAnalyzer.logicStateExportPath, config);
    if isempty(logicStateTable)
        error("Transmitter_Test:EmptyLogicStateExport", "logicStateExportPath did not contain any analyser samples.");
    end
    logicStateTable = normalizeLogicStateExportColumns(logicStateTable, config);
end

logicState = convertLogicStateTableToSamples(logicStateTable, config.logicAnalyzer.sampleRateHz);
warnIfSigrokCaptureDurationLooksShort(logicState, config);

referenceCapture = extractReferenceEdgesFromLogicState(logicState, config);
trainerPpmCapture = extractTrainerPpmCaptureFromLogicState(logicState, config);
receiverCapture = extractReceiverCaptureFromLogicState(logicState, config);
if isempty(referenceCapture) && isempty(trainerPpmCapture) && isempty(receiverCapture)
    error("Transmitter_Test:EmptyLogicStateDecode", "logicStateExportPath did not decode into any analyser events.");
end

function [logicStateTable, sampleRateHz, usedRawCaptureParse] = tryBuildLogicStateTableFromSigrokRawCapture(config)
logicStateTable = table();
sampleRateHz = double(config.logicAnalyzer.sampleRateHz);
usedRawCaptureParse = false;

try
    [logicStateTable, sampleRateHz] = readSigrokRawCaptureAsLogicStateTable(config);
    usedRawCaptureParse = ~isempty(logicStateTable);
catch rawCaptureError
    fprintf("Raw .sr parse fallback to sigrok CSV export: %s\n", rawCaptureError.message);
    logicStateTable = table();
    sampleRateHz = double(config.logicAnalyzer.sampleRateHz);
    usedRawCaptureParse = false;
end
end

function [logicStateTable, sampleRateHz] = readSigrokRawCaptureAsLogicStateTable(config)
rawCapturePath = string(config.logicAnalyzer.rawCapturePath);
if ~isfile(rawCapturePath)
    error("Transmitter_Test:MissingSigrokCapture", "Raw sigrok capture file does not exist: %s", char(rawCapturePath));
end

extractDirectory = string(tempname);
mkdir(extractDirectory);
cleanupDirectory = onCleanup(@() removeDirectoryIfPresent(extractDirectory));
unzip(rawCapturePath, extractDirectory);

metadataPath = fullfile(extractDirectory, "metadata");
if ~isfile(metadataPath)
    error("Transmitter_Test:MissingSigrokMetadata", "Raw sigrok capture is missing the metadata entry.");
end

metadataText = fileread(metadataPath);
[sampleRateHz, probeNames, unitSizeBytes] = parseSigrokRawMetadata(metadataText, config.logicAnalyzer.sampleRateHz);
if unitSizeBytes ~= 1
    error("Transmitter_Test:UnsupportedSigrokUnitSize", "Raw sigrok parsing currently supports only unitsize=1 captures.");
end

logicFiles = dir(fullfile(extractDirectory, "logic-1-*"));
if isempty(logicFiles)
    error("Transmitter_Test:MissingSigrokLogicChunks", "Raw sigrok capture does not contain logic sample chunks.");
end
logicFiles = sortSigrokLogicChunkFiles(logicFiles);
[changeSampleIndex, changeBytes] = readSigrokLogicTransitions(logicFiles);
if isempty(changeSampleIndex)
    logicStateTable = table();
    return;
end

timeSeconds = sampleIndexToTimeSeconds(changeSampleIndex, sampleRateHz);
logicStateTable = table(timeSeconds, changeSampleIndex, 'VariableNames', {'time_s', 'sample_index'});

for channelIndex = 1:numel(config.logicAnalyzer.channelNames)
    configuredName = string(config.logicAnalyzer.channelNames(channelIndex));
    probeIndex = find(probeNames == configuredName, 1, "first");
    if isempty(probeIndex)
        error( ...
            "Transmitter_Test:MissingSigrokProbe", ...
            "Raw sigrok capture metadata does not contain probe %s.", ...
            char(configuredName));
    end
    logicStateTable.(char(configuredName)) = double(bitget(changeBytes, probeIndex));
end
end

function [sampleRateHz, probeNames, unitSizeBytes] = parseSigrokRawMetadata(metadataText, defaultSampleRateHz)
sampleRateHz = double(defaultSampleRateHz);
probeNames = strings(0, 1);
unitSizeBytes = 1;

metadataLines = splitlines(string(metadataText));
probeNumbers = zeros(0, 1);
probeValues = strings(0, 1);
for lineIndex = 1:numel(metadataLines)
    lineText = strtrim(metadataLines(lineIndex));
    if startsWith(lineText, "samplerate=", "IgnoreCase", true)
        sampleRateHz = parseSigrokSampleRateText(extractAfter(lineText, "="), defaultSampleRateHz);
    elseif startsWith(lineText, "unitsize=", "IgnoreCase", true)
        unitSizeBytes = round(str2double(extractAfter(lineText, "=")));
    elseif startsWith(lineText, "probe", "IgnoreCase", true)
        token = regexp(char(lineText), '^probe(\d+)=(.*)$', 'tokens', 'once');
        if ~isempty(token)
            probeNumbers(end + 1, 1) = str2double(token{1}); %#ok<AGROW>
            probeValues(end + 1, 1) = string(strtrim(token{2})); %#ok<AGROW>
        end
    end
end

if ~isempty(probeNumbers)
    probeNames = strings(max(probeNumbers), 1);
    for probeIndex = 1:numel(probeNumbers)
        probeNames(probeNumbers(probeIndex)) = probeValues(probeIndex);
    end
end
end

function sampleRateHz = parseSigrokSampleRateText(sampleRateText, defaultSampleRateHz)
sampleRateHz = double(defaultSampleRateHz);
matchTokens = regexp(strtrim(char(sampleRateText)), '^([0-9]*\.?[0-9]+)\s*([kmg]?)(?:hz)?$', 'tokens', 'once', 'ignorecase');
if isempty(matchTokens)
    return;
end

sampleRateValue = str2double(matchTokens{1});
scaleToken = lower(string(matchTokens{2}));
scaleFactor = 1;
if scaleToken == "k"
    scaleFactor = 1e3;
elseif scaleToken == "m"
    scaleFactor = 1e6;
elseif scaleToken == "g"
    scaleFactor = 1e9;
end
sampleRateHz = sampleRateValue .* scaleFactor;
end

function logicFiles = sortSigrokLogicChunkFiles(logicFiles)
fileNames = string({logicFiles.name}');
chunkNumbers = nan(numel(fileNames), 1);
for fileIndex = 1:numel(fileNames)
    token = regexp(char(fileNames(fileIndex)), '^logic-\d+-(\d+)$', 'tokens', 'once');
    if ~isempty(token)
        chunkNumbers(fileIndex) = str2double(token{1});
    end
end
[~, sortOrder] = sort(chunkNumbers);
logicFiles = logicFiles(sortOrder);
end

function [changeSampleIndex, changeBytes] = readSigrokLogicTransitions(logicFiles)
changeSampleIndexCells = cell(numel(logicFiles), 1);
changeBytesCells = cell(numel(logicFiles), 1);
sampleOffset = 0;
lastByte = uint8(0);
hasPreviousSample = false;

for fileIndex = 1:numel(logicFiles)
    logicFilePath = fullfile(logicFiles(fileIndex).folder, logicFiles(fileIndex).name);
    fileIdentifier = fopen(logicFilePath, 'r');
    if fileIdentifier < 0
        error("Transmitter_Test:OpenSigrokChunkFailed", "Unable to open raw sigrok logic chunk: %s", logicFilePath);
    end
    closeFile = onCleanup(@() fclose(fileIdentifier));
    chunkBytes = fread(fileIdentifier, inf, '*uint8');
    clear closeFile

    if isempty(chunkBytes)
        continue;
    end

    if hasPreviousSample
        changeMask = [chunkBytes(1) ~= lastByte; chunkBytes(2:end) ~= chunkBytes(1:end-1)];
    else
        changeMask = [true; chunkBytes(2:end) ~= chunkBytes(1:end-1)];
    end

    changeIndices = sampleOffset + find(changeMask) - 1;
    changeSampleIndexCells{fileIndex} = reshape(double(changeIndices), [], 1);
    changeBytesCells{fileIndex} = reshape(chunkBytes(changeMask), [], 1);

    sampleOffset = sampleOffset + numel(chunkBytes);
    lastByte = chunkBytes(end);
    hasPreviousSample = true;
end

changeSampleIndex = vertcat(changeSampleIndexCells{:});
changeBytes = vertcat(changeBytesCells{:});
end

function removeDirectoryIfPresent(directoryPath)
if isfolder(directoryPath)
    rmdir(directoryPath, 's');
end
end
printTrainerDecodeSummary(trainerPpmCapture, config);
validateDecodedAnalyzerTables(referenceCapture, trainerPpmCapture, receiverCapture, config);

writetable(referenceCapture, config.logicAnalyzer.decodedReferencePath);
writetable(trainerPpmCapture, config.logicAnalyzer.decodedTrainerPpmPath);
writetable(receiverCapture, config.logicAnalyzer.decodedReceiverPath);

config.logicAnalyzer.referenceCapturePath = config.logicAnalyzer.decodedReferencePath;
config.logicAnalyzer.trainerPpmCapturePath = config.logicAnalyzer.decodedTrainerPpmPath;
config.logicAnalyzer.receiverCapturePath = config.logicAnalyzer.decodedReceiverPath;

if config.logicAnalyzer.deleteRawAfterDecode
    deleteFileIfPresent(config.logicAnalyzer.rawCapturePath);
end
end

function commandText = buildSigrokRawCaptureCommand(config, captureDurationSeconds)
driverSpec = buildSigrokDriverSpec(config.logicAnalyzer);
channelAssignments = buildSigrokChannelAssignments(config.logicAnalyzer.channels, config.logicAnalyzer.channelNames);
captureDurationMs = max(1, ceil(1000 .* captureDurationSeconds));
captureDurationText = string(captureDurationMs);

commandText = strjoin([ ...
    quoteWindowsArgument(config.logicAnalyzer.sigrokCliPath), ...
    "--driver", quoteWindowsArgument(driverSpec), ...
    "--config", quoteWindowsArgument("samplerate=" + string(round(config.logicAnalyzer.sampleRateHz))), ...
    "--channels", quoteWindowsArgument(channelAssignments), ...
    "--time", quoteWindowsArgument(captureDurationText), ...
    "--output-file", quoteWindowsArgument(config.logicAnalyzer.rawCapturePath)], " ");

if config.logicAnalyzer.useTrigger
    commandText = strjoin([ ...
        commandText, ...
        "--triggers", quoteWindowsArgument(config.logicAnalyzer.triggerChannel + "=" + config.logicAnalyzer.triggerType), ...
        "--wait-trigger"], ...
        " ");
end
end

function commandText = buildSigrokLogicStateExportCommand(config)
outputFormatText = resolveSigrokLogicStateOutputFormat(config.logicAnalyzer.outputFormat);
commandText = strjoin([ ...
    quoteWindowsArgument(config.logicAnalyzer.sigrokCliPath), ...
    "--input-file", quoteWindowsArgument(config.logicAnalyzer.rawCapturePath), ...
    "--output-file", quoteWindowsArgument(config.logicAnalyzer.logicStateExportPath), ...
    "--output-format", quoteWindowsArgument(outputFormatText)], " ");
end

function sigrokCliPath = validateSigrokExecutable(sigrokCliPath)
sigrokCliPath = strtrim(string(sigrokCliPath));
if strlength(sigrokCliPath) > 0
    sigrokCliPath = resolveSigrokExecutablePath(sigrokCliPath);
end
if strlength(sigrokCliPath) == 0
    sigrokCliPath = resolveSigrokExecutablePath("C:\Program Files\sigrok\sigrok-cli\sigrok-cli.exe");
end
if strlength(sigrokCliPath) == 0
    sigrokCliPath = resolveSigrokExecutablePath("C:\Program Files\sigrok\sigrok-cli");
end
if strlength(sigrokCliPath) == 0
    [status, whereOutput] = system('where.exe sigrok-cli');
    if status ~= 0
        error("Transmitter_Test:MissingSigrokCli", "Unable to locate sigrok-cli via where.exe.");
    end
    whereLines = splitlines(string(whereOutput));
    whereLines = strtrim(whereLines(strlength(strtrim(whereLines)) > 0));
    if isempty(whereLines)
        error("Transmitter_Test:MissingSigrokCli", "where.exe did not return a sigrok-cli path.");
    end
    sigrokCliPath = resolveSigrokExecutablePath(whereLines(1));
end

if ~isfile(sigrokCliPath)
    error("Transmitter_Test:InvalidSigrokCliPath", "sigrok-cli executable was not found: %s", char(sigrokCliPath));
end

[status, outputText] = runWindowsCommand(quoteWindowsArgument(sigrokCliPath) + " --version");
if status ~= 0
    error("Transmitter_Test:SigrokCliValidationFailed", "sigrok-cli validation failed. %s", strtrim(outputText));
end
end

function sigrokCliPath = resolveSigrokExecutablePath(sigrokCliPath)
sigrokCliPath = strtrim(string(sigrokCliPath));
if strlength(sigrokCliPath) == 0
    return;
end

candidatePaths = strings(0, 1);
candidatePaths(end + 1, 1) = sigrokCliPath; %#ok<AGROW>

if isfolder(sigrokCliPath)
    candidatePaths(end + 1, 1) = fullfile(sigrokCliPath, "sigrok-cli", "sigrok-cli.exe"); %#ok<AGROW>
    candidatePaths(end + 1, 1) = fullfile(sigrokCliPath, "sigrok-cli.exe"); %#ok<AGROW>
elseif ~endsWith(lower(sigrokCliPath), ".exe")
    candidatePaths(end + 1, 1) = sigrokCliPath + ".exe"; %#ok<AGROW>
    candidatePaths(end + 1, 1) = fullfile(sigrokCliPath, "sigrok-cli.exe"); %#ok<AGROW>
end

for candidateIndex = 1:numel(candidatePaths)
    candidatePath = strtrim(candidatePaths(candidateIndex));
    if isfile(candidatePath)
        sigrokCliPath = candidatePath;
        return;
    end
end

sigrokCliPath = "";
end

function validateSigrokDevice(config)
sigrokCliPath = resolveSigrokCliPath(config.logicAnalyzer.sigrokCliPath);
driverSpec = buildSigrokDriverSpec(config.logicAnalyzer);

openCommand = strjoin([ ...
    quoteWindowsArgument(sigrokCliPath), ...
    "--driver", quoteWindowsArgument(driverSpec), ...
    "--show"], " ");
[openStatus, openOutputText] = runWindowsCommand(openCommand);
raiseSigrokUsbAccessErrorIfNeeded(openOutputText);
if openStatus == 0
    return;
end

[scanStatus, scanOutputText] = runWindowsCommand(quoteWindowsArgument(sigrokCliPath) + " --scan");
raiseSigrokUsbAccessErrorIfNeeded(scanOutputText);
if scanStatus ~= 0 || ~contains(lower(string(scanOutputText)), lower(config.logicAnalyzer.deviceDriver))
    error("Transmitter_Test:SigrokDeviceValidationFailed", "sigrok-cli scan did not confirm driver %s.", char(config.logicAnalyzer.deviceDriver));
end

error("Transmitter_Test:SigrokDeviceValidationFailed", "sigrok-cli found driver %s but could not open the device. %s", char(config.logicAnalyzer.deviceDriver), strtrim(openOutputText));
end

function raiseSigrokUsbAccessErrorIfNeeded(outputText)
outputText = string(outputText);
outputTextLower = lower(outputText);

if contains(outputTextLower, "libusb_error_access") || contains(outputTextLower, "libusb_error_not_supported")
    error( ...
        "Transmitter_Test:SigrokUsbAccessDenied", ...
        [ ...
        "sigrok-cli could not open the logic analyser because Windows USB access is not configured for libusb (%s). " + ...
        "Close PulseView and any other analyser software, unplug/replug the analyser, and install a libusb-compatible driver such as WinUSB for this analyser in Zadig before retrying."], ...
        strtrim(outputText));
end
end

function logicStateTable = readLogicStateExportCsv(filePath, config)
if ~isfile(filePath)
    error("Transmitter_Test:MissingLogicStateExport", "logicStateExportPath file was not created: %s", char(filePath));
end

validateLogicStateExportFileSize(filePath, config.logicAnalyzer.maximumLogicStateExportBytes);

options = detectImportOptions(filePath, "FileType", "text", "CommentStyle", "#");
options.CommentStyle = "#";
if isprop(options, "VariableNamingRule")
    options.VariableNamingRule = "preserve";
end
logicStateTable = readtable(filePath, options);
end

function logicStateTable = normalizeLogicStateExportColumns(rawTable, config)
variableNames = string(rawTable.Properties.VariableNames);
canonicalNames = canonicalizeLogicStateNames(variableNames);
timeIndex = find(contains(canonicalNames, "time"), 1, "first");
if isempty(timeIndex)
    error("Transmitter_Test:MissingLogicStateTimeColumn", "logicStateExportPath is missing an explicit time column.");
end

rawTimeData = convertLogicStateColumnToNumeric(rawTable.(char(variableNames(timeIndex))));
timeSeconds = normalizeSigrokTimeColumnToSeconds(rawTimeData, config.logicAnalyzer.sampleRateHz);
logicStateTable = table(timeSeconds, 'VariableNames', {'time_s'});
sampleIndexColumn = find(ismember(canonicalNames, ["sampleindex", "samplenum", "sample"]), 1, "first");
if ~isempty(sampleIndexColumn)
    logicStateTable.sample_index = convertLogicStateColumnToNumeric(rawTable.(char(variableNames(sampleIndexColumn))));
end

for channelIndex = 1:numel(config.logicAnalyzer.channels)
    targetName = string(config.logicAnalyzer.channelNames(channelIndex));
    headerIndex = findMatchingLogicStateColumnIndex( ...
        canonicalNames, ...
        targetName, ...
        config.logicAnalyzer.channels(channelIndex));
    if isempty(headerIndex)
        error( ...
            "Transmitter_Test:MissingLogicStateChannel", ...
            "logicStateExportPath is missing a logic column for channel %s.", ...
            char(targetName));
    end
    logicStateTable.(char(targetName)) = convertLogicStateColumnToNumeric(rawTable.(char(variableNames(headerIndex))));
end
end

function logicState = convertLogicStateTableToSamples(logicStateTable, sampleRateHz)
variableNames = string(logicStateTable.Properties.VariableNames);
sampleColumnIndex = find(ismember(variableNames, ["sample_index", "samplenum"]), 1, "first");
if isempty(sampleColumnIndex)
    sampleIndex = round(double(logicStateTable.time_s) .* double(sampleRateHz));
else
    sampleIndex = round(double(logicStateTable.(char(variableNames(sampleColumnIndex)))));
end
channelMask = variableNames ~= "time_s" & variableNames ~= "sample_index";

logicState = struct( ...
    "sampleIndex", reshape(sampleIndex, [], 1), ...
    "sampleRateHz", double(sampleRateHz), ...
    "channelNames", variableNames(channelMask), ...
    "stateMatrix", double(logicStateTable{:, channelMask}));
end

function warnIfSigrokCaptureDurationLooksShort(logicState, config)
requestedDurationSeconds = double(config.logicAnalyzer.requestedCaptureDurationSeconds);
if ~isfinite(requestedDurationSeconds) || requestedDurationSeconds <= 0
    return;
end

if isempty(logicState.sampleIndex)
    return;
end

observedDurationSeconds = ...
    sampleIndexToTimeSeconds(max(double(logicState.sampleIndex)), logicState.sampleRateHz);
if ~isfinite(observedDurationSeconds) || observedDurationSeconds <= 0
    return;
end

captureCoverage = observedDurationSeconds ./ requestedDurationSeconds;
if captureCoverage >= 0.90
    return;
end

warning( ...
    "Transmitter_Test:SigrokCaptureShort", ...
    [ ...
    "Sigrok capture appears truncated (observed %.3f s, requested %.3f s, coverage %.1f%%). " + ...
    "Increase logicAnalyzer.maximumCaptureSamples (or set to 0 for unlimited) and/or reduce logicAnalyzer.sampleRateHz."], ...
    observedDurationSeconds, ...
    requestedDurationSeconds, ...
    100 .* captureCoverage);
end

function referenceCapture = extractReferenceEdgesFromLogicState(logicState, config)
referenceColumnIndex = resolveLogicStateRoleColumnIndex(logicState, config, "reference");
referenceStates = logicState.stateMatrix(:, referenceColumnIndex);
edgeMask = [false; diff(referenceStates) > 0];
edgeSamples = logicState.sampleIndex(edgeMask);
debounceSamples = round(double(config.matching.referenceDebounceUs) .* logicState.sampleRateHz ./ 1e6);
edgeSamples = applySampleDebounce(edgeSamples, debounceSamples);
referenceCapture = table( ...
    sampleIndexToTimeSeconds(edgeSamples, logicState.sampleRateHz), ...
    edgeSamples, ...
    repmat(logicState.sampleRateHz, numel(edgeSamples), 1), ...
    'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'});
end

function trainerPpmCapture = extractTrainerPpmCaptureFromLogicState(logicState, config)
trainerColumnIndex = resolveLogicStateRoleColumnIndex(logicState, config, "trainerPpm");
trainerStates = logicState.stateMatrix(:, trainerColumnIndex);
markStartSamples = extractTrainerMarkStartSamples(trainerStates, logicState.sampleIndex, logicState.sampleRateHz, config);
if numel(markStartSamples) < 2
    trainerPpmCapture = buildEmptyPulseCaptureTable(true);
    return;
end

slotSampleCounts = diff(markStartSamples);
syncGapThresholdSamples = round( ...
    double(max(config.trainerPpm.maximumPulseUs + config.trainerPpm.markWidthUs, 2 .* config.trainerPpm.maximumPulseUs)) .* ...
    logicState.sampleRateHz ./ ...
    1e6);
minimumValidSlotSamples = round( ...
    double(max(0.8 .* config.trainerPpm.minimumPulseUs, config.trainerPpm.minimumPulseUs - config.trainerPpm.markWidthUs)) .* ...
    logicState.sampleRateHz ./ ...
    1e6);

rowCapacity = numel(slotSampleCounts);
surfaceNames = strings(rowCapacity, 1);
timeSeconds = nan(rowCapacity, 1);
sampleIndex = nan(rowCapacity, 1);
sampleCount = nan(rowCapacity, 1);
sampleRate = nan(rowCapacity, 1);
frameIndex = nan(rowCapacity, 1);
rowCount = 0;

currentFrameIndex = 0;
currentChannelIndex = 1;
hasSeenFrameBoundary = false;
for slotIndex = 1:numel(slotSampleCounts)
    slotCount = slotSampleCounts(slotIndex);
    if slotCount > syncGapThresholdSamples
        currentFrameIndex = currentFrameIndex + 1;
        currentChannelIndex = 1;
        hasSeenFrameBoundary = true;
        continue;
    end
    if ~hasSeenFrameBoundary
        continue;
    end
    if slotCount < minimumValidSlotSamples
        continue;
    end

    if currentChannelIndex <= numel(config.trainerPpm.channelSurfaceMap)
        mappedSurfaceName = string(config.trainerPpm.channelSurfaceMap(currentChannelIndex));
        if any(mappedSurfaceName == config.surfaceNames)
            rowCount = rowCount + 1;
            surfaceNames(rowCount) = mappedSurfaceName;
            sampleIndex(rowCount) = markStartSamples(slotIndex);
            % The Uno schedules each slot as one mark followed by the
            % remaining gap, so consecutive mark-start timestamps already
            % span the full commanded slot width.
            sampleCount(rowCount) = slotCount;
            sampleRate(rowCount) = logicState.sampleRateHz;
            timeSeconds(rowCount) = sampleIndexToTimeSeconds(markStartSamples(slotIndex), logicState.sampleRateHz);
            frameIndex(rowCount) = currentFrameIndex;
        end
    end
    currentChannelIndex = currentChannelIndex + 1;
end

trainerPpmCapture = table( ...
    surfaceNames(1:rowCount), ...
    timeSeconds(1:rowCount), ...
    sampleCountToPulseUs(sampleCount(1:rowCount), logicState.sampleRateHz), ...
    sampleIndex(1:rowCount), ...
    sampleCount(1:rowCount), ...
    sampleRate(1:rowCount), ...
    frameIndex(1:rowCount), ...
    'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz', 'frame_index'});
end

function markStartSamples = extractTrainerMarkStartSamples(trainerStates, sampleIndex, sampleRateHz, config)
risingSamples = sampleIndex([false; diff(trainerStates) > 0]);
fallingSamples = sampleIndex([false; diff(trainerStates) < 0]);
if config.trainerPpm.idleHigh
    [candidateMarkStarts, candidateMarkCounts] = pairEdgeSamples(fallingSamples, risingSamples);
else
    [candidateMarkStarts, candidateMarkCounts] = pairEdgeSamples(risingSamples, fallingSamples);
end

markWidthSamples = double(config.trainerPpm.markWidthUs) .* double(sampleRateHz) ./ 1e6;
markToleranceSamples = max(ceil(0.35 .* markWidthSamples), round(60 .* double(sampleRateHz) ./ 1e6));
minimumMarkSamples = max(1, round(markWidthSamples - markToleranceSamples));
maximumMarkSamples = round(markWidthSamples + markToleranceSamples);
validMarkMask = ...
    isfinite(candidateMarkCounts) & ...
    candidateMarkCounts >= minimumMarkSamples & ...
    candidateMarkCounts <= maximumMarkSamples;
markStartSamples = candidateMarkStarts(validMarkMask);
end

function receiverCapture = extractReceiverCaptureFromLogicState(logicState, config)
surfaceCount = numel(config.surfaceNames);
surfaceCaptureTables = cell(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    roleChannel = config.logicAnalyzer.channelRoleMap.receiver(surfaceIndex);
    channelColumnIndex = resolveLogicStateChannelColumnIndex(logicState, config, roleChannel);
    channelStates = logicState.stateMatrix(:, channelColumnIndex);
    risingSamples = logicState.sampleIndex([false; diff(channelStates) > 0]);
    fallingSamples = logicState.sampleIndex([false; diff(channelStates) < 0]);
    [pulseStartSamples, pulseSampleCounts] = pairEdgeSamples(risingSamples, fallingSamples);
    pulseWidthsUs = sampleCountToPulseUs(pulseSampleCounts, logicState.sampleRateHz);
    validPulseMask = isPlausibleRcPulseWidthUs(pulseWidthsUs, config);
    pulseStartSamples = pulseStartSamples(validPulseMask);
    pulseSampleCounts = pulseSampleCounts(validPulseMask);
    pulseCount = numel(pulseStartSamples);
    if pulseCount == 0
        continue;
    end

    surfaceCaptureTables{surfaceIndex} = table( ...
        repmat(config.surfaceNames(surfaceIndex), pulseCount, 1), ...
        sampleIndexToTimeSeconds(pulseStartSamples, logicState.sampleRateHz), ...
        sampleCountToPulseUs(pulseSampleCounts, logicState.sampleRateHz), ...
        pulseStartSamples, ...
        pulseSampleCounts, ...
        repmat(logicState.sampleRateHz, pulseCount, 1), ...
        'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz'});
end

surfaceCaptureTables = surfaceCaptureTables(~cellfun(@isempty, surfaceCaptureTables));
if isempty(surfaceCaptureTables)
    receiverCapture = buildEmptyPulseCaptureTable(false);
    return;
end

receiverCapture = vertcat(surfaceCaptureTables{:});
if ~isempty(receiverCapture)
    receiverCapture = sortrows(receiverCapture, {'surface_name', 'sample_index'});
end
end

function timeSeconds = sampleIndexToTimeSeconds(sampleIndex, sampleRateHz)
timeSeconds = double(sampleIndex) ./ double(sampleRateHz);
end

function pulseUs = sampleCountToPulseUs(sampleCount, sampleRateHz)
pulseUs = 1e6 .* double(sampleCount) ./ double(sampleRateHz);
end

function printSigrokAnalyzerConfiguration(config, captureDurationSeconds)
fprintf("Sigrok analyser configuration\n");
fprintf("  Sample rate: %.0f Hz\n", double(config.logicAnalyzer.sampleRateHz));
fprintf("  Capture duration: %.3f s\n", double(captureDurationSeconds));
if isfinite(config.logicAnalyzer.maximumCaptureSamples) && config.logicAnalyzer.maximumCaptureSamples > 0
    fprintf("  Max capture samples: %.0f\n", double(config.logicAnalyzer.maximumCaptureSamples));
else
    fprintf("  Max capture samples: unlimited\n");
end
fprintf("  Uno trainer pin: %s\n", char(config.trainerPpm.outputPin));
fprintf("  Uno reference pin: %s\n", char(config.trainerPpm.referencePin));
fprintf("  Trainer idle level: %s\n", char(string(ternaryText(config.trainerPpm.idleHigh, "HIGH", "LOW"))));
fprintf("  Logic analyser receiver channels: %s\n", char(strjoin("D" + string(config.logicAnalyzer.channelRoleMap.receiver), ", ")));
fprintf("  Logic analyser reference channel: D%d (%s)\n", ...
    double(config.logicAnalyzer.channelRoleMap.reference), ...
    char(resolveConfiguredLogicAnalyzerChannelName(config, config.logicAnalyzer.channelRoleMap.reference)));
fprintf("  Logic analyser trainer channel: D%d (%s)\n", ...
    double(config.logicAnalyzer.channelRoleMap.trainerPpm), ...
    char(resolveConfiguredLogicAnalyzerChannelName(config, config.logicAnalyzer.channelRoleMap.trainerPpm)));
fprintf("\n");
end

function printTrainerDecodeSummary(trainerPpmCapture, config)
fprintf("Trainer decode summary\n");
fprintf("  Configured trainer analyser channel: D%d (%s)\n", ...
    double(config.logicAnalyzer.channelRoleMap.trainerPpm), ...
    char(resolveConfiguredLogicAnalyzerChannelName(config, config.logicAnalyzer.channelRoleMap.trainerPpm)));
fprintf("  Decoded trainer rows: %d\n", height(trainerPpmCapture));
if isempty(trainerPpmCapture)
    fprintf("  Distinct trainer frames: 0\n");
    fprintf("  Pulse width range: NaN to NaN us\n\n");
    return;
end

if ismember("frame_index", string(trainerPpmCapture.Properties.VariableNames))
    validFrameIndex = trainerPpmCapture.frame_index(isfinite(trainerPpmCapture.frame_index));
    trainerFrameCount = numel(unique(validFrameIndex));
else
    trainerFrameCount = 0;
end
minimumPulseUs = min(trainerPpmCapture.pulse_us);
maximumPulseUs = max(trainerPpmCapture.pulse_us);
fprintf("  Distinct trainer frames: %d\n", trainerFrameCount);
fprintf("  Pulse width range: %.1f to %.1f us\n", double(minimumPulseUs), double(maximumPulseUs));
fprintf("  First decoded rows:\n");
previewRowCount = min(4, height(trainerPpmCapture));
for rowIndex = 1:previewRowCount
    if ismember("frame_index", string(trainerPpmCapture.Properties.VariableNames))
        frameIndex = trainerPpmCapture.frame_index(rowIndex);
    else
        frameIndex = NaN;
    end
    fprintf("    %s time=%.6f s pulse=%.1f us frame=%g\n", ...
        char(trainerPpmCapture.surface_name(rowIndex)), ...
        double(trainerPpmCapture.time_s(rowIndex)), ...
        double(trainerPpmCapture.pulse_us(rowIndex)), ...
        double(frameIndex));
end
fprintf("\n");
end

function validateDecodedAnalyzerTables(referenceCapture, trainerPpmCapture, receiverCapture, config)
if ~isSigrokAutoMode(config)
    return;
end
if isempty(referenceCapture)
    error("Transmitter_Test:MissingReferenceCapture", "No valid reference edges were decoded from the configured reference analyser channel.");
end
if isempty(trainerPpmCapture)
    error( ...
        "Transmitter_Test:MissingTrainerPpmCapture", ...
        [ ...
        "No valid trainer PPM pulses were decoded from the configured trainer analyser channel. " + ...
        "This usually means the probe is on the wrong signal, the Uno D3 trainer channel wiring or ground is bad, or the captured waveform is too noisy to match a %d-%d us RC pulse train."], ...
        config.trainerPpm.minimumPulseUs, ...
        config.trainerPpm.maximumPulseUs);
end
minimumTrainerRows = 3 .* numel(config.surfaceNames);
if height(trainerPpmCapture) < minimumTrainerRows
    error( ...
        "Transmitter_Test:InsufficientTrainerPpmCapture", ...
        [ ...
        "Trainer PPM decode produced only %d rows, which is too few for a valid %d-surface Uno trainer capture. " + ...
        "Check the analyser probe mapped to TRAINER_PPM_D3 and verify it is actually connected to Uno D3."], ...
        height(trainerPpmCapture), ...
        numel(config.surfaceNames));
end
trainerFrameCount = countTrainerCaptureFrames(trainerPpmCapture);
if trainerFrameCount < 3
    error( ...
        "Transmitter_Test:InsufficientTrainerPpmFrames", ...
        [ ...
        "Trainer PPM decode produced only %d distinct frame(s), which is not credible for the automated run. " + ...
        "This indicates the configured TRAINER_PPM_D3 analyser channel is still not a valid Uno trainer waveform."], ...
        trainerFrameCount);
end
end

function validPulseMask = isPlausibleRcPulseWidthUs(pulseWidthsUs, config)
minimumPulseUs = double(config.trainerPpm.minimumPulseUs) - 150;
maximumPulseUs = double(config.trainerPpm.maximumPulseUs) + 150;
validPulseMask = isfinite(pulseWidthsUs) & ...
    pulseWidthsUs >= minimumPulseUs & ...
    pulseWidthsUs <= maximumPulseUs;
end

function trainerFrameCount = countTrainerCaptureFrames(trainerPpmCapture)
trainerFrameCount = 0;
if ~ismember("frame_index", string(trainerPpmCapture.Properties.VariableNames))
    return;
end

frameIndex = double(trainerPpmCapture.frame_index);
frameIndex = frameIndex(isfinite(frameIndex));
if isempty(frameIndex)
    return;
end

trainerFrameCount = numel(unique(frameIndex));
end

function configuredName = resolveConfiguredLogicAnalyzerChannelName(config, roleChannel)
configurationIndex = find(double(config.logicAnalyzer.channels) == double(roleChannel), 1, "first");
if isempty(configurationIndex)
    configuredName = "";
    return;
end
configuredName = string(config.logicAnalyzer.channelNames(configurationIndex));
end

function valueText = ternaryText(conditionValue, trueText, falseText)
if conditionValue
    valueText = trueText;
else
    valueText = falseText;
end
end

function cleanupSigrokSession(sigrokSession)
if ~isstruct(sigrokSession) || ~isfield(sigrokSession, "process")
    return;
end
if isempty(sigrokSession.process)
    return;
end
try
    if ~sigrokSession.process.HasExited
        sigrokSession.process.Kill();
    end
catch
end
end

function hostDispatchLog = buildHostDispatchLogFromSession(loggerSession)
hostDispatchLog = table( ...
    loggerSession.dispatchSampleIndex(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchSurfaceName(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchCommandSequence(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchSampleSequence(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchScheduledTimeSeconds(1:loggerSession.dispatchCount), ...
    loggerSession.dispatchCommandUs(1:loggerSession.dispatchCount), ...
    (loggerSession.dispatchCommandUs(1:loggerSession.dispatchCount) - double(loggerSession.testStartOffsetUs)) ./ 1e6, ...
    loggerSession.dispatchPosition(1:loggerSession.dispatchCount), ...
    'VariableNames', { ...
        'sample_index', ...
        'surface_name', ...
        'command_sequence', ...
        'sample_sequence', ...
        'scheduled_time_s', ...
        'command_dispatch_us', ...
        'command_dispatch_s', ...
        'position_norm'});
end

function boardRxLog = buildBoardRxLogFromSession(loggerSession, surfaceNames)
if loggerSession.rxEventCount <= 0
    boardRxLog = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        strings(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'active_surface_mask', ...
            'surface_name', ...
            'command_sequence', ...
            'host_rx_us', ...
            'rx_us', ...
            'received_position_code', ...
            'received_position_norm'});
    return;
end

surfaceCount = numel(surfaceNames);
rowCapacity = max(1, loggerSession.rxEventCount .* surfaceCount);
sampleSequence = nan(rowCapacity, 1);
activeSurfaceMask = nan(rowCapacity, 1);
surfaceName = strings(rowCapacity, 1);
commandSequence = nan(rowCapacity, 1);
hostRxUs = nan(rowCapacity, 1);
rxUs = nan(rowCapacity, 1);
receivedPositionCode = nan(rowCapacity, 1);
receivedPositionNorm = nan(rowCapacity, 1);
rowIndex = 0;

for rxIndex = 1:loggerSession.rxEventCount
    activeMaskValue = uint8(max(0, round(loggerSession.rxActiveMask(rxIndex))));
    for surfaceIndex = 1:surfaceCount
        if ~bitget(activeMaskValue, surfaceIndex)
            continue;
        end

        rowIndex = rowIndex + 1;
        sampleSequence(rowIndex) = loggerSession.rxSampleSequence(rxIndex);
        activeSurfaceMask(rowIndex) = double(activeMaskValue);
        surfaceName(rowIndex) = surfaceNames(surfaceIndex);
        commandSequence(rowIndex) = loggerSession.rxSampleSequence(rxIndex);
        hostRxUs(rowIndex) = loggerSession.rxHostRxUs(rxIndex);
        rxUs(rowIndex) = loggerSession.rxBoardRxUs(rxIndex);
        receivedPositionCode(rowIndex) = loggerSession.rxPositionCode(rxIndex, surfaceIndex);
        receivedPositionNorm(rowIndex) = receivedPositionCode(rowIndex) ./ 65535.0;
    end
end

boardRxLog = table( ...
    sampleSequence(1:rowIndex), ...
    activeSurfaceMask(1:rowIndex), ...
    surfaceName(1:rowIndex), ...
    commandSequence(1:rowIndex), ...
    hostRxUs(1:rowIndex), ...
    rxUs(1:rowIndex), ...
    receivedPositionCode(1:rowIndex), ...
    receivedPositionNorm(1:rowIndex), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'active_surface_mask', ...
        'surface_name', ...
        'command_sequence', ...
        'host_rx_us', ...
        'rx_us', ...
        'received_position_code', ...
        'received_position_norm'});
end

function config = adjustSigrokSampleRateForCaptureDuration(config, captureDurationSeconds)
if ~isfinite(captureDurationSeconds) || captureDurationSeconds <= 0
    return;
end

requestedSampleRateHz = double(config.logicAnalyzer.sampleRateHz);
maximumCaptureSamples = double(config.logicAnalyzer.maximumCaptureSamples);
if ~isfinite(maximumCaptureSamples) || maximumCaptureSamples <= 0
    return;
end

requiredSamples = requestedSampleRateHz .* captureDurationSeconds;
if requiredSamples <= maximumCaptureSamples
    return;
end

maximumAllowedSampleRateHz = floor(maximumCaptureSamples ./ captureDurationSeconds);
supportedSampleRatesHz = [ ...
    250000, ...
    500000, ...
    1000000, ...
    2000000, ...
    4000000, ...
    8000000, ...
    12000000, ...
    16000000, ...
    24000000];
candidateRatesHz = supportedSampleRatesHz(supportedSampleRatesHz <= maximumAllowedSampleRateHz);
if isempty(candidateRatesHz)
    adjustedSampleRateHz = supportedSampleRatesHz(1);
else
    adjustedSampleRateHz = candidateRatesHz(end);
end

if adjustedSampleRateHz < requestedSampleRateHz
    fprintf([ ...
        "Sigrok sample-rate auto-adjustment\n" + ...
        "  requested: %.0f Hz\n" + ...
        "  adjusted: %.0f Hz\n" + ...
        "  capture duration: %.3f s\n" + ...
        "  max capture samples: %.0f\n"], ...
        requestedSampleRateHz, ...
        adjustedSampleRateHz, ...
        captureDurationSeconds, ...
        maximumCaptureSamples);
    config.logicAnalyzer.sampleRateHz = adjustedSampleRateHz;
end
end

function boardCommitLog = buildBoardCommitLogFromSession(loggerSession)
if loggerSession.commitEventCount <= 0
    boardCommitLog = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'active_surface_mask', ...
            'host_rx_us', ...
            'board_rx_us', ...
            'board_commit_us', ...
            'receive_to_commit_us', ...
            'strobe_us', ...
            'frame_index', ...
            'ppm_ch1_us', ...
            'ppm_ch2_us', ...
            'ppm_ch3_us', ...
            'ppm_ch4_us', ...
            'ppm_ch5_us', ...
            'ppm_ch6_us', ...
            'ppm_ch7_us', ...
            'ppm_ch8_us'});
    return;
end

receiveToCommitUs = loggerSession.commitReceiveToCommitUs(1:loggerSession.commitEventCount);
missingMask = ~isfinite(receiveToCommitUs);
if any(missingMask)
    receiveToCommitUs(missingMask) = ...
        loggerSession.commitBoardCommitUs(missingMask) - loggerSession.commitBoardRxUs(missingMask);
end

boardCommitLog = table( ...
    loggerSession.commitSampleSequence(1:loggerSession.commitEventCount), ...
    loggerSession.commitActiveMask(1:loggerSession.commitEventCount), ...
    loggerSession.commitHostRxUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitBoardRxUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitBoardCommitUs(1:loggerSession.commitEventCount), ...
    receiveToCommitUs, ...
    loggerSession.commitStrobeUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitFrameIndex(1:loggerSession.commitEventCount), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 1), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 2), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 3), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 4), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 5), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 6), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 7), ...
    loggerSession.commitPpmUs(1:loggerSession.commitEventCount, 8), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'active_surface_mask', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_commit_us', ...
        'receive_to_commit_us', ...
        'strobe_us', ...
        'frame_index', ...
        'ppm_ch1_us', ...
        'ppm_ch2_us', ...
        'ppm_ch3_us', ...
        'ppm_ch4_us', ...
        'ppm_ch5_us', ...
        'ppm_ch6_us', ...
        'ppm_ch7_us', ...
        'ppm_ch8_us'});
end

function boardSyncLog = buildBoardSyncLogFromSession(loggerSession)
boardSyncLog = table( ...
    loggerSession.boardSyncId(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncHostTxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncHostRxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardRxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardTxUs(1:loggerSession.boardSyncCount), ...
    loggerSession.boardSyncBoardTxUs(1:loggerSession.boardSyncCount) - loggerSession.boardSyncBoardRxUs(1:loggerSession.boardSyncCount), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us', ...
        'board_turnaround_us'});
end

function [referenceCapture, trainerPpmCapture, receiverCapture] = importTransmitterCaptureData(config)
referenceCapture = normalizeReferenceCaptureTable(table(), config.matching.referenceDebounceUs);
trainerPpmCapture = normalizePulseCaptureTable(table(), config);
receiverCapture = normalizePulseCaptureTable(table(), config);

if ~config.logicAnalyzer.enabled
    return;
end

isRequired = isSigrokAutoMode(config);
referenceCapture = normalizeReferenceCaptureTable( ...
    readCaptureTable(config.logicAnalyzer.referenceCapturePath, isRequired), ...
    config.matching.referenceDebounceUs);
trainerPpmCapture = normalizePulseCaptureTable( ...
    readCaptureTable(config.logicAnalyzer.trainerPpmCapturePath, isRequired), ...
    config);
receiverCapture = normalizePulseCaptureTable( ...
    readCaptureTable(config.logicAnalyzer.receiverCapturePath, isRequired), ...
    config);
end

function captureTable = readCaptureTable(filePath, isRequired)
captureTable = table();
if strlength(filePath) == 0 || ~isfile(filePath)
    if isRequired
        error("Transmitter_Test:MissingCaptureTable", "Capture file was not found: %s", char(filePath));
    end
    return;
end
captureTable = readCommentCsv(filePath);
end

function referenceCapture = normalizeReferenceCaptureTable(rawTable, debounceUs)
referenceCapture = table( ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'});
if isempty(rawTable)
    return;
end

timeColumn = resolveTableVariableName(rawTable, "time_s", false);
if strlength(timeColumn) == 0
    timeColumn = resolveTableVariableName(rawTable, "reference_time_s", false);
end
if strlength(timeColumn) == 0
    timeColumn = resolveTableVariableName(rawTable, "host_time_s", false);
end
if strlength(timeColumn) == 0
    return;
end

sampleIndexColumn = resolveTableVariableName(rawTable, "sample_index", false);
sampleRateColumn = resolveTableVariableName(rawTable, "sample_rate_hz", false);
sampleIndex = nan(height(rawTable), 1);
sampleRateHz = nan(height(rawTable), 1);
if strlength(sampleIndexColumn) > 0
    sampleIndex = reshape(double(rawTable.(char(sampleIndexColumn))), [], 1);
end
if strlength(sampleRateColumn) > 0
    sampleRateHz = reshape(double(rawTable.(char(sampleRateColumn))), [], 1);
end

timeSeconds = reshape(double(rawTable.(char(timeColumn))), [], 1);
sampleBasedMask = isfinite(sampleIndex) & isfinite(sampleRateHz) & sampleRateHz > 0;
timeSeconds(sampleBasedMask) = sampleIndex(sampleBasedMask) ./ sampleRateHz(sampleBasedMask);

validMask = isfinite(timeSeconds);
if ~any(validMask)
    return;
end

referenceMatrix = [timeSeconds(validMask), sampleIndex(validMask), sampleRateHz(validMask)];
referenceMatrix = sortrows(referenceMatrix, 1);

keepMask = false(size(referenceMatrix, 1), 1);
keepMask(1) = true;
minimumSpacingSeconds = double(debounceUs) ./ 1e6;
lastKeptTimeSeconds = referenceMatrix(1, 1);
for rowIndex = 2:size(referenceMatrix, 1)
    if (referenceMatrix(rowIndex, 1) - lastKeptTimeSeconds) >= minimumSpacingSeconds
        keepMask(rowIndex) = true;
        lastKeptTimeSeconds = referenceMatrix(rowIndex, 1);
    end
end

referenceCapture = table( ...
    referenceMatrix(keepMask, 1), ...
    referenceMatrix(keepMask, 2), ...
    referenceMatrix(keepMask, 3), ...
    'VariableNames', {'time_s', 'sample_index', 'sample_rate_hz'});
end

function pulseCapture = normalizePulseCaptureTable(rawTable, config)
pulseCapture = buildEmptyPulseCaptureTable(false);
if isempty(rawTable)
    return;
end

timeColumn = resolveTableVariableName(rawTable, "time_s", false);
if strlength(timeColumn) == 0
    timeColumn = resolveTableVariableName(rawTable, "host_time_s", false);
end

pulseColumn = resolveTableVariableName(rawTable, "pulse_us", false);
if strlength(pulseColumn) == 0
    pulseColumn = resolveTableVariableName(rawTable, "pulse_width_us", false);
end
if strlength(pulseColumn) == 0
    pulseColumn = resolveTableVariableName(rawTable, "width_us", false);
end

surfaceColumn = resolveTableVariableName(rawTable, "surface_name", false);
if strlength(surfaceColumn) == 0
    surfaceColumn = resolveTableVariableName(rawTable, "channel_name", false);
end

channelColumn = resolveTableVariableName(rawTable, "channel_index", false);
if strlength(channelColumn) == 0
    channelColumn = resolveTableVariableName(rawTable, "trainer_channel", false);
end
if strlength(channelColumn) == 0
    channelColumn = resolveTableVariableName(rawTable, "receiver_channel", false);
end

if strlength(timeColumn) == 0 || strlength(pulseColumn) == 0
    sampleIndexColumn = resolveTableVariableName(rawTable, "sample_index", false);
    sampleCountColumn = resolveTableVariableName(rawTable, "sample_count", false);
    sampleRateColumn = resolveTableVariableName(rawTable, "sample_rate_hz", false);
    if strlength(sampleIndexColumn) == 0 || strlength(sampleCountColumn) == 0 || strlength(sampleRateColumn) == 0
        return;
    end
end

rowCount = height(rawTable);
timeSeconds = nan(rowCount, 1);
if strlength(timeColumn) > 0
    timeSeconds = reshape(double(rawTable.(char(timeColumn))), [], 1);
end
pulseUs = nan(rowCount, 1);
if strlength(pulseColumn) > 0
    pulseUs = reshape(double(rawTable.(char(pulseColumn))), [], 1);
end

sampleIndexColumn = resolveTableVariableName(rawTable, "sample_index", false);
sampleCountColumn = resolveTableVariableName(rawTable, "sample_count", false);
sampleRateColumn = resolveTableVariableName(rawTable, "sample_rate_hz", false);
frameIndexColumn = resolveTableVariableName(rawTable, "frame_index", false);
sampleIndex = nan(rowCount, 1);
sampleCount = nan(rowCount, 1);
sampleRateHz = nan(rowCount, 1);
frameIndex = nan(rowCount, 1);
if strlength(sampleIndexColumn) > 0
    sampleIndex = reshape(double(rawTable.(char(sampleIndexColumn))), [], 1);
end
if strlength(sampleCountColumn) > 0
    sampleCount = reshape(double(rawTable.(char(sampleCountColumn))), [], 1);
end
if strlength(sampleRateColumn) > 0
    sampleRateHz = reshape(double(rawTable.(char(sampleRateColumn))), [], 1);
end
if strlength(frameIndexColumn) > 0
    frameIndex = reshape(double(rawTable.(char(frameIndexColumn))), [], 1);
end

sampleBasedTimeMask = isfinite(sampleIndex) & isfinite(sampleRateHz) & sampleRateHz > 0;
timeSeconds(sampleBasedTimeMask) = sampleIndex(sampleBasedTimeMask) ./ sampleRateHz(sampleBasedTimeMask);
sampleBasedPulseMask = isfinite(sampleCount) & isfinite(sampleRateHz) & sampleRateHz > 0;
pulseUs(sampleBasedPulseMask) = 1e6 .* sampleCount(sampleBasedPulseMask) ./ sampleRateHz(sampleBasedPulseMask);

if strlength(surfaceColumn) > 0
    surfaceNames = reshape(string(rawTable.(char(surfaceColumn))), [], 1);
elseif strlength(channelColumn) > 0
    channelIndex = reshape(double(rawTable.(char(channelColumn))), [], 1);
    channelSurfaceMap = config.trainerPpm.channelSurfaceMap;
    surfaceNames = strings(size(channelIndex));
    validChannelMask = ...
        isfinite(channelIndex) & ...
        channelIndex >= 1 & ...
        channelIndex <= numel(channelSurfaceMap);
    validChannelIndex = channelIndex(validChannelMask);
    surfaceNames(validChannelMask) = channelSurfaceMap(validChannelIndex);
else
    return;
end

validMask = ...
    isfinite(timeSeconds) & ...
    isfinite(pulseUs) & ...
    ismember(surfaceNames, config.surfaceNames);
if ~any(validMask)
    return;
end

pulseCapture = table( ...
    surfaceNames(validMask), ...
    timeSeconds(validMask), ...
    pulseUs(validMask), ...
    sampleIndex(validMask), ...
    sampleCount(validMask), ...
    sampleRateHz(validMask), ...
    frameIndex(validMask), ...
    'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz', 'frame_index'});
pulseCapture = sortrows(pulseCapture, {'surface_name', 'time_s'});
end

function storage = finalizeTransmitterStorage(storage, config, loggerData)
storage.rawLogs = loggerData;

boardRxLog = loggerData.boardRxLog;
boardCommitLog = loggerData.boardCommitLog;
boardSyncLog = loggerData.boardSyncLog;
if ~isempty(loggerData.hostDispatchLog) && ismember("surface_name", loggerData.hostDispatchLog.Properties.VariableNames)
    loggerData.hostDispatchLog.surface_name = normalizeSurfaceNameColumn( ...
        loggerData.hostDispatchLog.surface_name, ...
        height(loggerData.hostDispatchLog));
end
if ~isempty(boardRxLog) && ismember("surface_name", boardRxLog.Properties.VariableNames)
    boardRxLog.surface_name = normalizeSurfaceNameColumn( ...
        boardRxLog.surface_name, ...
        height(boardRxLog));
end
[clockSlope, clockIntercept] = estimateBoardToHostClockMap(boardSyncLog(:, {'sync_id', 'host_tx_us', 'host_rx_us', 'board_rx_us', 'board_tx_us'}));

if config.arduinoTransport.rxTimestampSource == "host_rx_us" && ...
        ismember("host_rx_us", string(boardRxLog.Properties.VariableNames))
    rxHostUs = double(boardRxLog.host_rx_us);
else
    rxHostUs = clockSlope .* double(boardRxLog.rx_us) + clockIntercept;
end
commitHostUs = clockSlope .* double(boardCommitLog.board_commit_us) + clockIntercept;
strobeHostUs = nan(height(boardCommitLog), 1);
strobeMask = isfinite(boardCommitLog.strobe_us) & boardCommitLog.strobe_us > 0;
strobeHostUs(strobeMask) = clockSlope .* double(boardCommitLog.strobe_us(strobeMask)) + clockIntercept;

boardRxLog.rx_time_s = (rxHostUs - loggerData.testStartOffsetUs) ./ 1e6;
boardCommitLog.commit_time_s = (commitHostUs - loggerData.testStartOffsetUs) ./ 1e6;
boardCommitLog.strobe_time_s = nan(height(boardCommitLog), 1);
boardCommitLog.strobe_time_s(strobeMask) = ...
    (strobeHostUs(strobeMask) - loggerData.testStartOffsetUs) ./ 1e6;

storage.rawLogs.boardRxLog = boardRxLog;
storage.rawLogs.boardCommitLog = boardCommitLog;
storage.rawLogs.boardSyncLog = boardSyncLog;
storage.rawLogs.referenceCapture = loggerData.referenceCapture;
storage.rawLogs.trainerPpmCapture = loggerData.trainerPpmCapture;
storage.rawLogs.receiverCapture = loggerData.receiverCapture;

storage.matchedEvents = buildTransmitterMatchedEvents( ...
    loggerData.hostDispatchLog, ...
    boardRxLog, ...
    boardCommitLog, ...
    loggerData.referenceCapture, ...
    loggerData.trainerPpmCapture, ...
    loggerData.receiverCapture, ...
    config);

for rowIndex = 1:height(storage.matchedEvents)
    sampleIndex = storage.matchedEvents.sample_index(rowIndex);
    surfaceIndex = storage.matchedEvents.surface_index(rowIndex);
    storage.boardRxSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.board_rx_s(rowIndex);
    storage.boardCommitSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.board_commit_s(rowIndex);
    storage.anchorSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.anchor_s(rowIndex);
    storage.referenceStrobeSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.reference_strobe_s(rowIndex);
    storage.trainerPpmSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.trainer_ppm_s(rowIndex);
    storage.receiverResponseSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.receiver_response_s(rowIndex);
    storage.anchorToPpmLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.anchor_to_ppm_latency_s(rowIndex);
    storage.anchorToReceiverLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.anchor_to_receiver_latency_s(rowIndex);
    storage.computerToArduinoRxLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.computer_to_arduino_rx_latency_s(rowIndex);
    storage.arduinoReceiveToPpmCommitLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.arduino_receive_to_ppm_commit_latency_s(rowIndex);
    storage.ppmToReceiverLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.ppm_to_receiver_latency_s(rowIndex);
    storage.computerToReceiverLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.computer_to_receiver_latency_s(rowIndex);
    storage.scheduledToReceiverLatencySeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.scheduled_to_receiver_latency_s(rowIndex);
    storage.receivedPositionNorm(sampleIndex, surfaceIndex) = storage.matchedEvents.received_position_norm(rowIndex);
    if isfinite(storage.matchedEvents.expected_ppm_us(rowIndex))
        storage.committedPpmUs(sampleIndex, surfaceIndex) = storage.matchedEvents.expected_ppm_us(rowIndex);
    end
end

for sampleIndex = 1:storage.sampleCount
    candidateStart = [storage.boardRxSeconds(sampleIndex, :), storage.boardCommitSeconds(sampleIndex, :)];
    candidateStop = [ ...
        candidateStart, ...
        storage.anchorSeconds(sampleIndex, :), ...
        storage.referenceStrobeSeconds(sampleIndex, :), ...
        storage.receiverResponseSeconds(sampleIndex, :)];
    if any(isfinite(candidateStart))
        storage.boardReadStartSeconds(sampleIndex) = min(candidateStart, [], "omitnan");
    end
    if any(isfinite(candidateStop))
        storage.boardReadStopSeconds(sampleIndex) = max(candidateStop, [], "omitnan");
    end
end

storage.integritySummary = buildTransmitterIntegritySummary(storage, config);
end

function surfaceNames = normalizeSurfaceNameColumn(surfaceNames, expectedRowCount)
if nargin < 2
    expectedRowCount = size(surfaceNames, 1);
end

if ischar(surfaceNames)
    surfaceNames = string(cellstr(surfaceNames));
    return;
end

if iscell(surfaceNames)
    if isvector(surfaceNames)
        surfaceNames = string(surfaceNames(:));
    else
        surfaceNames = join(string(surfaceNames), "", 2);
    end
elseif isstring(surfaceNames)
    if isvector(surfaceNames)
        surfaceNames = surfaceNames(:);
    else
        surfaceNames = join(surfaceNames, "", 2);
    end
elseif iscategorical(surfaceNames)
    surfaceNames = string(surfaceNames);
    if ~isvector(surfaceNames)
        surfaceNames = join(surfaceNames, "", 2);
    else
        surfaceNames = surfaceNames(:);
    end
else
    surfaceNames = string(surfaceNames);
    if ~isvector(surfaceNames)
        surfaceNames = join(surfaceNames, "", 2);
    else
        surfaceNames = surfaceNames(:);
    end
end

surfaceNames = surfaceNames(:);
if numel(surfaceNames) ~= expectedRowCount
    if expectedRowCount == 0
        surfaceNames = strings(0, 1);
        return;
    end
    if size(surfaceNames, 1) == expectedRowCount
        return;
    end
    error("Transmitter_Test:InvalidSurfaceNameShape", ...
        "Surface-name column could not be normalized to %d rows.", expectedRowCount);
end
end

function matchedEvents = buildTransmitterMatchedEvents( ...
    hostDispatchLog, ...
    boardRxLog, ...
    boardCommitLog, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    receiverCapture, ...
    config)
hostDispatchLog = sortrows(hostDispatchLog, {'sample_index', 'surface_name'});
matchedEvents = hostDispatchLog(:, {'sample_index', 'surface_name', 'command_sequence', 'sample_sequence', 'scheduled_time_s', 'command_dispatch_s', 'position_norm'});
matchedEvents.surface_index = zeros(height(matchedEvents), 1);
for rowIndex = 1:height(matchedEvents)
    matchedEvents.surface_index(rowIndex) = find(config.surfaceNames == matchedEvents.surface_name(rowIndex), 1, "first");
end

matchedEvents.received_position_norm = nan(height(matchedEvents), 1);
matchedEvents.board_rx_s = nan(height(matchedEvents), 1);
matchedEvents.board_commit_s = nan(height(matchedEvents), 1);
matchedEvents.reference_strobe_s = nan(height(matchedEvents), 1);
matchedEvents.anchor_s = nan(height(matchedEvents), 1);
matchedEvents.anchor_source = strings(height(matchedEvents), 1);
matchedEvents.trainer_ppm_s = nan(height(matchedEvents), 1);
matchedEvents.receiver_response_s = nan(height(matchedEvents), 1);
matchedEvents.reference_sample_index = nan(height(matchedEvents), 1);
matchedEvents.trainer_sample_index = nan(height(matchedEvents), 1);
matchedEvents.receiver_sample_index = nan(height(matchedEvents), 1);
matchedEvents.reference_time_from_samples_s = nan(height(matchedEvents), 1);
matchedEvents.trainer_time_from_samples_s = nan(height(matchedEvents), 1);
matchedEvents.receiver_time_from_samples_s = nan(height(matchedEvents), 1);
matchedEvents.analyzer_sample_rate_hz = nan(height(matchedEvents), 1);
matchedEvents.expected_ppm_us = nan(height(matchedEvents), 1);
matchedEvents.trainer_ppm_us = nan(height(matchedEvents), 1);
matchedEvents.receiver_pulse_us = nan(height(matchedEvents), 1);
matchedEvents.anchor_to_ppm_latency_s = nan(height(matchedEvents), 1);
matchedEvents.anchor_to_receiver_latency_s = nan(height(matchedEvents), 1);
matchedEvents.computer_to_arduino_rx_latency_s = nan(height(matchedEvents), 1);
matchedEvents.arduino_receive_to_ppm_commit_latency_s = nan(height(matchedEvents), 1);
matchedEvents.ppm_to_receiver_latency_s = nan(height(matchedEvents), 1);
matchedEvents.computer_to_receiver_latency_s = nan(height(matchedEvents), 1);
matchedEvents.scheduled_to_receiver_latency_s = nan(height(matchedEvents), 1);
matchedEvents.dropped_before_commit = false(height(matchedEvents), 1);
matchedEvents.is_observable_downstream = false(height(matchedEvents), 1);
matchedEvents.used_sample_based_timing = false(height(matchedEvents), 1);
matchedEvents.capture_source = repmat(config.logicAnalyzer.mode, height(matchedEvents), 1);

rxUnique = deduplicateEchoImportTable(boardRxLog(:, {'surface_name', 'command_sequence', 'received_position_norm', 'rx_time_s'}));
hostKeys = buildSurfaceSequenceKeys(matchedEvents.surface_name, matchedEvents.command_sequence);
rxKeys = buildSurfaceSequenceKeys(rxUnique.surface_name, rxUnique.command_sequence);
[isRxMatched, rxIndex] = ismember(hostKeys, rxKeys);
matchedEvents.received_position_norm(isRxMatched) = rxUnique.received_position_norm(rxIndex(isRxMatched));
matchedEvents.board_rx_s(isRxMatched) = rxUnique.rx_time_s(rxIndex(isRxMatched));
matchedEvents.computer_to_arduino_rx_latency_s(isRxMatched) = ...
    matchedEvents.board_rx_s(isRxMatched) - matchedEvents.command_dispatch_s(isRxMatched);

commitUnique = deduplicateCommitTable(boardCommitLog);
[commitUnique.reference_strobe_s, commitUnique.reference_sample_index, commitUnique.reference_sample_rate_hz] = matchReferenceCaptureTimes( ...
    commitUnique, ...
    referenceCapture, ...
    config.referenceStrobe.mode, ...
    config.matching.referenceAssociationWindowSeconds);
if ~isempty(commitUnique)
    [isCommitMatched, commitIndex] = ismember(double(matchedEvents.sample_sequence), double(commitUnique.sample_sequence));
    for rowIndex = 1:height(matchedEvents)
        if ~isCommitMatched(rowIndex)
            matchedEvents.dropped_before_commit(rowIndex) = isfinite(matchedEvents.board_rx_s(rowIndex));
            continue;
        end

        commitRow = commitIndex(rowIndex);
        commitTimeSeconds = commitUnique.commit_time_s(commitRow);
        if ~isCommitTimingConsistent( ...
                commitTimeSeconds, ...
                matchedEvents.command_dispatch_s(rowIndex), ...
                config.matching.maxCommitAssociationSeconds)
            matchedEvents.dropped_before_commit(rowIndex) = isfinite(matchedEvents.board_rx_s(rowIndex));
            continue;
        end

        matchedEvents.board_commit_s(rowIndex) = commitTimeSeconds;
        matchedEvents.reference_strobe_s(rowIndex) = commitUnique.reference_strobe_s(commitRow);
        matchedEvents.reference_sample_index(rowIndex) = commitUnique.reference_sample_index(commitRow);
        matchedEvents.analyzer_sample_rate_hz(rowIndex) = commitUnique.reference_sample_rate_hz(commitRow);
        if isfinite(commitUnique.reference_sample_index(commitRow)) && isfinite(commitUnique.reference_sample_rate_hz(commitRow))
            matchedEvents.reference_time_from_samples_s(rowIndex) = ...
                commitUnique.reference_sample_index(commitRow) ./ commitUnique.reference_sample_rate_hz(commitRow);
        end
        matchedEvents.expected_ppm_us(rowIndex) = getCommitPulseForSurface(commitUnique(commitRow, :), matchedEvents.surface_index(rowIndex));
        if isfinite(matchedEvents.board_rx_s(rowIndex))
            matchedEvents.arduino_receive_to_ppm_commit_latency_s(rowIndex) = ...
                matchedEvents.board_commit_s(rowIndex) - matchedEvents.board_rx_s(rowIndex);
        end
    end
end

matchedEvents = matchSharedClockMinimalLatency( ...
    matchedEvents, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    receiverCapture, ...
    config);
end

function isConsistent = isCommitTimingConsistent( ...
    commitTimeSeconds, ...
    dispatchTimeSeconds, ...
    maxCommitAssociationSeconds)
isConsistent = isfinite(commitTimeSeconds);
if ~isConsistent
    return;
end

if ~isfinite(maxCommitAssociationSeconds) || maxCommitAssociationSeconds <= 0
    maxCommitAssociationSeconds = 0.25;
end

if isfinite(dispatchTimeSeconds)
    dispatchToCommitSeconds = commitTimeSeconds - dispatchTimeSeconds;
    if dispatchToCommitSeconds < -0.005 || dispatchToCommitSeconds > maxCommitAssociationSeconds
        isConsistent = false;
        return;
    end
end
end

function commitTable = deduplicateCommitTable(commitTable)
if isempty(commitTable)
    return;
end
[~, firstIndex] = unique(double(commitTable.sample_sequence), "stable");
commitTable = commitTable(sort(firstIndex), :);
end

function matchedEvents = matchSharedClockMinimalLatency( ...
    matchedEvents, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    receiverCapture, ...
    config)
frameWindowSeconds = double(config.matching.downstreamFrameWindowSeconds);
if ~isfinite(frameWindowSeconds) || frameWindowSeconds <= 0
    frameWindowSeconds = max(0.02, double(config.trainerPpm.frameLengthUs) ./ 1e6);
end

anchorTable = buildSharedClockAnchorTable( ...
    matchedEvents, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    frameWindowSeconds);
[isAnchorMatched, anchorIndex] = ismember( ...
    double(matchedEvents.sample_sequence), ...
    double(anchorTable.sample_sequence));
for rowIndex = 1:height(matchedEvents)
    if ~isAnchorMatched(rowIndex)
        continue;
    end
    anchorRow = anchorIndex(rowIndex);
    matchedEvents.anchor_s(rowIndex) = anchorTable.anchor_s(anchorRow);
    matchedEvents.anchor_source(rowIndex) = anchorTable.anchor_source(anchorRow);
    if matchedEvents.anchor_source(rowIndex) == "D4"
        matchedEvents.reference_strobe_s(rowIndex) = anchorTable.anchor_s(anchorRow);
        matchedEvents.reference_sample_index(rowIndex) = anchorTable.anchor_sample_index(anchorRow);
        if isfinite(anchorTable.anchor_sample_index(anchorRow)) && isfinite(anchorTable.anchor_sample_rate_hz(anchorRow))
            matchedEvents.reference_time_from_samples_s(rowIndex) = ...
                anchorTable.anchor_sample_index(anchorRow) ./ anchorTable.anchor_sample_rate_hz(anchorRow);
        end
    elseif matchedEvents.anchor_source(rowIndex) == "D5"
        matchedEvents.reference_strobe_s(rowIndex) = NaN;
    end
    if isfinite(anchorTable.anchor_sample_rate_hz(anchorRow))
        matchedEvents.analyzer_sample_rate_hz(rowIndex) = anchorTable.anchor_sample_rate_hz(anchorRow);
    end
end

surfaceCount = numel(config.surfaceNames);
trainerBySurface = cell(surfaceCount, 1);
receiverBySurface = cell(surfaceCount, 1);
for surfaceIndex = 1:surfaceCount
    trainerRows = trainerPpmCapture(trainerPpmCapture.surface_name == config.surfaceNames(surfaceIndex), :);
    receiverRows = receiverCapture(receiverCapture.surface_name == config.surfaceNames(surfaceIndex), :);
    trainerBySurface{surfaceIndex} = sortrows(trainerRows, "time_s");
    receiverBySurface{surfaceIndex} = sortrows(receiverRows, "time_s");
end
trainerNextIndex = ones(surfaceCount, 1);
receiverNextIndex = ones(surfaceCount, 1);
previousExpectedUs = double(config.trainerPpm.neutralPulseUs) .* ones(surfaceCount, 1);

for rowIndex = 1:height(matchedEvents)
    surfaceIdx = matchedEvents.surface_index(rowIndex);
    if ~isfinite(surfaceIdx) || surfaceIdx < 1 || surfaceIdx > surfaceCount
        continue;
    end

    expectedUs = matchedEvents.expected_ppm_us(rowIndex);
    anchorTime = matchedEvents.anchor_s(rowIndex);
    if ~isfinite(expectedUs)
        continue;
    end

    matchedEvents.is_observable_downstream(rowIndex) = ...
        abs(expectedUs - previousExpectedUs(surfaceIdx)) >= double(config.matching.ppmChangeThresholdUs);
    if ~isfinite(anchorTime)
        previousExpectedUs(surfaceIdx) = expectedUs;
        continue;
    end

    [trainerTime, trainerPulse, trainerSampleIndex, trainerSampleRateHz, trainerNextIndex(surfaceIdx)] = ...
        findFirstPulseCaptureEventAfterAnchor( ...
            trainerBySurface{surfaceIdx}, ...
            trainerNextIndex(surfaceIdx), ...
            anchorTime, ...
            frameWindowSeconds);
    matchedEvents.trainer_ppm_s(rowIndex) = trainerTime;
    matchedEvents.trainer_ppm_us(rowIndex) = trainerPulse;
    matchedEvents.trainer_sample_index(rowIndex) = trainerSampleIndex;
    if isfinite(trainerSampleRateHz)
        matchedEvents.analyzer_sample_rate_hz(rowIndex) = trainerSampleRateHz;
    end
    if isfinite(trainerSampleIndex) && isfinite(trainerSampleRateHz)
        matchedEvents.trainer_time_from_samples_s(rowIndex) = trainerSampleIndex ./ trainerSampleRateHz;
    end

    [receiverTime, receiverPulse, receiverSampleIndex, receiverSampleRateHz, receiverNextIndex(surfaceIdx)] = ...
        findFirstPulseCaptureEventAfterAnchor( ...
            receiverBySurface{surfaceIdx}, ...
            receiverNextIndex(surfaceIdx), ...
            anchorTime, ...
            frameWindowSeconds);
    matchedEvents.receiver_response_s(rowIndex) = receiverTime;
    matchedEvents.receiver_pulse_us(rowIndex) = receiverPulse;
    matchedEvents.receiver_sample_index(rowIndex) = receiverSampleIndex;
    if isfinite(receiverSampleRateHz)
        matchedEvents.analyzer_sample_rate_hz(rowIndex) = receiverSampleRateHz;
    end
    if isfinite(receiverSampleIndex) && isfinite(receiverSampleRateHz)
        matchedEvents.receiver_time_from_samples_s(rowIndex) = receiverSampleIndex ./ receiverSampleRateHz;
    end

    if isfinite(trainerTime)
        matchedEvents.anchor_to_ppm_latency_s(rowIndex) = trainerTime - anchorTime;
    end
    if isfinite(receiverTime)
        matchedEvents.anchor_to_receiver_latency_s(rowIndex) = receiverTime - anchorTime;
        matchedEvents.computer_to_receiver_latency_s(rowIndex) = ...
            receiverTime - matchedEvents.command_dispatch_s(rowIndex);
        matchedEvents.scheduled_to_receiver_latency_s(rowIndex) = ...
            receiverTime - matchedEvents.scheduled_time_s(rowIndex);
    end
    if isfinite(trainerTime) && isfinite(receiverTime)
        matchedEvents.ppm_to_receiver_latency_s(rowIndex) = receiverTime - trainerTime;
    end

    matchedEvents.used_sample_based_timing(rowIndex) = ...
        isfinite(matchedEvents.reference_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.trainer_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.receiver_sample_index(rowIndex));
    previousExpectedUs(surfaceIdx) = expectedUs;
end
end

function anchorTable = buildSharedClockAnchorTable( ...
    matchedEvents, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    frameWindowSeconds)
if isempty(matchedEvents)
    anchorTable = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        strings(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'anchor_s', ...
            'anchor_source', ...
            'anchor_sample_index', ...
            'anchor_sample_rate_hz'});
    return;
end

[~, firstIndex] = unique(double(matchedEvents.sample_sequence), "stable");
sequenceRows = matchedEvents(sort(firstIndex), :);
sequenceCount = height(sequenceRows);
anchorTable = table( ...
    double(sequenceRows.sample_sequence), ...
    nan(sequenceCount, 1), ...
    strings(sequenceCount, 1), ...
    nan(sequenceCount, 1), ...
    nan(sequenceCount, 1), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'anchor_s', ...
        'anchor_source', ...
        'anchor_sample_index', ...
        'anchor_sample_rate_hz'});

referenceTimes = reshape(double(referenceCapture.time_s), [], 1);
referenceSampleIndex = reshape(double(referenceCapture.sample_index), [], 1);
referenceSampleRateHz = reshape(double(referenceCapture.sample_rate_hz), [], 1);
referenceCursor = 1;

trainerGlobal = trainerPpmCapture(:, {'time_s', 'sample_index', 'sample_rate_hz'});
trainerGlobal = trainerGlobal(isfinite(trainerGlobal.time_s), :);
trainerGlobal = sortrows(trainerGlobal, "time_s");
trainerTimes = reshape(double(trainerGlobal.time_s), [], 1);
trainerSampleIndex = reshape(double(trainerGlobal.sample_index), [], 1);
trainerSampleRateHz = reshape(double(trainerGlobal.sample_rate_hz), [], 1);
trainerCursor = 1;

for rowIndex = 1:sequenceCount
    commitTime = double(sequenceRows.board_commit_s(rowIndex));
    if ~isfinite(commitTime)
        continue;
    end

    [referenceEventIndex, referenceCursor] = findNearestSortedEventIndex( ...
        referenceTimes, ...
        referenceCursor, ...
        commitTime, ...
        frameWindowSeconds);
    if isfinite(referenceEventIndex)
        anchorTable.anchor_s(rowIndex) = referenceTimes(referenceEventIndex);
        anchorTable.anchor_source(rowIndex) = "D4";
        anchorTable.anchor_sample_index(rowIndex) = referenceSampleIndex(referenceEventIndex);
        anchorTable.anchor_sample_rate_hz(rowIndex) = referenceSampleRateHz(referenceEventIndex);
        continue;
    end

    [trainerEventIndex, trainerCursor] = findNearestSortedEventIndex( ...
        trainerTimes, ...
        trainerCursor, ...
        commitTime, ...
        frameWindowSeconds);
    if isfinite(trainerEventIndex)
        anchorTable.anchor_s(rowIndex) = trainerTimes(trainerEventIndex);
        anchorTable.anchor_source(rowIndex) = "D5";
        anchorTable.anchor_sample_index(rowIndex) = trainerSampleIndex(trainerEventIndex);
        anchorTable.anchor_sample_rate_hz(rowIndex) = trainerSampleRateHz(trainerEventIndex);
    end
end
end

function [eventIndex, updatedCursor] = findNearestSortedEventIndex( ...
    eventTimes, ...
    cursorIndex, ...
    targetTimeSeconds, ...
    maxDistanceSeconds)
eventIndex = NaN;
updatedCursor = max(1, cursorIndex);
eventCount = numel(eventTimes);
if eventCount == 0 || ~isfinite(targetTimeSeconds)
    return;
end

while updatedCursor <= eventCount && eventTimes(updatedCursor) < targetTimeSeconds
    updatedCursor = updatedCursor + 1;
end

previousIndex = updatedCursor - 1;
candidateIndex = NaN;
candidateDistance = Inf;
if previousIndex >= 1 && previousIndex <= eventCount
    previousDistance = abs(eventTimes(previousIndex) - targetTimeSeconds);
    if previousDistance < candidateDistance
        candidateIndex = previousIndex;
        candidateDistance = previousDistance;
    end
end
if updatedCursor >= 1 && updatedCursor <= eventCount
    nextDistance = abs(eventTimes(updatedCursor) - targetTimeSeconds);
    if nextDistance < candidateDistance
        candidateIndex = updatedCursor;
        candidateDistance = nextDistance;
    end
end

if isfinite(candidateIndex) && candidateDistance <= maxDistanceSeconds
    eventIndex = candidateIndex;
end
end

function [eventTime, eventPulseUs, eventSampleIndex, eventSampleRateHz, nextIndex] = ...
    findFirstPulseCaptureEventAfterAnchor( ...
    pulseCapture, ...
    startIndex, ...
    anchorTimeSeconds, ...
    maxWindowSeconds)
eventTime = NaN;
eventPulseUs = NaN;
eventSampleIndex = NaN;
eventSampleRateHz = NaN;
nextIndex = max(1, startIndex);
if isempty(pulseCapture) || ~isfinite(anchorTimeSeconds)
    return;
end

rowIndex = nextIndex;
windowEnd = anchorTimeSeconds + maxWindowSeconds;
while rowIndex <= height(pulseCapture) && pulseCapture.time_s(rowIndex) < anchorTimeSeconds
    rowIndex = rowIndex + 1;
end
if rowIndex > height(pulseCapture)
    nextIndex = rowIndex;
    return;
end
if pulseCapture.time_s(rowIndex) > windowEnd
    nextIndex = rowIndex;
    return;
end

eventTime = double(pulseCapture.time_s(rowIndex));
eventPulseUs = double(pulseCapture.pulse_us(rowIndex));
eventSampleIndex = double(pulseCapture.sample_index(rowIndex));
eventSampleRateHz = double(pulseCapture.sample_rate_hz(rowIndex));
nextIndex = rowIndex + 1;
end

function [referenceTime, referenceSampleIndex, referenceSampleRateHz] = matchReferenceCaptureTimes(commitTable, referenceCapture, referenceStrobeMode, referenceAssociationWindowSeconds)
referenceTime = selectBoardReferenceTimes(commitTable);
referenceSampleIndex = nan(height(commitTable), 1);
referenceSampleRateHz = nan(height(commitTable), 1);
if isempty(commitTable)
    return;
end
if referenceStrobeMode ~= "commit_only" || isempty(referenceCapture)
    return;
end

referenceIndex = 1;
for commitIndex = 1:height(commitTable)
    searchTime = commitTable.commit_time_s(commitIndex);
    while referenceIndex <= height(referenceCapture) && referenceCapture.time_s(referenceIndex) < searchTime
        referenceIndex = referenceIndex + 1;
    end

    previousIndex = referenceIndex - 1;
    candidateIndex = NaN;
    candidateDistance = Inf;

    if previousIndex >= 1 && previousIndex <= height(referenceCapture)
        previousDistance = abs(referenceCapture.time_s(previousIndex) - searchTime);
        if previousDistance < candidateDistance
            candidateIndex = previousIndex;
            candidateDistance = previousDistance;
        end
    end

    if referenceIndex >= 1 && referenceIndex <= height(referenceCapture)
        nextDistance = abs(referenceCapture.time_s(referenceIndex) - searchTime);
        if nextDistance < candidateDistance
            candidateIndex = referenceIndex;
            candidateDistance = nextDistance;
        end
    end

    if ~isfinite(candidateIndex)
        continue;
    end
    if candidateDistance > referenceAssociationWindowSeconds
        continue;
    end

    referenceTime(commitIndex) = referenceCapture.time_s(candidateIndex);
    if ismember("sample_index", string(referenceCapture.Properties.VariableNames))
        referenceSampleIndex(commitIndex) = referenceCapture.sample_index(candidateIndex);
    end
    if ismember("sample_rate_hz", string(referenceCapture.Properties.VariableNames))
        referenceSampleRateHz(commitIndex) = referenceCapture.sample_rate_hz(candidateIndex);
    end
end
end

function referenceTime = selectBoardReferenceTimes(commitTable)
referenceTime = nan(height(commitTable), 1);
if isempty(commitTable)
    return;
end

variableNames = string(commitTable.Properties.VariableNames);
if ismember("strobe_time_s", variableNames)
    referenceTime = commitTable.strobe_time_s;
    if any(isfinite(referenceTime))
        return;
    end
end

if ismember("commit_time_s", variableNames)
    referenceTime = commitTable.commit_time_s;
end
end

function pulseUs = getCommitPulseForSurface(commitRow, surfaceIndex)
pulseVariableName = "ppm_ch" + string(surfaceIndex) + "_us";
pulseUs = double(commitRow.(char(pulseVariableName)));
end

function matchedEvents = matchDownstreamPulseCaptures(matchedEvents, trainerPpmCapture, receiverCapture, config)
stateCentersBySurface = deriveExpectedStateCentersFromMatchedEvents( ...
    matchedEvents, ...
    config.surfaceNames, ...
    config.trainerPpm.neutralPulseUs);
trainerTables = buildStableStateTransitionTablesBySurface( ...
    trainerPpmCapture, ...
    config.surfaceNames, ...
    stateCentersBySurface, ...
    config.matching.stableStatePulseCount);
receiverStateCentersBySurface = derivePulseCaptureStateCenters( ...
    receiverCapture, ...
    config.surfaceNames, ...
    stateCentersBySurface);
receiverTables = buildStableStateTransitionTablesBySurface( ...
    receiverCapture, ...
    config.surfaceNames, ...
    receiverStateCentersBySurface, ...
    config.matching.stableStatePulseCount);
frameWindowSeconds = config.matching.downstreamFrameWindowSeconds;
trainerNextIndex = ones(numel(config.surfaceNames), 1);
receiverNextIndex = ones(numel(config.surfaceNames), 1);
previousExpectedUs = config.trainerPpm.neutralPulseUs .* ones(numel(config.surfaceNames), 1);

for rowIndex = 1:height(matchedEvents)
    surfaceIdx = matchedEvents.surface_index(rowIndex);
    expectedUs = matchedEvents.expected_ppm_us(rowIndex);
    if ~isfinite(matchedEvents.board_commit_s(rowIndex)) || ~isfinite(expectedUs)
        continue;
    end

    matchedEvents.is_observable_downstream(rowIndex) = ...
        abs(expectedUs - previousExpectedUs(surfaceIdx)) >= config.matching.ppmChangeThresholdUs;
    expectedState = classifyPulseState( ...
        expectedUs, ...
        stateCentersBySurface{surfaceIdx}, ...
        config.matching.stateToleranceUs);
    searchAnchorSeconds = selectDownstreamSearchAnchorTime( ...
        matchedEvents.board_commit_s(rowIndex), ...
        matchedEvents.reference_strobe_s(rowIndex));

    if matchedEvents.is_observable_downstream(rowIndex) && ...
            isfinite(expectedState)
        [trainerTime, trainerPulse, trainerSampleIndex, trainerSampleRateHz, trainerNextIndex(surfaceIdx)] = findFirstStableStateTransitionAfterAnchor( ...
            trainerTables{surfaceIdx}, ...
            trainerNextIndex(surfaceIdx), ...
            searchAnchorSeconds, ...
            NaN, ...
            expectedState, ...
            frameWindowSeconds, ...
            0.0);
        matchedEvents.trainer_ppm_s(rowIndex) = trainerTime;
        matchedEvents.trainer_ppm_us(rowIndex) = trainerPulse;
        matchedEvents.trainer_sample_index(rowIndex) = trainerSampleIndex;
        if isfinite(trainerSampleRateHz)
            matchedEvents.analyzer_sample_rate_hz(rowIndex) = trainerSampleRateHz;
        end
        if isfinite(trainerSampleIndex) && isfinite(trainerSampleRateHz)
            matchedEvents.trainer_time_from_samples_s(rowIndex) = trainerSampleIndex ./ trainerSampleRateHz;
        end

        receiverStartSeconds = searchAnchorSeconds;
        if isfinite(trainerTime)
            receiverStartSeconds = trainerTime;
        end
        if config.matching.requireTrainerBeforeReceiver && ~isfinite(trainerTime)
            previousExpectedUs(surfaceIdx) = expectedUs;
            continue;
        end

        receiverExpectedState = classifyPulseState( ...
            expectedUs, ...
            receiverStateCentersBySurface{surfaceIdx}, ...
            inf);
        if ~isfinite(receiverExpectedState)
            previousExpectedUs(surfaceIdx) = expectedUs;
            continue;
        end
        [receiverTime, receiverPulse, receiverSampleIndex, receiverSampleRateHz, receiverNextIndex(surfaceIdx)] = findFirstStableStateTransitionAfterAnchor( ...
            receiverTables{surfaceIdx}, ...
            receiverNextIndex(surfaceIdx), ...
            receiverStartSeconds, ...
            NaN, ...
            receiverExpectedState, ...
            frameWindowSeconds, ...
            0.0);
        matchedEvents.receiver_response_s(rowIndex) = receiverTime;
        matchedEvents.receiver_pulse_us(rowIndex) = receiverPulse;
        matchedEvents.receiver_sample_index(rowIndex) = receiverSampleIndex;
        if isfinite(receiverSampleRateHz)
            matchedEvents.analyzer_sample_rate_hz(rowIndex) = receiverSampleRateHz;
        end
        if isfinite(receiverSampleIndex) && isfinite(receiverSampleRateHz)
            matchedEvents.receiver_time_from_samples_s(rowIndex) = receiverSampleIndex ./ receiverSampleRateHz;
        end
    end

    if isfinite(matchedEvents.trainer_ppm_s(rowIndex)) && isfinite(matchedEvents.receiver_response_s(rowIndex))
        matchedEvents.ppm_to_receiver_latency_s(rowIndex) = ...
            matchedEvents.receiver_response_s(rowIndex) - matchedEvents.trainer_ppm_s(rowIndex);
    end
    if isfinite(matchedEvents.receiver_response_s(rowIndex))
        if config.matching.composeReceiverLatencyFromChain || config.analysis.composeReceiverLatencyFromChain
            commitToReceiverLatencySeconds = computeCommitToReceiverLatencySeconds( ...
                matchedEvents.receiver_response_s(rowIndex), ...
                matchedEvents.reference_strobe_s(rowIndex), ...
                matchedEvents.board_commit_s(rowIndex));
            [computerToReceiverLatencySeconds, scheduledToReceiverLatencySeconds] = ...
                composeReceiverLatenciesFromChain( ...
                    matchedEvents.scheduled_time_s(rowIndex), ...
                    matchedEvents.command_dispatch_s(rowIndex), ...
                    matchedEvents.board_rx_s(rowIndex), ...
                    matchedEvents.board_commit_s(rowIndex), ...
                    commitToReceiverLatencySeconds);
            matchedEvents.computer_to_receiver_latency_s(rowIndex) = computerToReceiverLatencySeconds;
            matchedEvents.scheduled_to_receiver_latency_s(rowIndex) = scheduledToReceiverLatencySeconds;
        else
            matchedEvents.computer_to_receiver_latency_s(rowIndex) = ...
                matchedEvents.receiver_response_s(rowIndex) - matchedEvents.command_dispatch_s(rowIndex);
            matchedEvents.scheduled_to_receiver_latency_s(rowIndex) = ...
                matchedEvents.receiver_response_s(rowIndex) - matchedEvents.scheduled_time_s(rowIndex);
        end
    end
    matchedEvents.used_sample_based_timing(rowIndex) = ...
        isfinite(matchedEvents.reference_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.trainer_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.receiver_sample_index(rowIndex));

    previousExpectedUs(surfaceIdx) = expectedUs;
end
end

function stateCentersBySurface = deriveExpectedStateCentersFromMatchedEvents(matchedEvents, surfaceNames, neutralPulseUs)
stateCentersBySurface = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    surfaceMask = matchedEvents.surface_name == surfaceNames(surfaceIndex);
    pulseValues = unique(double(matchedEvents.expected_ppm_us(surfaceMask)));
    pulseValues = pulseValues(isfinite(pulseValues));
    pulseValues = sort(pulseValues(:));
    neutralUs = double(neutralPulseUs);

    if numel(pulseValues) >= 2
        stateCentersBySurface{surfaceIndex} = reshape(pulseValues, 1, []);
    elseif numel(pulseValues) == 1
        centerUs = pulseValues(1);
        stateCentersBySurface{surfaceIndex} = [centerUs - 250, centerUs, centerUs + 250];
    else
        stateCentersBySurface{surfaceIndex} = [neutralUs - 250, neutralUs, neutralUs + 250];
    end
end
end

function searchStartSeconds = selectDownstreamSearchAnchorTime(boardCommitSeconds, referenceSeconds)
if isfinite(boardCommitSeconds) && isfinite(referenceSeconds)
    searchStartSeconds = max(double(boardCommitSeconds), double(referenceSeconds));
elseif isfinite(referenceSeconds)
    searchStartSeconds = double(referenceSeconds);
elseif isfinite(boardCommitSeconds)
    searchStartSeconds = double(boardCommitSeconds);
else
    searchStartSeconds = NaN;
end
end

function searchStartSeconds = selectDownstreamSearchStartTime(anchorSeconds, estimatedLagSeconds, useEstimatedLagForSearch)
searchStartSeconds = double(anchorSeconds);
if useEstimatedLagForSearch
    searchStartSeconds = applyEstimatedTransitionLag(searchStartSeconds, estimatedLagSeconds);
end
end

function commitToReceiverLatencySeconds = computeCommitToReceiverLatencySeconds( ...
    receiverTimeSeconds, ...
    referenceTimeSeconds, ...
    boardCommitTimeSeconds)
if ~isfinite(receiverTimeSeconds)
    commitToReceiverLatencySeconds = NaN;
    return;
end

if isfinite(referenceTimeSeconds)
    commitToReceiverLatencySeconds = receiverTimeSeconds - referenceTimeSeconds;
elseif isfinite(boardCommitTimeSeconds)
    % Fallback only when reference strobe timing is unavailable.
    commitToReceiverLatencySeconds = receiverTimeSeconds - boardCommitTimeSeconds;
else
    commitToReceiverLatencySeconds = NaN;
end
end

function [computerToReceiverLatencySeconds, scheduledToReceiverLatencySeconds] = composeReceiverLatenciesFromChain( ...
    scheduledTimeSeconds, ...
    dispatchTimeSeconds, ...
    boardRxTimeSeconds, ...
    boardCommitTimeSeconds, ...
    commitToReceiverLatencySeconds)
computerToReceiverLatencySeconds = NaN;
scheduledToReceiverLatencySeconds = NaN;

if ~( ...
        isfinite(dispatchTimeSeconds) && ...
        isfinite(boardRxTimeSeconds) && ...
        isfinite(boardCommitTimeSeconds) && ...
        isfinite(commitToReceiverLatencySeconds))
    return;
end

dispatchToRxLatencySeconds = boardRxTimeSeconds - dispatchTimeSeconds;
rxToCommitLatencySeconds = boardCommitTimeSeconds - boardRxTimeSeconds;
computerToReceiverLatencySeconds = ...
    dispatchToRxLatencySeconds + rxToCommitLatencySeconds + commitToReceiverLatencySeconds;

if isfinite(scheduledTimeSeconds)
    scheduleToDispatchLatencySeconds = dispatchTimeSeconds - scheduledTimeSeconds;
    scheduledToReceiverLatencySeconds = ...
        scheduleToDispatchLatencySeconds + computerToReceiverLatencySeconds;
end
end

function directEvents = buildReferenceAnchoredLatencyEvents( ...
    hostDispatchLog, ...
    boardRxLog, ...
    boardCommitLog, ...
    referenceCapture, ...
    trainerPpmCapture, ...
    receiverCapture, ...
    config)
directEvents = table( ...
    zeros(0, 1), ...
    strings(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    strings(0, 1), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'surface_name', ...
        'surface_index', ...
        'command_sequence', ...
        'scheduled_time_s', ...
        'command_dispatch_s', ...
        'board_rx_s', ...
        'board_commit_s', ...
        'reference_strobe_s', ...
        'anchor_s', ...
        'previous_ppm_us', ...
        'expected_ppm_us', ...
        'delta_ppm_us', ...
        'trainer_transition_s', ...
        'trainer_transition_us', ...
        'receiver_transition_s', ...
        'receiver_transition_us', ...
        'anchor_to_trainer_latency_s', ...
        'anchor_to_receiver_latency_s', ...
        'ppm_to_receiver_latency_s', ...
        'computer_to_receiver_latency_s', ...
        'scheduled_to_receiver_latency_s', ...
        'anchor_source'});
if ~config.matching.enableDownstreamMatching
    return;
end
if isempty(boardCommitLog)
    return;
end

commitUnique = deduplicateCommitTable(boardCommitLog);
[commitUnique.reference_strobe_s, commitUnique.reference_sample_index, commitUnique.reference_sample_rate_hz] = matchReferenceCaptureTimes( ...
    commitUnique, ...
    referenceCapture, ...
    config.referenceStrobe.mode, ...
    config.matching.referenceAssociationWindowSeconds);
if height(commitUnique) < 2
    return;
end

hostDispatchUnique = deduplicateHostDispatchBySampleSequence(hostDispatchLog);
rxUnique = deduplicateEchoImportTable(boardRxLog(:, {'surface_name', 'command_sequence', 'received_position_norm', 'rx_time_s'}));
ppmStateCentersBySurface = deriveSurfaceStateCenters(commitUnique, config.surfaceNames);
receiverStateCentersBySurface = derivePulseCaptureStateCenters(receiverCapture, config.surfaceNames, ppmStateCentersBySurface);
trainerTables = buildStableStateTransitionTablesBySurface( ...
    trainerPpmCapture, ...
    config.surfaceNames, ...
    ppmStateCentersBySurface, ...
    config.matching.stableStatePulseCount);
receiverTables = buildStableStateTransitionTablesBySurface( ...
    receiverCapture, ...
    config.surfaceNames, ...
    receiverStateCentersBySurface, ...
    config.matching.stableStatePulseCount);
commitTransitionTables = buildCommitTransitionTablesBySurface( ...
    commitUnique, ...
    config.surfaceNames, ...
    ppmStateCentersBySurface);
transitionAssociationWindowSeconds = max( ...
    1.0, ...
    config.matching.maxResponseWindowSeconds + 0.05);
trainerLagBySurface = estimateOrderedTransitionLagBySurface( ...
    commitTransitionTables, ...
    trainerTables, ...
    transitionAssociationWindowSeconds);
receiverLagBySurface = estimateOrderedTransitionLagBySurface( ...
    commitTransitionTables, ...
    receiverTables, ...
    transitionAssociationWindowSeconds);
if config.debug.strictDownstreamMatching
    validateEstimatedDownstreamLags(trainerLagBySurface, receiverLagBySurface, config);
end

maxEventCount = max(0, (height(commitUnique) - 1) .* numel(config.surfaceNames));
directEvents = directEvents(false(maxEventCount, 1), :);
eventCount = 0;

for commitIndex = 2:height(commitUnique)
    referenceTime = commitUnique.reference_strobe_s(commitIndex);
    commitTime = commitUnique.commit_time_s(commitIndex);
    if isfinite(referenceTime)
        anchorTime = referenceTime;
        anchorSource = "reference";
    elseif isfinite(commitTime)
        anchorTime = commitTime;
        anchorSource = "board_commit";
    else
        continue;
    end

    for surfaceIndex = 1:numel(config.surfaceNames)
        previousPulseUs = getCommitPulseForSurface(commitUnique(commitIndex - 1, :), surfaceIndex);
        expectedPulseUs = getCommitPulseForSurface(commitUnique(commitIndex, :), surfaceIndex);
        deltaPulseUs = expectedPulseUs - previousPulseUs;
        previousState = classifyPulseState(previousPulseUs, ppmStateCentersBySurface{surfaceIndex}, config.matching.stateToleranceUs);
        expectedState = classifyPulseState(expectedPulseUs, ppmStateCentersBySurface{surfaceIndex}, config.matching.stateToleranceUs);
        thresholdUs = config.matching.receiverChangeThresholdUs;
        if abs(deltaPulseUs) < thresholdUs || previousState == expectedState
            continue;
        end

        eventCount = eventCount + 1;
        directEvents(eventCount, :) = { ...
            NaN, "", NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ""};
        directEvents.sample_sequence(eventCount) = commitUnique.sample_sequence(commitIndex);
        directEvents.surface_name(eventCount) = config.surfaceNames(surfaceIndex);
        directEvents.surface_index(eventCount) = surfaceIndex;
        directEvents.board_commit_s(eventCount) = commitTime;
        directEvents.reference_strobe_s(eventCount) = referenceTime;
        directEvents.anchor_s(eventCount) = anchorTime;
        directEvents.previous_ppm_us(eventCount) = previousPulseUs;
        directEvents.expected_ppm_us(eventCount) = expectedPulseUs;
        directEvents.delta_ppm_us(eventCount) = deltaPulseUs;
        directEvents.anchor_source(eventCount) = anchorSource;

        hostRow = findHostDispatchRow(hostDispatchUnique, config.surfaceNames(surfaceIndex), commitUnique.sample_sequence(commitIndex));
        if hostRow > 0
            directEvents.command_sequence(eventCount) = hostDispatchUnique.command_sequence(hostRow);
            directEvents.scheduled_time_s(eventCount) = hostDispatchUnique.scheduled_time_s(hostRow);
            directEvents.command_dispatch_s(eventCount) = hostDispatchUnique.command_dispatch_s(hostRow);

            rxKey = buildSurfaceSequenceKeys(config.surfaceNames(surfaceIndex), hostDispatchUnique.command_sequence(hostRow));
            rxMatch = buildSurfaceSequenceKeys(rxUnique.surface_name, rxUnique.command_sequence);
            rxRow = find(rxMatch == rxKey, 1, "first");
            if ~isempty(rxRow)
                directEvents.board_rx_s(eventCount) = rxUnique.rx_time_s(rxRow);
            end
        end

        trainerSearchStartSeconds = selectDownstreamSearchStartTime( ...
            anchorTime, ...
            trainerLagBySurface(surfaceIndex), ...
            config.matching.useEstimatedLagForSearch);
        [trainerTime, trainerPulse] = findFirstTransitionToStateAfterAnchor( ...
            trainerTables{surfaceIndex}, ...
            trainerSearchStartSeconds, ...
            previousState, ...
            expectedState, ...
            config.matching.maxResponseWindowSeconds, ...
            0.0);
        if isfinite(trainerTime)
            directEvents.trainer_transition_s(eventCount) = trainerTime;
            directEvents.trainer_transition_us(eventCount) = trainerPulse;
            directEvents.anchor_to_trainer_latency_s(eventCount) = trainerTime - anchorTime;
        end

        % The true end-to-end metric is reference/commit to receiver output.
        % Trainer matching is diagnostic only and must not gate receiver
        % latency extraction because trainer decode can be noisier than the
        % receiver waveform.
        receiverSearchStartSeconds = selectDownstreamSearchStartTime( ...
            anchorTime, ...
            receiverLagBySurface(surfaceIndex), ...
            config.matching.useEstimatedLagForSearch);
        [receiverTime, receiverPulse] = findFirstTransitionToStateAfterAnchor( ...
            receiverTables{surfaceIndex}, ...
            receiverSearchStartSeconds, ...
            NaN, ...
            expectedState, ...
            config.matching.maxResponseWindowSeconds, ...
            0.0);
        if isfinite(receiverTime)
            directEvents.receiver_transition_s(eventCount) = receiverTime;
            directEvents.receiver_transition_us(eventCount) = receiverPulse;
            directEvents.anchor_to_receiver_latency_s(eventCount) = receiverTime - anchorTime;
            if config.matching.composeReceiverLatencyFromChain || config.analysis.composeReceiverLatencyFromChain
                commitToReceiverLatencySeconds = computeCommitToReceiverLatencySeconds( ...
                    receiverTime, ...
                    directEvents.reference_strobe_s(eventCount), ...
                    directEvents.board_commit_s(eventCount));
                [computerToReceiverLatencySeconds, scheduledToReceiverLatencySeconds] = ...
                    composeReceiverLatenciesFromChain( ...
                        directEvents.scheduled_time_s(eventCount), ...
                        directEvents.command_dispatch_s(eventCount), ...
                        directEvents.board_rx_s(eventCount), ...
                        directEvents.board_commit_s(eventCount), ...
                        commitToReceiverLatencySeconds);
                directEvents.computer_to_receiver_latency_s(eventCount) = computerToReceiverLatencySeconds;
                directEvents.scheduled_to_receiver_latency_s(eventCount) = scheduledToReceiverLatencySeconds;
            else
                if isfinite(directEvents.command_dispatch_s(eventCount))
                    directEvents.computer_to_receiver_latency_s(eventCount) = ...
                        receiverTime - directEvents.command_dispatch_s(eventCount);
                end
                if isfinite(directEvents.scheduled_time_s(eventCount))
                    directEvents.scheduled_to_receiver_latency_s(eventCount) = ...
                        receiverTime - directEvents.scheduled_time_s(eventCount);
                end
            end
        end
        if isfinite(directEvents.trainer_transition_s(eventCount)) && isfinite(directEvents.receiver_transition_s(eventCount))
            directEvents.ppm_to_receiver_latency_s(eventCount) = ...
                directEvents.receiver_transition_s(eventCount) - directEvents.trainer_transition_s(eventCount);
        end
    end
end

directEvents = directEvents(1:eventCount, :);
end

function truthSubsetEvents = buildTruthSubsetLatencyEvents( ...
    hostDispatchLog, ...
    boardRxLog, ...
    boardCommitLog, ...
    referenceCapture, ...
    receiverCapture, ...
    profileInfo, ...
    config)
truthSubsetEvents = buildEmptyTruthSubsetEventsTable();
if ~config.matching.enableDownstreamMatching
    return;
end
if ~shouldUseTruthSubsetAnalysis(profileInfo, config)
    return;
end
if isempty(boardCommitLog) || isempty(profileInfo) || ~isfield(profileInfo, "profileEvents") || isempty(profileInfo.profileEvents)
    return;
end

commitUnique = deduplicateCommitTable(boardCommitLog);
[commitUnique.reference_strobe_s, commitUnique.reference_sample_index, commitUnique.reference_sample_rate_hz] = matchReferenceCaptureTimes( ...
    commitUnique, ...
    referenceCapture, ...
    config.referenceStrobe.mode, ...
    config.matching.referenceAssociationWindowSeconds);
if height(commitUnique) < 2
    return;
end

profileEvents = profileInfo.profileEvents;
if height(profileEvents) < 2
    return;
end

hostDispatchUnique = deduplicateHostDispatchBySampleSequence(hostDispatchLog);
rxUnique = deduplicateEchoImportTable(boardRxLog(:, {'surface_name', 'command_sequence', 'received_position_norm', 'rx_time_s'}));
ppmStateCentersBySurface = deriveSurfaceStateCenters(commitUnique, config.surfaceNames);
receiverStateCentersBySurface = derivePulseCaptureStateCenters(receiverCapture, config.surfaceNames, ppmStateCentersBySurface);
commitTransitionTables = buildCommitTransitionTablesBySurface(commitUnique, config.surfaceNames, ppmStateCentersBySurface);
receiverTables = buildStableStateTransitionTablesBySurface( ...
    receiverCapture, ...
    config.surfaceNames, ...
    receiverStateCentersBySurface, ...
    config.matching.stableStatePulseCount);
transitionAssociationWindowSeconds = max( ...
    1.0, ...
    config.matching.maxResponseWindowSeconds + 0.05);
receiverLagBySurface = estimateOrderedTransitionLagBySurface( ...
    commitTransitionTables, ...
    receiverTables, ...
    transitionAssociationWindowSeconds);

activeSurfaceCount = sum(config.activeSurfaceMask);
maxEventCount = max(0, (height(profileEvents) - 1) .* max(1, activeSurfaceCount));
truthSubsetEvents = buildEmptyTruthSubsetEventsTable(maxEventCount);
eventCount = 0;

for eventRowIndex = 2:height(profileEvents)
    previousTargetDeg = double(profileEvents.TargetDeflection_deg(eventRowIndex - 1));
    currentTargetDeg = double(profileEvents.TargetDeflection_deg(eventRowIndex));
    if currentTargetDeg == previousTargetDeg
        continue;
    end

    boundaryTimeSeconds = double(profileInfo.commandStartSeconds) + double(profileEvents.StartTime_s(eventRowIndex));
    for surfaceIndex = 1:numel(config.surfaceNames)
        if ~config.activeSurfaceMask(surfaceIndex)
            continue;
        end

        previousState = classifyProfileBoundaryState(previousTargetDeg, surfaceIndex, config);
        expectedState = classifyProfileBoundaryState(currentTargetDeg, surfaceIndex, config);
        if ~isfinite(previousState) || ~isfinite(expectedState) || previousState == expectedState
            continue;
        end

        eventCount = eventCount + 1;
        truthSubsetEvents = initializeTruthSubsetEventRow(truthSubsetEvents, eventCount);
        truthSubsetEvents.profile_event_index(eventCount) = double(profileEvents.EventIndex(eventRowIndex));
        truthSubsetEvents.profile_event_label(eventCount) = string(profileEvents.EventLabel(eventRowIndex));
        truthSubsetEvents.boundary_time_s(eventCount) = boundaryTimeSeconds;
        truthSubsetEvents.previous_target_deg(eventCount) = previousTargetDeg;
        truthSubsetEvents.target_deflection_deg(eventCount) = currentTargetDeg;
        truthSubsetEvents.surface_name(eventCount) = config.surfaceNames(surfaceIndex);
        truthSubsetEvents.surface_index(eventCount) = surfaceIndex;
        truthSubsetEvents.analysis_mode(eventCount) = "isolated_profile_boundaries";

        [commitTime, referenceTime, anchorTime, sampleSequence, previousPulseUs, expectedPulseUs] = findFirstCommitTransitionAfterBoundary( ...
            commitTransitionTables{surfaceIndex}, ...
            boundaryTimeSeconds, ...
            previousState, ...
            expectedState, ...
            config.matching.maxResponseWindowSeconds);
        if ~isfinite(anchorTime)
            continue;
        end

        truthSubsetEvents.commit_transition_found(eventCount) = true;
        truthSubsetEvents.sample_sequence(eventCount) = sampleSequence;
        truthSubsetEvents.board_commit_s(eventCount) = commitTime;
        truthSubsetEvents.reference_strobe_s(eventCount) = referenceTime;
        truthSubsetEvents.anchor_s(eventCount) = anchorTime;
        truthSubsetEvents.previous_ppm_us(eventCount) = previousPulseUs;
        truthSubsetEvents.expected_ppm_us(eventCount) = expectedPulseUs;
        truthSubsetEvents.delta_ppm_us(eventCount) = expectedPulseUs - previousPulseUs;
        truthSubsetEvents.anchor_source(eventCount) = ternaryText(isfinite(referenceTime), "reference", "board_commit");

        hostRow = findHostDispatchRow(hostDispatchUnique, config.surfaceNames(surfaceIndex), sampleSequence);
        if hostRow > 0
            truthSubsetEvents.command_sequence(eventCount) = hostDispatchUnique.command_sequence(hostRow);
            truthSubsetEvents.scheduled_time_s(eventCount) = hostDispatchUnique.scheduled_time_s(hostRow);
            truthSubsetEvents.command_dispatch_s(eventCount) = hostDispatchUnique.command_dispatch_s(hostRow);

            rxKey = buildSurfaceSequenceKeys(config.surfaceNames(surfaceIndex), hostDispatchUnique.command_sequence(hostRow));
            rxMatch = buildSurfaceSequenceKeys(rxUnique.surface_name, rxUnique.command_sequence);
            rxRow = find(rxMatch == rxKey, 1, "first");
            if ~isempty(rxRow)
                truthSubsetEvents.board_rx_s(eventCount) = rxUnique.rx_time_s(rxRow);
            end
        end
    end
end

truthSubsetEvents = truthSubsetEvents(1:eventCount, :);
truthSubsetEvents = pairTruthSubsetReceiverTransitionsByOrder( ...
    truthSubsetEvents, ...
    receiverTables, ...
    receiverLagBySurface, ...
    receiverStateCentersBySurface, ...
    config.surfaceNames, ...
    config.matching.maxResponseWindowSeconds, ...
    config);
end

function summarySource = determineLatencySummarySource(directLatencyEvents, truthSubsetEvents, profileInfo, config)
directReceiverMatchCount = countFiniteLatencyEvents(directLatencyEvents, "receiver_transition_s");
truthSubsetReceiverMatchCount = countFiniteLatencyEvents(truthSubsetEvents, "receiver_transition_s");

if shouldUseTruthSubsetAnalysis(profileInfo, config)
    if isempty(truthSubsetEvents)
        summarySource = "truth_subset_profile_boundaries_empty";
    elseif directReceiverMatchCount == 0 && truthSubsetReceiverMatchCount > 0
        summarySource = "truth_subset_only_direct_matching_failed";
    else
        summarySource = "truth_subset_profile_boundaries";
    end
    return;
end

if isempty(directLatencyEvents) || directReceiverMatchCount == 0
    summarySource = "reference_anchored_direct_events_empty";
else
    summarySource = "reference_anchored_direct_events";
end
end

function matchCount = countFiniteLatencyEvents(latencyEvents, variableName)
matchCount = 0;
if isempty(latencyEvents)
    return;
end
if ~ismember(variableName, string(latencyEvents.Properties.VariableNames))
    return;
end
matchCount = sum(isfinite(double(latencyEvents.(char(variableName)))));
end

function latencyEventsForSummary = selectLatencyEventsForSummary(directLatencyEvents, truthSubsetEvents)
if ~isempty(truthSubsetEvents)
    latencyEventsForSummary = truthSubsetEvents;
else
    latencyEventsForSummary = directLatencyEvents;
end
end

function truthSubsetSummary = buildTruthSubsetSummary(truthSubsetEvents, config)
surfaceCount = numel(config.surfaceNames);
truthEventCount = zeros(surfaceCount, 1);
commitMatchCount = zeros(surfaceCount, 1);
receiverMatchCount = zeros(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceMask = truthSubsetEvents.surface_name == config.surfaceNames(surfaceIndex);
    truthEventCount(surfaceIndex) = sum(surfaceMask);
    commitMatchCount(surfaceIndex) = sum(truthSubsetEvents.commit_transition_found(surfaceMask));
    receiverMatchCount(surfaceIndex) = sum(truthSubsetEvents.receiver_transition_found(surfaceMask));
end

truthSubsetSummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    truthEventCount, ...
    commitMatchCount, ...
    max(0, truthEventCount - commitMatchCount), ...
    receiverMatchCount, ...
    max(0, truthEventCount - receiverMatchCount), ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'TruthEventCount', ...
        'TruthCommitMatchCount', ...
        'TruthCommitMissCount', ...
        'TruthReceiverMatchCount', ...
        'TruthReceiverMissCount'});
end

function truthSubsetEvents = pairTruthSubsetReceiverTransitionsByOrder( ...
    truthSubsetEvents, ...
    receiverTables, ...
    receiverLagBySurface, ...
    receiverStateCentersBySurface, ...
    surfaceNames, ...
    maxWindowSeconds, ...
    config)
if isempty(truthSubsetEvents)
    return;
end

for surfaceIndex = 1:numel(surfaceNames)
    surfaceRows = find( ...
        truthSubsetEvents.surface_name == surfaceNames(surfaceIndex) & ...
        truthSubsetEvents.commit_transition_found);
    if isempty(surfaceRows)
        continue;
    end

    receiverTable = receiverTables{surfaceIndex};
    if isempty(receiverTable)
        continue;
    end

    receiverSearchIndex = 1;
    stateCentersUs = receiverStateCentersBySurface{surfaceIndex};
    for rowCursor = 1:numel(surfaceRows)
        rowIndex = surfaceRows(rowCursor);
        previousState = classifyPulseState( ...
            truthSubsetEvents.previous_ppm_us(rowIndex), ...
            stateCentersUs, ...
            inf);
        expectedState = classifyPulseState( ...
            truthSubsetEvents.expected_ppm_us(rowIndex), ...
            stateCentersUs, ...
            inf);
        if ~isfinite(previousState) || ~isfinite(expectedState) || previousState == expectedState
            continue;
        end

        [matchedReceiverRow, receiverSearchIndex] = findNextOrderedStateTransition( ...
            receiverTable, ...
            receiverSearchIndex, ...
            selectDownstreamSearchStartTime( ...
                truthSubsetEvents.anchor_s(rowIndex), ...
                receiverLagBySurface(surfaceIndex), ...
                config.matching.useEstimatedLagForSearch), ...
            previousState, ...
            expectedState, ...
            maxWindowSeconds);
        if ~isfinite(matchedReceiverRow)
            continue;
        end

        receiverTime = receiverTable.time_s(matchedReceiverRow);
        receiverPulse = receiverTable.pulse_us(matchedReceiverRow);
        truthSubsetEvents.receiver_transition_found(rowIndex) = true;
        truthSubsetEvents.receiver_transition_s(rowIndex) = receiverTime;
        truthSubsetEvents.receiver_transition_us(rowIndex) = receiverPulse;
        truthSubsetEvents.anchor_to_receiver_latency_s(rowIndex) = receiverTime - truthSubsetEvents.anchor_s(rowIndex);
        if config.matching.composeReceiverLatencyFromChain || config.analysis.composeReceiverLatencyFromChain
            commitToReceiverLatencySeconds = computeCommitToReceiverLatencySeconds( ...
                receiverTime, ...
                truthSubsetEvents.reference_strobe_s(rowIndex), ...
                truthSubsetEvents.board_commit_s(rowIndex));
            [computerToReceiverLatencySeconds, scheduledToReceiverLatencySeconds] = ...
                composeReceiverLatenciesFromChain( ...
                    truthSubsetEvents.scheduled_time_s(rowIndex), ...
                    truthSubsetEvents.command_dispatch_s(rowIndex), ...
                    truthSubsetEvents.board_rx_s(rowIndex), ...
                    truthSubsetEvents.board_commit_s(rowIndex), ...
                    commitToReceiverLatencySeconds);
            truthSubsetEvents.computer_to_receiver_latency_s(rowIndex) = computerToReceiverLatencySeconds;
            truthSubsetEvents.scheduled_to_receiver_latency_s(rowIndex) = scheduledToReceiverLatencySeconds;
        else
            if isfinite(truthSubsetEvents.command_dispatch_s(rowIndex))
                truthSubsetEvents.computer_to_receiver_latency_s(rowIndex) = ...
                    receiverTime - truthSubsetEvents.command_dispatch_s(rowIndex);
            end
            if isfinite(truthSubsetEvents.scheduled_time_s(rowIndex))
                truthSubsetEvents.scheduled_to_receiver_latency_s(rowIndex) = ...
                    receiverTime - truthSubsetEvents.scheduled_time_s(rowIndex);
            end
        end
    end
end
end

function [matchedRowIndex, nextSearchIndex] = findNextOrderedStateTransition( ...
    transitionTable, ...
    startIndex, ...
    minimumTimeSeconds, ...
    previousState, ...
    expectedState, ...
    maxWindowSeconds)
matchedRowIndex = NaN;
nextSearchIndex = startIndex;
if isempty(transitionTable)
    return;
end

rowIndex = max(1, startIndex);
windowEndSeconds = NaN;
if isfinite(minimumTimeSeconds) && isfinite(maxWindowSeconds)
    windowEndSeconds = minimumTimeSeconds + maxWindowSeconds;
end
while rowIndex <= height(transitionTable)
    transitionTimeSeconds = transitionTable.time_s(rowIndex);
    if isfinite(minimumTimeSeconds) && transitionTimeSeconds < minimumTimeSeconds
        rowIndex = rowIndex + 1;
        continue;
    end
    if isfinite(windowEndSeconds) && transitionTimeSeconds > windowEndSeconds
        nextSearchIndex = rowIndex;
        return;
    end
    if transitionTable.previous_state(rowIndex) == previousState && ...
            transitionTable.new_state(rowIndex) == expectedState
        matchedRowIndex = rowIndex;
        nextSearchIndex = rowIndex + 1;
        return;
    end
    rowIndex = rowIndex + 1;
end

nextSearchIndex = rowIndex;
end

function commitTransitionTables = buildMatchedEventCommitTransitionTablesBySurface( ...
    matchedCommitRows, ...
    surfaceNames, ...
    stateCentersBySurface, ...
    toleranceUs)
commitTransitionTables = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    commitTransitionTables{surfaceIndex} = buildMatchedEventCommitTransitionTable( ...
        matchedCommitRows, ...
        surfaceNames(surfaceIndex), ...
        surfaceIndex, ...
        stateCentersBySurface{surfaceIndex}, ...
        toleranceUs);
end
end

function transitionTable = buildMatchedEventCommitTransitionTable( ...
    matchedCommitRows, ...
    surfaceName, ...
    surfaceIndex, ...
    stateCentersUs, ...
    toleranceUs)
if isempty(matchedCommitRows)
    transitionTable = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'commit_time_s', ...
            'reference_strobe_s', ...
            'anchor_s', ...
            'previous_pulse_us', ...
            'pulse_us', ...
            'previous_state', ...
            'new_state'});
    return;
end

surfaceRows = matchedCommitRows(matchedCommitRows.surface_name == surfaceName, :);
if height(surfaceRows) < 2
    transitionTable = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'commit_time_s', ...
            'reference_strobe_s', ...
            'anchor_s', ...
            'previous_pulse_us', ...
            'pulse_us', ...
            'previous_state', ...
            'new_state'});
    return;
end

rowCapacity = height(surfaceRows) - 1;
transitionTable = table( ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'commit_time_s', ...
        'reference_strobe_s', ...
        'anchor_s', ...
        'previous_pulse_us', ...
        'pulse_us', ...
        'previous_state', ...
        'new_state'});
transitionCount = 0;
for rowIndex = 2:height(surfaceRows)
    previousPulseUs = double(surfaceRows.expected_ppm_us(rowIndex - 1));
    currentPulseUs = double(surfaceRows.expected_ppm_us(rowIndex));
    previousState = classifyPulseState(previousPulseUs, stateCentersUs, toleranceUs);
    currentState = classifyPulseState(currentPulseUs, stateCentersUs, toleranceUs);
    if ~isfinite(previousState) || ~isfinite(currentState) || previousState == currentState
        continue;
    end

    transitionCount = transitionCount + 1;
    transitionTable.sample_sequence(transitionCount) = double(surfaceRows.sample_sequence(rowIndex));
    transitionTable.commit_time_s(transitionCount) = double(surfaceRows.board_commit_s(rowIndex));
    transitionTable.reference_strobe_s(transitionCount) = double(surfaceRows.reference_strobe_s(rowIndex));
    transitionTable.anchor_s(transitionCount) = selectDownstreamSearchAnchorTime( ...
        transitionTable.commit_time_s(transitionCount), ...
        transitionTable.reference_strobe_s(transitionCount));
    transitionTable.previous_pulse_us(transitionCount) = previousPulseUs;
    transitionTable.pulse_us(transitionCount) = currentPulseUs;
    transitionTable.previous_state(transitionCount) = previousState;
    transitionTable.new_state(transitionCount) = currentState;
end

transitionTable = transitionTable(1:transitionCount, :);
end

function lagBySurface = estimateOrderedTransitionLagBySurface( ...
    commitTransitionTables, ...
    pulseTransitionTables, ...
    associationWindowSeconds)
surfaceCount = numel(commitTransitionTables);
lagBySurface = nan(surfaceCount, 1);
for surfaceIndex = 1:surfaceCount
    lagBySurface(surfaceIndex) = estimateOrderedTransitionLag( ...
        commitTransitionTables{surfaceIndex}, ...
        pulseTransitionTables{surfaceIndex}, ...
        associationWindowSeconds);
end
end

function validateEstimatedDownstreamLags(trainerLagBySurface, receiverLagBySurface, config)
frameSeconds = double(config.trainerPpm.frameLengthUs) ./ 1e6;
maxTrainerLagSeconds = 3.0 .* frameSeconds;
maxReceiverLagSeconds = maxTrainerLagSeconds + 0.05;

if any(isfinite(trainerLagBySurface) & trainerLagBySurface > maxTrainerLagSeconds)
    error("Transmitter_Test:ImplausibleTrainerLag", ...
        "Estimated trainer lag exceeds %.6f s. Reference and trainer captures are likely misaligned or misidentified.", ...
        maxTrainerLagSeconds);
end

if any(isfinite(receiverLagBySurface) & receiverLagBySurface > maxReceiverLagSeconds)
    error("Transmitter_Test:ImplausibleReceiverLag", ...
        "Estimated receiver lag exceeds %.6f s. Downstream capture alignment is likely invalid.", ...
        maxReceiverLagSeconds);
end
end

function lagSeconds = estimateOrderedTransitionLag( ...
    commitTransitionTable, ...
    pulseTransitionTable, ...
    associationWindowSeconds)
lagSeconds = NaN;
if isempty(commitTransitionTable) || isempty(pulseTransitionTable)
    return;
end

lagSamples = nan(height(commitTransitionTable), 1);
lagCount = 0;
searchIndex = 1;
for rowIndex = 1:height(commitTransitionTable)
    [matchedRowIndex, searchIndex] = findNextOrderedStateTransition( ...
        pulseTransitionTable, ...
        searchIndex, ...
        commitTransitionTable.anchor_s(rowIndex), ...
        commitTransitionTable.previous_state(rowIndex), ...
        commitTransitionTable.new_state(rowIndex), ...
        associationWindowSeconds);
    if ~isfinite(matchedRowIndex)
        continue;
    end

    lagCount = lagCount + 1;
    lagSamples(lagCount) = ...
        pulseTransitionTable.time_s(matchedRowIndex) - commitTransitionTable.anchor_s(rowIndex);
end

lagSamples = lagSamples(1:lagCount);
lagSamples = lagSamples(isfinite(lagSamples));
if isempty(lagSamples)
    return;
end

lagSeconds = median(lagSamples);
end

function adjustedTimeSeconds = applyEstimatedTransitionLag(anchorTimeSeconds, lagSeconds)
adjustedTimeSeconds = anchorTimeSeconds;
if isfinite(adjustedTimeSeconds) && isfinite(lagSeconds)
    adjustedTimeSeconds = adjustedTimeSeconds + lagSeconds;
end
end

function useTruthSubset = shouldUseTruthSubsetAnalysis(profileInfo, config)
useTruthSubset = false;
truthSubsetMode = string(config.matching.truthSubsetMode);
if truthSubsetMode == "off"
    return;
end
if truthSubsetMode == "isolated_profile_boundaries"
    useTruthSubset = true;
    return;
end

useTruthSubset = ...
    isstruct(profileInfo) && ...
    isfield(profileInfo, "type") && ...
    string(profileInfo.type) == "latency_isolated_step";
end

function truthSubsetMode = canonicalizeTruthSubsetMode(truthSubsetMode)
truthSubsetMode = lower(string(truthSubsetMode));
validModes = ["auto", "isolated_profile_boundaries", "off"];
if ~any(truthSubsetMode == validModes)
    error("Transmitter_Test:InvalidTruthSubsetMode", ...
        "matching.truthSubsetMode must be one of: %s.", ...
        char(join(validModes, ", ")));
end
end

function transitionTables = buildCommitTransitionTablesBySurface(commitTable, surfaceNames, stateCentersBySurface)
transitionTables = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    transitionTables{surfaceIndex} = buildCommitTransitionTable( ...
        commitTable, ...
        surfaceIndex, ...
        stateCentersBySurface{surfaceIndex});
end
end

function transitionTable = buildCommitTransitionTable(commitTable, surfaceIndex, stateCentersUs)
if isempty(commitTable) || height(commitTable) < 2
    transitionTable = table( ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'sample_sequence', ...
            'commit_time_s', ...
            'reference_strobe_s', ...
            'anchor_s', ...
            'previous_pulse_us', ...
            'pulse_us', ...
            'previous_state', ...
            'new_state'});
    return;
end

rowCapacity = height(commitTable) - 1;
transitionTable = table( ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    nan(rowCapacity, 1), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'commit_time_s', ...
        'reference_strobe_s', ...
        'anchor_s', ...
        'previous_pulse_us', ...
        'pulse_us', ...
        'previous_state', ...
        'new_state'});
transitionCount = 0;
for commitIndex = 2:height(commitTable)
    previousPulseUs = getCommitPulseForSurface(commitTable(commitIndex - 1, :), surfaceIndex);
    currentPulseUs = getCommitPulseForSurface(commitTable(commitIndex, :), surfaceIndex);
    previousState = classifyPulseState(previousPulseUs, stateCentersUs, inf);
    currentState = classifyPulseState(currentPulseUs, stateCentersUs, inf);
    if ~isfinite(previousState) || ~isfinite(currentState) || previousState == currentState
        continue;
    end

    transitionCount = transitionCount + 1;
    transitionTable.sample_sequence(transitionCount) = double(commitTable.sample_sequence(commitIndex));
    transitionTable.commit_time_s(transitionCount) = double(commitTable.commit_time_s(commitIndex));
    transitionTable.reference_strobe_s(transitionCount) = double(commitTable.reference_strobe_s(commitIndex));
    transitionTable.anchor_s(transitionCount) = selectDownstreamSearchAnchorTime( ...
        transitionTable.commit_time_s(transitionCount), ...
        transitionTable.reference_strobe_s(transitionCount));
    transitionTable.previous_pulse_us(transitionCount) = previousPulseUs;
    transitionTable.pulse_us(transitionCount) = currentPulseUs;
    transitionTable.previous_state(transitionCount) = previousState;
    transitionTable.new_state(transitionCount) = currentState;
end

transitionTable = transitionTable(1:transitionCount, :);
end

function [commitTime, referenceTime, anchorTime, sampleSequence, previousPulseUs, expectedPulseUs] = findFirstCommitTransitionAfterBoundary( ...
    transitionTable, ...
    boundaryTimeSeconds, ...
    previousState, ...
    expectedState, ...
    maxWindowSeconds)
commitTime = NaN;
referenceTime = NaN;
anchorTime = NaN;
sampleSequence = NaN;
previousPulseUs = NaN;
expectedPulseUs = NaN;
if isempty(transitionTable)
    return;
end

windowEndSeconds = boundaryTimeSeconds + maxWindowSeconds;
for rowIndex = 1:height(transitionTable)
    if transitionTable.anchor_s(rowIndex) < boundaryTimeSeconds
        continue;
    end
    if transitionTable.anchor_s(rowIndex) > windowEndSeconds
        return;
    end
    if transitionTable.previous_state(rowIndex) ~= previousState || transitionTable.new_state(rowIndex) ~= expectedState
        continue;
    end

    commitTime = transitionTable.commit_time_s(rowIndex);
    referenceTime = transitionTable.reference_strobe_s(rowIndex);
    anchorTime = transitionTable.anchor_s(rowIndex);
    sampleSequence = transitionTable.sample_sequence(rowIndex);
    previousPulseUs = transitionTable.previous_pulse_us(rowIndex);
    expectedPulseUs = transitionTable.pulse_us(rowIndex);
    return;
end
end

function stateIndex = classifyProfileBoundaryState(baseTargetDeflectionDeg, surfaceIndex, config)
neutralDeflectionDeg = double(config.commandDeflectionOffsetsDegrees(surfaceIndex));
surfaceTargetDeflectionDeg = ...
    double(config.commandDeflectionScales(surfaceIndex)) .* double(baseTargetDeflectionDeg) + ...
    neutralDeflectionDeg;
stateIndex = 2;
if surfaceTargetDeflectionDeg > neutralDeflectionDeg
    stateIndex = 3;
elseif surfaceTargetDeflectionDeg < neutralDeflectionDeg
    stateIndex = 1;
end
end

function truthSubsetEvents = buildEmptyTruthSubsetEventsTable(rowCount)
if nargin < 1
    rowCount = 0;
end
rowCount = max(0, double(rowCount));
truthSubsetEvents = table( ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    false(rowCount, 1), ...
    false(rowCount, 1), ...
    strings(rowCount, 1), ...
    'VariableNames', { ...
        'profile_event_index', ...
        'profile_event_label', ...
        'boundary_time_s', ...
        'previous_target_deg', ...
        'target_deflection_deg', ...
        'sample_sequence', ...
        'surface_name', ...
        'surface_index', ...
        'command_sequence', ...
        'scheduled_time_s', ...
        'command_dispatch_s', ...
        'board_rx_s', ...
        'board_commit_s', ...
        'reference_strobe_s', ...
        'anchor_s', ...
        'previous_ppm_us', ...
        'expected_ppm_us', ...
        'delta_ppm_us', ...
        'trainer_transition_s', ...
        'trainer_transition_us', ...
        'receiver_transition_s', ...
        'receiver_transition_us', ...
        'anchor_source', ...
        'commit_transition_found', ...
        'receiver_transition_found', ...
        'analysis_mode'});
truthSubsetEvents.anchor_to_trainer_latency_s = nan(rowCount, 1);
truthSubsetEvents.anchor_to_receiver_latency_s = nan(rowCount, 1);
truthSubsetEvents.ppm_to_receiver_latency_s = nan(rowCount, 1);
truthSubsetEvents.computer_to_receiver_latency_s = nan(rowCount, 1);
truthSubsetEvents.scheduled_to_receiver_latency_s = nan(rowCount, 1);
end

function truthSubsetEvents = initializeTruthSubsetEventRow(truthSubsetEvents, rowIndex)
if rowIndex > height(truthSubsetEvents)
    truthSubsetEvents = [truthSubsetEvents; buildEmptyTruthSubsetEventsTable(rowIndex - height(truthSubsetEvents))];
end
truthSubsetEvents(rowIndex, :) = buildEmptyTruthSubsetEventsTable(1);
end

function hostDispatchUnique = deduplicateHostDispatchBySampleSequence(hostDispatchLog)
hostDispatchUnique = hostDispatchLog(:, {'surface_name', 'command_sequence', 'sample_sequence', 'scheduled_time_s', 'command_dispatch_s', 'position_norm'});
if isempty(hostDispatchUnique)
    return;
end
[~, firstIndex] = unique(buildSurfaceSequenceKeys(hostDispatchUnique.surface_name, hostDispatchUnique.sample_sequence), "stable");
hostDispatchUnique = hostDispatchUnique(sort(firstIndex), :);
end

function rowIndex = findHostDispatchRow(hostDispatchUnique, surfaceName, sampleSequence)
rowIndex = 0;
if isempty(hostDispatchUnique)
    return;
end
key = buildSurfaceSequenceKeys(surfaceName, sampleSequence);
keys = buildSurfaceSequenceKeys(hostDispatchUnique.surface_name, hostDispatchUnique.sample_sequence);
match = find(keys == key, 1, "first");
if ~isempty(match)
    rowIndex = match;
end
end

function stateCentersBySurface = deriveSurfaceStateCenters(commitTable, surfaceNames)
stateCentersBySurface = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    pulseVariableName = "ppm_ch" + string(surfaceIndex) + "_us";
    pulseValues = unique(double(commitTable.(char(pulseVariableName))));
    pulseValues = pulseValues(isfinite(pulseValues));
    pulseValues = sort(pulseValues(:));
    if numel(pulseValues) >= 3
        stateCentersBySurface{surfaceIndex} = [pulseValues(1), pulseValues(round((numel(pulseValues) + 1) / 2)), pulseValues(end)];
    elseif numel(pulseValues) == 2
        neutralGuess = mean(pulseValues);
        stateCentersBySurface{surfaceIndex} = [pulseValues(1), neutralGuess, pulseValues(2)];
    elseif numel(pulseValues) == 1
        center = pulseValues(1);
        stateCentersBySurface{surfaceIndex} = [center - 250, center, center + 250];
    else
        stateCentersBySurface{surfaceIndex} = [1250, 1500, 1750];
    end
end
end

function stateCentersBySurface = derivePulseCaptureStateCenters(pulseCapture, surfaceNames, fallbackCentersBySurface)
if nargin < 3 || isempty(fallbackCentersBySurface)
    fallbackCentersBySurface = repmat({[1250, 1500, 1750]}, numel(surfaceNames), 1);
end

stateCentersBySurface = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    fallbackCenters = reshape(double(fallbackCentersBySurface{surfaceIndex}), 1, []);
    fallbackCenters = sort(unique(fallbackCenters(isfinite(fallbackCenters))));
    if ~isempty(fallbackCenters)
        stateCentersBySurface{surfaceIndex} = fallbackCenters;
        continue;
    end

    surfaceCapture = pulseCapture(pulseCapture.surface_name == surfaceNames(surfaceIndex), :);
    pulseValues = sort(double(surfaceCapture.pulse_us));
    pulseValues = pulseValues(isfinite(pulseValues));
    if numel(pulseValues) < 3
        stateCentersBySurface{surfaceIndex} = [1250, 1500, 1750];
        continue;
    end

    lowerCenter = computeSortedVectorQuantile(pulseValues, 0.10);
    neutralCenter = computeSortedVectorQuantile(pulseValues, 0.50);
    upperCenter = computeSortedVectorQuantile(pulseValues, 0.90);
    stateCentersBySurface{surfaceIndex} = sort([lowerCenter, neutralCenter, upperCenter]);
end
end

function quantileValue = computeSortedVectorQuantile(sortedValues, quantileLevel)
quantileValue = NaN;
if isempty(sortedValues)
    return;
end

sortedValues = reshape(double(sortedValues), [], 1);
quantileLevel = min(1, max(0, double(quantileLevel)));
fractionalIndex = 1 + (numel(sortedValues) - 1) .* quantileLevel;
lowerIndex = floor(fractionalIndex);
upperIndex = ceil(fractionalIndex);
interpolationFraction = fractionalIndex - lowerIndex;

if lowerIndex == upperIndex
    quantileValue = sortedValues(lowerIndex);
    return;
end

quantileValue = ...
    (1 - interpolationFraction) .* sortedValues(lowerIndex) + ...
    interpolationFraction .* sortedValues(upperIndex);
end

function stateIndex = classifyPulseState(pulseUs, stateCentersUs, toleranceUs)
stateIndex = NaN;
if ~isfinite(pulseUs) || isempty(stateCentersUs)
    return;
end
[distanceUs, nearestIndex] = min(abs(double(stateCentersUs(:)) - double(pulseUs)));
if isfinite(distanceUs) && distanceUs <= toleranceUs
    stateIndex = nearestIndex;
end
end

function transitionTables = buildStableStateTransitionTablesBySurface( ...
    pulseCapture, ...
    surfaceNames, ...
    stateCentersBySurface, ...
    stablePulseCount)
transitionTables = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    surfaceCapture = pulseCapture(pulseCapture.surface_name == surfaceNames(surfaceIndex), :);
    transitionTables{surfaceIndex} = buildStableStateTransitionTable( ...
        surfaceCapture, ...
        stateCentersBySurface{surfaceIndex}, ...
        stablePulseCount);
end
end

function transitionTable = buildStableStateTransitionTable(surfaceCapture, stateCentersUs, stablePulseCount)
transitionTable = table( ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', { ...
        'time_s', ...
        'pulse_us', ...
        'previous_state', ...
        'new_state', ...
        'sample_rate_hz', ...
        'sample_index'});
if isempty(surfaceCapture)
    return;
end

pulseUs = double(surfaceCapture.pulse_us);
stateSequence = nan(size(pulseUs));
for pulseIndex = 1:numel(pulseUs)
    stateSequence(pulseIndex) = classifyPulseState(pulseUs(pulseIndex), stateCentersUs, inf);
end

stableState = stateSequence(1);
candidateState = stableState;
candidateCount = 1;
candidateStartIndex = 1;
transitionCount = 0;
maxTransitionCount = max(0, numel(stateSequence) - 1);
transitionTable = table( ...
    nan(maxTransitionCount, 1), ...
    nan(maxTransitionCount, 1), ...
    nan(maxTransitionCount, 1), ...
    nan(maxTransitionCount, 1), ...
    nan(maxTransitionCount, 1), ...
    nan(maxTransitionCount, 1), ...
    'VariableNames', { ...
        'time_s', ...
        'pulse_us', ...
        'previous_state', ...
        'new_state', ...
        'sample_rate_hz', ...
        'sample_index'});

for pulseIndex = 2:numel(stateSequence)
    currentState = stateSequence(pulseIndex);
    if ~isfinite(currentState)
        continue;
    end
    if currentState == candidateState
        candidateCount = candidateCount + 1;
    else
        candidateState = currentState;
        candidateCount = 1;
        candidateStartIndex = pulseIndex;
    end

    if candidateState ~= stableState && candidateCount >= stablePulseCount
        transitionCount = transitionCount + 1;
        transitionTable.time_s(transitionCount) = surfaceCapture.time_s(candidateStartIndex);
        transitionTable.pulse_us(transitionCount) = surfaceCapture.pulse_us(candidateStartIndex);
        transitionTable.previous_state(transitionCount) = stableState;
        transitionTable.new_state(transitionCount) = candidateState;
        if ismember("sample_rate_hz", string(surfaceCapture.Properties.VariableNames))
            transitionTable.sample_rate_hz(transitionCount) = surfaceCapture.sample_rate_hz(candidateStartIndex);
        else
            transitionTable.sample_rate_hz(transitionCount) = NaN;
        end
        if ismember("sample_index", string(surfaceCapture.Properties.VariableNames))
            transitionTable.sample_index(transitionCount) = surfaceCapture.sample_index(candidateStartIndex);
        else
            transitionTable.sample_index(transitionCount) = NaN;
        end
        stableState = candidateState;
    end
end

transitionTable = transitionTable(1:transitionCount, :);
end

function [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz, nextIndex] = findFirstStableStateTransitionAfterAnchor( ...
    transitionTable, ...
    startIndex, ...
    startTimeSeconds, ...
    previousState, ...
    expectedState, ...
    maxWindowSeconds, ...
    leadSeconds)
matchedTime = NaN;
matchedPulse = NaN;
matchedSampleIndex = NaN;
matchedSampleRateHz = NaN;
nextIndex = startIndex;
if isempty(transitionTable) || ~isfinite(expectedState)
    return;
end
if isfinite(previousState) && previousState == expectedState
    return;
end

rowIndex = max(1, startIndex);
windowStartSeconds = startTimeSeconds - leadSeconds;
windowEndSeconds = startTimeSeconds + maxWindowSeconds;
while rowIndex <= height(transitionTable)
    transitionTime = transitionTable.time_s(rowIndex);
    if transitionTime < windowStartSeconds
        rowIndex = rowIndex + 1;
        continue;
    end
    if transitionTime > windowEndSeconds
        nextIndex = rowIndex;
        return;
    end
    previousStateMatches = true;
    if isfinite(previousState)
        previousStateMatches = transitionTable.previous_state(rowIndex) == previousState;
    end
    if previousStateMatches && transitionTable.new_state(rowIndex) == expectedState
        matchedTime = transitionTime;
        matchedPulse = transitionTable.pulse_us(rowIndex);
        matchedSampleIndex = transitionTable.sample_index(rowIndex);
        matchedSampleRateHz = transitionTable.sample_rate_hz(rowIndex);
        nextIndex = rowIndex + 1;
        return;
    end
    rowIndex = rowIndex + 1;
end

nextIndex = rowIndex;
end

function [matchedTime, matchedPulse] = findFirstTransitionToStateAfterAnchor( ...
    transitionTable, ...
    anchorTimeSeconds, ...
    previousState, ...
    expectedState, ...
    maxWindowSeconds, ...
    leadSeconds)
matchedTime = NaN;
matchedPulse = NaN;
if isempty(transitionTable) || ~isfinite(expectedState)
    return;
end

windowStartSeconds = anchorTimeSeconds - leadSeconds;
windowEndSeconds = anchorTimeSeconds + maxWindowSeconds;
for rowIndex = 1:height(transitionTable)
    transitionTime = transitionTable.time_s(rowIndex);
    if transitionTime < windowStartSeconds
        continue;
    end
    if transitionTime > windowEndSeconds
        return;
    end
    if transitionTable.new_state(rowIndex) ~= expectedState
        continue;
    end
    if isfinite(previousState) && transitionTable.previous_state(rowIndex) ~= previousState
        continue;
    end
    matchedTime = transitionTable.time_s(rowIndex);
    matchedPulse = transitionTable.pulse_us(rowIndex);
    return;
end
end

function expectedPpmUs = buildExpectedPpmWidthsUs(servoPositions, config)
expectedPpmUs = config.trainerPpm.neutralPulseUs .* ones(1, config.trainerPpm.channelCount);
surfaceCount = min(numel(config.surfaceNames), 4);
servoPositions = min(max(reshape(servoPositions, 1, []), 0.0), 1.0);
expectedPpmUs(1:surfaceCount) = round( ...
    config.trainerPpm.minimumPulseUs + ...
    (config.trainerPpm.maximumPulseUs - config.trainerPpm.minimumPulseUs) .* servoPositions(1:surfaceCount));
end

function directEvents = buildSharedClockDirectLatencyEventsFromMatchedEvents(matchedEvents)
directEvents = buildEmptyDirectLatencyEventsTable(0);
if isempty(matchedEvents)
    return;
end

eventMask = isfinite(matchedEvents.anchor_s);
if ~any(eventMask)
    return;
end
eventRows = matchedEvents(eventMask, :);
eventCount = height(eventRows);
directEvents = buildEmptyDirectLatencyEventsTable(eventCount);

directEvents.sample_sequence = double(eventRows.sample_sequence);
directEvents.surface_name = string(eventRows.surface_name);
directEvents.surface_index = double(eventRows.surface_index);
directEvents.command_sequence = double(eventRows.command_sequence);
directEvents.scheduled_time_s = double(eventRows.scheduled_time_s);
directEvents.command_dispatch_s = double(eventRows.command_dispatch_s);
directEvents.board_rx_s = double(eventRows.board_rx_s);
directEvents.board_commit_s = double(eventRows.board_commit_s);
directEvents.reference_strobe_s = double(eventRows.reference_strobe_s);
directEvents.anchor_s = double(eventRows.anchor_s);
directEvents.expected_ppm_us = double(eventRows.expected_ppm_us);
directEvents.trainer_transition_s = double(eventRows.trainer_ppm_s);
directEvents.trainer_transition_us = double(eventRows.trainer_ppm_us);
directEvents.receiver_transition_s = double(eventRows.receiver_response_s);
directEvents.receiver_transition_us = double(eventRows.receiver_pulse_us);
directEvents.anchor_to_trainer_latency_s = double(eventRows.anchor_to_ppm_latency_s);
directEvents.anchor_to_receiver_latency_s = double(eventRows.anchor_to_receiver_latency_s);
directEvents.ppm_to_receiver_latency_s = double(eventRows.ppm_to_receiver_latency_s);
directEvents.computer_to_receiver_latency_s = double(eventRows.computer_to_receiver_latency_s);
directEvents.scheduled_to_receiver_latency_s = double(eventRows.scheduled_to_receiver_latency_s);
directEvents.anchor_source = string(eventRows.anchor_source);

finiteSurfaceIndex = directEvents.surface_index(isfinite(directEvents.surface_index));
if isempty(finiteSurfaceIndex)
    surfaceCount = 1;
else
    surfaceCount = max(1, max(finiteSurfaceIndex));
end
previousBySurface = nan(surfaceCount, 1);
for rowIndex = 1:eventCount
    surfaceIndex = directEvents.surface_index(rowIndex);
    if ~isfinite(surfaceIndex) || surfaceIndex < 1 || surfaceIndex > surfaceCount
        continue;
    end

    expectedPulseUs = directEvents.expected_ppm_us(rowIndex);
    previousPulseUs = previousBySurface(surfaceIndex);
    directEvents.previous_ppm_us(rowIndex) = previousPulseUs;
    if isfinite(previousPulseUs) && isfinite(expectedPulseUs)
        directEvents.delta_ppm_us(rowIndex) = expectedPulseUs - previousPulseUs;
    end
    previousBySurface(surfaceIndex) = expectedPulseUs;
end
end

function directEvents = buildEmptyDirectLatencyEventsTable(rowCount)
rowCount = max(0, double(rowCount));
directEvents = table( ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    'VariableNames', { ...
        'sample_sequence', ...
        'surface_name', ...
        'surface_index', ...
        'command_sequence', ...
        'scheduled_time_s', ...
        'command_dispatch_s', ...
        'board_rx_s', ...
        'board_commit_s', ...
        'reference_strobe_s', ...
        'anchor_s', ...
        'previous_ppm_us', ...
        'expected_ppm_us', ...
        'delta_ppm_us', ...
        'trainer_transition_s', ...
        'trainer_transition_us', ...
        'receiver_transition_s', ...
        'receiver_transition_us', ...
        'anchor_to_trainer_latency_s', ...
        'anchor_to_receiver_latency_s', ...
        'ppm_to_receiver_latency_s', ...
        'computer_to_receiver_latency_s', ...
        'scheduled_to_receiver_latency_s', ...
        'anchor_source'});
end

function logs = buildTransmitterLogs(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceNames = config.surfaceNames;
rowTimesSeconds = chooseFiniteRowTimes(storage.commandWriteStopSeconds(sampleIndices), storage.scheduledTimeSeconds(sampleIndices));
directLatencyEvents = buildSharedClockDirectLatencyEventsFromMatchedEvents(storage.matchedEvents);
truthSubsetEvents = buildEmptyTruthSubsetEventsTable(0);
latencyEventsForSummary = directLatencyEvents;
alignmentSummary = buildAlignmentSummary(directLatencyEvents, config);
printAlignmentSummary(alignmentSummary);

inputSignal = array2timetable( ...
    [ ...
        storage.baseCommandDegrees(sampleIndices), ...
        storage.desiredDeflectionsDegrees(sampleIndices, :), ...
        storage.commandedServoPositions(sampleIndices, :), ...
        double(storage.commandSaturated(sampleIndices, :))], ...
    'RowTimes', seconds(rowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        "base_command_deg", ...
        buildSurfaceVariableNames(surfaceNames, "desired_deg"), ...
        buildSurfaceVariableNames(surfaceNames, "command_position"), ...
        buildSurfaceVariableNames(surfaceNames, "command_saturated")]));
inputSignal.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
inputSignal.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
inputSignal.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);

hostSchedulingDelay = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        repmat(storage.scheduledTimeSeconds(sampleIndices), 1, numel(surfaceNames)), ...
        storage.commandDispatchSeconds(sampleIndices, :), ...
        storage.hostSchedulingDelaySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(rowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "scheduled_time_s"), ...
        buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
        buildSurfaceVariableNames(surfaceNames, "host_scheduling_delay_s")]));

    boardRowTimesSeconds = chooseFiniteRowTimes(storage.boardReadStopSeconds(sampleIndices), rowTimesSeconds);
computerToArduinoRxLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.commandDispatchSeconds(sampleIndices, :), ...
        storage.boardRxSeconds(sampleIndices, :), ...
        storage.computerToArduinoRxLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(boardRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
        buildSurfaceVariableNames(surfaceNames, "arduino_rx_s"), ...
        buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_rx_latency_s")]));

arduinoReceiveToPpmCommitLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.boardRxSeconds(sampleIndices, :), ...
        storage.boardCommitSeconds(sampleIndices, :), ...
        storage.arduinoReceiveToPpmCommitLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(boardRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "arduino_rx_s"), ...
        buildSurfaceVariableNames(surfaceNames, "ppm_commit_s"), ...
        buildSurfaceVariableNames(surfaceNames, "arduino_receive_to_ppm_commit_latency_s")]));

anchorRowTimesSeconds = chooseFiniteRowTimes( ...
    max(storage.anchorSeconds(sampleIndices, :), [], 2, "omitnan"), ...
    boardRowTimesSeconds);
anchorToPpmLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.anchorSeconds(sampleIndices, :), ...
        storage.trainerPpmSeconds(sampleIndices, :), ...
        storage.anchorToPpmLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(anchorRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "anchor_s"), ...
        buildSurfaceVariableNames(surfaceNames, "trainer_ppm_s"), ...
        buildSurfaceVariableNames(surfaceNames, "anchor_to_ppm_latency_s")]));

anchorToReceiverLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.anchorSeconds(sampleIndices, :), ...
        storage.receiverResponseSeconds(sampleIndices, :), ...
        storage.anchorToReceiverLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(anchorRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "anchor_s"), ...
        buildSurfaceVariableNames(surfaceNames, "receiver_response_s"), ...
        buildSurfaceVariableNames(surfaceNames, "anchor_to_receiver_latency_s")]));

receiverRowTimesSeconds = chooseFiniteRowTimes( ...
    max(storage.receiverResponseSeconds(sampleIndices, :), [], 2, "omitnan"), ...
    boardRowTimesSeconds);
ppmToReceiverLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.trainerPpmSeconds(sampleIndices, :), ...
        storage.receiverResponseSeconds(sampleIndices, :), ...
        storage.ppmToReceiverLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(receiverRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "trainer_ppm_s"), ...
        buildSurfaceVariableNames(surfaceNames, "receiver_response_s"), ...
        buildSurfaceVariableNames(surfaceNames, "ppm_to_receiver_latency_s")]));

computerToReceiverLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        storage.commandDispatchSeconds(sampleIndices, :), ...
        storage.receiverResponseSeconds(sampleIndices, :), ...
        storage.computerToReceiverLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(receiverRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
        buildSurfaceVariableNames(surfaceNames, "receiver_response_s"), ...
        buildSurfaceVariableNames(surfaceNames, "computer_to_receiver_latency_s")]));

scheduledToReceiverLatency = array2timetable( ...
    [ ...
        storage.commandSequenceNumbers(sampleIndices, :), ...
        repmat(storage.scheduledTimeSeconds(sampleIndices), 1, numel(surfaceNames)), ...
        storage.receiverResponseSeconds(sampleIndices, :), ...
        storage.scheduledToReceiverLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(receiverRowTimesSeconds), ...
    'VariableNames', cellstr([ ...
        buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
        buildSurfaceVariableNames(surfaceNames, "scheduled_time_s"), ...
        buildSurfaceVariableNames(surfaceNames, "receiver_response_s"), ...
        buildSurfaceVariableNames(surfaceNames, "scheduled_to_receiver_latency_s")]));

sampleSummary = table( ...
    storage.scheduledTimeSeconds(sampleIndices), ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.commandWriteStartSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices), ...
    storage.boardReadStartSeconds(sampleIndices), ...
    storage.boardReadStopSeconds(sampleIndices), ...
    any(isfinite(storage.boardRxSeconds(sampleIndices, :)), 2), ...
    any(isfinite(storage.boardCommitSeconds(sampleIndices, :)), 2), ...
    any(isfinite(storage.receiverResponseSeconds(sampleIndices, :)), 2), ...
    'VariableNames', { ...
        'scheduled_time_s', ...
        'base_command_deg', ...
        'command_write_start_s', ...
        'command_write_stop_s', ...
        'board_read_start_s', ...
        'board_read_stop_s', ...
        'arduino_rx_available', ...
        'ppm_commit_available', ...
        'receiver_available'});

logs = struct( ...
    "inputSignal", inputSignal, ...
    "hostSchedulingDelay", hostSchedulingDelay, ...
    "boardRxLog", storage.rawLogs.boardRxLog, ...
    "boardCommitLog", storage.rawLogs.boardCommitLog, ...
    "boardSyncLog", storage.rawLogs.boardSyncLog, ...
    "referenceCapture", storage.rawLogs.referenceCapture, ...
    "trainerPpmCapture", storage.rawLogs.trainerPpmCapture, ...
    "receiverCapture", storage.rawLogs.receiverCapture, ...
    "computerToArduinoRxLatency", computerToArduinoRxLatency, ...
    "arduinoReceiveToPpmCommitLatency", arduinoReceiveToPpmCommitLatency, ...
    "anchorToPpmLatency", anchorToPpmLatency, ...
    "anchorToReceiverLatency", anchorToReceiverLatency, ...
    "ppmToReceiverLatency", ppmToReceiverLatency, ...
    "computerToReceiverLatency", computerToReceiverLatency, ...
    "scheduledToReceiverLatency", scheduledToReceiverLatency, ...
    "directLatencyEvents", directLatencyEvents, ...
    "truthSubsetEvents", truthSubsetEvents, ...
    "alignmentSummary", alignmentSummary, ...
    "matchedEvents", storage.matchedEvents, ...
    "latencySummary", buildTransmitterLatencySummary(storage, config, latencyEventsForSummary), ...
    "truthSubsetSummary", buildTruthSubsetSummary(truthSubsetEvents, config), ...
    "latencySummarySource", "shared_clock_minimal", ...
    "integritySummary", storage.integritySummary, ...
    "profileEvents", storage.profileInfo.profileEvents, ...
    "sampleSummary", sampleSummary, ...
    "hostDispatchLog", storage.rawLogs.hostDispatchLog, ...
    "serialTelemetryLog", storage.rawLogs.serialTelemetryLog);
end

function validateStrictDownstreamMatching(directLatencyEvents, truthSubsetEvents)
directReceiverMatchCount = countFiniteLatencyEvents(directLatencyEvents, "receiver_transition_s");
truthSubsetReceiverMatchCount = countFiniteLatencyEvents(truthSubsetEvents, "receiver_transition_s");
if directReceiverMatchCount == 0 && truthSubsetReceiverMatchCount > 0
    error("Transmitter_Test:StrictDownstreamMatchingFailed", ...
        "Direct receiver matching produced zero events while truth-subset matching produced %d events. Capture identity or anchor alignment is invalid.", ...
        truthSubsetReceiverMatchCount);
end
end

function alignmentSummary = buildAlignmentSummary(directLatencyEvents, config)
surfaceCount = numel(config.surfaceNames);
boardCommitToReferenceCount = zeros(surfaceCount, 1);
boardCommitToReferenceMedian = nan(surfaceCount, 1);
boardCommitToReferenceP95 = nan(surfaceCount, 1);
referenceToTrainerCount = zeros(surfaceCount, 1);
referenceToTrainerMedian = nan(surfaceCount, 1);
referenceToTrainerP95 = nan(surfaceCount, 1);
trainerToReceiverCount = zeros(surfaceCount, 1);
trainerToReceiverMedian = nan(surfaceCount, 1);
trainerToReceiverP95 = nan(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceRows = directLatencyEvents(directLatencyEvents.surface_name == config.surfaceNames(surfaceIndex), :);
    boardCommitToReference = surfaceRows.anchor_s - surfaceRows.board_commit_s;
    referenceToTrainer = surfaceRows.anchor_to_trainer_latency_s;
    trainerToReceiver = surfaceRows.ppm_to_receiver_latency_s;

    boardCommitToReferenceStats = computeLatencyStats(boardCommitToReference);
    referenceToTrainerStats = computeLatencyStats(referenceToTrainer);
    trainerToReceiverStats = computeLatencyStats(trainerToReceiver);

    boardCommitToReferenceCount(surfaceIndex) = boardCommitToReferenceStats.sampleCount;
    boardCommitToReferenceMedian(surfaceIndex) = boardCommitToReferenceStats.medianValue;
    boardCommitToReferenceP95(surfaceIndex) = boardCommitToReferenceStats.p95Value;

    referenceToTrainerCount(surfaceIndex) = referenceToTrainerStats.sampleCount;
    referenceToTrainerMedian(surfaceIndex) = referenceToTrainerStats.medianValue;
    referenceToTrainerP95(surfaceIndex) = referenceToTrainerStats.p95Value;

    trainerToReceiverCount(surfaceIndex) = trainerToReceiverStats.sampleCount;
    trainerToReceiverMedian(surfaceIndex) = trainerToReceiverStats.medianValue;
    trainerToReceiverP95(surfaceIndex) = trainerToReceiverStats.p95Value;
end

alignmentSummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    boardCommitToReferenceCount, ...
    boardCommitToReferenceMedian, ...
    boardCommitToReferenceP95, ...
    referenceToTrainerCount, ...
    referenceToTrainerMedian, ...
    referenceToTrainerP95, ...
    trainerToReceiverCount, ...
    trainerToReceiverMedian, ...
    trainerToReceiverP95, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'BoardCommitToReferenceCount', ...
        'BoardCommitToReferenceMedian_s', ...
        'BoardCommitToReferenceP95_s', ...
        'ReferenceToTrainerCount', ...
        'ReferenceToTrainerMedian_s', ...
        'ReferenceToTrainerP95_s', ...
        'TrainerToReceiverCount', ...
        'TrainerToReceiverMedian_s', ...
        'TrainerToReceiverP95_s'});
end

function printAlignmentSummary(alignmentSummary)
if isempty(alignmentSummary)
    return;
end

disp("AlignmentSummary (s):");
disp(alignmentSummary(:, { ...
    'SurfaceName', ...
    'BoardCommitToReferenceCount', ...
    'BoardCommitToReferenceMedian_s', ...
    'BoardCommitToReferenceP95_s', ...
    'ReferenceToTrainerCount', ...
    'ReferenceToTrainerMedian_s', ...
    'ReferenceToTrainerP95_s', ...
    'TrainerToReceiverCount', ...
    'TrainerToReceiverMedian_s', ...
    'TrainerToReceiverP95_s'}));
end

function latencySummary = buildTransmitterLatencySummary(storage, config, directLatencyEvents)
if nargin < 3
    directLatencyEvents = buildReferenceAnchoredLatencyEvents( ...
        storage.rawLogs.hostDispatchLog, ...
        storage.rawLogs.boardRxLog, ...
        storage.rawLogs.boardCommitLog, ...
        storage.rawLogs.referenceCapture, ...
        storage.rawLogs.trainerPpmCapture, ...
        storage.rawLogs.receiverCapture, ...
        config);
end
surfaceCount = numel(config.surfaceNames);
hostSchedulingStats = repmat(computeLatencyStats([]), surfaceCount, 1);
rxStats = repmat(computeLatencyStats([]), surfaceCount, 1);
commitStats = repmat(computeLatencyStats([]), surfaceCount, 1);
anchorToTrainerStats = repmat(computeLatencyStats([]), surfaceCount, 1);
anchorToReceiverStats = repmat(computeLatencyStats([]), surfaceCount, 1);
ppmToReceiverStats = repmat(computeLatencyStats([]), surfaceCount, 1);
computerToReceiverStats = repmat(computeLatencyStats([]), surfaceCount, 1);
scheduledToReceiverStats = repmat(computeLatencyStats([]), surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    hostSchedulingStats(surfaceIndex) = computeLatencyStats(storage.hostSchedulingDelaySeconds(:, surfaceIndex));
    rxStats(surfaceIndex) = computeLatencyStats(storage.computerToArduinoRxLatencySeconds(:, surfaceIndex));
    commitStats(surfaceIndex) = computeLatencyStats(storage.arduinoReceiveToPpmCommitLatencySeconds(:, surfaceIndex));
    surfaceMask = directLatencyEvents.surface_name == config.surfaceNames(surfaceIndex);
    anchorToTrainerStats(surfaceIndex) = computeLatencyStats(directLatencyEvents.anchor_to_trainer_latency_s(surfaceMask));
    anchorToReceiverStats(surfaceIndex) = computeLatencyStats(directLatencyEvents.anchor_to_receiver_latency_s(surfaceMask));
    ppmToReceiverStats(surfaceIndex) = computeLatencyStats(directLatencyEvents.ppm_to_receiver_latency_s(surfaceMask));
    computerToReceiverStats(surfaceIndex) = computeLatencyStats(directLatencyEvents.computer_to_receiver_latency_s(surfaceMask));
    scheduledToReceiverStats(surfaceIndex) = computeLatencyStats(directLatencyEvents.scheduled_to_receiver_latency_s(surfaceMask));
end

latencySummary = table(config.surfaceNames, config.activeSurfaceMask, 'VariableNames', {'SurfaceName', 'IsActive'});
latencySummary = addLatencyStatsColumns(latencySummary, hostSchedulingStats, "HostSchedulingDelay");
latencySummary = addLatencyStatsColumns(latencySummary, rxStats, "ComputerToArduinoRxLatency");
latencySummary = addLatencyStatsColumns(latencySummary, commitStats, "ArduinoReceiveToPpmCommitLatency");
latencySummary = addLatencyStatsColumns(latencySummary, anchorToTrainerStats, "AnchorToTrainerLatency");
latencySummary = addLatencyStatsColumns(latencySummary, anchorToTrainerStats, "AnchorToPpmLatency");
latencySummary = addLatencyStatsColumns(latencySummary, anchorToReceiverStats, "AnchorToReceiverLatency");
latencySummary = addLatencyStatsColumns(latencySummary, ppmToReceiverStats, "PpmToReceiverLatency");
latencySummary = addLatencyStatsColumns(latencySummary, computerToReceiverStats, "ComputerToReceiverLatency");
latencySummary = addLatencyStatsColumns(latencySummary, scheduledToReceiverStats, "ScheduledToReceiverLatency");
end

function latencySummary = addLatencyStatsColumns(latencySummary, statsArray, prefix)
sampleCount = zeros(numel(statsArray), 1);
meanValue = nan(numel(statsArray), 1);
stdValue = nan(numel(statsArray), 1);
medianValue = nan(numel(statsArray), 1);
p95Value = nan(numel(statsArray), 1);
p99Value = nan(numel(statsArray), 1);
maxValue = nan(numel(statsArray), 1);
for rowIndex = 1:numel(statsArray)
    [sampleCount(rowIndex), meanValue(rowIndex), stdValue(rowIndex), medianValue(rowIndex), p95Value(rowIndex), p99Value(rowIndex), maxValue(rowIndex)] = unpackLatencyStats(statsArray(rowIndex));
end
latencySummary.(prefix + "SampleCount") = sampleCount;
latencySummary.(prefix + "Mean_s") = meanValue;
latencySummary.(prefix + "Std_s") = stdValue;
latencySummary.(prefix + "Median_s") = medianValue;
latencySummary.(prefix + "P95_s") = p95Value;
latencySummary.(prefix + "P99_s") = p99Value;
latencySummary.(prefix + "Max_s") = maxValue;
end

function integritySummary = buildTransmitterIntegritySummary(storage, config)
surfaceCount = numel(config.surfaceNames);
dispatchedCommandCount = zeros(surfaceCount, 1);
matchedRxCount = zeros(surfaceCount, 1);
unmatchedRxCount = zeros(surfaceCount, 1);
matchedCommitCount = zeros(surfaceCount, 1);
droppedBeforeCommitCount = zeros(surfaceCount, 1);
matchedReceiverCount = zeros(surfaceCount, 1);
unmatchedReceiverCount = zeros(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    dispatchedCommandCount(surfaceIndex) = sum(isfinite(storage.commandSequenceNumbers(:, surfaceIndex)));
    matchedRxCount(surfaceIndex) = sum(isfinite(storage.boardRxSeconds(:, surfaceIndex)));
    unmatchedRxCount(surfaceIndex) = max(0, dispatchedCommandCount(surfaceIndex) - matchedRxCount(surfaceIndex));
    matchedCommitCount(surfaceIndex) = sum(isfinite(storage.boardCommitSeconds(:, surfaceIndex)));
    droppedBeforeCommitCount(surfaceIndex) = max(0, matchedRxCount(surfaceIndex) - matchedCommitCount(surfaceIndex));
    matchedReceiverCount(surfaceIndex) = sum(isfinite(storage.receiverResponseSeconds(:, surfaceIndex)));
    unmatchedReceiverCount(surfaceIndex) = max(0, dispatchedCommandCount(surfaceIndex) - matchedReceiverCount(surfaceIndex));
end

integritySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    dispatchedCommandCount, ...
    matchedRxCount, ...
    unmatchedRxCount, ...
    matchedCommitCount, ...
    droppedBeforeCommitCount, ...
    matchedReceiverCount, ...
    unmatchedReceiverCount, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'DispatchedCommandCount', ...
        'MatchedRxCount', ...
        'UnmatchedRxCount', ...
        'MatchedCommitCount', ...
        'DroppedBeforeCommitCount', ...
        'MatchedReceiverCount', ...
        'UnmatchedReceiverCount'});
end

function surfaceSummary = buildTransmitterSurfaceSummaryFromLogs(logs)
surfaceSummary = logs.integritySummary;
surfaceSummary = [surfaceSummary logs.latencySummary(:, 3:end)];
if isfield(logs, "truthSubsetSummary") && ~isempty(logs.truthSubsetSummary)
    surfaceSummary = [surfaceSummary logs.truthSubsetSummary(:, 3:end)];
end
if isfield(logs, "latencySummarySource")
    surfaceSummary.LatencySummarySource = repmat(string(logs.latencySummarySource), height(surfaceSummary), 1);
end
end

function surfaceSetup = buildTransmitterSurfaceSetupTable(config)
surfaceSetup = table( ...
    config.surfaceNames, ...
    (1:numel(config.surfaceNames)).', ...
    config.activeSurfaceMask, ...
    config.servoNeutralPositions, ...
    config.servoUnitsPerDegree, ...
    config.servoMinimumPositions, ...
    config.servoMaximumPositions, ...
    config.commandDeflectionScales, ...
    config.commandDeflectionOffsetsDegrees, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'TrainerChannel', ...
        'IsActive', ...
        'ServoNeutralPosition', ...
        'ServoUnitsPerDegree', ...
        'ServoMinimumPosition', ...
        'ServoMaximumPosition', ...
        'CommandScale', ...
        'CommandOffsetDeg'});
end

function artifacts = exportTransmitterRunData(runData)
matFilePath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".mat");
requestedWorkbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");
workbookPath = resolveWritableWorkbookPath(requestedWorkbookPath);
loggerFolderPath = runData.config.arduinoTransport.loggerOutputFolder;

if ~isfolder(loggerFolderPath)
    mkdir(loggerFolderPath);
end

artifacts = struct( ...
    "matFilePath", string(matFilePath), ...
    "workbookPath", string(workbookPath), ...
    "loggerFolderPath", string(loggerFolderPath), ...
    "rawCapturePath", string(runData.config.logicAnalyzer.rawCapturePath), ...
    "logicStateExportPath", string(runData.config.logicAnalyzer.logicStateExportPath), ...
    "decodedReferencePath", string(runData.config.logicAnalyzer.decodedReferencePath), ...
    "decodedTrainerPpmPath", string(runData.config.logicAnalyzer.decodedTrainerPpmPath), ...
    "decodedReceiverPath", string(runData.config.logicAnalyzer.decodedReceiverPath), ...
    "sigrokStdoutPath", string(runData.config.logicAnalyzer.storeStdoutPath), ...
    "sigrokStderrPath", string(runData.config.logicAnalyzer.storeStderrPath));

save(matFilePath, "runData", "-v7.3");

writetable(runData.logs.hostDispatchLog, fullfile(loggerFolderPath, "host_dispatch_log.csv"));
writetable(runData.logs.boardSyncLog(:, {'sync_id', 'host_tx_us', 'host_rx_us', 'board_rx_us', 'board_tx_us'}), fullfile(loggerFolderPath, "host_sync_roundtrip.csv"));
writetable(runData.logs.boardRxLog, fullfile(loggerFolderPath, "board_rx_log.csv"));
writetable(runData.logs.boardCommitLog, fullfile(loggerFolderPath, "board_commit_log.csv"));
writetable(runData.logs.boardSyncLog, fullfile(loggerFolderPath, "board_sync_log.csv"));
writetable(runData.logs.referenceCapture, fullfile(loggerFolderPath, "reference_capture.csv"));
writetable(runData.logs.trainerPpmCapture, fullfile(loggerFolderPath, "trainer_ppm_capture.csv"));
writetable(runData.logs.receiverCapture, fullfile(loggerFolderPath, "receiver_capture.csv"));
writetable(runData.logs.directLatencyEvents, fullfile(loggerFolderPath, "direct_latency_events.csv"));
writetable(timetableToExportTable(runData.logs.anchorToPpmLatency), fullfile(loggerFolderPath, "anchor_to_ppm_latency.csv"));
writetable(timetableToExportTable(runData.logs.anchorToReceiverLatency), fullfile(loggerFolderPath, "anchor_to_receiver_latency.csv"));
writetable(runData.logs.truthSubsetEvents, fullfile(loggerFolderPath, "truth_subset_events.csv"));
writetable(runData.logs.truthSubsetSummary, fullfile(loggerFolderPath, "truth_subset_summary.csv"));
writetable(runData.logs.alignmentSummary, fullfile(loggerFolderPath, "alignment_summary.csv"));
writetable(runData.logs.matchedEvents, fullfile(loggerFolderPath, "matched_events.csv"));
if ~isempty(runData.logs.serialTelemetryLog)
    writelines(runData.logs.serialTelemetryLog, fullfile(loggerFolderPath, "serial_telemetry_log.txt"));
end

writeWorkbookSheet(buildTransmitterCriticalSettingsTable(runData), workbookPath, "CriticalSettings");
writeWorkbookSheet(runData.logs.inputSignal, workbookPath, "InputSignal");
writeWorkbookSheet(runData.logs.hostSchedulingDelay, workbookPath, "HostSchedulingDelay");
writeWorkbookSheet(runData.logs.latencySummary, workbookPath, "LatencySummary");
writeWorkbookSheet(runData.logs.integritySummary, workbookPath, "IntegritySummary");
writeWorkbookSheet(runData.logs.profileEvents, workbookPath, "ProfileEvents");
writeWorkbookSheet(runData.surfaceSetup, workbookPath, "SurfaceSetup");
writeWorkbookSheet(runData.logs.sampleSummary, workbookPath, "SampleSummary");
writeWorkbookSheet(runData.surfaceSummary, workbookPath, "SurfaceSummary");
writeWorkbookSheet(buildTransmitterLogicAnalyzerConfigTable(runData), workbookPath, "LogicAnalyzerConfig");
writeWorkbookSheet(runData.logs.boardRxLog, workbookPath, "BoardRxLog");
writeWorkbookSheet(runData.logs.boardCommitLog, workbookPath, "BoardCommitLog");
writeWorkbookSheet(runData.logs.referenceCapture, workbookPath, "ReferenceCapture");
writeWorkbookSheet(runData.logs.trainerPpmCapture, workbookPath, "TrainerPpmCapture");
writeWorkbookSheet(runData.logs.receiverCapture, workbookPath, "ReceiverCapture");
writeWorkbookSheet(runData.logs.computerToArduinoRxLatency, workbookPath, "ComputerToArduinoRxLatency");
writeWorkbookSheet(runData.logs.arduinoReceiveToPpmCommitLatency, workbookPath, "ArduinoReceiveToPpmCommitLatency");
writeWorkbookSheet(runData.logs.anchorToPpmLatency, workbookPath, "AnchorToPpmLatency");
writeWorkbookSheet(runData.logs.anchorToReceiverLatency, workbookPath, "AnchorToReceiverLatency");
writeWorkbookSheet(runData.logs.ppmToReceiverLatency, workbookPath, "PpmToReceiverLatency");
writeWorkbookSheet(runData.logs.computerToReceiverLatency, workbookPath, "ComputerToReceiverLatency");
writeWorkbookSheet(runData.logs.scheduledToReceiverLatency, workbookPath, "ScheduledToReceiverLatency");
writeWorkbookSheet(runData.logs.directLatencyEvents, workbookPath, "DirectLatencyEvents");
writeWorkbookSheet(runData.logs.truthSubsetEvents, workbookPath, "TruthSubsetEvents");
writeWorkbookSheet(runData.logs.truthSubsetSummary, workbookPath, "TruthSubsetSummary");
writeWorkbookSheet(runData.logs.alignmentSummary, workbookPath, "AlignmentSummary");
writeWorkbookSheet(runData.logs.matchedEvents, workbookPath, "MatchedEvents");
end

function workbookPath = resolveWritableWorkbookPath(requestedWorkbookPath)
workbookPath = string(requestedWorkbookPath);
if ~isfile(workbookPath)
    return;
end

fileId = fopen(workbookPath, "a");
if fileId ~= -1
    fclose(fileId);
    return;
end

[folderPath, fileName, fileExtension] = fileparts(workbookPath);
timestampText = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
workbookPath = fullfile(folderPath, fileName + "_locked_" + timestampText + fileExtension);
warning("Transmitter_Test:WorkbookLocked", ...
    "Workbook '%s' is locked. Writing to '%s' instead.", ...
    char(requestedWorkbookPath), ...
    char(workbookPath));
end

function criticalSettingsTable = buildTransmitterCriticalSettingsTable(runData)
config = runData.config;
arduinoInfo = runData.connectionInfo.arduino;
settings = { ...
    "Run", "RunLabel", formatSettingValue(config.runLabel); ...
    "Run", "Status", formatSettingValue(runData.runInfo.status); ...
    "Run", "OutputFolder", formatSettingValue(config.outputFolder); ...
    "Run", "LoggerFolder", formatSettingValue(config.arduinoTransport.loggerOutputFolder); ...
    "Transport", "SerialPort", formatSettingValue(config.arduinoTransport.serialPort); ...
    "Transport", "BaudRate", formatSettingValue(config.arduinoTransport.baudRate); ...
    "Transport", "CommandEncoding", formatSettingValue(config.arduinoTransport.commandEncoding); ...
    "Transport", "RxTimestampSource", formatSettingValue(config.arduinoTransport.rxTimestampSource); ...
    "Transport", "FirmwareVersion", formatSettingValue(arduinoInfo.loggerFirmwareVersion); ...
    "TrainerPPM", "OutputPin", formatSettingValue(config.trainerPpm.outputPin); ...
    "TrainerPPM", "ReferencePin", formatSettingValue(config.trainerPpm.referencePin); ...
    "TrainerPPM", "FrameLengthUs", formatSettingValue(config.trainerPpm.frameLengthUs); ...
    "TrainerPPM", "MarkWidthUs", formatSettingValue(config.trainerPpm.markWidthUs); ...
    "TrainerPPM", "ChannelCount", formatSettingValue(config.trainerPpm.channelCount); ...
    "ReferenceStrobe", "Enabled", formatSettingValue(config.referenceStrobe.enabled); ...
    "ReferenceStrobe", "Mode", formatSettingValue(config.referenceStrobe.mode); ...
    "ReferenceStrobe", "PulseWidthUs", formatSettingValue(config.referenceStrobe.pulseWidthUs); ...
    "LogicAnalyzer", "Enabled", formatSettingValue(config.logicAnalyzer.enabled); ...
    "LogicAnalyzer", "Mode", formatSettingValue(config.logicAnalyzer.mode); ...
    "LogicAnalyzer", "SigrokCliPath", formatSettingValue(config.logicAnalyzer.sigrokCliPath); ...
    "LogicAnalyzer", "DeviceDriver", formatSettingValue(config.logicAnalyzer.deviceDriver); ...
    "LogicAnalyzer", "DeviceId", formatSettingValue(config.logicAnalyzer.deviceId); ...
    "LogicAnalyzer", "SampleRateHz", formatSettingValue(config.logicAnalyzer.sampleRateHz); ...
    "LogicAnalyzer", "LogicStateExportPath", formatSettingValue(config.logicAnalyzer.logicStateExportPath); ...
    "Matching", "Mode", formatSettingValue(config.matching.mode); ...
    "Matching", "AnchorPriority", formatSettingValue(config.matching.anchorPriority); ...
    "Matching", "PpmChangeThresholdUs", formatSettingValue(config.matching.ppmChangeThresholdUs); ...
    "Matching", "EnableDownstreamMatching", formatSettingValue(config.matching.enableDownstreamMatching); ...
    "Matching", "ReceiverChangeThresholdUs", formatSettingValue(config.matching.receiverChangeThresholdUs); ...
    "Matching", "TransitionLeadSeconds", formatSettingValue(config.matching.transitionLeadSeconds); ...
    "Matching", "StateToleranceUs", formatSettingValue(config.matching.stateToleranceUs); ...
    "Matching", "StableStatePulseCount", formatSettingValue(config.matching.stableStatePulseCount); ...
    "Matching", "TruthSubsetMode", formatSettingValue(config.matching.truthSubsetMode); ...
    "Matching", "TransitionTargetToleranceUs", formatSettingValue(config.matching.transitionTargetToleranceUs); ...
    "Matching", "TransitionPreviousToleranceUs", formatSettingValue(config.matching.transitionPreviousToleranceUs); ...
    "Matching", "DownstreamFrameWindowSeconds", formatSettingValue(config.matching.downstreamFrameWindowSeconds); ...
    "Matching", "MaxCommitAssociationSeconds", formatSettingValue(config.matching.maxCommitAssociationSeconds); ...
    "Matching", "UseEstimatedLagForSearch", formatSettingValue(config.matching.useEstimatedLagForSearch); ...
    "Matching", "RequireTrainerBeforeReceiver", formatSettingValue(config.matching.requireTrainerBeforeReceiver); ...
    "Matching", "ComposeReceiverLatencyFromChain", formatSettingValue(config.matching.composeReceiverLatencyFromChain); ...
    "Matching", "MaxResponseWindowSeconds", formatSettingValue(config.matching.maxResponseWindowSeconds); ...
    "Debug", "StrictDownstreamMatching", formatSettingValue(config.debug.strictDownstreamMatching); ...
    "Analysis", "ComposeReceiverLatencyFromChain", formatSettingValue(config.analysis.composeReceiverLatencyFromChain); ...
    "Analysis", "LatencySummarySource", formatSettingValue(runData.logs.latencySummarySource)};
criticalSettingsTable = cell2table(settings, 'VariableNames', {'Category', 'Setting', 'Value'});
end

function logicAnalyzerConfigTable = buildTransmitterLogicAnalyzerConfigTable(runData)
config = runData.config.logicAnalyzer;
settings = { ...
    "enabled", formatSettingValue(config.enabled); ...
    "mode", formatSettingValue(config.mode); ...
    "automationMode", formatSettingValue(config.automationMode); ...
    "sigrokCliPath", formatSettingValue(config.sigrokCliPath); ...
    "deviceDriver", formatSettingValue(config.deviceDriver); ...
    "deviceId", formatSettingValue(config.deviceId); ...
    "sampleRateHz", formatSettingValue(config.sampleRateHz); ...
    "maximumCaptureSamples", formatSettingValue(config.maximumCaptureSamples); ...
    "captureStartLeadSeconds", formatSettingValue(config.captureStartLeadSeconds); ...
    "captureStopLagSeconds", formatSettingValue(config.captureStopLagSeconds); ...
    "channels", formatSettingValue(config.channels); ...
    "channelNames", formatSettingValue(config.channelNames); ...
    "logicStateExportPath", formatSettingValue(config.logicStateExportPath); ...
    "rawCapturePath", formatSettingValue(config.rawCapturePath); ...
    "decodedReferencePath", formatSettingValue(config.decodedReferencePath); ...
    "decodedTrainerPpmPath", formatSettingValue(config.decodedTrainerPpmPath); ...
    "decodedReceiverPath", formatSettingValue(config.decodedReceiverPath); ...
    "storeStdoutPath", formatSettingValue(config.storeStdoutPath); ...
    "storeStderrPath", formatSettingValue(config.storeStderrPath)};
logicAnalyzerConfigTable = cell2table(settings, 'VariableNames', {'Setting', 'Value'});
end

function writeWorkbookSheet(sheetData, workbookPath, requestedSheetName)
sheetName = mapWorkbookSheetName(requestedSheetName);
if istimetable(sheetData)
    writetable(timetableToExportTable(sheetData), workbookPath, 'Sheet', char(sheetName));
else
    writetable(sheetData, workbookPath, 'Sheet', char(sheetName));
end
end

function sheetName = mapWorkbookSheetName(requestedSheetName)
sheetName = string(requestedSheetName);
if sheetName == "ArduinoReceiveToPpmCommitLatency"
    sheetName = "ArduinoRxToPpmCommitLatency";
end
if strlength(sheetName) > 31
    sheetName = extractBefore(sheetName, 32);
end
end

function createFolderIfMissing(folderPath)
if strlength(folderPath) == 0
    return;
end
if ~isfolder(folderPath)
    mkdir(folderPath);
end
end

function deleteFileIfPresent(filePath)
if strlength(filePath) == 0 || ~isfile(filePath)
    return;
end
delete(filePath);
end

function quotedText = quoteWindowsArgument(argumentText)
argumentText = string(argumentText);
quotedText = '"' + replace(argumentText, '"', '""') + '"';
end

function argumentsText = buildCmdExeArguments(commandText)
argumentsText = '/d /s /c "' + string(commandText) + '"';
end

function [status, outputText] = runWindowsCommand(commandText)
[status, outputText] = system(char("cmd.exe " + buildCmdExeArguments(commandText)));
outputText = string(outputText);
end

function sigrokCliPath = resolveSigrokCliPath(sigrokCliPath)
sigrokCliPath = strtrim(string(sigrokCliPath));
if strlength(sigrokCliPath) == 0
    error("Transmitter_Test:MissingSigrokCliPath", "sigrok-cli path must be resolved before use.");
end
end

function driverSpec = buildSigrokDriverSpec(logicAnalyzer)
driverSpec = string(logicAnalyzer.deviceDriver);
deviceId = strtrim(string(logicAnalyzer.deviceId));
if strlength(deviceId) == 0
    return;
end
if contains(deviceId, "=")
    driverSpec = driverSpec + ":" + deviceId;
else
    driverSpec = driverSpec + ":conn=" + deviceId;
end
end

function channelAssignments = buildSigrokChannelAssignments(channels, channelNames)
channelAssignments = strings(numel(channels), 1);
for channelIndex = 1:numel(channels)
    channelAssignments(channelIndex) = formatSigrokChannelName(channels(channelIndex)) + "=" + string(channelNames(channelIndex));
end
channelAssignments = join(channelAssignments, ",");
end

function channelName = formatSigrokChannelName(channelIndex)
channelName = "D" + string(round(double(channelIndex)));
end

function outputFormatText = resolveSigrokLogicStateOutputFormat(outputFormatSetting)
outputFormatSetting = string(outputFormatSetting);
if outputFormatSetting == "logic_state_csv"
    outputFormatText = "csv:header=true:label=channel:time=true:dedup=true:comment=#";
else
    outputFormatText = outputFormatSetting;
end
end

function validateLogicStateExportFileSize(filePath, maximumBytes)
fileInfo = dir(filePath);
if isempty(fileInfo)
    error("Transmitter_Test:MissingLogicStateExport", "logicStateExportPath file was not created: %s", char(filePath));
end

if fileInfo.bytes > maximumBytes
    error( ...
        "Transmitter_Test:LogicStateExportTooLarge", ...
        [ ...
        "logicStateExportPath grew to %.3f GB, which indicates sigrok exported dense sample-by-sample CSV instead of a compact deduplicated logic-state table. " + ...
        "Reduce capture duration or sample rate, or adjust the local sigrok CSV format."], ...
        double(fileInfo.bytes) ./ 1e9);
end
end

function canonicalNames = canonicalizeLogicStateNames(variableNames)
canonicalNames = lower(regexprep(string(variableNames), '[^a-zA-Z0-9]+', ''));
end

function columnIndex = findMatchingLogicStateColumnIndex(canonicalNames, targetName, channelNumber)
aliasNames = unique([ ...
    canonicalizeLogicStateNames(targetName), ...
    canonicalizeLogicStateNames("D" + string(channelNumber)), ...
    canonicalizeLogicStateNames("CH" + string(channelNumber)), ...
    canonicalizeLogicStateNames("Channel" + string(channelNumber)), ...
    canonicalizeLogicStateNames("logic" + string(channelNumber)), ...
    canonicalizeLogicStateNames(string(channelNumber))]);
columnIndex = [];
for aliasIndex = 1:numel(aliasNames)
    matchedIndex = find(canonicalNames == aliasNames(aliasIndex), 1, "first");
    if ~isempty(matchedIndex)
        columnIndex = matchedIndex;
        return;
    end
end
end

function numericData = convertLogicStateColumnToNumeric(columnData)
if isnumeric(columnData) || islogical(columnData)
    numericData = reshape(double(columnData), [], 1);
    return;
end

columnText = lower(strtrim(string(columnData)));
numericData = nan(numel(columnText), 1);
numericData(columnText == "1" | columnText == "high" | columnText == "true") = 1;
numericData(columnText == "0" | columnText == "low" | columnText == "false") = 0;
remainingMask = isnan(numericData) & strlength(columnText) > 0;
if any(remainingMask)
    numericData(remainingMask) = str2double(columnText(remainingMask));
end
end

function timeSeconds = normalizeSigrokTimeColumnToSeconds(rawTimeData, sampleRateHz)
timeSeconds = reshape(double(rawTimeData), [], 1);
validMask = isfinite(timeSeconds);
if ~any(validMask)
    return;
end

validTimes = timeSeconds(validMask);
if max(validTimes) <= 1e3
    return;
end

positiveDiffs = diff(validTimes);
positiveDiffs = positiveDiffs(positiveDiffs > 0);
if isempty(positiveDiffs)
    return;
end

candidateScales = [1, 1e-3, 1e-6, 1e-9];
bestScale = 1;
bestScore = inf;
bestDuration = inf;
for scaleIndex = 1:numel(candidateScales)
    scale = candidateScales(scaleIndex);
    scaledDiffs = positiveDiffs .* scale .* double(sampleRateHz);
    roundedDiffs = round(scaledDiffs);
    integerResidual = mean(abs(scaledDiffs - roundedDiffs));
    medianStepSamples = median(scaledDiffs);
    totalDurationSeconds = max(validTimes) .* scale;
    if medianStepSamples < 1
        continue;
    end
    durationPenalty = 0;
    if totalDurationSeconds < 0.1
        durationPenalty = durationPenalty + 1e6;
    end
    if totalDurationSeconds > 1e3
        durationPenalty = durationPenalty + 1e6;
    end
    score = integerResidual + durationPenalty;
    if score < bestScore || (abs(score - bestScore) < 1e-9 && totalDurationSeconds < bestDuration)
        bestScale = scale;
        bestScore = score;
        bestDuration = totalDurationSeconds;
    end
end

timeSeconds(validMask) = validTimes .* bestScale;
end

function columnIndex = resolveLogicStateRoleColumnIndex(logicState, config, roleName)
switch string(roleName)
    case "reference"
        roleChannel = config.logicAnalyzer.channelRoleMap.reference;
    case "trainerPpm"
        roleChannel = config.logicAnalyzer.channelRoleMap.trainerPpm;
    otherwise
        error("Transmitter_Test:UnsupportedLogicStateRole", "Unsupported logic-state role: %s", char(roleName));
end
columnIndex = resolveLogicStateChannelColumnIndex(logicState, config, roleChannel);
end

function columnIndex = resolveLogicStateChannelColumnIndex(logicState, config, roleChannel)
configurationIndex = find(double(config.logicAnalyzer.channels) == double(roleChannel), 1, "first");
if isempty(configurationIndex)
    error("Transmitter_Test:MissingLogicStateChannelRole", "Configured logic analyser channel %d was not found.", roleChannel);
end
configuredName = string(config.logicAnalyzer.channelNames(configurationIndex));
columnIndex = find(logicState.channelNames == configuredName, 1, "first");
if isempty(columnIndex)
    error("Transmitter_Test:MissingLogicStateChannelColumn", "Logic-state data is missing channel %s.", char(configuredName));
end
end

function debouncedSamples = applySampleDebounce(edgeSamples, debounceSamples)
debouncedSamples = edgeSamples;
if isempty(edgeSamples)
    return;
end

keepMask = false(size(edgeSamples));
keepMask(1) = true;
lastSample = edgeSamples(1);
for sampleIndex = 2:numel(edgeSamples)
    if (edgeSamples(sampleIndex) - lastSample) >= debounceSamples
        keepMask(sampleIndex) = true;
        lastSample = edgeSamples(sampleIndex);
    end
end
debouncedSamples = edgeSamples(keepMask);
end

function [pulseStartSamples, pulseSampleCounts] = pairEdgeSamples(risingSamples, fallingSamples)
pulseStartSamples = zeros(0, 1);
pulseSampleCounts = zeros(0, 1);
if isempty(risingSamples) || isempty(fallingSamples)
    return;
end

fallIndex = 1;
for riseIndex = 1:numel(risingSamples)
    while fallIndex <= numel(fallingSamples) && fallingSamples(fallIndex) <= risingSamples(riseIndex)
        fallIndex = fallIndex + 1;
    end
    if fallIndex > numel(fallingSamples)
        break;
    end
    pulseStartSamples(end + 1, 1) = risingSamples(riseIndex); %#ok<AGROW>
    pulseSampleCounts(end + 1, 1) = fallingSamples(fallIndex) - risingSamples(riseIndex); %#ok<AGROW>
    fallIndex = fallIndex + 1;
end
end

function pulseCapture = buildEmptyPulseCaptureTable(includeFrameIndex)
if nargin < 1
    includeFrameIndex = false;
end

pulseCapture = table( ...
    strings(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', {'surface_name', 'time_s', 'pulse_us', 'sample_index', 'sample_count', 'sample_rate_hz'});
if includeFrameIndex
    pulseCapture.frame_index = zeros(0, 1);
end
end

