function runData = Arduino_Test(config)
%ARDUINO_TEST Execute an Arduino-only servo command test over WiFi.
%   runData = Arduino_Test(config) connects to the Arduino Nano 33 IoT,
%   commands one or all four servos with a configurable profile, logs the
%   desired commands, and logs the Arduino-applied commands via
%   readPosition.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);
runData.runInfo.startTime = datetime("now");

assignin("base", "ArduinoTestLatestState", struct([]));
assignin("base", "ArduinoTestRunData", runData);

arduinoConnection = [];
servoObjects = {};
cleanupHandle = onCleanup(@() cleanupResources(arduinoConnection, servoObjects, config));

try
    [arduinoConnection, servoObjects, arduinoInfo] = connectToArduino(config);
    runData.connectionInfo.arduino = arduinoInfo;
    assignin("base", "ArduinoTestRunData", runData);

    if ~arduinoInfo.isConnected
        runData.runInfo.status = "arduino_connection_failed";
        runData.runInfo.reason = arduinoInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ArduinoTestRunData", runData);
        printConnectionStatus(runData);
        return;
    end

    printConnectionStatus(runData);

    moveServosToNeutral(servoObjects, config.servoNeutralPositions);
    pause(config.neutralSettleSeconds);

    [storage, runInfo] = executeArduinoTest(servoObjects, config);
    runData.runInfo = runInfo;
    runData.logs = buildLogTimetables(storage, config);
    runData.surfaceSummary = buildSurfaceSummary(storage, config);
    runData.artifacts = exportRunData(runData);

    assignin("base", "ArduinoTestRunData", runData);
    assignin("base", "ArduinoTestLatestState", buildLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupResources(arduinoConnection, servoObjects, config);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    assignin("base", "ArduinoTestRunData", runData);
    rethrow(executionException);
end
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));
defaultSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];
defaultSurfacePins = ["D9"; "D10"; "D11"; "D12"];

config.arduinoIPAddress = getTextScalarField(config, "arduinoIPAddress", "192.168.0.33");
config.arduinoBoard = getTextScalarField(config, "arduinoBoard", "Nano33IoT");
config.arduinoPort = getOptionalScalarField(config, "arduinoPort", NaN);

config.surfaceNames = getStringArrayField(config, "surfaceNames", defaultSurfaceNames);
config.surfacePins = getStringArrayField(config, "surfacePins", defaultSurfacePins);
config.servoNeutralPositions = getNumericColumnField(config, "servoNeutralPositions", 0.5 .* ones(numel(config.surfaceNames), 1));
config.servoUnitsPerDegree = getNumericColumnField(config, "servoUnitsPerDegree", (1 / 180) .* ones(numel(config.surfaceNames), 1));
config.servoMinimumPositions = getNumericColumnField(config, "servoMinimumPositions", zeros(numel(config.surfaceNames), 1));
config.servoMaximumPositions = getNumericColumnField(config, "servoMaximumPositions", ones(numel(config.surfaceNames), 1));
config.servoMinPulseDurationSeconds = getNumericColumnField(config, "servoMinPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.servoMaxPulseDurationSeconds = getNumericColumnField(config, "servoMaxPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.commandDeflectionScales = getNumericColumnField(config, "commandDeflectionScales", ones(numel(config.surfaceNames), 1));
config.commandDeflectionOffsetsDegrees = getNumericColumnField(config, "commandDeflectionOffsetsDegrees", zeros(numel(config.surfaceNames), 1));

config.commandMode = getTextScalarField(config, "commandMode", "single");
config.singleSurfaceName = getTextScalarField(config, "singleSurfaceName", "Aileron_L");
config.commandProfile = normalizeCommandProfile(getFieldOrDefault(config, "commandProfile", struct()));

config.neutralSettleSeconds = getPositiveScalarField(config, "neutralSettleSeconds", 1.0);
config.returnToNeutralOnExit = getLogicalField(config, "returnToNeutralOnExit", true);

config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "C_Arduino_Test"));
config.runLabel = getTextScalarField(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    timeStamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    config.runLabel = timeStamp + "_Test";
end

validateattributes(config.arduinoPort, {"numeric"}, {"real", "scalar"}, char(mfilename), 'arduinoPort');

surfaceCount = numel(config.surfaceNames);
mustHaveMatchingLength(config.surfacePins, surfaceCount, "surfacePins");

validateattributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoNeutralPositions');
validateattributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoUnitsPerDegree');
validateattributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMinimumPositions');
validateattributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMaximumPositions');
validateattributes(config.servoMinPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, char(mfilename), 'servoMinPulseDurationSeconds');
validateattributes(config.servoMaxPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, char(mfilename), 'servoMaxPulseDurationSeconds');
validateattributes(config.commandDeflectionScales, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionScales');
validateattributes(config.commandDeflectionOffsetsDegrees, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionOffsetsDegrees');

if any(abs(config.servoUnitsPerDegree) <= eps)
    error("Arduino_Test:InvalidServoScale", "servoUnitsPerDegree must be nonzero for every surface.");
end

if any(config.servoMinimumPositions > config.servoMaximumPositions)
    error("Arduino_Test:InvalidServoLimits", "servoMinimumPositions must be less than or equal to servoMaximumPositions.");
end

if any(config.servoNeutralPositions < config.servoMinimumPositions) || any(config.servoNeutralPositions > config.servoMaximumPositions)
    error("Arduino_Test:NeutralOutsideLimits", "servoNeutralPositions must lie inside the configured servo limits.");
end

validCommandModes = ["single", "all"];
if ~any(config.commandMode == validCommandModes)
    error("Arduino_Test:InvalidCommandMode", "commandMode must be 'single' or 'all'.");
end

if ~any(config.singleSurfaceName == config.surfaceNames)
    error("Arduino_Test:InvalidSingleSurface", ...
        "singleSurfaceName must match one of: %s.", ...
        char(join(config.surfaceNames, ", ")));
end

activeSurfaceMask = false(surfaceCount, 1);
if config.commandMode == "single"
    activeSurfaceMask(config.surfaceNames == config.singleSurfaceName) = true;
else
    activeSurfaceMask(:) = true;
end

config.activeSurfaceMask = activeSurfaceMask;
config.activeSurfaceNames = config.surfaceNames(activeSurfaceMask);
config.surfaceSetup = buildSurfaceSetupTable(config);
end

function commandProfile = normalizeCommandProfile(commandProfileConfig)
if isempty(commandProfileConfig)
    commandProfileConfig = struct();
end

commandProfile.type = getTextScalarField(commandProfileConfig, "type", "sine");
commandProfile.sampleTimeSeconds = getPositiveScalarField(commandProfileConfig, "sampleTimeSeconds", 0.10);
commandProfile.preCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "preCommandNeutralSeconds", 0.0);
commandProfile.postCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "postCommandNeutralSeconds", 1.0);
commandProfile.durationSeconds = getPositiveScalarField(commandProfileConfig, "durationSeconds", 6.0);
commandProfile.amplitudeDegrees = getScalarField(commandProfileConfig, "amplitudeDegrees", 45.0);
commandProfile.offsetDegrees = getScalarField(commandProfileConfig, "offsetDegrees", 0.0);
commandProfile.frequencyHz = getPositiveScalarField(commandProfileConfig, "frequencyHz", 0.5);
commandProfile.phaseDegrees = getScalarField(commandProfileConfig, "phaseDegrees", 90.0);
commandProfile.doubletHoldSeconds = getPositiveScalarField(commandProfileConfig, "doubletHoldSeconds", 0.5);
commandProfile.customTimeSeconds = getNumericVectorField(commandProfileConfig, "customTimeSeconds", []);
commandProfile.customDeflectionDegrees = getNumericVectorField(commandProfileConfig, "customDeflectionDegrees", []);
commandProfile.customInterpolationMethod = getTextScalarField(commandProfileConfig, "customInterpolationMethod", "linear");
commandProfile.customFunction = getFieldOrDefault(commandProfileConfig, "customFunction", []);

validProfileTypes = ["sine", "square", "doublet", "custom", "function"];
if ~any(commandProfile.type == validProfileTypes)
    error("Arduino_Test:InvalidProfileType", ...
        "commandProfile.type must be one of: %s.", ...
        char(join(validProfileTypes, ", ")));
end

if ~isempty(commandProfile.customFunction) && ~isa(commandProfile.customFunction, "function_handle")
    error("Arduino_Test:InvalidCustomFunction", "commandProfile.customFunction must be a function handle when provided.");
end

if commandProfile.type == "doublet"
    commandProfile.durationSeconds = max(commandProfile.durationSeconds, 2 .* commandProfile.doubletHoldSeconds);
end

if commandProfile.type == "custom"
    validateattributes(commandProfile.customTimeSeconds, {"numeric"}, {"real", "finite", "vector", "nonempty"}, char(mfilename), 'commandProfile.customTimeSeconds');
    validateattributes(commandProfile.customDeflectionDegrees, {"numeric"}, {"real", "finite", "vector", "numel", numel(commandProfile.customTimeSeconds)}, char(mfilename), 'commandProfile.customDeflectionDegrees');

    commandProfile.customTimeSeconds = reshape(double(commandProfile.customTimeSeconds), [], 1);
    commandProfile.customDeflectionDegrees = reshape(double(commandProfile.customDeflectionDegrees), [], 1);

    if any(diff(commandProfile.customTimeSeconds) <= 0)
        error("Arduino_Test:InvalidCustomTimeVector", "commandProfile.customTimeSeconds must be strictly increasing.");
    end

    commandProfile.durationSeconds = commandProfile.customTimeSeconds(end);
end

if commandProfile.type == "function" && isempty(commandProfile.customFunction)
    error("Arduino_Test:MissingCustomFunction", "commandProfile.customFunction is required when type is 'function'.");
end
end

function runData = initializeRunData(config)
runData = struct( ...
    "config", config, ...
    "connectionInfo", struct("arduino", struct()), ...
    "runInfo", struct( ...
        "status", "initialized", ...
        "reason", "", ...
        "startTime", NaT, ...
        "stopTime", NaT, ...
        "sampleCount", 0, ...
        "scheduledDurationSeconds", NaN), ...
    "surfaceSetup", config.surfaceSetup, ...
    "logs", struct(), ...
    "surfaceSummary", table(), ...
    "artifacts", struct( ...
        "matFilePath", "", ...
        "workbookPath", ""));
end

function [arduinoConnection, servoObjects, connectionInfo] = connectToArduino(config)
surfaceCount = numel(config.surfaceNames);
servoObjects = cell(surfaceCount, 1);
arduinoConnection = [];

connectionInfo = struct( ...
    "ipAddress", config.arduinoIPAddress, ...
    "boardName", config.arduinoBoard, ...
    "port", config.arduinoPort, ...
    "isConnected", false, ...
    "connectElapsedSeconds", NaN, ...
    "connectionMessage", "", ...
    "availableLibraries", strings(0, 1), ...
    "surfaceNames", config.surfaceNames, ...
    "surfacePins", config.surfacePins);

connectStart = tic;
try
    arduinoConnection = createArduinoConnection(config);

    if isprop(arduinoConnection, "Libraries")
        connectionInfo.availableLibraries = reshape(string(arduinoConnection.Libraries), [], 1);
    end

    for surfaceIndex = 1:surfaceCount
        servoObjects{surfaceIndex} = createServoObject( ...
            arduinoConnection, ...
            config.surfacePins(surfaceIndex), ...
            config.servoMinPulseDurationSeconds(surfaceIndex), ...
            config.servoMaxPulseDurationSeconds(surfaceIndex));
    end

    connectionInfo.isConnected = true;
    connectionInfo.connectionMessage = "Connected";
catch connectionException
    connectionInfo.connectionMessage = string(connectionException.message);
end

connectionInfo.connectElapsedSeconds = toc(connectStart);
end

function arduinoConnection = createArduinoConnection(config)
if isnan(config.arduinoPort)
    arduinoConnection = arduino(char(config.arduinoIPAddress), char(config.arduinoBoard));
else
    arduinoConnection = arduino(char(config.arduinoIPAddress), char(config.arduinoBoard), config.arduinoPort);
end
end

function servoObject = createServoObject(arduinoConnection, pinName, minimumPulseDuration, maximumPulseDuration)
if isfinite(minimumPulseDuration) && isfinite(maximumPulseDuration)
    servoObject = servo( ...
        arduinoConnection, ...
        char(pinName), ...
        "MinPulseDuration", minimumPulseDuration, ...
        "MaxPulseDuration", maximumPulseDuration);
else
    servoObject = servo(arduinoConnection, char(pinName));
end
end

function printConnectionStatus(runData)
fprintf("\nArduino_Test connection summary\n");
fprintf("  Arduino (%s): %s\n", ...
    runData.config.arduinoIPAddress, ...
    char(getStatusText(runData.connectionInfo.arduino)));
fprintf("  Active command surfaces: %s\n\n", char(join(runData.config.activeSurfaceNames, ", ")));
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

function [storage, runInfo] = executeArduinoTest(servoObjects, config)
[scheduledTimeSeconds, profileInfo] = buildCommandSchedule(config.commandProfile);
surfaceCount = numel(config.surfaceNames);
sampleCount = numel(scheduledTimeSeconds);

storage = initializeStorage(sampleCount, surfaceCount);
storage.scheduledTimeSeconds = scheduledTimeSeconds;
storage.profileInfo = profileInfo;

runInfo = struct( ...
    "status", "running", ...
    "reason", "", ...
    "startTime", datetime("now"), ...
    "stopTime", NaT, ...
    "sampleCount", 0, ...
    "scheduledDurationSeconds", profileInfo.totalDurationSeconds);

testStart = tic;

for sampleIndex = 1:sampleCount
    scheduledSampleTimeSeconds = scheduledTimeSeconds(sampleIndex);
    waitForScheduledTime(testStart, scheduledSampleTimeSeconds);

    baseCommandDegrees = evaluateBaseCommandDegrees(profileInfo, scheduledSampleTimeSeconds);
    desiredDeflectionsDegrees = zeros(1, surfaceCount);

    desiredDeflectionsDegrees(config.activeSurfaceMask.') = ...
        config.commandDeflectionScales(config.activeSurfaceMask).' .* baseCommandDegrees + ...
        config.commandDeflectionOffsetsDegrees(config.activeSurfaceMask).';

    commandedServoPositions = ...
        config.servoNeutralPositions.' + desiredDeflectionsDegrees .* config.servoUnitsPerDegree.';
    commandedServoPositionsClipped = min( ...
        max(commandedServoPositions, config.servoMinimumPositions.'), ...
        config.servoMaximumPositions.');
    saturatedMask = abs(commandedServoPositionsClipped - commandedServoPositions) > 10 .* eps;

    storage.commandWriteStartSeconds(sampleIndex) = toc(testStart);
    commandDispatchSeconds = writeServoPositions( ...
        servoObjects, ...
        commandedServoPositionsClipped, ...
        testStart, ...
        config.activeSurfaceMask.');
    storage.commandDispatchSeconds(sampleIndex, :) = commandDispatchSeconds;
    storage.commandWriteStopSeconds(sampleIndex) = lastFiniteTimestamp( ...
        commandDispatchSeconds, ...
        storage.commandWriteStartSeconds(sampleIndex));

    storage.arduinoReadStartSeconds(sampleIndex) = toc(testStart);
    [appliedServoPositions, arduinoEchoSeconds] = readServoPositions( ...
        servoObjects, ...
        testStart, ...
        config.activeSurfaceMask.');
    appliedServoPositions(~config.activeSurfaceMask.') = config.servoNeutralPositions(~config.activeSurfaceMask).';
    storage.arduinoEchoSeconds(sampleIndex, :) = arduinoEchoSeconds;
    storage.arduinoReadStopSeconds(sampleIndex) = lastFiniteTimestamp( ...
        arduinoEchoSeconds, ...
        storage.arduinoReadStartSeconds(sampleIndex));

    storage.sampleCount = sampleIndex;
    storage.baseCommandDegrees(sampleIndex) = baseCommandDegrees;
    storage.desiredDeflectionsDegrees(sampleIndex, :) = desiredDeflectionsDegrees;
    storage.commandedServoPositions(sampleIndex, :) = commandedServoPositionsClipped;
    storage.commandSaturated(sampleIndex, :) = saturatedMask;
    storage.appliedServoPositions(sampleIndex, :) = appliedServoPositions;
    storage.appliedEquivalentDegrees(sampleIndex, :) = ...
        (appliedServoPositions - config.servoNeutralPositions.') ./ config.servoUnitsPerDegree.';

    assignin("base", "ArduinoTestLatestState", buildLatestState(storage, config, sampleIndex));
end

runInfo.status = "completed";
runInfo.stopTime = datetime("now");
runInfo.sampleCount = storage.sampleCount;

if config.returnToNeutralOnExit
    moveServosToNeutral(servoObjects, config.servoNeutralPositions);
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

function storage = initializeStorage(sampleCount, surfaceCount)
storage = struct( ...
    "sampleCount", 0, ...
    "scheduledTimeSeconds", nan(sampleCount, 1), ...
    "baseCommandDegrees", nan(sampleCount, 1), ...
    "commandWriteStartSeconds", nan(sampleCount, 1), ...
    "commandWriteStopSeconds", nan(sampleCount, 1), ...
    "commandDispatchSeconds", nan(sampleCount, surfaceCount), ...
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

function moveServosToNeutral(servoObjects, neutralPositions)
neutralRow = reshape(neutralPositions, 1, []);
writeServoPositions(servoObjects, neutralRow);
end

function dispatchTimesSeconds = writeServoPositions(servoObjects, servoPositions, referenceTimer, activeSurfaceMask)
if nargin < 3
    referenceTimer = [];
end

if nargin < 4 || isempty(activeSurfaceMask)
    activeSurfaceMask = true(1, numel(servoObjects));
end

dispatchTimesSeconds = nan(1, numel(servoObjects));

for surfaceIndex = 1:numel(servoObjects)
    if ~activeSurfaceMask(surfaceIndex)
        continue;
    end

    writePosition(servoObjects{surfaceIndex}, servoPositions(surfaceIndex));

    if ~isempty(referenceTimer)
        dispatchTimesSeconds(surfaceIndex) = toc(referenceTimer);
    end
end
end

function [servoPositions, echoTimesSeconds] = readServoPositions(servoObjects, referenceTimer, activeSurfaceMask)
if nargin < 2
    referenceTimer = [];
end

if nargin < 3 || isempty(activeSurfaceMask)
    activeSurfaceMask = true(1, numel(servoObjects));
end

servoPositions = nan(1, numel(servoObjects));
echoTimesSeconds = nan(1, numel(servoObjects));

for surfaceIndex = 1:numel(servoObjects)
    if ~activeSurfaceMask(surfaceIndex)
        continue;
    end

    try
        servoPositions(surfaceIndex) = readPosition(servoObjects{surfaceIndex});
    catch
        servoPositions(surfaceIndex) = NaN;
    end

    if ~isempty(referenceTimer)
        echoTimesSeconds(surfaceIndex) = toc(referenceTimer);
    end
end
end

function logs = buildLogTimetables(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceNames = config.surfaceNames;
commandVariableNames = [ ...
    "base_command_deg", ...
    buildSurfaceVariableNames(surfaceNames, "desired_deg"), ...
    buildSurfaceVariableNames(surfaceNames, "command_position"), ...
    buildSurfaceVariableNames(surfaceNames, "command_saturated")];
commandLog = array2timetable( ...
    [ ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.desiredDeflectionsDegrees(sampleIndices, :), ...
    storage.commandedServoPositions(sampleIndices, :), ...
    double(storage.commandSaturated(sampleIndices, :))], ...
    'RowTimes', seconds(storage.commandWriteStopSeconds(sampleIndices)), ...
    'VariableNames', cellstr(commandVariableNames));

commandLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
commandLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
commandLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);

arduinoVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "applied_position"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_equivalent_deg")];
arduinoLog = array2timetable( ...
    [ ...
    storage.appliedServoPositions(sampleIndices, :), ...
    storage.appliedEquivalentDegrees(sampleIndices, :)], ...
    'RowTimes', seconds(storage.arduinoReadStopSeconds(sampleIndices)), ...
    'VariableNames', cellstr(arduinoVariableNames));

arduinoLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
arduinoLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
arduinoLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

computerToArduinoLatencySeconds = ...
    storage.arduinoEchoSeconds(sampleIndices, :) - storage.commandDispatchSeconds(sampleIndices, :);
latencyVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_echo_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_latency_s")];
computerToArduinoLatencyLog = array2timetable( ...
    [ ...
    storage.commandDispatchSeconds(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    computerToArduinoLatencySeconds], ...
    'RowTimes', seconds(storage.arduinoReadStopSeconds(sampleIndices)), ...
    'VariableNames', cellstr(latencyVariableNames));

computerToArduinoLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
computerToArduinoLatencyLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
computerToArduinoLatencyLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);
computerToArduinoLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
computerToArduinoLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

sampleSummary = table( ...
    storage.scheduledTimeSeconds(sampleIndices), ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.commandWriteStartSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices), ...
    storage.arduinoReadStartSeconds(sampleIndices), ...
    storage.arduinoReadStopSeconds(sampleIndices), ...
    'VariableNames', { ...
        'scheduled_time_s', ...
        'base_command_deg', ...
        'command_write_start_s', ...
        'command_write_stop_s', ...
        'arduino_read_start_s', ...
        'arduino_read_stop_s'});

logs = struct( ...
    "inputSignal", commandLog, ...
    "command", commandLog, ...
    "arduinoEcho", arduinoLog, ...
    "arduino", arduinoLog, ...
    "computerToArduinoLatency", computerToArduinoLatencyLog, ...
    "sampleSummary", sampleSummary);
end

function surfaceSummary = buildSurfaceSummary(storage, config)
surfaceCount = numel(config.surfaceNames);
appliedSampleCount = zeros(surfaceCount, 1);
saturationCount = zeros(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    appliedSampleCount(surfaceIndex) = sum(isfinite(storage.appliedServoPositions(:, surfaceIndex)));
    saturationCount(surfaceIndex) = sum(storage.commandSaturated(:, surfaceIndex));
end

surfaceSummary = table( ...
    config.surfaceNames, ...
    config.surfacePins, ...
    config.activeSurfaceMask, ...
    appliedSampleCount, ...
    saturationCount, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'ArduinoPin', ...
        'IsActive', ...
        'AppliedCommandSampleCount', ...
        'SaturationSampleCount'});
end

function latestState = buildLatestState(storage, config, sampleIndex)
surfaceCount = numel(config.surfaceNames);
surfaceStates = repmat(struct( ...
    "name", "", ...
    "desired_deg", NaN, ...
    "command_position", NaN, ...
    "applied_position", NaN, ...
    "applied_equivalent_deg", NaN), surfaceCount, 1);

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

function surfaceSetup = buildSurfaceSetupTable(config)
surfaceSetup = table( ...
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
        'SurfaceName', ...
        'ArduinoPin', ...
        'IsActive', ...
        'ServoNeutralPosition', ...
        'ServoUnitsPerDegree', ...
        'ServoMinimumPosition', ...
        'ServoMaximumPosition', ...
        'CommandScale', ...
        'CommandOffsetDeg'});
end

function artifacts = exportRunData(runData)
matFilePath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".mat");
workbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");

artifacts = struct( ...
    "matFilePath", string(matFilePath), ...
    "workbookPath", string(workbookPath));

save(matFilePath, "runData", "-v7.3");

writetable(buildCriticalSettingsTable(runData), workbookPath, 'Sheet', 'CriticalSettings');
writetable(timetableToExportTable(runData.logs.inputSignal), workbookPath, 'Sheet', 'InputSignal');
writetable(timetableToExportTable(runData.logs.arduinoEcho), workbookPath, 'Sheet', 'ArduinoEcho');
writetable( ...
    timetableToExportTable(runData.logs.computerToArduinoLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ComputerToArduinoLatency');
writetable(runData.surfaceSetup, workbookPath, 'Sheet', 'SurfaceSetup');
writetable(runData.logs.sampleSummary, workbookPath, 'Sheet', 'SampleSummary');
writetable(runData.surfaceSummary, workbookPath, 'Sheet', 'SurfaceSummary');
end

function criticalSettingsTable = buildCriticalSettingsTable(runData)
config = runData.config;
profile = config.commandProfile;
arduinoInfo = runData.connectionInfo.arduino;

settings = [ ...
    "Run", "RunLabel", formatSettingValue(config.runLabel); ...
    "Run", "Status", formatSettingValue(runData.runInfo.status); ...
    "Run", "Reason", formatSettingValue(runData.runInfo.reason); ...
    "Run", "StartTime", formatSettingValue(runData.runInfo.startTime); ...
    "Run", "StopTime", formatSettingValue(runData.runInfo.stopTime); ...
    "Run", "SampleCount", formatSettingValue(runData.runInfo.sampleCount); ...
    "Run", "ScheduledDurationSeconds", formatSettingValue(runData.runInfo.scheduledDurationSeconds); ...
    "Run", "OutputFolder", formatSettingValue(config.outputFolder); ...
    "Arduino", "IPAddress", formatSettingValue(config.arduinoIPAddress); ...
    "Arduino", "Board", formatSettingValue(config.arduinoBoard); ...
    "Arduino", "Port", formatSettingValue(config.arduinoPort); ...
    "Arduino", "IsConnected", formatSettingValue(arduinoInfo.isConnected); ...
    "Arduino", "ConnectionMessage", formatSettingValue(arduinoInfo.connectionMessage); ...
    "Arduino", "ConnectElapsedSeconds", formatSettingValue(arduinoInfo.connectElapsedSeconds); ...
    "Command", "Mode", formatSettingValue(config.commandMode); ...
    "Command", "SingleSurfaceName", formatSettingValue(config.singleSurfaceName); ...
    "Command", "ActiveSurfaces", formatSettingValue(config.activeSurfaceNames); ...
    "Command", "NeutralSettleSeconds", formatSettingValue(config.neutralSettleSeconds); ...
    "Command", "ReturnToNeutralOnExit", formatSettingValue(config.returnToNeutralOnExit); ...
    "Profile", "Type", formatSettingValue(profile.type); ...
    "Profile", "SampleTimeSeconds", formatSettingValue(profile.sampleTimeSeconds); ...
    "Profile", "PreCommandNeutralSeconds", formatSettingValue(profile.preCommandNeutralSeconds); ...
    "Profile", "DurationSeconds", formatSettingValue(profile.durationSeconds); ...
    "Profile", "PostCommandNeutralSeconds", formatSettingValue(profile.postCommandNeutralSeconds); ...
    "Profile", "AmplitudeDegrees", formatSettingValue(profile.amplitudeDegrees); ...
    "Profile", "OffsetDegrees", formatSettingValue(profile.offsetDegrees); ...
    "Profile", "FrequencyHz", formatSettingValue(profile.frequencyHz); ...
    "Profile", "PhaseDegrees", formatSettingValue(profile.phaseDegrees); ...
    "Profile", "DoubletHoldSeconds", formatSettingValue(profile.doubletHoldSeconds); ...
    "Surfaces", "Names", formatSettingValue(config.surfaceNames); ...
    "Surfaces", "Pins", formatSettingValue(config.surfacePins); ...
    "Surfaces", "ServoNeutralPositions", formatSettingValue(config.servoNeutralPositions); ...
    "Surfaces", "ServoUnitsPerDegree", formatSettingValue(config.servoUnitsPerDegree); ...
    "Surfaces", "ServoMinimumPositions", formatSettingValue(config.servoMinimumPositions); ...
    "Surfaces", "ServoMaximumPositions", formatSettingValue(config.servoMaximumPositions); ...
    "Surfaces", "CommandDeflectionScales", formatSettingValue(config.commandDeflectionScales); ...
    "Surfaces", "CommandDeflectionOffsetsDegrees", formatSettingValue(config.commandDeflectionOffsetsDegrees)];

criticalSettingsTable = array2table( ...
    settings, ...
    'VariableNames', {'Category', 'Setting', 'Value'});
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

function cleanupResources(arduinoConnection, servoObjects, config)
if ~isempty(servoObjects) && config.returnToNeutralOnExit
    try
        moveServosToNeutral(servoObjects, config.servoNeutralPositions);
    catch
    end
end

if ~isempty(arduinoConnection)
    try
        clear arduinoConnection %#ok<NASGU>
    catch
    end
end
end

function timestampSeconds = lastFiniteTimestamp(timeValuesSeconds, fallbackSeconds)
finiteTimestamps = timeValuesSeconds(isfinite(timeValuesSeconds));

if isempty(finiteTimestamps)
    timestampSeconds = fallbackSeconds;
else
    timestampSeconds = finiteTimestamps(end);
end
end

function waitForScheduledTime(referenceTimer, targetTimeSeconds)
while true
    remainingSeconds = targetTimeSeconds - toc(referenceTimer);
    if remainingSeconds <= 0
        return;
    end

    pause(min(0.001, remainingSeconds));
end
end

function output = squareWave(phaseRadians)
output = ones(size(phaseRadians));
output(sin(phaseRadians) < 0) = -1;
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
    error("Arduino_Test:InvalidConfigType", "%s must be a text scalar.", fieldName);
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
    error("Arduino_Test:InvalidConfigType", ...
        "%s must be text, a string array, or a cell array of character vectors.", ...
        fieldName);
end

value = reshape(value, [], 1);
end

function value = getLogicalField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"logical", "numeric"}, {"scalar"}, char(mfilename), char(fieldName));
value = logical(value);
end

function value = getPositiveScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "positive"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getPositiveIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "positive"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNonnegativeScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "nonnegative"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getOptionalScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = defaultValue;
end

validateattributes(value, {"numeric"}, {"real", "scalar"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNumericColumnField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "column"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNumericVectorField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = [];
    return;
end

validateattributes(value, {"numeric"}, {"real", "finite", "vector"}, char(mfilename), char(fieldName));
value = reshape(double(value), [], 1);
end

function mustHaveMatchingLength(values, expectedLength, fieldName)
if numel(values) ~= expectedLength
    error("Arduino_Test:InvalidArrayLength", ...
        "%s must contain exactly %d elements.", ...
        fieldName, ...
        expectedLength);
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
