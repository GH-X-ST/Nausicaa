function runData = Servo_Test(config)
%SERVO_TEST Execute a servo-command latency test with Arduino and Vicon.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);
runData.runInfo.startTime = datetime("now");

assignin("base", "ServoTestLatestState", struct([]));
assignin("base", "ServoTestRunData", runData);

arduinoConnection = [];
servoObjects = {};
viconClient = [];
cleanupHandle = onCleanup(@() cleanupResources( ...
    arduinoConnection, ...
    servoObjects, ...
    viconClient, ...
    config));

try
    [arduinoConnection, servoObjects, arduinoInfo] = connectToArduino(config);
    runData.connectionInfo.arduino = arduinoInfo;
    assignin("base", "ServoTestRunData", runData);

    if ~arduinoInfo.isConnected
        runData.runInfo.status = "arduino_connection_failed";
        runData.runInfo.reason = arduinoInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ServoTestRunData", runData);
        printConnectionStatus(runData);
        return;
    end

    [viconClient, viconInfo, trackedSubjects] = connectToVicon(config);
    runData.connectionInfo.vicon = viconInfo;
    runData.surfaceSetup.ViconSegmentName = reshape(string({trackedSubjects.segmentName}), [], 1);
    assignin("base", "ServoTestRunData", runData);

    if ~viconInfo.isConnected
        runData.runInfo.status = "vicon_connection_failed";
        runData.runInfo.reason = viconInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ServoTestRunData", runData);
        printConnectionStatus(runData);
        return;
    end

    if config.requireAllViconSubjects && any(~[trackedSubjects.isAvailable].')
        missingSubjects = string({trackedSubjects(~[trackedSubjects.isAvailable]).subjectName});
        runData.runInfo.status = "vicon_subjects_missing";
        runData.runInfo.reason = "Requested Vicon subjects are missing: " + join(missingSubjects, ", ");
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ServoTestRunData", runData);
        printConnectionStatus(runData);
        return;
    end

    printConnectionStatus(runData);

    moveServosToNeutral(servoObjects, config.servoNeutralPositions);
    pause(config.neutralSettleSeconds);

    [neutralReference, neutralInfo] = captureNeutralReference(viconClient, trackedSubjects, config);
    runData.neutralReference = neutralReference;
    runData.neutralInfo = neutralInfo;
    assignin("base", "ServoTestRunData", runData);

    if ~neutralInfo.isSuccessful
        runData.runInfo.status = "neutral_reference_failed";
        runData.runInfo.reason = neutralInfo.message;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "ServoTestRunData", runData);
        return;
    end

    [storage, runInfo] = executeLatencyTest( ...
        servoObjects, ...
        viconClient, ...
        trackedSubjects, ...
        neutralReference, ...
        config);

    runData.runInfo = runInfo;
    runData.logs = buildLogTimetables(storage, config);
    runData.surfaceSummary = buildSurfaceSummary(storage, config);
    runData.artifacts = exportRunData(runData);

    assignin("base", "ServoTestRunData", runData);
    assignin("base", "ServoTestLatestState", buildLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupResources(arduinoConnection, servoObjects, viconClient, config);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    assignin("base", "ServoTestRunData", runData);
    rethrow(executionException);
end
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));
defaultSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];
defaultSurfacePins = ["D9"; "D10"; "D11"; "D12"];
defaultHingeAxes = [0, 1, 0; 0, 1, 0; 0, 0, 1; 0, 1, 0];

config.arduinoIPAddress = getTextScalarField(config, "arduinoIPAddress", "192.168.0.33");
config.arduinoBoard = getTextScalarField(config, "arduinoBoard", "Nano33IoT");
config.arduinoPort = getOptionalScalarField(config, "arduinoPort", NaN);

config.viconHostName = getTextScalarField(config, "viconHostName", "192.168.0.100:801");
config.viconStreamMode = getTextScalarField(config, "viconStreamMode", "ServerPush");
config.viconAxisMapping = getTextScalarField(config, "viconAxisMapping", "ZUp");
config.connectTimeoutSeconds = getPositiveScalarField(config, "connectTimeoutSeconds", 5);
config.connectRetryPauseSeconds = getNonnegativeScalarField(config, "connectRetryPauseSeconds", 0.25);
config.maxConnectionAttempts = getPositiveIntegerField(config, "maxConnectionAttempts", 3);
config.frameWaitTimeoutSeconds = getPositiveScalarField(config, "frameWaitTimeoutSeconds", 0.2);
config.requireAllViconSubjects = getLogicalField(config, "requireAllViconSubjects", true);

config.surfaceNames = getStringArrayField(config, "surfaceNames", defaultSurfaceNames);
config.surfacePins = getStringArrayField(config, "surfacePins", defaultSurfacePins);
config.viconSubjectNames = getStringArrayField(config, "viconSubjectNames", config.surfaceNames);
config.viconSegmentNames = getStringArrayField(config, "viconSegmentNames", strings(numel(config.surfaceNames), 1));
config.servoNeutralPositions = getNumericColumnField(config, "servoNeutralPositions", 0.5 .* ones(numel(config.surfaceNames), 1));
config.servoUnitsPerDegree = getNumericColumnField(config, "servoUnitsPerDegree", (1 / 180) .* ones(numel(config.surfaceNames), 1));
config.servoMinimumPositions = getNumericColumnField(config, "servoMinimumPositions", zeros(numel(config.surfaceNames), 1));
config.servoMaximumPositions = getNumericColumnField(config, "servoMaximumPositions", ones(numel(config.surfaceNames), 1));
config.servoMinPulseDurationSeconds = getNumericColumnField(config, "servoMinPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.servoMaxPulseDurationSeconds = getNumericColumnField(config, "servoMaxPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.viconHingeAxesNeutralFrame = getNumericMatrixField(config, "viconHingeAxesNeutralFrame", defaultHingeAxes, [numel(config.surfaceNames), 3]);
config.viconMeasurementSigns = getNumericColumnField(config, "viconMeasurementSigns", ones(numel(config.surfaceNames), 1));
config.commandDeflectionScales = getNumericColumnField(config, "commandDeflectionScales", ones(numel(config.surfaceNames), 1));
config.commandDeflectionOffsetsDegrees = getNumericColumnField(config, "commandDeflectionOffsetsDegrees", zeros(numel(config.surfaceNames), 1));

config.commandMode = getTextScalarField(config, "commandMode", "single");
config.singleSurfaceName = getTextScalarField(config, "singleSurfaceName", "Aileron_L");
config.commandProfile = normalizeCommandProfile(getFieldOrDefault(config, "commandProfile", struct()));

config.neutralSettleSeconds = getPositiveScalarField(config, "neutralSettleSeconds", 1.0);
config.neutralCalibrationSeconds = getPositiveScalarField(config, "neutralCalibrationSeconds", 1.0);
config.returnToNeutralOnExit = getLogicalField(config, "returnToNeutralOnExit", true);

config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "C_Servo_Test"));
config.runLabel = getTextScalarField(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    timeStamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    config.runLabel = timeStamp + "_ServoLatencyTest";
end

validateattributes(config.arduinoPort, {"numeric"}, {"real", "scalar"}, mfilename, "arduinoPort");

surfaceCount = numel(config.surfaceNames);
mustHaveMatchingLength(config.surfacePins, surfaceCount, "surfacePins");
mustHaveMatchingLength(config.viconSubjectNames, surfaceCount, "viconSubjectNames");
mustHaveMatchingLength(config.viconSegmentNames, surfaceCount, "viconSegmentNames");

validateattributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "servoNeutralPositions");
validateattributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "servoUnitsPerDegree");
validateattributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "servoMinimumPositions");
validateattributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "servoMaximumPositions");
validateattributes(config.servoMinPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, mfilename, "servoMinPulseDurationSeconds");
validateattributes(config.servoMaxPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, mfilename, "servoMaxPulseDurationSeconds");
validateattributes(config.viconMeasurementSigns, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "viconMeasurementSigns");
validateattributes(config.commandDeflectionScales, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "commandDeflectionScales");
validateattributes(config.commandDeflectionOffsetsDegrees, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, mfilename, "commandDeflectionOffsetsDegrees");

if any(abs(config.servoUnitsPerDegree) <= eps)
    error("Servo_Test:InvalidServoScale", "servoUnitsPerDegree must be nonzero for every surface.");
end

if any(config.servoMinimumPositions > config.servoMaximumPositions)
    error("Servo_Test:InvalidServoLimits", "servoMinimumPositions must be less than or equal to servoMaximumPositions.");
end

if any(config.servoNeutralPositions < config.servoMinimumPositions) || any(config.servoNeutralPositions > config.servoMaximumPositions)
    error("Servo_Test:NeutralOutsideLimits", "servoNeutralPositions must lie inside the configured servo limits.");
end

if any(~ismember(config.viconMeasurementSigns, [-1; 1]))
    error("Servo_Test:InvalidMeasurementSign", "viconMeasurementSigns must contain only -1 or 1.");
end

config.viconHingeAxesNeutralFrame = normalizeRowVectors(config.viconHingeAxesNeutralFrame, "viconHingeAxesNeutralFrame");

validAxisMappings = ["XUp", "YUp", "ZUp"];
if ~any(config.viconAxisMapping == validAxisMappings)
    error("Servo_Test:InvalidAxisMapping", "viconAxisMapping must be XUp, YUp, or ZUp.");
end

config.viconStreamMode = resolveStreamModeName(config.viconStreamMode);

validCommandModes = ["single", "all"];
if ~any(config.commandMode == validCommandModes)
    error("Servo_Test:InvalidCommandMode", "commandMode must be 'single' or 'all'.");
end

if ~any(config.singleSurfaceName == config.surfaceNames)
    error("Servo_Test:InvalidSingleSurface", ...
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
config.surfaceSetup = buildSurfaceSetupTable(config, strings(surfaceCount, 1));
end

function commandProfile = normalizeCommandProfile(commandProfileConfig)
if isempty(commandProfileConfig)
    commandProfileConfig = struct();
end

commandProfile.type = getTextScalarField(commandProfileConfig, "type", "sine");
commandProfile.sampleTimeSeconds = getPositiveScalarField(commandProfileConfig, "sampleTimeSeconds", 0.02);
commandProfile.preCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "preCommandNeutralSeconds", 1.0);
commandProfile.postCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "postCommandNeutralSeconds", 1.0);
commandProfile.durationSeconds = getPositiveScalarField(commandProfileConfig, "durationSeconds", 6.0);
commandProfile.amplitudeDegrees = getScalarField(commandProfileConfig, "amplitudeDegrees", 15.0);
commandProfile.offsetDegrees = getScalarField(commandProfileConfig, "offsetDegrees", 0.0);
commandProfile.frequencyHz = getPositiveScalarField(commandProfileConfig, "frequencyHz", 0.5);
commandProfile.phaseDegrees = getScalarField(commandProfileConfig, "phaseDegrees", 0.0);
commandProfile.doubletHoldSeconds = getPositiveScalarField(commandProfileConfig, "doubletHoldSeconds", 0.5);
commandProfile.customTimeSeconds = getNumericVectorField(commandProfileConfig, "customTimeSeconds", []);
commandProfile.customDeflectionDegrees = getNumericVectorField(commandProfileConfig, "customDeflectionDegrees", []);
commandProfile.customInterpolationMethod = getTextScalarField(commandProfileConfig, "customInterpolationMethod", "linear");
commandProfile.customFunction = getFieldOrDefault(commandProfileConfig, "customFunction", []);

validProfileTypes = ["sine", "square", "doublet", "custom", "function"];
if ~any(commandProfile.type == validProfileTypes)
    error("Servo_Test:InvalidProfileType", ...
        "commandProfile.type must be one of: %s.", ...
        char(join(validProfileTypes, ", ")));
end

if ~isempty(commandProfile.customFunction) && ~isa(commandProfile.customFunction, "function_handle")
    error("Servo_Test:InvalidCustomFunction", "commandProfile.customFunction must be a function handle when provided.");
end

if commandProfile.type == "doublet"
    commandProfile.durationSeconds = max(commandProfile.durationSeconds, 2 .* commandProfile.doubletHoldSeconds);
end

if commandProfile.type == "custom"
    validateattributes(commandProfile.customTimeSeconds, {"numeric"}, {"real", "finite", "vector", "nonempty"}, mfilename, "commandProfile.customTimeSeconds");
    validateattributes(commandProfile.customDeflectionDegrees, {"numeric"}, {"real", "finite", "vector", "numel", numel(commandProfile.customTimeSeconds)}, mfilename, "commandProfile.customDeflectionDegrees");

    commandProfile.customTimeSeconds = reshape(double(commandProfile.customTimeSeconds), [], 1);
    commandProfile.customDeflectionDegrees = reshape(double(commandProfile.customDeflectionDegrees), [], 1);

    if any(diff(commandProfile.customTimeSeconds) <= 0)
        error("Servo_Test:InvalidCustomTimeVector", "commandProfile.customTimeSeconds must be strictly increasing.");
    end

    commandProfile.durationSeconds = commandProfile.customTimeSeconds(end);
end

if commandProfile.type == "function" && isempty(commandProfile.customFunction)
    error("Servo_Test:MissingCustomFunction", "commandProfile.customFunction is required when type is 'function'.");
end
end

function runData = initializeRunData(config)
runData = struct( ...
    "config", config, ...
    "connectionInfo", struct( ...
        "arduino", struct(), ...
        "vicon", struct()), ...
    "runInfo", struct( ...
        "status", "initialized", ...
        "reason", "", ...
        "startTime", NaT, ...
        "stopTime", NaT, ...
        "sampleCount", 0, ...
        "scheduledDurationSeconds", NaN), ...
    "surfaceSetup", config.surfaceSetup, ...
    "neutralReference", struct([]), ...
    "neutralInfo", struct(), ...
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

function [client, connectionInfo, trackedSubjects] = connectToVicon(config)
surfaceCount = numel(config.surfaceNames);
trackedSubjects = repmat(struct( ...
    "surfaceName", "", ...
    "subjectName", "", ...
    "rootSegmentName", "", ...
    "segmentName", "", ...
    "isAvailable", false), surfaceCount, 1);

connectionInfo = struct( ...
    "dllPath", "", ...
    "hostName", config.viconHostName, ...
    "isConnected", false, ...
    "connectAttempts", 0, ...
    "connectElapsedSeconds", NaN, ...
    "connectionMessage", "", ...
    "diagnostics", strings(0, 1), ...
    "sdkVersion", "", ...
    "streamMode", "", ...
    "axisMapping", config.viconAxisMapping, ...
    "frameRateHz", NaN, ...
    "latencySeconds", NaN, ...
    "subjectCount", 0, ...
    "availableSubjectNames", strings(0, 1), ...
    "missingSubjectNames", strings(0, 1), ...
    "allRequestedSubjectsAvailable", false);

client = [];

try
    dllPath = resolveSdkAssembly();
    connectionInfo.dllPath = dllPath;
    loadSdkAssembly(dllPath);
catch connectionException
    connectionInfo.connectionMessage = string(connectionException.message);
    return;
end

client = ViconDataStreamSDK.DotNET.Client();
connectionDiagnostics = strings(config.maxConnectionAttempts + 1, 1);
diagnosticCount = 0;

try
    client.SetConnectionTimeout(uint32(round(config.connectTimeoutSeconds .* 1000)));
catch
    diagnosticCount = diagnosticCount + 1;
    connectionDiagnostics(diagnosticCount) = "The SDK client does not expose SetConnectionTimeout in this MATLAB session.";
end

connectStart = tic;
lastResult = "NotAttempted";

for attemptIndex = 1:config.maxConnectionAttempts
    try
        connectOutput = client.Connect(char(config.viconHostName));
        lastResult = netValueToString(connectOutput.Result);
    catch connectionException
        lastResult = string(connectionException.message);
    end

    if adaptLogical(client.IsConnected().Connected)
        break;
    end

    diagnosticCount = diagnosticCount + 1;
    connectionDiagnostics(diagnosticCount) = "Attempt " + attemptIndex + " failed: " + lastResult;
    pause(config.connectRetryPauseSeconds);
end

connectionInfo.connectAttempts = attemptIndex;
connectionInfo.connectElapsedSeconds = toc(connectStart);
connectionInfo.connectionMessage = lastResult;
connectionInfo.diagnostics = connectionDiagnostics(1:diagnosticCount);
connectionInfo.isConnected = adaptLogical(client.IsConnected().Connected);

if ~connectionInfo.isConnected
    return;
end

client.EnableSegmentData();
client.SetBufferSize(uint32(1));
client.SetStreamMode(resolveStreamMode(config.viconStreamMode));
applyAxisMapping(client, config.viconAxisMapping);

versionOutput = client.GetVersion();
connectionInfo.sdkVersion = join(string([versionOutput.Major, versionOutput.Minor, versionOutput.Point]), ".");
connectionInfo.streamMode = resolveStreamModeName(config.viconStreamMode);
connectionInfo.axisMapping = getAxisMappingString(client);

if waitForFrame(client, config.frameWaitTimeoutSeconds)
    connectionInfo.frameRateHz = double(client.GetFrameRate().FrameRateHz);
    connectionInfo.latencySeconds = double(client.GetLatencyTotal().Total);
end

trackedSubjects = discoverTrackedSubjects(client, config);
availableMask = reshape([trackedSubjects.isAvailable], [], 1);

connectionInfo.subjectCount = double(client.GetSubjectCount().SubjectCount);
connectionInfo.availableSubjectNames = reshape(string({trackedSubjects(availableMask).subjectName}), [], 1);
connectionInfo.missingSubjectNames = reshape(string({trackedSubjects(~availableMask).subjectName}), [], 1);
connectionInfo.allRequestedSubjectsAvailable = all(availableMask);

if all(availableMask)
    connectionInfo.connectionMessage = "Connected";
else
    connectionInfo.connectionMessage = "Connected, but one or more requested Vicon subjects are missing.";
end
end

function dllPath = resolveSdkAssembly()
rootFolder = fileparts(mfilename("fullpath"));
dllPath = fullfile(rootFolder, "A_Vicon_Example", "dotNET", "ViconDataStreamSDK_DotNET.dll");

if ~isfile(dllPath)
    error("Servo_Test:MissingSdkDll", "Vicon SDK DLL not found at %s.", dllPath);
end
end

function loadSdkAssembly(dllPath)
NET.addAssembly(char(dllPath));
end

function trackedSubjects = discoverTrackedSubjects(client, config)
surfaceCount = numel(config.surfaceNames);
trackedSubjects = repmat(struct( ...
    "surfaceName", "", ...
    "subjectName", "", ...
    "rootSegmentName", "", ...
    "segmentName", "", ...
    "isAvailable", false), surfaceCount, 1);

availableSubjectCount = double(client.GetSubjectCount().SubjectCount);
availableSubjectNames = strings(availableSubjectCount, 1);
availableRootSegments = strings(availableSubjectCount, 1);

for subjectIndex = 1:availableSubjectCount
    availableSubjectNames(subjectIndex) = string(client.GetSubjectName(uint32(subjectIndex - 1)).SubjectName);
    availableRootSegments(subjectIndex) = string(client.GetSubjectRootSegmentName(char(availableSubjectNames(subjectIndex))).SegmentName);
end

for surfaceIndex = 1:surfaceCount
    subjectName = config.viconSubjectNames(surfaceIndex);
    matchIndex = find(availableSubjectNames == subjectName, 1, "first");

    trackedSubjects(surfaceIndex).surfaceName = config.surfaceNames(surfaceIndex);
    trackedSubjects(surfaceIndex).subjectName = subjectName;

    if isempty(matchIndex)
        trackedSubjects(surfaceIndex).rootSegmentName = "";
        trackedSubjects(surfaceIndex).segmentName = "";
        trackedSubjects(surfaceIndex).isAvailable = false;
        continue;
    end

    trackedSubjects(surfaceIndex).rootSegmentName = availableRootSegments(matchIndex);
    trackedSubjects(surfaceIndex).segmentName = config.viconSegmentNames(surfaceIndex);
    if strlength(trackedSubjects(surfaceIndex).segmentName) == 0
        trackedSubjects(surfaceIndex).segmentName = trackedSubjects(surfaceIndex).rootSegmentName;
    end
    trackedSubjects(surfaceIndex).isAvailable = true;
end
end

function printConnectionStatus(runData)
fprintf("\nServo_Test connection summary\n");
fprintf("  Arduino (%s): %s\n", ...
    runData.config.arduinoIPAddress, ...
    char(getStatusText(runData.connectionInfo.arduino)));
fprintf("  Vicon   (%s): %s\n", ...
    runData.config.viconHostName, ...
    char(getStatusText(runData.connectionInfo.vicon)));

if isfield(runData.connectionInfo, "vicon") && isfield(runData.connectionInfo.vicon, "missingSubjectNames") && ~isempty(runData.connectionInfo.vicon.missingSubjectNames)
    fprintf("  Missing Vicon subjects: %s\n", char(join(runData.connectionInfo.vicon.missingSubjectNames, ", ")));
end

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

function [neutralReference, neutralInfo] = captureNeutralReference(client, trackedSubjects, config)
surfaceCount = numel(config.surfaceNames);
neutralReference = repmat(struct( ...
    "surfaceName", "", ...
    "subjectName", "", ...
    "segmentName", "", ...
    "quaternionXYZW", nan(1, 4), ...
    "sampleCount", 0), surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    neutralReference(surfaceIndex).surfaceName = config.surfaceNames(surfaceIndex);
    neutralReference(surfaceIndex).subjectName = config.viconSubjectNames(surfaceIndex);
    neutralReference(surfaceIndex).segmentName = trackedSubjects(surfaceIndex).segmentName;
end

neutralInfo = struct( ...
    "isSuccessful", false, ...
    "message", "", ...
    "durationSeconds", config.neutralCalibrationSeconds, ...
    "surfaceValidSampleCounts", zeros(surfaceCount, 1));

quaternionSamples = cell(surfaceCount, 1);
calibrationStart = tic;

while toc(calibrationStart) < config.neutralCalibrationSeconds
    frameReady = waitForFrame(client, config.frameWaitTimeoutSeconds);
    if ~frameReady
        continue;
    end

    snapshot = readTrackedSurfaceSample( ...
        client, ...
        trackedSubjects, ...
        struct([]), ...
        config.viconHingeAxesNeutralFrame, ...
        config.viconMeasurementSigns);

    for surfaceIndex = 1:surfaceCount
        if snapshot.isOccluded(surfaceIndex)
            continue;
        end

        quaternionSamples{surfaceIndex}(end + 1, :) = snapshot.quaternionXYZW(surfaceIndex, :); %#ok<AGROW>
    end
end

for surfaceIndex = 1:surfaceCount
    neutralReference(surfaceIndex).sampleCount = size(quaternionSamples{surfaceIndex}, 1);
    neutralInfo.surfaceValidSampleCounts(surfaceIndex) = neutralReference(surfaceIndex).sampleCount;

    if ~trackedSubjects(surfaceIndex).isAvailable
        neutralReference(surfaceIndex).quaternionXYZW = [NaN, NaN, NaN, NaN];
        continue;
    end

    if neutralReference(surfaceIndex).sampleCount == 0
        neutralInfo.message = "Neutral reference failed because no valid Vicon samples were captured for " + config.surfaceNames(surfaceIndex) + ".";
        return;
    end

    neutralReference(surfaceIndex).quaternionXYZW = averageQuaternions(quaternionSamples{surfaceIndex});
end

neutralInfo.isSuccessful = true;
neutralInfo.message = "Neutral reference captured successfully.";
end

function [storage, runInfo] = executeLatencyTest(servoObjects, client, trackedSubjects, neutralReference, config)
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
    waitForScheduledTime(testStart, scheduledTimeSeconds(sampleIndex));

    sampleTime = toc(testStart);
    baseCommandDegrees = evaluateBaseCommandDegrees(profileInfo, sampleTime);
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
    writeServoPositions(servoObjects, commandedServoPositionsClipped);
    storage.commandWriteStopSeconds(sampleIndex) = toc(testStart);

    storage.arduinoReadStartSeconds(sampleIndex) = toc(testStart);
    appliedServoPositions = readServoPositions(servoObjects);
    storage.arduinoReadStopSeconds(sampleIndex) = toc(testStart);

    frameReady = waitForFrame(client, config.frameWaitTimeoutSeconds);
    storage.viconSampleTimeSeconds(sampleIndex) = toc(testStart);

    if frameReady
        snapshot = readTrackedSurfaceSample( ...
            client, ...
            trackedSubjects, ...
            neutralReference, ...
            config.viconHingeAxesNeutralFrame, ...
            config.viconMeasurementSigns);
    else
        snapshot = createEmptyViconSnapshot(surfaceCount);
    end

    storage.sampleCount = sampleIndex;
    storage.baseCommandDegrees(sampleIndex) = baseCommandDegrees;
    storage.desiredDeflectionsDegrees(sampleIndex, :) = desiredDeflectionsDegrees;
    storage.commandedServoPositions(sampleIndex, :) = commandedServoPositionsClipped;
    storage.commandSaturated(sampleIndex, :) = saturatedMask;
    storage.appliedServoPositions(sampleIndex, :) = appliedServoPositions;
    storage.appliedEquivalentDegrees(sampleIndex, :) = ...
        (appliedServoPositions - config.servoNeutralPositions.') ./ config.servoUnitsPerDegree.';
    storage.viconFrameNumbers(sampleIndex) = snapshot.frameNumber;
    storage.viconLatencySeconds(sampleIndex) = snapshot.latencySeconds;
    storage.measuredDeflectionsDegrees(sampleIndex, :) = snapshot.measuredAnglesDegrees;
    storage.viconOccluded(sampleIndex, :) = snapshot.isOccluded;
    storage.viconQuaternionXYZW(sampleIndex, :, :) = snapshot.quaternionXYZW;

    assignin("base", "ServoTestLatestState", buildLatestState(storage, config, sampleIndex));
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
    "arduinoReadStartSeconds", nan(sampleCount, 1), ...
    "arduinoReadStopSeconds", nan(sampleCount, 1), ...
    "viconSampleTimeSeconds", nan(sampleCount, 1), ...
    "viconFrameNumbers", nan(sampleCount, 1), ...
    "viconLatencySeconds", nan(sampleCount, 1), ...
    "desiredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "commandedServoPositions", nan(sampleCount, surfaceCount), ...
    "commandSaturated", false(sampleCount, surfaceCount), ...
    "appliedServoPositions", nan(sampleCount, surfaceCount), ...
    "appliedEquivalentDegrees", nan(sampleCount, surfaceCount), ...
    "measuredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "viconOccluded", true(sampleCount, surfaceCount), ...
    "viconQuaternionXYZW", nan(sampleCount, surfaceCount, 4), ...
    "profileInfo", struct());
end

function moveServosToNeutral(servoObjects, neutralPositions)
neutralRow = reshape(neutralPositions, 1, []);
writeServoPositions(servoObjects, neutralRow);
end

function writeServoPositions(servoObjects, servoPositions)
for surfaceIndex = 1:numel(servoObjects)
    writePosition(servoObjects{surfaceIndex}, servoPositions(surfaceIndex));
end
end

function servoPositions = readServoPositions(servoObjects)
servoPositions = nan(1, numel(servoObjects));

for surfaceIndex = 1:numel(servoObjects)
    try
        servoPositions(surfaceIndex) = readPosition(servoObjects{surfaceIndex});
    catch
        servoPositions(surfaceIndex) = NaN;
    end
end
end

function snapshot = readTrackedSurfaceSample(client, trackedSubjects, neutralReference, hingeAxesNeutralFrame, measurementSigns)
surfaceCount = numel(trackedSubjects);
snapshot = createEmptyViconSnapshot(surfaceCount);

snapshot.frameNumber = double(client.GetFrameNumber().FrameNumber);
snapshot.latencySeconds = double(client.GetLatencyTotal().Total);

for surfaceIndex = 1:surfaceCount
    if ~trackedSubjects(surfaceIndex).isAvailable
        continue;
    end

    subjectName = char(trackedSubjects(surfaceIndex).subjectName);
    segmentName = char(trackedSubjects(surfaceIndex).segmentName);

    quaternionOutput = client.GetSegmentGlobalRotationQuaternion(subjectName, segmentName);
    isValid = isSuccessOutput(quaternionOutput) && ~adaptLogical(quaternionOutput.Occluded);

    if ~isValid
        continue;
    end

    quaternionXYZW = reshape(double(quaternionOutput.Rotation), 1, 4);
    snapshot.isOccluded(surfaceIndex) = false;
    snapshot.quaternionXYZW(surfaceIndex, :) = normalizeQuaternion(quaternionXYZW);

    if ~isempty(neutralReference)
        relativeQuaternion = multiplyQuaternions( ...
            conjugateQuaternion(neutralReference(surfaceIndex).quaternionXYZW), ...
            snapshot.quaternionXYZW(surfaceIndex, :));

        hingeAngleRadians = extractTwistAngle(relativeQuaternion, hingeAxesNeutralFrame(surfaceIndex, :));
        snapshot.measuredAnglesDegrees(surfaceIndex) = measurementSigns(surfaceIndex) .* rad2deg(hingeAngleRadians);
    end
end
end

function snapshot = createEmptyViconSnapshot(surfaceCount)
snapshot = struct( ...
    "frameNumber", NaN, ...
    "latencySeconds", NaN, ...
    "measuredAnglesDegrees", nan(1, surfaceCount), ...
    "isOccluded", true(1, surfaceCount), ...
    "quaternionXYZW", nan(surfaceCount, 4));
end

function logs = buildLogTimetables(storage, config)
surfaceNames = config.surfaceNames;
commandVariableNames = [ ...
    "base_command_deg", ...
    buildSurfaceVariableNames(surfaceNames, "desired_deg"), ...
    buildSurfaceVariableNames(surfaceNames, "command_position"), ...
    buildSurfaceVariableNames(surfaceNames, "command_saturated")];
commandLog = array2timetable( ...
    [storage.baseCommandDegrees, storage.desiredDeflectionsDegrees, storage.commandedServoPositions, double(storage.commandSaturated)], ...
    "RowTimes", seconds(storage.commandWriteStopSeconds), ...
    "VariableNames", cellstr(commandVariableNames));

commandLog.scheduled_time_s = storage.scheduledTimeSeconds;
commandLog.command_write_start_s = storage.commandWriteStartSeconds;
commandLog.command_write_stop_s = storage.commandWriteStopSeconds;

arduinoVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "applied_position"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_equivalent_deg")];
arduinoLog = array2timetable( ...
    [storage.appliedServoPositions, storage.appliedEquivalentDegrees], ...
    "RowTimes", seconds(storage.arduinoReadStopSeconds), ...
    "VariableNames", cellstr(arduinoVariableNames));

arduinoLog.scheduled_time_s = storage.scheduledTimeSeconds;
arduinoLog.arduino_read_start_s = storage.arduinoReadStartSeconds;
arduinoLog.arduino_read_stop_s = storage.arduinoReadStopSeconds;

viconDataMatrix = [ ...
    storage.measuredDeflectionsDegrees, ...
    double(storage.viconOccluded), ...
    reshape(storage.viconQuaternionXYZW(:, :, 1), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(:, :, 2), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(:, :, 3), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(:, :, 4), [], numel(surfaceNames))];

viconVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "measured_deg"), ...
    buildSurfaceVariableNames(surfaceNames, "is_occluded"), ...
    buildSurfaceVariableNames(surfaceNames, "qx"), ...
    buildSurfaceVariableNames(surfaceNames, "qy"), ...
    buildSurfaceVariableNames(surfaceNames, "qz"), ...
    buildSurfaceVariableNames(surfaceNames, "qw")];

viconLog = array2timetable( ...
    viconDataMatrix, ...
    "RowTimes", seconds(storage.viconSampleTimeSeconds), ...
    "VariableNames", cellstr(viconVariableNames));

viconLog.scheduled_time_s = storage.scheduledTimeSeconds;
viconLog.vicon_sample_time_s = storage.viconSampleTimeSeconds;
viconLog.vicon_frame_number = storage.viconFrameNumbers;
viconLog.vicon_latency_s = storage.viconLatencySeconds;

sampleSummary = table( ...
    storage.scheduledTimeSeconds, ...
    storage.baseCommandDegrees, ...
    storage.commandWriteStartSeconds, ...
    storage.commandWriteStopSeconds, ...
    storage.arduinoReadStartSeconds, ...
    storage.arduinoReadStopSeconds, ...
    storage.viconSampleTimeSeconds, ...
    storage.viconFrameNumbers, ...
    storage.viconLatencySeconds, ...
    "VariableNames", { ...
        "scheduled_time_s", ...
        "base_command_deg", ...
        "command_write_start_s", ...
        "command_write_stop_s", ...
        "arduino_read_start_s", ...
        "arduino_read_stop_s", ...
        "vicon_sample_time_s", ...
        "vicon_frame_number", ...
        "vicon_latency_s"});

logs = struct( ...
    "command", commandLog, ...
    "arduino", arduinoLog, ...
    "vicon", viconLog, ...
    "sampleSummary", sampleSummary);
end

function surfaceSummary = buildSurfaceSummary(storage, config)
surfaceCount = numel(config.surfaceNames);
validViconSampleCount = zeros(surfaceCount, 1);
appliedSampleCount = zeros(surfaceCount, 1);
saturationCount = zeros(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    validViconSampleCount(surfaceIndex) = sum(~storage.viconOccluded(:, surfaceIndex) & isfinite(storage.measuredDeflectionsDegrees(:, surfaceIndex)));
    appliedSampleCount(surfaceIndex) = sum(isfinite(storage.appliedServoPositions(:, surfaceIndex)));
    saturationCount(surfaceIndex) = sum(storage.commandSaturated(:, surfaceIndex));
end

surfaceSummary = table( ...
    config.surfaceNames, ...
    config.surfacePins, ...
    config.activeSurfaceMask, ...
    validViconSampleCount, ...
    appliedSampleCount, ...
    saturationCount, ...
    "VariableNames", { ...
        "SurfaceName", ...
        "ArduinoPin", ...
        "IsActive", ...
        "ValidViconSampleCount", ...
        "AppliedCommandSampleCount", ...
        "SaturationSampleCount"});
end

function latestState = buildLatestState(storage, config, sampleIndex)
surfaceCount = numel(config.surfaceNames);
surfaceStates = repmat(struct( ...
    "name", "", ...
    "desired_deg", NaN, ...
    "command_position", NaN, ...
    "applied_position", NaN, ...
    "applied_equivalent_deg", NaN, ...
    "measured_deg", NaN, ...
    "is_occluded", true), surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceStates(surfaceIndex).name = config.surfaceNames(surfaceIndex);
    surfaceStates(surfaceIndex).desired_deg = storage.desiredDeflectionsDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).command_position = storage.commandedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_position = storage.appliedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_equivalent_deg = storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).measured_deg = storage.measuredDeflectionsDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).is_occluded = storage.viconOccluded(sampleIndex, surfaceIndex);
end

latestState = struct( ...
    "sampleIndex", sampleIndex, ...
    "scheduledTimeSeconds", storage.scheduledTimeSeconds(sampleIndex), ...
    "commandWriteTimeSeconds", storage.commandWriteStopSeconds(sampleIndex), ...
    "arduinoReadTimeSeconds", storage.arduinoReadStopSeconds(sampleIndex), ...
    "viconSampleTimeSeconds", storage.viconSampleTimeSeconds(sampleIndex), ...
    "viconFrameNumber", storage.viconFrameNumbers(sampleIndex), ...
    "viconLatencySeconds", storage.viconLatencySeconds(sampleIndex), ...
    "surfaces", surfaceStates);
end

function surfaceSetup = buildSurfaceSetupTable(config, viconSegmentNames)
surfaceSetup = table( ...
    config.surfaceNames, ...
    config.surfacePins, ...
    config.viconSubjectNames, ...
    viconSegmentNames, ...
    config.activeSurfaceMask, ...
    config.servoNeutralPositions, ...
    config.servoUnitsPerDegree, ...
    config.servoMinimumPositions, ...
    config.servoMaximumPositions, ...
    config.viconHingeAxesNeutralFrame(:, 1), ...
    config.viconHingeAxesNeutralFrame(:, 2), ...
    config.viconHingeAxesNeutralFrame(:, 3), ...
    config.viconMeasurementSigns, ...
    config.commandDeflectionScales, ...
    config.commandDeflectionOffsetsDegrees, ...
    "VariableNames", { ...
        "SurfaceName", ...
        "ArduinoPin", ...
        "ViconSubjectName", ...
        "ViconSegmentName", ...
        "IsActive", ...
        "ServoNeutralPosition", ...
        "ServoUnitsPerDegree", ...
        "ServoMinimumPosition", ...
        "ServoMaximumPosition", ...
        "HingeAxisX", ...
        "HingeAxisY", ...
        "HingeAxisZ", ...
        "MeasurementSign", ...
        "CommandScale", ...
        "CommandOffsetDeg"});
end

function artifacts = exportRunData(runData)
matFilePath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".mat");
workbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");

artifacts = struct( ...
    "matFilePath", string(matFilePath), ...
    "workbookPath", string(workbookPath));

save(matFilePath, "runData", "-v7.3");

metadata = {
    "Run Label", char(runData.config.runLabel); ...
    "Status", char(runData.runInfo.status); ...
    "Reason", char(runData.runInfo.reason); ...
    "Arduino IP", char(runData.config.arduinoIPAddress); ...
    "Arduino Board", char(runData.config.arduinoBoard); ...
    "Vicon Host", char(runData.config.viconHostName); ...
    "Vicon Stream Mode", char(runData.config.viconStreamMode); ...
    "Vicon Axis Mapping", char(runData.config.viconAxisMapping); ...
    "Command Mode", char(runData.config.commandMode); ...
    "Active Surfaces", char(join(runData.config.activeSurfaceNames, ", ")); ...
    "Sample Count", runData.runInfo.sampleCount; ...
    "Scheduled Duration (s)", runData.runInfo.scheduledDurationSeconds; ...
    "Start Time", char(string(runData.runInfo.startTime)); ...
    "Stop Time", char(string(runData.runInfo.stopTime)); ...
    "Neutral Calibration Success", runData.neutralInfo.isSuccessful; ...
    "Neutral Calibration Message", char(runData.neutralInfo.message)};

metadataTable = cell2table(metadata, "VariableNames", {"Field", "Value"});
neutralReferenceTable = buildNeutralReferenceTable(runData.neutralReference);

writetable(metadataTable, workbookPath, "Sheet", "Metadata");
writetable(runData.surfaceSetup, workbookPath, "Sheet", "SurfaceSetup");
writetable(neutralReferenceTable, workbookPath, "Sheet", "NeutralReference");
writetable(timetableToExportTable(runData.logs.command), workbookPath, "Sheet", "CommandLog");
writetable(timetableToExportTable(runData.logs.arduino), workbookPath, "Sheet", "ArduinoLog");
writetable(timetableToExportTable(runData.logs.vicon), workbookPath, "Sheet", "ViconLog");
writetable(runData.logs.sampleSummary, workbookPath, "Sheet", "SampleSummary");
writetable(runData.surfaceSummary, workbookPath, "Sheet", "SurfaceSummary");
end

function neutralReferenceTable = buildNeutralReferenceTable(neutralReference)
surfaceCount = numel(neutralReference);
surfaceNames = strings(surfaceCount, 1);
subjectNames = strings(surfaceCount, 1);
segmentNames = strings(surfaceCount, 1);
sampleCounts = zeros(surfaceCount, 1);
quaternionXYZW = nan(surfaceCount, 4);

for surfaceIndex = 1:surfaceCount
    surfaceNames(surfaceIndex) = neutralReference(surfaceIndex).surfaceName;
    subjectNames(surfaceIndex) = neutralReference(surfaceIndex).subjectName;
    segmentNames(surfaceIndex) = neutralReference(surfaceIndex).segmentName;
    sampleCounts(surfaceIndex) = neutralReference(surfaceIndex).sampleCount;
    quaternionXYZW(surfaceIndex, :) = neutralReference(surfaceIndex).quaternionXYZW;
end

neutralReferenceTable = table( ...
    surfaceNames, ...
    subjectNames, ...
    segmentNames, ...
    sampleCounts, ...
    quaternionXYZW(:, 1), ...
    quaternionXYZW(:, 2), ...
    quaternionXYZW(:, 3), ...
    quaternionXYZW(:, 4), ...
    "VariableNames", { ...
        "SurfaceName", ...
        "SubjectName", ...
        "SegmentName", ...
        "SampleCount", ...
        "qx", ...
        "qy", ...
        "qz", ...
        "qw"});
end

function exportTable = timetableToExportTable(timetableData)
exportTable = timetable2table(timetableData);
exportTable.Properties.VariableNames{1} = "time_s";
exportTable.time_s = seconds(exportTable.time_s);
end

function cleanupResources(arduinoConnection, servoObjects, client, config)
if ~isempty(servoObjects) && config.returnToNeutralOnExit
    try
        moveServosToNeutral(servoObjects, config.servoNeutralPositions);
    catch
    end
end

if ~isempty(client)
    try
        if adaptLogical(client.IsConnected().Connected)
            client.Disconnect();
        end
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

function waitForScheduledTime(referenceTimer, targetTimeSeconds)
while true
    remainingSeconds = targetTimeSeconds - toc(referenceTimer);
    if remainingSeconds <= 0
        return;
    end

    pause(min(0.001, remainingSeconds));
end
end

function frameReady = waitForFrame(client, timeoutSeconds)
frameReady = false;
waitStart = tic;

while toc(waitStart) < timeoutSeconds
    getFrameOutput = client.GetFrame();
    if isSuccessResult(getFrameOutput.Result)
        frameReady = true;
        return;
    end

    pause(0.005);
end
end

function streamMode = resolveStreamMode(streamModeConfig)
streamModeName = resolveStreamModeName(streamModeConfig);

switch streamModeName
    case "ClientPull"
        streamMode = ViconDataStreamSDK.DotNET.StreamMode.ClientPull;
    case "ClientPullPreFetch"
        streamMode = ViconDataStreamSDK.DotNET.StreamMode.ClientPullPreFetch;
    otherwise
        streamMode = ViconDataStreamSDK.DotNET.StreamMode.ServerPush;
end
end

function streamModeName = resolveStreamModeName(streamModeConfig)
if isstring(streamModeConfig) || ischar(streamModeConfig)
    streamModeName = string(streamModeConfig);
else
    streamModeName = netValueToString(streamModeConfig);
end

validModes = ["ClientPull", "ClientPullPreFetch", "ServerPush"];
if ~any(streamModeName == validModes)
    error("Servo_Test:InvalidStreamMode", ...
        "Unsupported Vicon stream mode '%s'. Use ClientPull, ClientPullPreFetch, or ServerPush.", ...
        streamModeName);
end
end

function applyAxisMapping(client, axisMapping)
switch axisMapping
    case "XUp"
        client.SetAxisMapping( ...
            ViconDataStreamSDK.DotNET.Direction.Up, ...
            ViconDataStreamSDK.DotNET.Direction.Forward, ...
            ViconDataStreamSDK.DotNET.Direction.Left);
    case "YUp"
        client.SetAxisMapping( ...
            ViconDataStreamSDK.DotNET.Direction.Forward, ...
            ViconDataStreamSDK.DotNET.Direction.Up, ...
            ViconDataStreamSDK.DotNET.Direction.Right);
    otherwise
        client.SetAxisMapping( ...
            ViconDataStreamSDK.DotNET.Direction.Forward, ...
            ViconDataStreamSDK.DotNET.Direction.Left, ...
            ViconDataStreamSDK.DotNET.Direction.Up);
end
end

function axisMappingString = getAxisMappingString(client)
axisMappingOutput = client.GetAxisMapping();
axisMappingString = "X-" + netValueToString(axisMappingOutput.XAxis) + ...
    ", Y-" + netValueToString(axisMappingOutput.YAxis) + ...
    ", Z-" + netValueToString(axisMappingOutput.ZAxis);
end

function success = isSuccessOutput(output)
success = true;
if isprop(output, "Result")
    success = isSuccessResult(output.Result);
end
end

function success = isSuccessResult(resultValue)
success = netValueToString(resultValue) == "Success";
end

function valueString = netValueToString(value)
if isstring(value)
    valueString = value;
elseif ischar(value)
    valueString = string(value);
else
    valueString = string(char(value.ToString()));
end
end

function value = adaptLogical(netLogical)
if islogical(netLogical)
    value = netLogical;
else
    value = strcmpi(netValueToString(netLogical), "True");
end
end

function averageQuaternion = averageQuaternions(quaternionSamples)
quaternionSamples = reshape(quaternionSamples, [], 4);
accumulator = zeros(4, 4);

for sampleIndex = 1:size(quaternionSamples, 1)
    normalizedQuaternion = normalizeQuaternion(quaternionSamples(sampleIndex, :)).';
    accumulator = accumulator + normalizedQuaternion * normalizedQuaternion.';
end

[eigenVectors, eigenValues] = eig(accumulator);
[~, dominantIndex] = max(diag(eigenValues));
averageQuaternion = eigenVectors(:, dominantIndex).';
averageQuaternion = normalizeQuaternion(averageQuaternion);
end

function quaternion = normalizeQuaternion(quaternion)
quaternion = reshape(quaternion, 1, 4);
quaternionNorm = norm(quaternion);

if quaternionNorm <= eps
    quaternion = [0, 0, 0, 1];
    return;
end

quaternion = quaternion ./ quaternionNorm;
if quaternion(4) < 0
    quaternion = -quaternion;
end
end

function quaternionConjugate = conjugateQuaternion(quaternion)
normalizedQuaternion = normalizeQuaternion(quaternion);
quaternionConjugate = [-normalizedQuaternion(1:3), normalizedQuaternion(4)];
end

function productQuaternion = multiplyQuaternions(firstQuaternion, secondQuaternion)
q1 = normalizeQuaternion(firstQuaternion);
q2 = normalizeQuaternion(secondQuaternion);

x1 = q1(1);
y1 = q1(2);
z1 = q1(3);
w1 = q1(4);

x2 = q2(1);
y2 = q2(2);
z2 = q2(3);
w2 = q2(4);

productQuaternion = [ ...
    w1 .* x2 + x1 .* w2 + y1 .* z2 - z1 .* y2, ...
    w1 .* y2 - x1 .* z2 + y1 .* w2 + z1 .* x2, ...
    w1 .* z2 + x1 .* y2 - y1 .* x2 + z1 .* w2, ...
    w1 .* w2 - x1 .* x2 - y1 .* y2 - z1 .* z2];

productQuaternion = normalizeQuaternion(productQuaternion);
end

function twistAngleRadians = extractTwistAngle(relativeQuaternion, hingeAxis)
hingeAxis = hingeAxis ./ norm(hingeAxis);
relativeQuaternion = normalizeQuaternion(relativeQuaternion);

projectedVector = dot(relativeQuaternion(1:3), hingeAxis) .* hingeAxis;
twistQuaternion = normalizeQuaternion([projectedVector, relativeQuaternion(4)]);

twistAngleRadians = 2 .* atan2(dot(twistQuaternion(1:3), hingeAxis), twistQuaternion(4));
twistAngleRadians = wrapAngleToPi(twistAngleRadians);
end

function wrappedAngle = wrapAngleToPi(angleRadians)
wrappedAngle = mod(angleRadians + pi, 2 .* pi) - pi;
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

function normalizedRows = normalizeRowVectors(rowVectors, fieldName)
normalizedRows = rowVectors;

for rowIndex = 1:size(rowVectors, 1)
    rowNorm = norm(rowVectors(rowIndex, :));
    if rowNorm <= eps
        error("Servo_Test:ZeroVector", "%s contains a zero-length vector in row %d.", fieldName, rowIndex);
    end

    normalizedRows(rowIndex, :) = rowVectors(rowIndex, :) ./ rowNorm;
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
    error("Servo_Test:InvalidConfigType", "%s must be a text scalar.", fieldName);
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
    error("Servo_Test:InvalidConfigType", ...
        "%s must be text, a string array, or a cell array of character vectors.", ...
        fieldName);
end

value = reshape(value, [], 1);
end

function value = getLogicalField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"logical", "numeric"}, {"scalar"}, mfilename, fieldName);
value = logical(value);
end

function value = getPositiveIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "positive"}, mfilename, fieldName);
value = double(value);
end

function value = getPositiveScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "positive"}, mfilename, fieldName);
value = double(value);
end

function value = getNonnegativeScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "nonnegative"}, mfilename, fieldName);
value = double(value);
end

function value = getOptionalScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = defaultValue;
end

validateattributes(value, {"numeric"}, {"real", "scalar"}, mfilename, fieldName);
value = double(value);
end

function value = getScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar"}, mfilename, fieldName);
value = double(value);
end

function value = getNumericColumnField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "column"}, mfilename, fieldName);
value = double(value);
end

function value = getNumericMatrixField(config, fieldName, defaultValue, expectedSize)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "size", expectedSize}, mfilename, fieldName);
value = double(value);
end

function value = getNumericVectorField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = [];
    return;
end

validateattributes(value, {"numeric"}, {"real", "finite", "vector"}, mfilename, fieldName);
value = reshape(double(value), [], 1);
end

function mustHaveMatchingLength(values, expectedLength, fieldName)
if numel(values) ~= expectedLength
    error("Servo_Test:InvalidArrayLength", ...
        "%s must contain exactly %d elements.", ...
        fieldName, ...
        expectedLength);
end
end
