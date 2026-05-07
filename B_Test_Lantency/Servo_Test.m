function runData = Servo_Test(config)
%SERVO_TEST Execute a servo-command latency test with Arduino and Vicon.
%   Optionally imports an Arduino-side echo log during post-processing so
%   latency can be analysed without extra work in the inner loop.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);
runData.runInfo.startTime = datetime("now");

assignin("base", "ServoTestLatestState", struct([]));
assignin("base", "ServoTestRunData", runData);

commandInterface = struct();
viconClient = [];
cleanupHandle = onCleanup(@() cleanupResources( ...
    commandInterface, ...
    viconClient, ...
    config));

try
    [commandInterface, arduinoInfo] = connectToArduino(config);
    config.arduinoTransport.resolvedMode = arduinoInfo.transportResolvedMode;
    if arduinoInfo.isConnected && config.arduinoTransport.resolvedMode ~= "nano_logger_udp"
        config.arduinoTransport.captureMessage = ...
            "No integrated Nano logger capture; transport resolved to " + config.arduinoTransport.resolvedMode + ".";
    end
    runData.config = config;
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

    moveServosToNeutral(commandInterface, config.servoNeutralPositions);
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

    [storage, runInfo, config] = executeLatencyTest( ...
        commandInterface, ...
        viconClient, ...
        trackedSubjects, ...
        neutralReference, ...
        config);

    runData.config = config;
    runData.runInfo = runInfo;
    runData.logs = buildLogTimetables(storage, config);
    runData.surfaceSummary = buildSurfaceSummary(storage, config);
    runData.artifacts = exportRunData(runData);

    assignin("base", "ServoTestRunData", runData);
    assignin("base", "ServoTestLatestState", buildLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupResources(commandInterface, viconClient, config);
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
config.arduinoEchoImport = normalizeArduinoEchoImportConfig( ...
    getFieldOrDefault(config, "arduinoEchoImport", struct()), ...
    rootFolder);

config.neutralSettleSeconds = getPositiveScalarField(config, "neutralSettleSeconds", 1.0);
config.neutralCalibrationSeconds = getPositiveScalarField(config, "neutralCalibrationSeconds", 1.0);
config.returnToNeutralOnExit = getLogicalField(config, "returnToNeutralOnExit", true);

config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "D_Servo_Test"));
config.runLabel = getTextScalarField(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    timeStamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    config.runLabel = timeStamp + "_Test";
end

config.arduinoTransport = normalizeArduinoTransportConfig( ...
    getFieldOrDefault(config, "arduinoTransport", struct()), ...
    config.outputFolder, ...
    config.runLabel);

validateattributes(config.arduinoPort, {"numeric"}, {"real", "scalar"}, char(mfilename), 'arduinoPort');

surfaceCount = numel(config.surfaceNames);
mustHaveMatchingLength(config.surfacePins, surfaceCount, "surfacePins");
mustHaveMatchingLength(config.viconSubjectNames, surfaceCount, "viconSubjectNames");
mustHaveMatchingLength(config.viconSegmentNames, surfaceCount, "viconSegmentNames");

validateattributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoNeutralPositions');
validateattributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoUnitsPerDegree');
validateattributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMinimumPositions');
validateattributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMaximumPositions');
validateattributes(config.servoMinPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, char(mfilename), 'servoMinPulseDurationSeconds');
validateattributes(config.servoMaxPulseDurationSeconds, {"numeric"}, {"real", "column", "numel", surfaceCount}, char(mfilename), 'servoMaxPulseDurationSeconds');
validateattributes(config.viconMeasurementSigns, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'viconMeasurementSigns');
validateattributes(config.commandDeflectionScales, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionScales');
validateattributes(config.commandDeflectionOffsetsDegrees, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionOffsetsDegrees');

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
    validateattributes(commandProfile.customTimeSeconds, {"numeric"}, {"real", "finite", "vector", "nonempty"}, char(mfilename), 'commandProfile.customTimeSeconds');
    validateattributes(commandProfile.customDeflectionDegrees, {"numeric"}, {"real", "finite", "vector", "numel", numel(commandProfile.customTimeSeconds)}, char(mfilename), 'commandProfile.customDeflectionDegrees');

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

function arduinoEchoImport = normalizeArduinoEchoImportConfig(arduinoEchoImportConfig, rootFolder)
if isempty(arduinoEchoImportConfig)
    arduinoEchoImportConfig = struct();
end

arduinoEchoImport = struct( ...
    "filePath", getOptionalTextScalarField(arduinoEchoImportConfig, "filePath", ""), ...
    "sheetName", getOptionalTextScalarField(arduinoEchoImportConfig, "sheetName", ""), ...
    "surfaceColumn", getTextScalarField(arduinoEchoImportConfig, "surfaceColumn", "surface_name"), ...
    "sequenceColumn", getTextScalarField(arduinoEchoImportConfig, "sequenceColumn", "command_sequence"), ...
    "latencyColumn", getTextScalarField(arduinoEchoImportConfig, "latencyColumn", "computer_to_arduino_latency_s"), ...
    "echoTimeColumn", getOptionalTextScalarField(arduinoEchoImportConfig, "echoTimeColumn", ""), ...
    "matlabTimeOffsetSeconds", getScalarField(arduinoEchoImportConfig, "matlabTimeOffsetSeconds", 0.0), ...
    "appliedPositionColumn", getOptionalTextScalarField(arduinoEchoImportConfig, "appliedPositionColumn", "applied_position"), ...
    "appliedEquivalentDegreesColumn", getOptionalTextScalarField(arduinoEchoImportConfig, "appliedEquivalentDegreesColumn", "applied_equivalent_deg"), ...
    "autoDiscoverLatest", getLogicalField(arduinoEchoImportConfig, "autoDiscoverLatest", true), ...
    "loggerOutputFolder", getOptionalTextScalarField( ...
        arduinoEchoImportConfig, ...
        "loggerOutputFolder", ...
        fullfile(rootFolder, "E_Nano33IoT_Echo_Logger", "output")), ...
    "tableData", table());

if isfield(arduinoEchoImportConfig, "tableData")
    tableData = arduinoEchoImportConfig.tableData;
    if isempty(tableData)
        tableData = table();
    end

    if ~istable(tableData)
        error("Servo_Test:InvalidArduinoEchoImport", "arduinoEchoImport.tableData must be a table when provided.");
    end

    arduinoEchoImport.tableData = tableData;
end
end

function arduinoTransport = normalizeArduinoTransportConfig(arduinoTransportConfig, outputFolder, runLabel)
if isempty(arduinoTransportConfig)
    arduinoTransportConfig = struct();
end

defaultLoggerOutputFolder = fullfile(outputFolder, runLabel + "_ArduinoLogger");
requestedMode = canonicalizeArduinoTransportMode( ...
    getTextScalarField(arduinoTransportConfig, "mode", "nano_logger_udp"));
arduinoTransport = struct( ...
    "mode", requestedMode, ...
    "resolvedMode", "", ...
    "loggerPort", getPositiveIntegerField(arduinoTransportConfig, "loggerPort", 9500), ...
    "loggerProbeTimeoutSeconds", getPositiveScalarField(arduinoTransportConfig, "loggerProbeTimeoutSeconds", 1.0), ...
    "loggerTimeoutSeconds", getPositiveScalarField(arduinoTransportConfig, "loggerTimeoutSeconds", 5.0), ...
    "syncCountBeforeRun", getNonnegativeIntegerField(arduinoTransportConfig, "syncCountBeforeRun", 10), ...
    "syncCountAfterRun", getNonnegativeIntegerField(arduinoTransportConfig, "syncCountAfterRun", 10), ...
    "syncPauseSeconds", getNonnegativeScalarField(arduinoTransportConfig, "syncPauseSeconds", 0.05), ...
    "postRunSettleSeconds", getNonnegativeScalarField(arduinoTransportConfig, "postRunSettleSeconds", 0.25), ...
    "clearLogsBeforeRun", getLogicalField(arduinoTransportConfig, "clearLogsBeforeRun", true), ...
    "loggerOutputFolder", getOptionalTextScalarField(arduinoTransportConfig, "loggerOutputFolder", defaultLoggerOutputFolder), ...
    "captureSucceeded", false, ...
    "captureMessage", "", ...
    "captureRowCount", 0);

validTransportModes = ["auto", "mathworks_servo", "nano_logger_udp"];
if ~any(arduinoTransport.mode == validTransportModes)
    error("Servo_Test:InvalidArduinoTransportMode", ...
        "arduinoTransport.mode must be one of: %s.", ...
        char(join(validTransportModes, ", ")));
end
end

function transportMode = canonicalizeArduinoTransportMode(transportMode)
transportMode = string(transportMode);
if transportMode == "nano_logger_tcp"
    transportMode = "nano_logger_udp";
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

function [commandInterface, connectionInfo] = connectToArduino(config)
commandInterface = struct();

connectionInfo = struct( ...
    "ipAddress", config.arduinoIPAddress, ...
    "boardName", config.arduinoBoard, ...
    "port", config.arduinoPort, ...
    "isConnected", false, ...
    "connectElapsedSeconds", NaN, ...
    "connectionMessage", "", ...
    "availableLibraries", strings(0, 1), ...
    "surfaceNames", config.surfaceNames, ...
    "surfacePins", config.surfacePins, ...
    "transportRequestedMode", config.arduinoTransport.mode, ...
    "transportResolvedMode", "", ...
    "loggerPort", config.arduinoTransport.loggerPort, ...
    "loggerGreeting", "", ...
    "loggerFirmwareVersion", "", ...
    "loggerBackend", "", ...
    "transportDiagnostics", strings(0, 1));

connectStart = tic;
attemptModes = resolveArduinoTransportAttemptModes(config.arduinoTransport.mode);
transportDiagnostics = strings(0, 1);

for attemptIndex = 1:numel(attemptModes)
    attemptMode = attemptModes(attemptIndex);

    try
        switch attemptMode
            case "nano_logger_udp"
                loggerTimeoutSeconds = config.arduinoTransport.loggerTimeoutSeconds;
                if config.arduinoTransport.mode == "auto"
                    loggerTimeoutSeconds = config.arduinoTransport.loggerProbeTimeoutSeconds;
                end

                [commandInterface, loggerGreeting, firmwareVersion, loggerBackend] = ...
                    createNanoLoggerCommandInterface(config, loggerTimeoutSeconds);
                connectionInfo.loggerGreeting = loggerGreeting;
                connectionInfo.loggerFirmwareVersion = firmwareVersion;
                connectionInfo.loggerBackend = loggerBackend;
                connectionInfo.connectionMessage = "Prepared Nano logger UDP datagram transport (" + loggerBackend + ").";
            otherwise
                [commandInterface, availableLibraries] = createMathWorksCommandInterface(config);
                connectionInfo.availableLibraries = availableLibraries;
                connectionInfo.connectionMessage = "Connected via MathWorks arduino()/servo().";
        end

        connectionInfo.isConnected = true;
        connectionInfo.transportResolvedMode = attemptMode;
        break;
    catch connectionException
        transportDiagnostics(end + 1, 1) = ...
            attemptMode + ": " + string(connectionException.message); %#ok<AGROW>
    end
end

if ~connectionInfo.isConnected
    if ~isempty(transportDiagnostics)
        connectionInfo.connectionMessage = transportDiagnostics(end);
    else
        connectionInfo.connectionMessage = "No Arduino transport attempts were made.";
    end
end

connectionInfo.transportDiagnostics = transportDiagnostics;
connectionInfo.connectElapsedSeconds = toc(connectStart);
end

function attemptModes = resolveArduinoTransportAttemptModes(requestedMode)
switch requestedMode
    case "auto"
        attemptModes = ["nano_logger_udp"; "mathworks_servo"];
    case "nano_logger_udp"
        attemptModes = "nano_logger_udp";
    otherwise
        attemptModes = "mathworks_servo";
end
end

function [commandInterface, availableLibraries] = createMathWorksCommandInterface(config)
surfaceCount = numel(config.surfaceNames);
servoObjects = cell(surfaceCount, 1);
arduinoConnection = createArduinoConnection(config);
availableLibraries = strings(0, 1);

if isprop(arduinoConnection, "Libraries")
    availableLibraries = reshape(string(arduinoConnection.Libraries), [], 1);
end

for surfaceIndex = 1:surfaceCount
    servoObjects{surfaceIndex} = createServoObject( ...
        arduinoConnection, ...
        config.surfacePins(surfaceIndex), ...
        config.servoMinPulseDurationSeconds(surfaceIndex), ...
        config.servoMaxPulseDurationSeconds(surfaceIndex));
end

commandInterface = struct( ...
    "transportMode", "mathworks_servo", ...
    "connection", arduinoConnection, ...
    "servoObjects", {servoObjects}, ...
    "surfaceNames", config.surfaceNames);
end

function [commandInterface, greeting, firmwareVersion, backendName] = createNanoLoggerCommandInterface(config, timeoutSeconds)
[loggerConnection, backendName] = createNanoLoggerConnection( ...
    config.arduinoIPAddress, ...
    config.arduinoTransport.loggerPort, ...
    timeoutSeconds);

greeting = "UDP_DATAGRAM_TRANSPORT";
firmwareVersion = "unknown";
commandInterface = struct( ...
    "transportMode", "nano_logger_udp", ...
    "connection", loggerConnection, ...
    "servoObjects", {cell(numel(config.surfaceNames), 1)}, ...
    "surfaceNames", config.surfaceNames);
end

function [loggerConnection, backendName] = createNanoLoggerConnection(ipAddress, loggerPort, timeoutSeconds)
timeoutMilliseconds = int32(max(1, ceil(timeoutSeconds .* 1000)));
socket = javaObject("java.net.DatagramSocket");
socket.setSoTimeout(timeoutMilliseconds);
socket.setReceiveBufferSize(int32(262144));
remoteAddress = javaMethod("getByName", "java.net.InetAddress", char(ipAddress));
socket.connect(remoteAddress, int32(loggerPort));

loggerConnection = struct( ...
    "backend", "java_datagram_socket", ...
    "timeoutSeconds", timeoutSeconds, ...
    "socket", socket, ...
    "remoteAddress", remoteAddress, ...
    "remotePort", loggerPort, ...
    "packetBufferBytes", 2048);
backendName = "java_datagram_socket";
end

function sendNanoLoggerDatagram(loggerConnection, payloadText)
payloadBytes = uint8(char(string(payloadText)));
packet = javaObject( ...
    "java.net.DatagramPacket", ...
    int8(payloadBytes), ...
    numel(payloadBytes), ...
    loggerConnection.remoteAddress, ...
    int32(loggerConnection.remotePort));
loggerConnection.socket.send(packet);
end

function nextDatagram = tryReadNanoLoggerDatagram(loggerConnection, timeoutSeconds)
nextDatagram = "";
timeoutMilliseconds = int32(max(1, ceil(timeoutSeconds .* 1000)));
loggerConnection.socket.setSoTimeout(timeoutMilliseconds);

receivePacket = javaObject( ...
    "java.net.DatagramPacket", ...
    int8(zeros(1, loggerConnection.packetBufferBytes)), ...
    loggerConnection.packetBufferBytes);

try
    loggerConnection.socket.receive(receivePacket);
    packetLength = double(receivePacket.getLength());
    if packetLength <= 0
        return;
    end

    packetBytes = uint8(receivePacket.getData());
    packetBytes = reshape(packetBytes(1:packetLength), 1, []);
    nextDatagram = string(strtrim(char(packetBytes)));
catch readException
    if ~isNanoLoggerTimeoutException(readException)
        rethrow(readException);
    end
end
end

function sendNanoLoggerControlBurst(loggerConnection, payloadText, repeatCount, pauseSeconds)
for repeatIndex = 1:repeatCount
    sendNanoLoggerDatagram(loggerConnection, payloadText);
    if repeatIndex < repeatCount && pauseSeconds > 0
        pause(pauseSeconds);
    end
end
end

function isTimeout = isNanoLoggerTimeoutException(readException)
exceptionMessage = string(readException.message);
isTimeout = ...
    contains(exceptionMessage, "timed out", 'IgnoreCase', true) || ...
    contains(exceptionMessage, "timeout", 'IgnoreCase', true);
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
if isfield(runData.connectionInfo.arduino, "transportResolvedMode") && ...
        strlength(runData.connectionInfo.arduino.transportResolvedMode) > 0
    fprintf("  Transport mode: %s\n", char(runData.connectionInfo.arduino.transportResolvedMode));
end
if isfield(runData.connectionInfo.arduino, "loggerBackend") && ...
        strlength(runData.connectionInfo.arduino.loggerBackend) > 0
    fprintf("  Logger backend: %s\n", char(runData.connectionInfo.arduino.loggerBackend));
end
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

function [storage, runInfo, config] = executeLatencyTest(commandInterface, client, trackedSubjects, neutralReference, config)
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

if isNanoLoggerTransport(commandInterface)
    loggerSession = startNanoLoggerSession(commandInterface, config, sampleCount);
    testStart = loggerSession.hostTimer;
    testStartOffsetSeconds = loggerSession.testStartOffsetSeconds;
else
    loggerSession = createEmptyNanoLoggerSession();
    testStart = tic;
    testStartOffsetSeconds = 0;
end

surfaceCommandCounts = zeros(1, surfaceCount);

for sampleIndex = 1:sampleCount
    scheduledSampleTimeSeconds = scheduledTimeSeconds(sampleIndex);
    waitForScheduledTime(testStart, testStartOffsetSeconds + scheduledSampleTimeSeconds);

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

    nextCommandSequenceNumbers = nan(1, surfaceCount);
    nextCommandSequenceNumbers(config.activeSurfaceMask.') = ...
        surfaceCommandCounts(config.activeSurfaceMask.') + 1;

    storage.commandWriteStartSeconds(sampleIndex) = max(0, toc(testStart) - testStartOffsetSeconds);
    [commandDispatchSeconds, commandWriteStopSeconds, loggerSession] = writeServoPositions( ...
        commandInterface, ...
        commandedServoPositionsClipped, ...
        testStart, ...
        config.activeSurfaceMask.', ...
        config.surfaceNames, ...
        nextCommandSequenceNumbers, ...
        loggerSession);
    storage.commandDispatchSeconds(sampleIndex, :) = commandDispatchSeconds;
    storage.commandWriteStopSeconds(sampleIndex) = commandWriteStopSeconds;
    surfaceCommandCounts(config.activeSurfaceMask.') = nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.commandSequenceNumbers(sampleIndex, config.activeSurfaceMask.') = ...
        nextCommandSequenceNumbers(config.activeSurfaceMask.');

    frameReady = waitForFrame(client, config.frameWaitTimeoutSeconds);
    storage.viconSampleTimeSeconds(sampleIndex) = max(0, toc(testStart) - testStartOffsetSeconds);

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
    storage.viconFrameNumbers(sampleIndex) = snapshot.frameNumber;
    storage.viconLatencySeconds(sampleIndex) = snapshot.latencySeconds;
    storage.measuredDeflectionsDegrees(sampleIndex, :) = snapshot.measuredAnglesDegrees;
    storage.viconPositionMillimeters(sampleIndex, :, :) = snapshot.positionMillimeters;
    storage.viconPositionOccluded(sampleIndex, :) = snapshot.positionOccluded;
    storage.viconOccluded(sampleIndex, :) = snapshot.isOccluded;
    storage.viconQuaternionXYZW(sampleIndex, :, :) = snapshot.quaternionXYZW;

    assignin("base", "ServoTestLatestState", buildLatestState(storage, config, sampleIndex));
end

if isNanoLoggerTransport(commandInterface)
    config = finalizeNanoLoggerSession(commandInterface, loggerSession, config);
end

storage = finalizeArduinoEcho(storage, config);
runInfo.status = "completed";
runInfo.stopTime = datetime("now");
runInfo.sampleCount = storage.sampleCount;

if config.returnToNeutralOnExit
    moveServosToNeutral(commandInterface, config.servoNeutralPositions);
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
    "commandSequenceNumbers", nan(sampleCount, surfaceCount), ...
    "arduinoReadStartSeconds", nan(sampleCount, 1), ...
    "arduinoReadStopSeconds", nan(sampleCount, 1), ...
    "arduinoEchoSeconds", nan(sampleCount, surfaceCount), ...
    "viconSampleTimeSeconds", nan(sampleCount, 1), ...
    "viconFrameNumbers", nan(sampleCount, 1), ...
    "viconLatencySeconds", nan(sampleCount, 1), ...
    "desiredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "commandedServoPositions", nan(sampleCount, surfaceCount), ...
    "commandSaturated", false(sampleCount, surfaceCount), ...
    "appliedServoPositions", nan(sampleCount, surfaceCount), ...
    "appliedEquivalentDegrees", nan(sampleCount, surfaceCount), ...
    "measuredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "viconPositionMillimeters", nan(sampleCount, surfaceCount, 3), ...
    "viconPositionOccluded", true(sampleCount, surfaceCount), ...
    "viconOccluded", true(sampleCount, surfaceCount), ...
    "viconQuaternionXYZW", nan(sampleCount, surfaceCount, 4), ...
    "profileInfo", struct());
end

function moveServosToNeutral(commandInterface, neutralPositions)
if isempty(fieldnames(commandInterface))
    return;
end

if isNanoLoggerTransport(commandInterface)
    sendNanoLoggerControlBurst(commandInterface.connection, "SET_NEUTRAL", 2, 0.02);
    return;
end

neutralRow = reshape(neutralPositions, 1, []);
writeServoPositions(commandInterface, neutralRow);
end

function [dispatchTimesSeconds, writeStopSeconds, loggerSession] = writeServoPositions( ...
    commandInterface, ...
    servoPositions, ...
    referenceTimer, ...
    activeSurfaceMask, ...
    surfaceNames, ...
    commandSequenceNumbers, ...
    loggerSession)
if nargin < 3
    referenceTimer = [];
end

if nargin < 4 || isempty(activeSurfaceMask)
    activeSurfaceMask = true(1, numel(commandInterface.surfaceNames));
end

if nargin < 5 || isempty(surfaceNames)
    surfaceNames = commandInterface.surfaceNames;
end

if nargin < 6 || isempty(commandSequenceNumbers)
    commandSequenceNumbers = nan(1, numel(surfaceNames));
end

if nargin < 7 || isempty(loggerSession)
    loggerSession = createEmptyNanoLoggerSession();
end

dispatchTimesSeconds = nan(1, numel(surfaceNames));
writeStopSeconds = NaN;

switch commandInterface.transportMode
    case "nano_logger_udp"
        for surfaceIndex = 1:numel(surfaceNames)
            if ~activeSurfaceMask(surfaceIndex)
                continue;
            end

            commandSequence = commandSequenceNumbers(surfaceIndex);
            if ~isfinite(commandSequence)
                error("Servo_Test:MissingNanoLoggerSequence", ...
                    "Missing command sequence for surface %s.", ...
                    char(surfaceNames(surfaceIndex)));
            end

            dispatchAbsoluteUs = hostNowUs(loggerSession.hostTimer);
            commandLine = sprintf( ...
                "SET,%s,%u,%.6f", ...
                char(surfaceNames(surfaceIndex)), ...
                uint32(commandSequence), ...
                servoPositions(surfaceIndex));
            sendNanoLoggerDatagram(commandInterface.connection, commandLine);

            loggerSession = appendNanoLoggerDispatchRow( ...
                loggerSession, ...
                surfaceNames(surfaceIndex), ...
                commandSequence, ...
                dispatchAbsoluteUs, ...
                servoPositions(surfaceIndex));

            if ~isempty(referenceTimer)
                dispatchTimesSeconds(surfaceIndex) = max( ...
                    0, ...
                    (double(dispatchAbsoluteUs) - double(loggerSession.testStartOffsetUs)) ./ 1e6);
            end
        end

        if ~isempty(referenceTimer)
            writeStopSeconds = max(0, toc(referenceTimer) - loggerSession.testStartOffsetSeconds);
        end
    otherwise
        for surfaceIndex = 1:numel(commandInterface.servoObjects)
            if ~activeSurfaceMask(surfaceIndex)
                continue;
            end

            writePosition(commandInterface.servoObjects{surfaceIndex}, servoPositions(surfaceIndex));

            if ~isempty(referenceTimer)
                dispatchTimesSeconds(surfaceIndex) = toc(referenceTimer);
            end
        end

        if ~isempty(referenceTimer)
            writeStopSeconds = toc(referenceTimer);
        end
end
end

function loggerSession = createEmptyNanoLoggerSession()
loggerSession = struct( ...
    "isEnabled", false, ...
    "hostTimer", [], ...
    "testStartOffsetUs", uint32(0), ...
    "testStartOffsetSeconds", 0, ...
    "dispatchCount", 0, ...
    "dispatchSurfaceName", strings(0, 1), ...
    "dispatchCommandSequence", zeros(0, 1), ...
    "dispatchCommandUs", zeros(0, 1), ...
    "dispatchPosition", zeros(0, 1), ...
    "syncCount", 0, ...
    "syncId", zeros(0, 1), ...
    "syncHostTxUs", zeros(0, 1), ...
    "syncHostRxUs", zeros(0, 1), ...
    "syncBoardRxUs", zeros(0, 1), ...
    "syncBoardTxUs", zeros(0, 1));
end

function loggerSession = startNanoLoggerSession(commandInterface, config, sampleCount)
activeSurfaceCount = nnz(config.activeSurfaceMask);
dispatchCapacity = sampleCount .* activeSurfaceCount;
syncCapacity = config.arduinoTransport.syncCountBeforeRun + config.arduinoTransport.syncCountAfterRun;

loggerSession = struct( ...
    "isEnabled", true, ...
    "hostTimer", tic, ...
    "testStartOffsetUs", uint32(0), ...
    "testStartOffsetSeconds", 0, ...
    "dispatchCount", 0, ...
    "dispatchSurfaceName", strings(dispatchCapacity, 1), ...
    "dispatchCommandSequence", nan(dispatchCapacity, 1), ...
    "dispatchCommandUs", nan(dispatchCapacity, 1), ...
    "dispatchPosition", nan(dispatchCapacity, 1), ...
    "syncCount", 0, ...
    "syncId", nan(syncCapacity, 1), ...
    "syncHostTxUs", nan(syncCapacity, 1), ...
    "syncHostRxUs", nan(syncCapacity, 1), ...
    "syncBoardRxUs", nan(syncCapacity, 1), ...
    "syncBoardTxUs", nan(syncCapacity, 1));

if config.arduinoTransport.clearLogsBeforeRun
    sendNanoLoggerControlBurst(commandInterface.connection, "CLEAR_LOGS", 3, 0.02);
end

for syncIndex = 1:config.arduinoTransport.syncCountBeforeRun
    loggerSession = appendNanoLoggerSyncRow( ...
        loggerSession, ...
        sendNanoLoggerSync(commandInterface.connection, loggerSession.hostTimer, uint32(syncIndex)));
    pause(config.arduinoTransport.syncPauseSeconds);
end

loggerSession.testStartOffsetUs = hostNowUs(loggerSession.hostTimer);
loggerSession.testStartOffsetSeconds = double(loggerSession.testStartOffsetUs) ./ 1e6;
end

function config = finalizeNanoLoggerSession(commandInterface, loggerSession, config)
if ~loggerSession.isEnabled
    return;
end

try
    if config.returnToNeutralOnExit
        sendNanoLoggerControlBurst(commandInterface.connection, "SET_NEUTRAL", 2, 0.02);
    end

    pause(config.arduinoTransport.postRunSettleSeconds);

    for syncIndex = 1:config.arduinoTransport.syncCountAfterRun
        syncId = uint32(config.arduinoTransport.syncCountBeforeRun + syncIndex);
        loggerSession = appendNanoLoggerSyncRow( ...
            loggerSession, ...
            sendNanoLoggerSync(commandInterface.connection, loggerSession.hostTimer, syncId));
        pause(config.arduinoTransport.syncPauseSeconds);
    end

    loggerOutputFolder = config.arduinoTransport.loggerOutputFolder;
    if strlength(loggerOutputFolder) == 0
        loggerOutputFolder = fullfile(config.outputFolder, config.runLabel + "_ArduinoLogger");
        config.arduinoTransport.loggerOutputFolder = loggerOutputFolder;
    end

    if ~isfolder(loggerOutputFolder)
        mkdir(loggerOutputFolder);
    end

    hostDispatchLog = buildNanoLoggerDispatchTable(loggerSession);
    [telemetryLines, boardCommandLog, boardSyncLog] = collectNanoLoggerTelemetry( ...
        commandInterface.connection, ...
        config.arduinoTransport.loggerTimeoutSeconds);

    if isempty(boardSyncLog)
        error("Servo_Test:MissingNanoLoggerSyncTelemetry", ...
            "No Nano logger SYNC_EVENT datagrams were received.");
    end

    if loggerSession.dispatchCount > 0 && isempty(boardCommandLog)
        error("Servo_Test:MissingNanoLoggerCommandTelemetry", ...
            "No Nano logger COMMAND_EVENT datagrams were received.");
    end

    syncRoundTripLog = buildNanoLoggerSyncRoundTripTableFromEvents(boardSyncLog);

    hostDispatchCsvPath = fullfile(loggerOutputFolder, "host_dispatch_log.csv");
    syncRoundTripCsvPath = fullfile(loggerOutputFolder, "host_sync_roundtrip.csv");
    boardCommandLogCsvPath = fullfile(loggerOutputFolder, "board_command_log.csv");
    boardSyncLogCsvPath = fullfile(loggerOutputFolder, "board_sync_log.csv");
    echoImportCsvPath = fullfile(loggerOutputFolder, "arduino_echo_import.csv");
    rawTelemetryLogPath = fullfile(loggerOutputFolder, "udp_telemetry_log.txt");

    writetable(hostDispatchLog, hostDispatchCsvPath);
    writetable(syncRoundTripLog, syncRoundTripCsvPath);
    writetable(boardCommandLog, boardCommandLogCsvPath);
    writetable(boardSyncLog, boardSyncLogCsvPath);
    if ~isempty(telemetryLines)
        writelines(telemetryLines, rawTelemetryLogPath);
    end

    echoImportTable = buildArduinoEchoImportTableFromLoggerRawFiles( ...
        hostDispatchCsvPath, ...
        syncRoundTripCsvPath, ...
        boardCommandLogCsvPath, ...
        config.activeSurfaceNames);

    if ismember("arduino_echo_time_s", string(echoImportTable.Properties.VariableNames))
        echoImportTable.arduino_echo_time_s(:) = nan(height(echoImportTable), 1);
    end

    writetable(echoImportTable, echoImportCsvPath);

    config.arduinoEchoImport.filePath = string(echoImportCsvPath);
    config.arduinoEchoImport.loggerOutputFolder = string(loggerOutputFolder);
    config.arduinoEchoImport.tableData = echoImportTable;
    config.arduinoTransport.captureSucceeded = true;
    config.arduinoTransport.captureMessage = ...
        "Captured " + height(boardCommandLog) + " command telemetry rows and " + ...
        height(boardSyncLog) + " sync telemetry rows.";
    config.arduinoTransport.captureRowCount = height(echoImportTable);
catch captureException
    warning("Servo_Test:ArduinoLoggerCaptureFailed", ...
        "Failed to capture Arduino logger output: %s", ...
        captureException.message);
    config.arduinoEchoImport.tableData = table();
    config.arduinoTransport.captureSucceeded = false;
    config.arduinoTransport.captureMessage = string(captureException.message);
    config.arduinoTransport.captureRowCount = 0;
end
end

function loggerSession = appendNanoLoggerDispatchRow( ...
    loggerSession, ...
    surfaceName, ...
    commandSequence, ...
    commandDispatchUs, ...
    commandPosition)
loggerSession.dispatchCount = loggerSession.dispatchCount + 1;
rowIndex = loggerSession.dispatchCount;

loggerSession.dispatchSurfaceName(rowIndex) = string(surfaceName);
loggerSession.dispatchCommandSequence(rowIndex) = double(commandSequence);
loggerSession.dispatchCommandUs(rowIndex) = double(commandDispatchUs);
loggerSession.dispatchPosition(rowIndex) = double(commandPosition);
end

function loggerSession = appendNanoLoggerSyncRow(loggerSession, syncRow)
loggerSession.syncCount = loggerSession.syncCount + 1;
rowIndex = loggerSession.syncCount;

loggerSession.syncId(rowIndex) = syncRow.sync_id;
loggerSession.syncHostTxUs(rowIndex) = syncRow.host_tx_us;
loggerSession.syncHostRxUs(rowIndex) = syncRow.host_rx_us;
loggerSession.syncBoardRxUs(rowIndex) = syncRow.board_rx_us;
loggerSession.syncBoardTxUs(rowIndex) = syncRow.board_tx_us;
end

function dispatchLog = buildNanoLoggerDispatchTable(loggerSession)
rowIndices = 1:loggerSession.dispatchCount;
dispatchLog = table( ...
    loggerSession.dispatchSurfaceName(rowIndices), ...
    loggerSession.dispatchCommandSequence(rowIndices), ...
    loggerSession.dispatchCommandUs(rowIndices), ...
    loggerSession.dispatchPosition(rowIndices), ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'command_dispatch_us', ...
        'position_norm'});
end

function syncRoundTripLog = buildNanoLoggerSyncRoundTripTable(loggerSession)
rowIndices = 1:loggerSession.syncCount;
syncRoundTripLog = table( ...
    loggerSession.syncId(rowIndices), ...
    loggerSession.syncHostTxUs(rowIndices), ...
    loggerSession.syncHostRxUs(rowIndices), ...
    loggerSession.syncBoardRxUs(rowIndices), ...
    loggerSession.syncBoardTxUs(rowIndices), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});
end

function syncRoundTripLog = buildNanoLoggerSyncRoundTripTableFromEvents(boardSyncLog)
syncRoundTripLog = table( ...
    boardSyncLog.sync_id, ...
    boardSyncLog.host_tx_us, ...
    nan(height(boardSyncLog), 1), ...
    boardSyncLog.board_rx_us, ...
    boardSyncLog.board_tx_us, ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});
end

function syncRow = sendNanoLoggerSync(loggerConnection, hostTimer, syncId)
hostTxUs = hostNowUs(hostTimer);
sendNanoLoggerDatagram(loggerConnection, sprintf("SYNC,%u,%u", uint32(syncId), uint32(hostTxUs)));

syncRow = table( ...
    double(syncId), ...
    double(hostTxUs), ...
    nan(1, 1), ...
    nan(1, 1), ...
    nan(1, 1), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});
end

function [telemetryLines, boardCommandLog, boardSyncLog] = collectNanoLoggerTelemetry(loggerConnection, maxWaitSeconds)
telemetryLines = strings(0, 1);
collectStart = tic;
lastReceiveElapsedSeconds = 0;
idleTimeoutSeconds = 0.25;

while toc(collectStart) < maxWaitSeconds
    remainingSeconds = maxWaitSeconds - toc(collectStart);
    perReadTimeoutSeconds = min(0.05, remainingSeconds);
    nextDatagram = tryReadNanoLoggerDatagram(loggerConnection, perReadTimeoutSeconds);

    if strlength(nextDatagram) > 0
        telemetryLines(end + 1, 1) = nextDatagram; %#ok<AGROW>
        lastReceiveElapsedSeconds = toc(collectStart);
        continue;
    end

    if toc(collectStart) >= idleTimeoutSeconds && ...
            (toc(collectStart) - lastReceiveElapsedSeconds) >= idleTimeoutSeconds
        break;
    end
end

[boardCommandLog, boardSyncLog] = parseNanoLoggerTelemetryDatagrams(telemetryLines);
end

function [boardCommandLog, boardSyncLog] = parseNanoLoggerTelemetryDatagrams(telemetryLines)
commandSurfaceNames = strings(0, 1);
commandSequence = nan(0, 1);
commandRxUs = nan(0, 1);
commandApplyUs = nan(0, 1);
commandAppliedPosition = nan(0, 1);
commandPulseUs = nan(0, 1);

syncId = nan(0, 1);
syncHostTxUs = nan(0, 1);
syncBoardRxUs = nan(0, 1);
syncBoardTxUs = nan(0, 1);

for lineIndex = 1:numel(telemetryLines)
    telemetryParts = split(string(strtrim(telemetryLines(lineIndex))), ",");
    if isempty(telemetryParts)
        continue;
    end

    switch telemetryParts(1)
        case "COMMAND_EVENT"
            if numel(telemetryParts) < 7
                continue;
            end

            commandSurfaceNames(end + 1, 1) = telemetryParts(2); %#ok<AGROW>
            commandSequence(end + 1, 1) = double(str2double(telemetryParts(3))); %#ok<AGROW>
            commandRxUs(end + 1, 1) = double(str2double(telemetryParts(4))); %#ok<AGROW>
            commandApplyUs(end + 1, 1) = double(str2double(telemetryParts(5))); %#ok<AGROW>
            commandAppliedPosition(end + 1, 1) = double(str2double(telemetryParts(6))); %#ok<AGROW>
            commandPulseUs(end + 1, 1) = double(str2double(telemetryParts(7))); %#ok<AGROW>
        case "SYNC_EVENT"
            if numel(telemetryParts) < 5
                continue;
            end

            syncId(end + 1, 1) = double(str2double(telemetryParts(2))); %#ok<AGROW>
            syncHostTxUs(end + 1, 1) = double(str2double(telemetryParts(3))); %#ok<AGROW>
            syncBoardRxUs(end + 1, 1) = double(str2double(telemetryParts(4))); %#ok<AGROW>
            syncBoardTxUs(end + 1, 1) = double(str2double(telemetryParts(5))); %#ok<AGROW>
    end
end

boardCommandLog = table( ...
    commandSurfaceNames, ...
    commandSequence, ...
    commandRxUs, ...
    commandApplyUs, ...
    commandApplyUs - commandRxUs, ...
    commandAppliedPosition, ...
    commandPulseUs, ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'rx_us', ...
        'apply_us', ...
        'receive_to_apply_us', ...
        'applied_position', ...
        'pulse_us'});

boardSyncLog = table( ...
    syncId, ...
    syncHostTxUs, ...
    syncBoardRxUs, ...
    syncBoardTxUs, ...
    syncBoardTxUs - syncBoardRxUs, ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'board_rx_us', ...
        'board_tx_us', ...
        'board_turnaround_us'});
end

function nowUs = hostNowUs(hostTimer)
nowUs = uint32(round(toc(hostTimer) .* 1e6));
end

function isLogger = isNanoLoggerTransport(commandInterface)
isLogger = ...
    isstruct(commandInterface) && ...
    isfield(commandInterface, "transportMode") && ...
    commandInterface.transportMode == "nano_logger_udp";
end

function storage = finalizeArduinoEcho(storage, config)
sampleIndices = 1:storage.sampleCount;

if isempty(sampleIndices)
    return;
end

storage.appliedServoPositions(sampleIndices, :) = nan(numel(sampleIndices), numel(config.surfaceNames));
storage.appliedEquivalentDegrees(sampleIndices, :) = nan(numel(sampleIndices), numel(config.surfaceNames));
storage.arduinoReadStartSeconds(sampleIndices) = nan(numel(sampleIndices), 1);
storage.arduinoReadStopSeconds(sampleIndices) = nan(numel(sampleIndices), 1);
storage.arduinoEchoSeconds(sampleIndices, :) = nan(numel(sampleIndices), numel(config.surfaceNames));

echoImportTable = loadArduinoEchoImportTable(config, config.activeSurfaceNames);
if isempty(echoImportTable)
    return;
end

commandLookup = buildArduinoCommandLookupTable(storage, config, sampleIndices);
echoAssignments = assignArduinoEchoToCommands(commandLookup, echoImportTable);

for assignmentIndex = 1:height(echoAssignments)
    sampleIndex = echoAssignments.sample_index(assignmentIndex);
    surfaceIndex = echoAssignments.surface_index(assignmentIndex);
    appliedPosition = echoAssignments.applied_position(assignmentIndex);
    appliedEquivalentDegrees = echoAssignments.applied_equivalent_deg(assignmentIndex);

    if ~isfinite(appliedEquivalentDegrees) && isfinite(appliedPosition)
        appliedEquivalentDegrees = ...
            (appliedPosition - config.servoNeutralPositions(surfaceIndex)) ./ ...
            config.servoUnitsPerDegree(surfaceIndex);
    elseif ~isfinite(appliedPosition) && isfinite(appliedEquivalentDegrees)
        appliedPosition = ...
            config.servoNeutralPositions(surfaceIndex) + ...
            appliedEquivalentDegrees .* config.servoUnitsPerDegree(surfaceIndex);
    end

    storage.arduinoReadStartSeconds(sampleIndex) = storage.commandWriteStopSeconds(sampleIndex);
    storage.arduinoEchoSeconds(sampleIndex, surfaceIndex) = echoAssignments.arduino_echo_time_s(assignmentIndex);
    storage.appliedServoPositions(sampleIndex, surfaceIndex) = appliedPosition;
    storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex) = appliedEquivalentDegrees;
end

for sampleIndex = sampleIndices
    if any(isfinite(storage.arduinoEchoSeconds(sampleIndex, :)))
        storage.arduinoReadStopSeconds(sampleIndex) = lastFiniteTimestamp( ...
            storage.arduinoEchoSeconds(sampleIndex, :), ...
            storage.commandWriteStopSeconds(sampleIndex));
    end
end
end

function echoImportTable = loadArduinoEchoImportTable(config, activeSurfaceNames)
echoImportTable = table();
importConfig = config.arduinoEchoImport;

if istable(importConfig.tableData) && ~isempty(importConfig.tableData)
    rawEchoTable = importConfig.tableData;
elseif strlength(importConfig.filePath) > 0
    rawEchoTable = readArduinoEchoImportFile(importConfig);
elseif importConfig.autoDiscoverLatest
    [echoImportTable, isCanonicalTable] = discoverArduinoEchoImportTable(config, activeSurfaceNames);
    if ~isempty(echoImportTable)
        if isCanonicalTable
            return;
        end

        rawEchoTable = echoImportTable;
    else
        return;
    end
else
    return;
end

if isempty(rawEchoTable)
    return;
end

echoImportTable = canonicalizeArduinoEchoImportTable(rawEchoTable, importConfig, activeSurfaceNames);
end

function [echoImportTable, isCanonicalTable] = discoverArduinoEchoImportTable(config, activeSurfaceNames)
echoImportTable = table();
isCanonicalTable = false;
loggerOutputFolder = config.arduinoEchoImport.loggerOutputFolder;

if strlength(loggerOutputFolder) == 0 || ~isfolder(loggerOutputFolder)
    return;
end

latestImportCsvPath = findLatestMatchingFile(loggerOutputFolder, "arduino_echo_import*.csv");
if strlength(latestImportCsvPath) > 0
    echoImportTable = readtable(latestImportCsvPath);
    return;
end

hostDispatchCsvPath = fullfile(loggerOutputFolder, "host_dispatch_log.csv");
syncRoundTripCsvPath = fullfile(loggerOutputFolder, "host_sync_roundtrip.csv");
boardCommandLogCsvPath = fullfile(loggerOutputFolder, "board_command_log.csv");

if isfile(hostDispatchCsvPath) && isfile(syncRoundTripCsvPath) && isfile(boardCommandLogCsvPath)
    echoImportTable = buildArduinoEchoImportTableFromLoggerRawFiles( ...
        hostDispatchCsvPath, ...
        syncRoundTripCsvPath, ...
        boardCommandLogCsvPath, ...
        activeSurfaceNames);
    isCanonicalTable = true;
end
end

function latestFilePath = findLatestMatchingFile(folderPath, filePattern)
directoryListing = dir(fullfile(folderPath, char(filePattern)));

if isempty(directoryListing)
    latestFilePath = "";
    return;
end

[~, latestIndex] = max([directoryListing.datenum]);
latestFilePath = string(fullfile(directoryListing(latestIndex).folder, directoryListing(latestIndex).name));
end

function echoImportTable = buildArduinoEchoImportTableFromLoggerRawFiles( ...
    hostDispatchCsvPath, ...
    syncRoundTripCsvPath, ...
    boardCommandLogCsvPath, ...
    activeSurfaceNames)
hostDispatchLog = readtable(hostDispatchCsvPath);
syncRoundTripLog = readtable(syncRoundTripCsvPath);
boardCommandLog = readCommentCsv(boardCommandLogCsvPath);

validateLoggerHostDispatchLog(hostDispatchLog);
validateLoggerSyncRoundTripLog(syncRoundTripLog);
validateLoggerBoardCommandLog(boardCommandLog);

[clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog);

hostDispatchLog.surface_name = reshape(string(hostDispatchLog.surface_name), [], 1);
hostDispatchLog.command_sequence = reshape(double(hostDispatchLog.command_sequence), [], 1);
boardCommandLog.surface_name = reshape(string(boardCommandLog.surface_name), [], 1);
boardCommandLog.command_sequence = reshape(double(boardCommandLog.command_sequence), [], 1);

if ~ismember("surface_name", string(boardCommandLog.Properties.VariableNames)) && numel(activeSurfaceNames) == 1
    boardCommandLog.surface_name = repmat(activeSurfaceNames(1), height(boardCommandLog), 1);
end

joinedTable = innerjoin( ...
    hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'}), ...
    boardCommandLog, ...
    'Keys', {'surface_name', 'command_sequence'});

applyHostUs = clockSlope .* double(joinedTable.apply_us) + clockIntercept;
latencyUs = applyHostUs - double(joinedTable.command_dispatch_us);
minimumLatencyUs = min(latencyUs);
if minimumLatencyUs < 0
    % Re-anchor the fitted clock so matched commands cannot precede dispatch.
    applyHostUs = applyHostUs - minimumLatencyUs;
    latencyUs = latencyUs - minimumLatencyUs;
end
echoImportTable = table( ...
    joinedTable.surface_name, ...
    joinedTable.command_sequence, ...
    nan(height(joinedTable), 1), ...
    latencyUs ./ 1e6, ...
    joinedTable.applied_position, ...
    nan(height(joinedTable), 1), ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'arduino_echo_time_s', ...
        'computer_to_arduino_latency_s', ...
        'applied_position', ...
        'applied_equivalent_deg'});
end

function tableData = readCommentCsv(filePath)
options = detectImportOptions(filePath, "FileType", "text", "CommentStyle", "#");
options.CommentStyle = "#";
tableData = readtable(filePath, options);
end

function [clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog)
% Calibrate against host send time and board receive time only.
% This avoids reply-side skew from delayed host reads.
hostTxUs = double(syncRoundTripLog.host_tx_us);
boardRxUs = double(syncRoundTripLog.board_rx_us);

if numel(hostTxUs) >= 2
    fitCoefficients = polyfit(boardRxUs, hostTxUs, 1);
    clockSlope = fitCoefficients(1);
    clockIntercept = fitCoefficients(2);
else
    clockSlope = 1.0;
    clockIntercept = hostTxUs(1) - boardRxUs(1);
end

syncForwardLatencyUs = clockSlope .* boardRxUs + clockIntercept - hostTxUs;
minimumForwardLatencyUs = min(syncForwardLatencyUs);
if minimumForwardLatencyUs < 0
    clockIntercept = clockIntercept - minimumForwardLatencyUs;
end
end

function validateLoggerHostDispatchLog(hostDispatchLog)
assertHasLoggerColumns(hostDispatchLog, ["surface_name", "command_sequence", "command_dispatch_us"], "host dispatch log");
end

function validateLoggerSyncRoundTripLog(syncRoundTripLog)
assertHasLoggerColumns( ...
    syncRoundTripLog, ...
    ["sync_id", "host_tx_us", "host_rx_us", "board_rx_us", "board_tx_us"], ...
    "sync round-trip log");
end

function validateLoggerBoardCommandLog(boardCommandLog)
assertHasLoggerColumns(boardCommandLog, ["surface_name", "command_sequence", "rx_us", "apply_us", "applied_position"], "board command log");
end

function assertHasLoggerColumns(tableData, requiredColumns, tableLabel)
variableNames = string(tableData.Properties.VariableNames);
missingColumns = requiredColumns(~ismember(requiredColumns, variableNames));
if ~isempty(missingColumns)
    error("Servo_Test:MissingArduinoLoggerColumns", ...
        "The %s is missing: %s", ...
        tableLabel, ...
        char(join(missingColumns, ", ")));
end
end

function rawEchoTable = readArduinoEchoImportFile(importConfig)
filePath = importConfig.filePath;
if ~isfile(filePath)
    error("Servo_Test:MissingArduinoEchoImport", ...
        "arduinoEchoImport.filePath does not exist: %s", ...
        char(filePath));
end

[~, ~, fileExtension] = fileparts(char(filePath));
fileExtension = string(lower(fileExtension));
switch fileExtension
    case {".xlsx", ".xls", ".xlsm"}
        if strlength(importConfig.sheetName) == 0
            rawEchoTable = readtable(filePath);
        else
            rawEchoTable = readtable(filePath, "Sheet", char(importConfig.sheetName));
        end
    case ".csv"
        rawEchoTable = readtable(filePath);
    case {".txt", ".tsv"}
        rawEchoTable = readtable(filePath, "FileType", "text", "Delimiter", "\t");
    otherwise
        error("Servo_Test:UnsupportedArduinoEchoImport", ...
            "Unsupported arduinoEchoImport file extension '%s'.", ...
            fileExtension);
end
end

function echoImportTable = canonicalizeArduinoEchoImportTable(rawEchoTable, importConfig, activeSurfaceNames)
rowCount = height(rawEchoTable);
echoImportTable = table();

surfaceColumnName = resolveTableVariableName(rawEchoTable, importConfig.surfaceColumn, false);
if strlength(surfaceColumnName) == 0
    if numel(activeSurfaceNames) ~= 1
        error("Servo_Test:AmbiguousArduinoEchoSurface", ...
            "arduinoEchoImport.surfaceColumn is required when more than one surface is active.");
    end

    echoImportTable.surface_name = repmat(activeSurfaceNames(1), rowCount, 1);
else
    echoImportTable.surface_name = reshape(string(rawEchoTable.(char(surfaceColumnName))), [], 1);
end

sequenceColumnName = resolveTableVariableName(rawEchoTable, importConfig.sequenceColumn, false);
if strlength(sequenceColumnName) == 0
    echoImportTable.command_sequence = buildDefaultArduinoEchoSequences(echoImportTable.surface_name);
else
    echoImportTable.command_sequence = reshape(double(rawEchoTable.(char(sequenceColumnName))), [], 1);
end

latencyColumnName = resolveTableVariableName(rawEchoTable, importConfig.latencyColumn, false);
echoTimeColumnName = resolveTableVariableName(rawEchoTable, importConfig.echoTimeColumn, false);
if strlength(latencyColumnName) == 0 && strlength(echoTimeColumnName) == 0
    error("Servo_Test:MissingArduinoEchoTiming", ...
        "arduinoEchoImport must provide either '%s' or '%s'.", ...
        char(importConfig.latencyColumn), ...
        char(importConfig.echoTimeColumn));
end

echoImportTable.computer_to_arduino_latency_s = nan(rowCount, 1);
if strlength(latencyColumnName) > 0
    echoImportTable.computer_to_arduino_latency_s = reshape(double(rawEchoTable.(char(latencyColumnName))), [], 1);
end

echoImportTable.arduino_echo_time_s = nan(rowCount, 1);
if strlength(echoTimeColumnName) > 0
    echoImportTable.arduino_echo_time_s = ...
        reshape(double(rawEchoTable.(char(echoTimeColumnName))), [], 1) + importConfig.matlabTimeOffsetSeconds;
end

appliedPositionColumnName = resolveTableVariableName(rawEchoTable, importConfig.appliedPositionColumn, false);
echoImportTable.applied_position = nan(rowCount, 1);
if strlength(appliedPositionColumnName) > 0
    echoImportTable.applied_position = reshape(double(rawEchoTable.(char(appliedPositionColumnName))), [], 1);
end

appliedEquivalentDegreesColumnName = resolveTableVariableName(rawEchoTable, importConfig.appliedEquivalentDegreesColumn, false);
echoImportTable.applied_equivalent_deg = nan(rowCount, 1);
if strlength(appliedEquivalentDegreesColumnName) > 0
    echoImportTable.applied_equivalent_deg = reshape(double(rawEchoTable.(char(appliedEquivalentDegreesColumnName))), [], 1);
end
end

function commandSequences = buildDefaultArduinoEchoSequences(surfaceNames)
surfaceNames = reshape(string(surfaceNames), [], 1);
commandSequences = nan(numel(surfaceNames), 1);
uniqueSurfaceNames = unique(surfaceNames, "stable");

for surfaceIndex = 1:numel(uniqueSurfaceNames)
    surfaceMask = surfaceNames == uniqueSurfaceNames(surfaceIndex);
    commandSequences(surfaceMask) = (1:nnz(surfaceMask)).';
end
end

function commandLookup = buildArduinoCommandLookupTable(storage, config, sampleIndices)
surfaceCount = numel(config.surfaceNames);
rowCount = nnz(isfinite(storage.commandSequenceNumbers(sampleIndices, :)));

commandLookup = table( ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    strings(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    nan(rowCount, 1), ...
    'VariableNames', { ...
        'sample_index', ...
        'surface_index', ...
        'surface_name', ...
        'command_sequence', ...
        'command_dispatch_s', ...
        'command_position'});

rowIndex = 0;
for sampleOffset = 1:numel(sampleIndices)
    sampleIndex = sampleIndices(sampleOffset);
    for surfaceIndex = 1:surfaceCount
        commandSequence = storage.commandSequenceNumbers(sampleIndex, surfaceIndex);
        if ~isfinite(commandSequence)
            continue;
        end

        rowIndex = rowIndex + 1;
        commandLookup.sample_index(rowIndex) = sampleIndex;
        commandLookup.surface_index(rowIndex) = surfaceIndex;
        commandLookup.surface_name(rowIndex) = config.surfaceNames(surfaceIndex);
        commandLookup.command_sequence(rowIndex) = commandSequence;
        commandLookup.command_dispatch_s(rowIndex) = storage.commandDispatchSeconds(sampleIndex, surfaceIndex);
        commandLookup.command_position(rowIndex) = storage.commandedServoPositions(sampleIndex, surfaceIndex);
    end
end
end

function echoAssignments = assignArduinoEchoToCommands(commandLookup, echoImportTable)
echoAssignments = commandLookup(:, {'sample_index', 'surface_index'});
rowCount = height(commandLookup);
echoAssignments.arduino_echo_time_s = nan(rowCount, 1);
echoAssignments.applied_position = nan(rowCount, 1);
echoAssignments.applied_equivalent_deg = nan(rowCount, 1);

for commandIndex = 1:rowCount
    matchMask = ...
        echoImportTable.surface_name == commandLookup.surface_name(commandIndex) & ...
        echoImportTable.command_sequence == commandLookup.command_sequence(commandIndex);

    if ~any(matchMask)
        continue;
    end

    matchedRow = find(matchMask, 1, "first");
    echoAssignments.arduino_echo_time_s(commandIndex) = resolveArduinoEchoTime( ...
        echoImportTable, ...
        matchedRow, ...
        commandLookup.command_dispatch_s(commandIndex));
    echoAssignments.applied_position(commandIndex) = echoImportTable.applied_position(matchedRow);
    echoAssignments.applied_equivalent_deg(commandIndex) = echoImportTable.applied_equivalent_deg(matchedRow);
end
end

function arduinoEchoTimeSeconds = resolveArduinoEchoTime(echoImportTable, matchedRow, commandDispatchSeconds)
arduinoEchoTimeSeconds = echoImportTable.arduino_echo_time_s(matchedRow);
if isfinite(arduinoEchoTimeSeconds)
    return;
end

latencySeconds = echoImportTable.computer_to_arduino_latency_s(matchedRow);
if isfinite(latencySeconds)
    arduinoEchoTimeSeconds = commandDispatchSeconds + latencySeconds;
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
    error("Servo_Test:MissingArduinoEchoColumn", ...
        "The Arduino echo import is missing column '%s'.", ...
        char(requestedName));
end

variableName = "";
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

    translationOutput = client.GetSegmentGlobalTranslation(subjectName, segmentName);
    quaternionOutput = client.GetSegmentGlobalRotationQuaternion(subjectName, segmentName);
    hasValidTranslation = isSuccessOutput(translationOutput) && ~adaptLogical(translationOutput.Occluded);
    hasValidQuaternion = isSuccessOutput(quaternionOutput) && ~adaptLogical(quaternionOutput.Occluded);

    if hasValidTranslation
        snapshot.positionOccluded(surfaceIndex) = false;
        snapshot.positionMillimeters(surfaceIndex, :) = reshape(double(translationOutput.Translation), 1, 3);
    end

    if ~hasValidQuaternion
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
    "positionMillimeters", nan(surfaceCount, 3), ...
    "positionOccluded", true(1, surfaceCount), ...
    "isOccluded", true(1, surfaceCount), ...
    "quaternionXYZW", nan(surfaceCount, 4));
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

arduinoRowTimesSeconds = chooseFiniteRowTimes( ...
    storage.arduinoReadStopSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices));
arduinoVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_position"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_equivalent_deg")];
arduinoLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.appliedServoPositions(sampleIndices, :), ...
    storage.appliedEquivalentDegrees(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(arduinoVariableNames));

arduinoLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
arduinoLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
arduinoLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);
arduinoLog.arduino_echo_available = any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2);

computerToArduinoLatencySeconds = ...
    storage.arduinoEchoSeconds(sampleIndices, :) - storage.commandDispatchSeconds(sampleIndices, :);
computerToArduinoVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_echo_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_latency_s")];
computerToArduinoLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.commandDispatchSeconds(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    computerToArduinoLatencySeconds], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(computerToArduinoVariableNames));

computerToArduinoLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
computerToArduinoLatencyLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
computerToArduinoLatencyLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);
computerToArduinoLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
computerToArduinoLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);
computerToArduinoLatencyLog.arduino_echo_available = any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2);

viconPositionDataMatrix = [ ...
    reshape(storage.viconPositionMillimeters(sampleIndices, :, 1), [], numel(surfaceNames)), ...
    reshape(storage.viconPositionMillimeters(sampleIndices, :, 2), [], numel(surfaceNames)), ...
    reshape(storage.viconPositionMillimeters(sampleIndices, :, 3), [], numel(surfaceNames)), ...
    double(storage.viconPositionOccluded(sampleIndices, :))];
viconPositionVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "x_mm"), ...
    buildSurfaceVariableNames(surfaceNames, "y_mm"), ...
    buildSurfaceVariableNames(surfaceNames, "z_mm"), ...
    buildSurfaceVariableNames(surfaceNames, "position_occluded")];
viconPositionLog = array2timetable( ...
    viconPositionDataMatrix, ...
    'RowTimes', seconds(storage.viconSampleTimeSeconds(sampleIndices)), ...
    'VariableNames', cellstr(viconPositionVariableNames));

viconPositionLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
viconPositionLog.vicon_sample_time_s = storage.viconSampleTimeSeconds(sampleIndices);
viconPositionLog.vicon_frame_number = storage.viconFrameNumbers(sampleIndices);
viconPositionLog.vicon_reported_latency_s = storage.viconLatencySeconds(sampleIndices);

viconDataMatrix = [ ...
    storage.measuredDeflectionsDegrees(sampleIndices, :), ...
    double(storage.viconOccluded(sampleIndices, :)), ...
    reshape(storage.viconQuaternionXYZW(sampleIndices, :, 1), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(sampleIndices, :, 2), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(sampleIndices, :, 3), [], numel(surfaceNames)), ...
    reshape(storage.viconQuaternionXYZW(sampleIndices, :, 4), [], numel(surfaceNames))];

viconTrackingVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "measured_deg"), ...
    buildSurfaceVariableNames(surfaceNames, "tracking_occluded"), ...
    buildSurfaceVariableNames(surfaceNames, "qx"), ...
    buildSurfaceVariableNames(surfaceNames, "qy"), ...
    buildSurfaceVariableNames(surfaceNames, "qz"), ...
    buildSurfaceVariableNames(surfaceNames, "qw")];

viconTrackingOutputLog = array2timetable( ...
    viconDataMatrix, ...
    'RowTimes', seconds(storage.viconSampleTimeSeconds(sampleIndices)), ...
    'VariableNames', cellstr(viconTrackingVariableNames));

viconTrackingOutputLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
viconTrackingOutputLog.vicon_sample_time_s = storage.viconSampleTimeSeconds(sampleIndices);
viconTrackingOutputLog.vicon_frame_number = storage.viconFrameNumbers(sampleIndices);
viconTrackingOutputLog.vicon_reported_latency_s = storage.viconLatencySeconds(sampleIndices);

arduinoToViconLatencySeconds = ...
    storage.viconSampleTimeSeconds(sampleIndices) - storage.arduinoEchoSeconds(sampleIndices, :);
arduinoToViconVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_echo_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_to_vicon_latency_s")];
arduinoToViconLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    arduinoToViconLatencySeconds], ...
    'RowTimes', seconds(storage.viconSampleTimeSeconds(sampleIndices)), ...
    'VariableNames', cellstr(arduinoToViconVariableNames));

arduinoToViconLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
arduinoToViconLatencyLog.vicon_sample_time_s = storage.viconSampleTimeSeconds(sampleIndices);
arduinoToViconLatencyLog.vicon_frame_number = storage.viconFrameNumbers(sampleIndices);
arduinoToViconLatencyLog.vicon_reported_latency_s = storage.viconLatencySeconds(sampleIndices);
arduinoToViconLatencyLog.arduino_echo_available = any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2);

sampleSummary = table( ...
    storage.scheduledTimeSeconds(sampleIndices), ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.commandWriteStartSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices), ...
    storage.arduinoReadStartSeconds(sampleIndices), ...
    storage.arduinoReadStopSeconds(sampleIndices), ...
    any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2), ...
    storage.viconSampleTimeSeconds(sampleIndices), ...
    storage.viconFrameNumbers(sampleIndices), ...
    storage.viconLatencySeconds(sampleIndices), ...
    'VariableNames', { ...
        'scheduled_time_s', ...
        'base_command_deg', ...
        'command_write_start_s', ...
        'command_write_stop_s', ...
        'arduino_read_start_s', ...
        'arduino_read_stop_s', ...
        'arduino_echo_available', ...
        'vicon_sample_time_s', ...
        'vicon_frame_number', ...
        'vicon_latency_s'});

latencySummary = buildComputerToArduinoLatencySummary(storage, config);

logs = struct( ...
    "inputSignal", commandLog, ...
    "command", commandLog, ...
    "arduinoEcho", arduinoLog, ...
    "arduino", arduinoLog, ...
    "computerToArduinoLatency", computerToArduinoLatencyLog, ...
    "latencySummary", latencySummary, ...
    "viconPosition", viconPositionLog, ...
    "viconTrackingOutput", viconTrackingOutputLog, ...
    "vicon", viconTrackingOutputLog, ...
    "arduinoToViconLatency", arduinoToViconLatencyLog, ...
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
    'VariableNames', { ...
        'SurfaceName', ...
        'ArduinoPin', ...
        'IsActive', ...
        'ValidViconSampleCount', ...
        'AppliedCommandSampleCount', ...
        'SaturationSampleCount'});

latencySummary = buildComputerToArduinoLatencySummary(storage, config);
surfaceSummary = [ ...
    surfaceSummary ...
    latencySummary(:, { ...
        'DispatchedCommandCount', ...
        'MatchedCommandCount', ...
        'UnmatchedCommandCount', ...
        'UnmatchedCommandFraction', ...
        'ComputerToArduinoLatencyMean_s', ...
        'ComputerToArduinoLatencyStd_s', ...
        'ComputerToArduinoLatencyMedian_s', ...
        'ComputerToArduinoLatencyP95_s', ...
        'ComputerToArduinoLatencyMax_s'})];
end

function latencySummary = buildComputerToArduinoLatencySummary(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceCount = numel(config.surfaceNames);

dispatchedCommandCount = zeros(surfaceCount, 1);
matchedCommandCount = zeros(surfaceCount, 1);
unmatchedCommandCount = zeros(surfaceCount, 1);
unmatchedCommandFraction = nan(surfaceCount, 1);
latencyMeanSeconds = nan(surfaceCount, 1);
latencyStdSeconds = nan(surfaceCount, 1);
latencyMedianSeconds = nan(surfaceCount, 1);
latencyP95Seconds = nan(surfaceCount, 1);
latencyMaxSeconds = nan(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    dispatchedMask = isfinite(storage.commandSequenceNumbers(sampleIndices, surfaceIndex));
    latencyValuesSeconds = ...
        storage.arduinoEchoSeconds(sampleIndices, surfaceIndex) - ...
        storage.commandDispatchSeconds(sampleIndices, surfaceIndex);
    latencyValuesSeconds = latencyValuesSeconds(isfinite(latencyValuesSeconds));

    dispatchedCommandCount(surfaceIndex) = nnz(dispatchedMask);
    matchedCommandCount(surfaceIndex) = numel(latencyValuesSeconds);
    unmatchedCommandCount(surfaceIndex) = ...
        max(0, dispatchedCommandCount(surfaceIndex) - matchedCommandCount(surfaceIndex));

    if dispatchedCommandCount(surfaceIndex) > 0
        unmatchedCommandFraction(surfaceIndex) = ...
            unmatchedCommandCount(surfaceIndex) ./ dispatchedCommandCount(surfaceIndex);
    end

    if isempty(latencyValuesSeconds)
        continue;
    end

    latencyMeanSeconds(surfaceIndex) = mean(latencyValuesSeconds);
    if numel(latencyValuesSeconds) >= 2
        latencyStdSeconds(surfaceIndex) = std(latencyValuesSeconds, 0);
    end
    latencyMedianSeconds(surfaceIndex) = computeSamplePercentile(latencyValuesSeconds, 50);
    latencyP95Seconds(surfaceIndex) = computeSamplePercentile(latencyValuesSeconds, 95);
    latencyMaxSeconds(surfaceIndex) = max(latencyValuesSeconds);
end

latencySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    dispatchedCommandCount, ...
    matchedCommandCount, ...
    unmatchedCommandCount, ...
    unmatchedCommandFraction, ...
    latencyMeanSeconds, ...
    latencyStdSeconds, ...
    latencyMedianSeconds, ...
    latencyP95Seconds, ...
    latencyMaxSeconds, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'DispatchedCommandCount', ...
        'MatchedCommandCount', ...
        'UnmatchedCommandCount', ...
        'UnmatchedCommandFraction', ...
        'ComputerToArduinoLatencyMean_s', ...
        'ComputerToArduinoLatencyStd_s', ...
        'ComputerToArduinoLatencyMedian_s', ...
        'ComputerToArduinoLatencyP95_s', ...
        'ComputerToArduinoLatencyMax_s'});
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

function latestState = buildLatestState(storage, config, sampleIndex)
surfaceCount = numel(config.surfaceNames);
surfaceStates = repmat(struct( ...
    "name", "", ...
    "desired_deg", NaN, ...
    "command_position", NaN, ...
    "applied_position", NaN, ...
    "applied_equivalent_deg", NaN, ...
    "vicon_position_mm", nan(1, 3), ...
    "position_occluded", true, ...
    "measured_deg", NaN, ...
    "is_occluded", true), surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceStates(surfaceIndex).name = config.surfaceNames(surfaceIndex);
    surfaceStates(surfaceIndex).desired_deg = storage.desiredDeflectionsDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).command_position = storage.commandedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_position = storage.appliedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).applied_equivalent_deg = storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).vicon_position_mm = reshape(storage.viconPositionMillimeters(sampleIndex, surfaceIndex, :), 1, 3);
    surfaceStates(surfaceIndex).position_occluded = storage.viconPositionOccluded(sampleIndex, surfaceIndex);
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
    'VariableNames', { ...
        'SurfaceName', ...
        'ArduinoPin', ...
        'ViconSubjectName', ...
        'ViconSegmentName', ...
        'IsActive', ...
        'ServoNeutralPosition', ...
        'ServoUnitsPerDegree', ...
        'ServoMinimumPosition', ...
        'ServoMaximumPosition', ...
        'HingeAxisX', ...
        'HingeAxisY', ...
        'HingeAxisZ', ...
        'MeasurementSign', ...
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

neutralReferenceTable = buildNeutralReferenceTable(runData.neutralReference);

writetable(buildCriticalSettingsTable(runData), workbookPath, 'Sheet', 'CriticalSettings');
writetable(timetableToExportTable(runData.logs.inputSignal), workbookPath, 'Sheet', 'InputSignal');
writetable(timetableToExportTable(runData.logs.arduinoEcho), workbookPath, 'Sheet', 'ArduinoEcho');
writetable( ...
    timetableToExportTable(runData.logs.computerToArduinoLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ComputerToArduinoLatency');
writetable(runData.logs.latencySummary, workbookPath, 'Sheet', 'LatencySummary');
writetable(timetableToExportTable(runData.logs.viconPosition), workbookPath, 'Sheet', 'ViconPosition');
writetable( ...
    timetableToExportTable(runData.logs.viconTrackingOutput), ...
    workbookPath, ...
    'Sheet', ...
    'ViconTrackingOutput');
writetable( ...
    timetableToExportTable(runData.logs.arduinoToViconLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ArduinoToViconLatency');
writetable(runData.surfaceSetup, workbookPath, 'Sheet', 'SurfaceSetup');
writetable(neutralReferenceTable, workbookPath, 'Sheet', 'NeutralReference');
writetable(runData.logs.sampleSummary, workbookPath, 'Sheet', 'SampleSummary');
writetable(runData.surfaceSummary, workbookPath, 'Sheet', 'SurfaceSummary');
end

function criticalSettingsTable = buildCriticalSettingsTable(runData)
config = runData.config;
profile = config.commandProfile;
arduinoInfo = runData.connectionInfo.arduino;
viconInfo = runData.connectionInfo.vicon;

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
    "Arduino", "LoggerBackend", formatSettingValue(getFieldOrDefault(arduinoInfo, "loggerBackend", "")); ...
    "ArduinoEchoImport", "FilePath", formatSettingValue(config.arduinoEchoImport.filePath); ...
    "ArduinoEchoImport", "SheetName", formatSettingValue(config.arduinoEchoImport.sheetName); ...
    "ArduinoEchoImport", "SurfaceColumn", formatSettingValue(config.arduinoEchoImport.surfaceColumn); ...
    "ArduinoEchoImport", "SequenceColumn", formatSettingValue(config.arduinoEchoImport.sequenceColumn); ...
    "ArduinoEchoImport", "LatencyColumn", formatSettingValue(config.arduinoEchoImport.latencyColumn); ...
    "ArduinoEchoImport", "EchoTimeColumn", formatSettingValue(config.arduinoEchoImport.echoTimeColumn); ...
    "ArduinoEchoImport", "MatlabTimeOffsetSeconds", formatSettingValue(config.arduinoEchoImport.matlabTimeOffsetSeconds); ...
    "ArduinoEchoImport", "AppliedPositionColumn", formatSettingValue(config.arduinoEchoImport.appliedPositionColumn); ...
    "ArduinoEchoImport", "AppliedEquivalentDegreesColumn", formatSettingValue(config.arduinoEchoImport.appliedEquivalentDegreesColumn); ...
    "ArduinoEchoImport", "AutoDiscoverLatest", formatSettingValue(config.arduinoEchoImport.autoDiscoverLatest); ...
    "ArduinoEchoImport", "LoggerOutputFolder", formatSettingValue(config.arduinoEchoImport.loggerOutputFolder); ...
    "ArduinoTransport", "RequestedMode", formatSettingValue(config.arduinoTransport.mode); ...
    "ArduinoTransport", "ResolvedMode", formatSettingValue(config.arduinoTransport.resolvedMode); ...
    "ArduinoTransport", "LoggerPort", formatSettingValue(config.arduinoTransport.loggerPort); ...
    "ArduinoTransport", "LoggerProbeTimeoutSeconds", formatSettingValue(config.arduinoTransport.loggerProbeTimeoutSeconds); ...
    "ArduinoTransport", "LoggerTimeoutSeconds", formatSettingValue(config.arduinoTransport.loggerTimeoutSeconds); ...
    "ArduinoTransport", "SyncCountBeforeRun", formatSettingValue(config.arduinoTransport.syncCountBeforeRun); ...
    "ArduinoTransport", "SyncCountAfterRun", formatSettingValue(config.arduinoTransport.syncCountAfterRun); ...
    "ArduinoTransport", "SyncPauseSeconds", formatSettingValue(config.arduinoTransport.syncPauseSeconds); ...
    "ArduinoTransport", "PostRunSettleSeconds", formatSettingValue(config.arduinoTransport.postRunSettleSeconds); ...
    "ArduinoTransport", "ClearLogsBeforeRun", formatSettingValue(config.arduinoTransport.clearLogsBeforeRun); ...
    "ArduinoTransport", "LoggerOutputFolder", formatSettingValue(config.arduinoTransport.loggerOutputFolder); ...
    "ArduinoTransport", "CaptureSucceeded", formatSettingValue(config.arduinoTransport.captureSucceeded); ...
    "ArduinoTransport", "CaptureMessage", formatSettingValue(config.arduinoTransport.captureMessage); ...
    "ArduinoTransport", "CaptureRowCount", formatSettingValue(config.arduinoTransport.captureRowCount); ...
    "Vicon", "HostName", formatSettingValue(config.viconHostName); ...
    "Vicon", "StreamMode", formatSettingValue(config.viconStreamMode); ...
    "Vicon", "AxisMapping", formatSettingValue(config.viconAxisMapping); ...
    "Vicon", "IsConnected", formatSettingValue(viconInfo.isConnected); ...
    "Vicon", "ConnectionMessage", formatSettingValue(viconInfo.connectionMessage); ...
    "Vicon", "ConnectElapsedSeconds", formatSettingValue(viconInfo.connectElapsedSeconds); ...
    "Vicon", "RequireAllSubjects", formatSettingValue(config.requireAllViconSubjects); ...
    "NeutralReference", "IsSuccessful", formatSettingValue(runData.neutralInfo.isSuccessful); ...
    "NeutralReference", "Message", formatSettingValue(runData.neutralInfo.message); ...
    "NeutralReference", "CalibrationSeconds", formatSettingValue(config.neutralCalibrationSeconds); ...
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
    "Surfaces", "ViconSubjectNames", formatSettingValue(config.viconSubjectNames); ...
    "Surfaces", "ViconSegmentNames", formatSettingValue(runData.surfaceSetup.ViconSegmentName); ...
    "Surfaces", "ServoNeutralPositions", formatSettingValue(config.servoNeutralPositions); ...
    "Surfaces", "ServoUnitsPerDegree", formatSettingValue(config.servoUnitsPerDegree); ...
    "Surfaces", "ServoMinimumPositions", formatSettingValue(config.servoMinimumPositions); ...
    "Surfaces", "ServoMaximumPositions", formatSettingValue(config.servoMaximumPositions); ...
    "Surfaces", "ViconHingeAxesNeutralFrame", formatSettingValue(config.viconHingeAxesNeutralFrame); ...
    "Surfaces", "ViconMeasurementSigns", formatSettingValue(config.viconMeasurementSigns); ...
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
    'VariableNames', { ...
        'SurfaceName', ...
        'SubjectName', ...
        'SegmentName', ...
        'SampleCount', ...
        'qx', ...
        'qy', ...
        'qz', ...
        'qw'});
end

function exportTable = timetableToExportTable(timetableData)
exportTable = timetable2table(timetableData);
exportTable.Properties.VariableNames{1} = 'time_s';
exportTable.time_s = seconds(exportTable.time_s);
end

function cleanupResources(commandInterface, client, config)
if ~isempty(fieldnames(commandInterface)) && config.returnToNeutralOnExit
    try
        moveServosToNeutral(commandInterface, config.servoNeutralPositions);
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

if isempty(fieldnames(commandInterface))
    return;
end

switch commandInterface.transportMode
    case "nano_logger_udp"
        try
            closeNanoLoggerConnection(commandInterface.connection);
        catch
        end
    otherwise
        try
            arduinoConnection = commandInterface.connection;
            clear arduinoConnection %#ok<NASGU>
        catch
        end
end
end

function closeNanoLoggerConnection(loggerConnection)
switch loggerConnection.backend
    case "java_datagram_socket"
        try
            loggerConnection.socket.close();
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

function value = getOptionalTextScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if isempty(value)
    value = defaultValue;
end

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
validateattributes(value, {"logical", "numeric"}, {"scalar"}, char(mfilename), char(fieldName));
value = logical(value);
end

function value = getPositiveIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "positive"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getNonnegativeIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "nonnegative"}, char(mfilename), char(fieldName));
value = double(value);
end

function value = getPositiveScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "positive"}, char(mfilename), char(fieldName));
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

function value = getNumericMatrixField(config, fieldName, defaultValue, expectedSize)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "size", expectedSize}, char(mfilename), char(fieldName));
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
    error("Servo_Test:InvalidArrayLength", ...
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
