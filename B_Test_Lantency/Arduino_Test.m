function runData = Arduino_Test(config)
% ARDUINO_TEST Execute an Arduino-only servo command test over WiFi.
%   runData = Arduino_Test(config) connects to the Arduino Nano 33 IoT,
%   commands one or all four servos with a configurable profile, logs the
%   desired commands, and optionally imports an Arduino-side echo log
%   during post-processing to evaluate command latency without disturbing
%   the inner loop.
%
%% =============================================================================
% SECTION MAP
% =============================================================================
% 1) Public Entry Point
% 2) Configuration and Connection Helpers
% 3) Command Schedule and Run Execution
% 4) Nano Logger Telemetry Handling
% 5) Echo Import and Latency Summaries
% 6) Export, Profile Builders, and Utility Helpers
% =============================================================================
%
%% =============================================================================
% 1) Public Entry Point
% =============================================================================
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);
runData.runInfo.startTime = datetime("now");

assignin("base", "ArduinoTestLatestState", struct([]));
assignin("base", "ArduinoTestRunData", runData);

commandInterface = struct();
cleanupHandle = onCleanup(@() cleanupResources(commandInterface, config));

try
    [commandInterface, arduinoInfo] = connectToArduino(config);
    config.arduinoTransport.resolvedMode = arduinoInfo.transportResolvedMode;
    if arduinoInfo.isConnected && config.arduinoTransport.resolvedMode ~= "nano_logger_udp"
        config.arduinoTransport.captureMessage = ...
            "No integrated Nano logger capture; transport resolved to " + config.arduinoTransport.resolvedMode + ".";
    end
    runData.config = config;
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

    moveServosToNeutral(commandInterface, config.servoNeutralPositions);
    pause(config.neutralSettleSeconds);

    [storage, runInfo, config] = executeArduinoTest(commandInterface, config);
    runData.config = config;
    runData.runInfo = runInfo;
    runData.logs = buildLogTimetables(storage, config);
    runData.surfaceSummary = buildSurfaceSummary(storage, config);
    runData.artifacts = exportRunData(runData);

    assignin("base", "ArduinoTestRunData", runData);
    assignin("base", "ArduinoTestLatestState", buildLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupResources(commandInterface, config);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    assignin("base", "ArduinoTestRunData", runData);
    rethrow(executionException);
end
end

%% =============================================================================
% 2) Configuration and Connection Helpers
% =============================================================================
function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));
defaultSurfaceNames = ["Aileron_L"; "Aileron_R"; "Rudder"; "Elevator"];
defaultSurfacePins = ["D9"; "D10"; "D11"; "D12"];
arduinoTransportConfig = getFieldOrDefault(config, "arduinoTransport", struct());
requestedNanoLoggerOperatingMode = canonicalizeNanoLoggerOperatingMode( ...
    getTextScalarField(arduinoTransportConfig, "operatingMode", "controller")); % or instrumentation

config.arduinoIPAddress = getTextScalarField(config, "arduinoIPAddress", "192.168.0.33");
config.arduinoBoard = getTextScalarField(config, "arduinoBoard", "Nano33IoT");
config.arduinoPort = getOptionalScalarField(config, "arduinoPort", NaN);

% Surface order is the contract with the Nano logger packet layout and the
% plotting scripts: D9-D12 correspond to Aileron_L, Aileron_R, Rudder, Elevator.
config.surfaceNames = getStringArrayField(config, "surfaceNames", defaultSurfaceNames);
config.surfacePins = getStringArrayField(config, "surfacePins", defaultSurfacePins);
% Positions are normalized 0..1 on the wire; degrees enter only through
% servoUnitsPerDegree and the per-surface neutral offsets.
config.servoNeutralPositions = getNumericColumnField(config, "servoNeutralPositions", 0.5 .* ones(numel(config.surfaceNames), 1));
config.servoUnitsPerDegree = getNumericColumnField(config, "servoUnitsPerDegree", (1 / 180) .* ones(numel(config.surfaceNames), 1));
config.servoMinimumPositions = getNumericColumnField(config, "servoMinimumPositions", zeros(numel(config.surfaceNames), 1));
config.servoMaximumPositions = getNumericColumnField(config, "servoMaximumPositions", ones(numel(config.surfaceNames), 1));
config.servoMinPulseDurationSeconds = getNumericColumnField(config, "servoMinPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.servoMaxPulseDurationSeconds = getNumericColumnField(config, "servoMaxPulseDurationSeconds", nan(numel(config.surfaceNames), 1));
config.commandDeflectionScales = getNumericColumnField(config, "commandDeflectionScales", ones(numel(config.surfaceNames), 1));
config.commandDeflectionOffsetsDegrees = getNumericColumnField(config, "commandDeflectionOffsetsDegrees", zeros(numel(config.surfaceNames), 1));

config.commandMode = getTextScalarField(config, "commandMode", "all");
config.singleSurfaceName = getTextScalarField(config, "singleSurfaceName", "Aileron_L");
config.commandProfile = normalizeCommandProfile(getFieldOrDefault(config, "commandProfile", struct()));
config.arduinoEchoImport = normalizeArduinoEchoImportConfig( ...
    getFieldOrDefault(config, "arduinoEchoImport", struct()), ...
    rootFolder);

config.neutralSettleSeconds = getPositiveScalarField(config, "neutralSettleSeconds", 1.0);
config.returnToNeutralOnExit = getLogicalField(config, "returnToNeutralOnExit", true);

config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "C_Arduino_Test"));
config.runLabel = getTextScalarField(config, "runLabel", "");

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

if strlength(config.runLabel) == 0
    if isfinite(config.commandProfile.randomSeed)
        config.runLabel = ...
            formatSeedRunLabel(config.commandProfile.randomSeed) + "_" + ...
            formatNanoLoggerOperatingModeLabel(requestedNanoLoggerOperatingMode);
    else
        timeStamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
        config.runLabel = ...
            timeStamp + "_" + ...
            formatNanoLoggerOperatingModeLabel(requestedNanoLoggerOperatingMode);
    end
end

config.arduinoTransport = normalizeArduinoTransportConfig( ...
    arduinoTransportConfig, ...
    config.outputFolder, ...
    config.runLabel);

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

commandProfile.type = getTextScalarField(commandProfileConfig, "type", "latency_step_train");
commandProfile.sampleTimeSeconds = getPositiveScalarField(commandProfileConfig, "sampleTimeSeconds", 0.02);
commandProfile.preCommandNeutralSeconds = getNonnegativeScalarField(commandProfileConfig, "preCommandNeutralSeconds", 0.0);
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

validProfileTypes = ["latency_step_train", "latency_vector_step_train", "sine", "square", "doublet", "custom", "function"];
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

if isfinite(commandProfile.randomSeed)
    validateattributes(commandProfile.randomSeed, {"numeric"}, {"real", "finite", "scalar"}, char(mfilename), 'commandProfile.randomSeed');
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
    "latencyColumn", getTextScalarField(arduinoEchoImportConfig, "latencyColumn", "computer_to_arduino_rx_latency_s"), ...
    "echoTimeColumn", getOptionalTextScalarField(arduinoEchoImportConfig, "echoTimeColumn", "arduino_echo_time_s"), ...
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
        error("Arduino_Test:InvalidArduinoEchoImport", "arduinoEchoImport.tableData must be a table when provided.");
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
    "operatingMode", canonicalizeNanoLoggerOperatingMode( ...
        getTextScalarField(arduinoTransportConfig, "operatingMode", "controller")), ...
    "commandEncoding", canonicalizeNanoLoggerCommandEncoding( ...
        getTextScalarField(arduinoTransportConfig, "commandEncoding", "binary_vector")), ...
    "loggerPort", getPositiveIntegerField(arduinoTransportConfig, "loggerPort", 9500), ...
    "loggerProbeTimeoutSeconds", getPositiveScalarField(arduinoTransportConfig, "loggerProbeTimeoutSeconds", 1.0), ...
    "loggerTimeoutSeconds", getPositiveScalarField(arduinoTransportConfig, "loggerTimeoutSeconds", 5.0), ...
    "syncReplyTimeoutSeconds", getPositiveScalarField(arduinoTransportConfig, "syncReplyTimeoutSeconds", 1.5), ...
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
    error("Arduino_Test:InvalidArduinoTransportMode", ...
        "arduinoTransport.mode must be one of: %s.", ...
        char(join(validTransportModes, ", ")));
end
end

function operatingMode = canonicalizeNanoLoggerOperatingMode(operatingMode)
operatingMode = lower(string(operatingMode));
if ~any(operatingMode == ["controller", "instrumentation"])
    error("Arduino_Test:InvalidNanoLoggerOperatingMode", ...
        "arduinoTransport.operatingMode must be 'controller' or 'instrumentation'.");
end
end

function commandEncoding = canonicalizeNanoLoggerCommandEncoding(commandEncoding)
commandEncoding = lower(string(commandEncoding));
if ~any(commandEncoding == ["binary_vector", "text_set_all"])
    error("Arduino_Test:InvalidNanoLoggerCommandEncoding", ...
        "arduinoTransport.commandEncoding must be 'binary_vector' or 'text_set_all'.");
end
end

function modeLabel = formatNanoLoggerOperatingModeLabel(operatingMode)
switch operatingMode
    case "instrumentation"
        modeLabel = "Instrumentation";
    otherwise
        modeLabel = "Controller";
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
    "connectionInfo", struct("arduino", struct()), ...
    "runInfo", struct( ...
        "status", "initialized", ...
        "reason", "", ...
        "operatingMode", config.arduinoTransport.operatingMode, ...
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

[greeting, firmwareVersion] = probeNanoLoggerConnection(loggerConnection, timeoutSeconds);
commandInterface = struct( ...
    "transportMode", "nano_logger_udp", ...
    "connection", loggerConnection, ...
    "commandEncoding", config.arduinoTransport.commandEncoding, ...
    "operatingMode", config.arduinoTransport.operatingMode, ...
    "servoObjects", {cell(numel(config.surfaceNames), 1)}, ...
    "surfaceNames", config.surfaceNames);
end

function [loggerConnection, backendName] = createNanoLoggerConnection(ipAddress, loggerPort, timeoutSeconds)
timeoutMilliseconds = int32(max(1, ceil(timeoutSeconds .* 1000)));
socket = javaObject("java.net.DatagramSocket");
socket.setSoTimeout(timeoutMilliseconds);
socket.setReceiveBufferSize(int32(1048576));
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

function [greeting, firmwareVersion] = probeNanoLoggerConnection(loggerConnection, timeoutSeconds)
sendNanoLoggerDatagram(loggerConnection, "HELLO");
greeting = tryReadNanoLoggerDatagram(loggerConnection, timeoutSeconds);

if strlength(greeting) == 0
    error("Arduino_Test:NanoLoggerNoHello", ...
        "No HELLO_EVENT was received from the Nano logger within %.3f s.", ...
        timeoutSeconds);
end

telemetryParts = split(greeting, ",");
if isempty(telemetryParts) || telemetryParts(1) ~= "HELLO_EVENT"
    error("Arduino_Test:NanoLoggerUnexpectedHello", ...
        "Expected HELLO_EVENT but received: %s", ...
        char(greeting));
end

if numel(telemetryParts) >= 2
    firmwareVersion = telemetryParts(2);
else
    firmwareVersion = "unknown";
end
end

function sendNanoLoggerDatagram(loggerConnection, payloadText)
if ischar(payloadText) || (isstring(payloadText) && isscalar(payloadText))
    payloadBytes = uint8(char(string(payloadText)));
elseif isnumeric(payloadText)
    payloadBytes = reshape(uint8(payloadText), 1, []);
else
    error("Arduino_Test:InvalidNanoLoggerPayload", ...
        "Nano logger datagrams must be text or uint8 payloads.");
end

packet = javaObject( ...
    "java.net.DatagramPacket", ...
    typecast(payloadBytes, "int8"), ...
    numel(payloadBytes), ...
    loggerConnection.remoteAddress, ...
    int32(loggerConnection.remotePort));
loggerConnection.socket.send(packet);
end

function [nextDatagram, hostReceiveUs] = tryReadNanoLoggerDatagram(loggerConnection, timeoutSeconds, hostTimer)
nextDatagram = "";
hostReceiveUs = nan;
timeoutMilliseconds = int32(max(1, ceil(timeoutSeconds .* 1000)));
loggerConnection.socket.setSoTimeout(timeoutMilliseconds);

receivePacket = javaObject( ...
    "java.net.DatagramPacket", ...
    int8(zeros(1, loggerConnection.packetBufferBytes)), ...
    loggerConnection.packetBufferBytes);

try
    loggerConnection.socket.receive(receivePacket);
    if nargin >= 3 && ~isempty(hostTimer)
        hostReceiveUs = double(hostNowUs(hostTimer));
    end

    packetLength = double(receivePacket.getLength());
    if packetLength <= 0
        return;
    end

    packetBytes = typecast(receivePacket.getData(), "uint8");
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

function printConnectionStatus(runData)
fprintf("\nArduino_Test connection summary\n");
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

%% =============================================================================
% 3) Command Schedule and Run Execution
% =============================================================================
function [storage, runInfo, config] = executeArduinoTest(commandInterface, config)
[scheduledTimeSeconds, profileInfo] = buildCommandSchedule(config.commandProfile);
surfaceCount = numel(config.surfaceNames);
sampleCount = numel(scheduledTimeSeconds);

storage = initializeStorage(sampleCount, surfaceCount);
storage.scheduledTimeSeconds = scheduledTimeSeconds;
storage.profileInfo = profileInfo;

runInfo = struct( ...
    "status", "running", ...
    "reason", "", ...
    "operatingMode", config.arduinoTransport.operatingMode, ...
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
        loggerSession, ...
        scheduledSampleTimeSeconds);
    storage.commandDispatchSeconds(sampleIndex, :) = commandDispatchSeconds;
    storage.commandWriteStopSeconds(sampleIndex) = commandWriteStopSeconds;
    storage.hostSchedulingDelaySeconds(sampleIndex, :) = ...
        commandDispatchSeconds - scheduledSampleTimeSeconds;
    surfaceCommandCounts(config.activeSurfaceMask.') = nextCommandSequenceNumbers(config.activeSurfaceMask.');
    storage.commandSequenceNumbers(sampleIndex, config.activeSurfaceMask.') = ...
        nextCommandSequenceNumbers(config.activeSurfaceMask.');

    storage.sampleCount = sampleIndex;
    storage.baseCommandDegrees(sampleIndex) = baseCommandDegrees;
    storage.desiredDeflectionsDegrees(sampleIndex, :) = desiredDeflectionsDegrees;
    storage.commandedServoPositions(sampleIndex, :) = commandedServoPositionsClipped;
    storage.commandSaturated(sampleIndex, :) = saturatedMask;

    assignin("base", "ArduinoTestLatestState", buildLatestState(storage, config, sampleIndex));
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

function storage = initializeStorage(sampleCount, surfaceCount)
storage = struct( ...
    "sampleCount", 0, ...
    "scheduledTimeSeconds", nan(sampleCount, 1), ...
    "baseCommandDegrees", nan(sampleCount, 1), ...
    "commandWriteStartSeconds", nan(sampleCount, 1), ...
    "commandWriteStopSeconds", nan(sampleCount, 1), ...
    "commandDispatchSeconds", nan(sampleCount, surfaceCount), ...
    "hostSchedulingDelaySeconds", nan(sampleCount, surfaceCount), ...
    "commandSequenceNumbers", nan(sampleCount, surfaceCount), ...
    "arduinoReadStartSeconds", nan(sampleCount, 1), ...
    "arduinoReadStopSeconds", nan(sampleCount, 1), ...
    "arduinoEchoSeconds", nan(sampleCount, surfaceCount), ...              % RX mapped to host time
    "arduinoApplySeconds", nan(sampleCount, surfaceCount), ...             % APPLY mapped to host time
    "computerToArduinoRxLatencySeconds", nan(sampleCount, surfaceCount), ...
    "computerToArduinoApplyLatencySeconds", nan(sampleCount, surfaceCount), ...
    "arduinoReceiveToApplyLatencySeconds", nan(sampleCount, surfaceCount), ...
    "scheduledToApplyLatencySeconds", nan(sampleCount, surfaceCount), ...
    "desiredDeflectionsDegrees", nan(sampleCount, surfaceCount), ...
    "commandedServoPositions", nan(sampleCount, surfaceCount), ...
    "commandSaturated", false(sampleCount, surfaceCount), ...
    "appliedServoPositions", nan(sampleCount, surfaceCount), ...
    "appliedEquivalentDegrees", nan(sampleCount, surfaceCount), ...
    "profileInfo", struct(), ...
    "integritySummary", table());
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
    loggerSession, ...
    scheduledSampleTimeSeconds)

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

if nargin < 8
    scheduledSampleTimeSeconds = NaN;
end

dispatchTimesSeconds = nan(1, numel(surfaceNames));
writeStopSeconds = NaN;

switch commandInterface.transportMode
    case "nano_logger_udp"
        activeIndices = find(activeSurfaceMask);

        if isempty(activeIndices)
            if ~isempty(referenceTimer)
                writeStopSeconds = max(0, toc(referenceTimer) - loggerSession.testStartOffsetSeconds);
            end
            return;
        end

        % Guard against missing sequence numbers for active surfaces.
        for surfaceIndex = activeIndices(:).'
            commandSequence = commandSequenceNumbers(surfaceIndex);
            if ~isfinite(commandSequence)
                error("Arduino_Test:MissingNanoLoggerSequence", ...
                    "Missing command sequence for surface %s.", ...
                    char(surfaceNames(surfaceIndex)));
            end
        end

        % hostNowUs is the dispatch timestamp paired with board micros in
        % COMMAND_EVENT telemetry for clock-domain conversion.
        dispatchAbsoluteUs = hostNowUs(loggerSession.hostTimer);
        if commandInterface.commandEncoding == "binary_vector"
            commandPayload = buildNanoLoggerBinaryVectorCommand( ...
                commandSequenceNumbers, ...
                servoPositions, ...
                activeSurfaceMask);
        else
            commandPayload = buildNanoLoggerSetAllCommand( ...
                surfaceNames, ...
                commandSequenceNumbers, ...
                servoPositions, ...
                activeSurfaceMask);
        end

        sendNanoLoggerDatagram(commandInterface.connection, commandPayload);

        % Keep one dispatch log row per surface so downstream matching logic
        % can remain surface-wise.
        for surfaceIndex = activeIndices(:).'
            dispatchSeconds = NaN;
            if ~isempty(referenceTimer)
                dispatchSeconds = max( ...
                    0, ...
                    (double(dispatchAbsoluteUs) - double(loggerSession.testStartOffsetUs)) ./ 1e6);
                dispatchTimesSeconds(surfaceIndex) = dispatchSeconds;
            end

            loggerSession = appendNanoLoggerDispatchRow( ...
                loggerSession, ...
                surfaceNames(surfaceIndex), ...
                commandSequenceNumbers(surfaceIndex), ...
                dispatchAbsoluteUs, ...
                servoPositions(surfaceIndex), ...
                dispatchSeconds, ...
                scheduledSampleTimeSeconds);
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

%% =============================================================================
% 4) Nano Logger Telemetry Handling
% =============================================================================
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
    "dispatchCommandSeconds", zeros(0, 1), ...
    "dispatchScheduledTimeSeconds", zeros(0, 1), ...
    "dispatchHostSchedulingDelaySeconds", zeros(0, 1), ...
    "syncCount", 0, ...
    "syncId", zeros(0, 1), ...
    "syncHostTxUs", zeros(0, 1), ...
    "syncHostRxUs", zeros(0, 1), ...
    "syncBoardRxUs", zeros(0, 1), ...
    "syncBoardTxUs", zeros(0, 1), ...
    "bufferedTelemetryLines", strings(0, 1));
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
    "dispatchCommandSeconds", nan(dispatchCapacity, 1), ...
    "dispatchScheduledTimeSeconds", nan(dispatchCapacity, 1), ...
    "dispatchHostSchedulingDelaySeconds", nan(dispatchCapacity, 1), ...
    "syncCount", 0, ...
    "syncId", nan(syncCapacity, 1), ...
    "syncHostTxUs", nan(syncCapacity, 1), ...
    "syncHostRxUs", nan(syncCapacity, 1), ...
    "syncBoardRxUs", nan(syncCapacity, 1), ...
    "syncBoardTxUs", nan(syncCapacity, 1), ...
    "bufferedTelemetryLines", strings(0, 1));

sendNanoLoggerControlBurst( ...
    commandInterface.connection, ...
    "MODE," + upper(config.arduinoTransport.operatingMode), ...
    2, ...
    0.02);

if config.arduinoTransport.clearLogsBeforeRun
    sendNanoLoggerControlBurst(commandInterface.connection, "CLEAR_LOGS", 3, 0.02);
end

for syncIndex = 1:config.arduinoTransport.syncCountBeforeRun
    [syncRow, loggerSession.bufferedTelemetryLines] = sendNanoLoggerSync( ...
        commandInterface.connection, ...
        loggerSession.hostTimer, ...
        uint32(syncIndex), ...
        config.arduinoTransport.syncReplyTimeoutSeconds, ...
        loggerSession.bufferedTelemetryLines);
    loggerSession = appendNanoLoggerSyncRow( ...
        loggerSession, ...
        syncRow);
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
        [syncRow, loggerSession.bufferedTelemetryLines] = sendNanoLoggerSync( ...
            commandInterface.connection, ...
            loggerSession.hostTimer, ...
            syncId, ...
            config.arduinoTransport.syncReplyTimeoutSeconds, ...
            loggerSession.bufferedTelemetryLines);
        loggerSession = appendNanoLoggerSyncRow( ...
            loggerSession, ...
            syncRow);
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
        config.arduinoTransport.loggerTimeoutSeconds, ...
        config.surfaceNames, ...
        loggerSession.bufferedTelemetryLines);

    syncRoundTripLog = mergeNanoLoggerSyncRoundTripLogs( ...
        buildNanoLoggerSyncRoundTripTable(loggerSession), ...
        boardSyncLog);
    hasAnySyncTelemetry = ...
        ~isempty(syncRoundTripLog) && ...
        any(isfinite(syncRoundTripLog.board_rx_us)) && ...
        any(isfinite(syncRoundTripLog.board_tx_us));

    if ~hasAnySyncTelemetry
        error("Arduino_Test:MissingNanoLoggerSyncTelemetry", ...
            "No Nano logger SYNC_EVENT datagrams were received.");
    end

    % Missing command telemetry invalidates latency analysis because the
    % board micros-to-host clock map cannot identify individual commands.
    if loggerSession.dispatchCount > 0 && isempty(boardCommandLog)
        error("Arduino_Test:MissingNanoLoggerCommandTelemetry", ...
            "No Nano logger COMMAND_EVENT datagrams were received.");
    end

    hostDispatchCsvPath = fullfile(loggerOutputFolder, "host_dispatch_log.csv");
    syncRoundTripCsvPath = fullfile(loggerOutputFolder, "host_sync_roundtrip.csv");
    boardCommandLogCsvPath = fullfile(loggerOutputFolder, "board_command_log.csv");
    boardSyncLogCsvPath = fullfile(loggerOutputFolder, "board_sync_log.csv");
    echoImportCsvPath = fullfile(loggerOutputFolder, "arduino_echo_import.csv");
    rawTelemetryLogPath = fullfile(loggerOutputFolder, "udp_telemetry_log.txt");

    % Export raw host, board, and telemetry logs before canonical echo import
    % so failed or partial runs still retain audit evidence.
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
        config.activeSurfaceNames, ...
        double(loggerSession.testStartOffsetUs));

    writetable(echoImportTable, echoImportCsvPath);

    config.arduinoEchoImport.filePath = string(echoImportCsvPath);
    config.arduinoEchoImport.loggerOutputFolder = string(loggerOutputFolder);
    config.arduinoEchoImport.tableData = echoImportTable;
    config.arduinoTransport.captureSucceeded = true;
    config.arduinoTransport.captureMessage = ...
        "Captured " + height(boardCommandLog) + " command telemetry rows and " + ...
        height(syncRoundTripLog) + " sync telemetry rows.";
    config.arduinoTransport.captureRowCount = height(echoImportTable);
catch captureException
    warning("Arduino_Test:ArduinoLoggerCaptureFailed", ...
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
    commandPosition, ...
    commandDispatchSeconds, ...
    scheduledTimeSeconds)
loggerSession.dispatchCount = loggerSession.dispatchCount + 1;
rowIndex = loggerSession.dispatchCount;

loggerSession.dispatchSurfaceName(rowIndex) = string(surfaceName);
loggerSession.dispatchCommandSequence(rowIndex) = double(commandSequence);
loggerSession.dispatchCommandUs(rowIndex) = double(commandDispatchUs);
loggerSession.dispatchPosition(rowIndex) = double(commandPosition);
loggerSession.dispatchCommandSeconds(rowIndex) = double(commandDispatchSeconds);
loggerSession.dispatchScheduledTimeSeconds(rowIndex) = double(scheduledTimeSeconds);
loggerSession.dispatchHostSchedulingDelaySeconds(rowIndex) = ...
    double(commandDispatchSeconds) - double(scheduledTimeSeconds);
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
    loggerSession.dispatchCommandSeconds(rowIndices), ...
    loggerSession.dispatchScheduledTimeSeconds(rowIndices), ...
    loggerSession.dispatchHostSchedulingDelaySeconds(rowIndices), ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'command_dispatch_us', ...
        'position_norm', ...
        'command_dispatch_s', ...
        'scheduled_time_s', ...
        'host_scheduling_delay_s'});
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

function syncRoundTripLog = mergeNanoLoggerSyncRoundTripLogs(immediateSyncLog, deferredSyncLog)
if isempty(immediateSyncLog)
    syncRoundTripLog = buildNanoLoggerSyncRoundTripTableFromEvents(deferredSyncLog);
    return;
end

syncRoundTripLog = immediateSyncLog;
deferredRoundTripLog = buildNanoLoggerSyncRoundTripTableFromEvents(deferredSyncLog);
if isempty(deferredRoundTripLog)
    return;
end

deferredKeys = buildNanoLoggerSyncKeys( ...
    deferredRoundTripLog.sync_id, ...
    deferredRoundTripLog.host_tx_us);
immediateKeys = buildNanoLoggerSyncKeys( ...
    syncRoundTripLog.sync_id, ...
    syncRoundTripLog.host_tx_us);
[isMatched, matchedIndices] = ismember(immediateKeys, deferredKeys);

matchedImmediateRows = find(isMatched);
for rowOffset = 1:numel(matchedImmediateRows)
    rowIndex = matchedImmediateRows(rowOffset);
    matchedRowIndex = matchedIndices(rowIndex);

    if ~isfinite(syncRoundTripLog.board_rx_us(rowIndex))
        syncRoundTripLog.board_rx_us(rowIndex) = ...
            deferredRoundTripLog.board_rx_us(matchedRowIndex);
    end

    if ~isfinite(syncRoundTripLog.board_tx_us(rowIndex))
        syncRoundTripLog.board_tx_us(rowIndex) = ...
            deferredRoundTripLog.board_tx_us(matchedRowIndex);
    end
end

unmatchedDeferredMask = ~ismember(deferredKeys, immediateKeys);
if any(unmatchedDeferredMask)
    syncRoundTripLog = [syncRoundTripLog; deferredRoundTripLog(unmatchedDeferredMask, :)]; %#ok<AGROW>
end
end

function syncKeys = buildNanoLoggerSyncKeys(syncId, hostTxUs)
syncKeys = ...
    compose("%.0f", double(syncId)) + "|" + ...
    compose("%.0f", double(hostTxUs));
end

function [syncRow, bufferedTelemetryLines] = sendNanoLoggerSync( ...
    loggerConnection, ...
    hostTimer, ...
    syncId, ...
    timeoutSeconds, ...
    bufferedTelemetryLines)
if nargin < 4 || ~isfinite(timeoutSeconds) || timeoutSeconds <= 0
    timeoutSeconds = 1.5;
end

if nargin < 5
    bufferedTelemetryLines = strings(0, 1);
end

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

waitStart = tic;
while toc(waitStart) < timeoutSeconds
    remainingSeconds = timeoutSeconds - toc(waitStart);
    perReadTimeoutSeconds = min(0.05, remainingSeconds);
    [nextDatagram, hostRxUs] = tryReadNanoLoggerDatagram( ...
        loggerConnection, ...
        perReadTimeoutSeconds, ...
        hostTimer);

    if strlength(nextDatagram) == 0
        continue;
    end

    [isMatchingSync, matchedSyncRow] = tryParseNanoLoggerSyncEvent( ...
        nextDatagram, ...
        uint32(syncId), ...
        double(hostTxUs), ...
        hostRxUs);
    if isMatchingSync
        syncRow = matchedSyncRow;
        return;
    end

    bufferedTelemetryLines(end + 1, 1) = nextDatagram; %#ok<AGROW>
end

% Fall back to deferred post-run parsing if the immediate reply is late.
end

function [telemetryLines, boardCommandLog, boardSyncLog] = collectNanoLoggerTelemetry( ...
    loggerConnection, ...
    maxWaitSeconds, ...
    surfaceNames, ...
    initialTelemetryLines)
if nargin < 4
    telemetryLines = strings(0, 1);
else
    telemetryLines = reshape(string(initialTelemetryLines), [], 1);
end

collectStart = tic;
lastReceiveElapsedSeconds = 0;
if ~isempty(telemetryLines)
    lastReceiveElapsedSeconds = toc(collectStart);
end
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

[boardCommandLog, boardSyncLog] = parseNanoLoggerTelemetryDatagrams(telemetryLines, surfaceNames);
end

function [isMatchingSync, syncRow] = tryParseNanoLoggerSyncEvent( ...
    telemetryLine, ...
    expectedSyncId, ...
    expectedHostTxUs, ...
    hostRxUs)
isMatchingSync = false;
syncRow = table( ...
    nan(1, 1), ...
    nan(1, 1), ...
    nan(1, 1), ...
    nan(1, 1), ...
    nan(1, 1), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});

telemetryParts = split(string(strtrim(telemetryLine)), ",");
if numel(telemetryParts) < 5 || telemetryParts(1) ~= "SYNC_EVENT"
    return;
end

syncId = double(str2double(telemetryParts(2)));
hostTxUs = double(str2double(telemetryParts(3)));
boardRxUs = double(str2double(telemetryParts(4)));
boardTxUs = double(str2double(telemetryParts(5)));

if ~isfinite(syncId) || ~isfinite(hostTxUs) || ~isfinite(boardRxUs) || ~isfinite(boardTxUs)
    return;
end

if uint32(syncId) ~= uint32(expectedSyncId) || hostTxUs ~= expectedHostTxUs
    return;
end

isMatchingSync = true;
syncRow = table( ...
    syncId, ...
    hostTxUs, ...
    double(hostRxUs), ...
    boardRxUs, ...
    boardTxUs, ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});
end

function [boardCommandLog, boardSyncLog] = parseNanoLoggerTelemetryDatagrams(telemetryLines, surfaceNames)
surfaceNames = reshape(string(surfaceNames), [], 1);
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
        case "VECTOR_EVENT"
            expectedPartCount = 4 + 3 * numel(surfaceNames);
            if numel(telemetryParts) < expectedPartCount
                continue;
            end

            sampleSequence = double(str2double(telemetryParts(2)));
            activeSurfaceMask = uint32(str2double(telemetryParts(3)));
            vectorRxUs = double(str2double(telemetryParts(4)));
            payloadIndex = 5;

            for surfaceIndex = 1:numel(surfaceNames)
                applyUsValue = double(str2double(telemetryParts(payloadIndex)));
                positionCodeValue = double(str2double(telemetryParts(payloadIndex + 1)));
                pulseUsValue = double(str2double(telemetryParts(payloadIndex + 2)));
                payloadIndex = payloadIndex + 3;

                if bitand(activeSurfaceMask, bitshift(uint32(1), surfaceIndex - 1)) == 0
                    continue;
                end

                commandSurfaceNames(end + 1, 1) = surfaceNames(surfaceIndex); %#ok<AGROW>
                commandSequence(end + 1, 1) = sampleSequence; %#ok<AGROW>
                commandRxUs(end + 1, 1) = vectorRxUs; %#ok<AGROW>
                commandApplyUs(end + 1, 1) = applyUsValue; %#ok<AGROW>
                commandAppliedPosition(end + 1, 1) = positionCodeValue ./ 65535.0; %#ok<AGROW>
                commandPulseUs(end + 1, 1) = pulseUsValue; %#ok<AGROW>
            end
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

%% =============================================================================
% 5) Echo Import and Latency Summaries
% =============================================================================
function storage = finalizeArduinoEcho(storage, config)
sampleIndices = 1:storage.sampleCount;

if isempty(sampleIndices)
    return;
end

surfaceCount = numel(config.surfaceNames);

storage.appliedServoPositions(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.appliedEquivalentDegrees(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.arduinoReadStartSeconds(sampleIndices) = nan(numel(sampleIndices), 1);
storage.arduinoReadStopSeconds(sampleIndices) = nan(numel(sampleIndices), 1);
storage.arduinoEchoSeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.arduinoApplySeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.computerToArduinoRxLatencySeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.computerToArduinoApplyLatencySeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.arduinoReceiveToApplyLatencySeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);
storage.scheduledToApplyLatencySeconds(sampleIndices, :) = nan(numel(sampleIndices), surfaceCount);

echoImportTable = loadArduinoEchoImportTable(config, config.activeSurfaceNames);
if isempty(echoImportTable)
    storage.integritySummary = buildEmptyIntegritySummary(config);
    return;
end

commandLookup = buildArduinoCommandLookupTable(storage, config, sampleIndices);
[echoAssignments, integritySummary] = assignArduinoEchoToCommands(commandLookup, echoImportTable, config);
storage.integritySummary = integritySummary;

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

    dispatchSeconds = storage.commandDispatchSeconds(sampleIndex, surfaceIndex);

    rxLatencySeconds = getAssignmentField( ...
        echoAssignments, "computer_to_arduino_rx_latency_s", assignmentIndex);
    applyLatencySeconds = getAssignmentField( ...
        echoAssignments, "computer_to_arduino_apply_latency_s", assignmentIndex);
    receiveToApplyLatencySeconds = getAssignmentField( ...
        echoAssignments, "arduino_receive_to_apply_latency_s", assignmentIndex);

    rxHostSeconds = echoAssignments.arduino_echo_time_s(assignmentIndex);
    if ~isfinite(rxHostSeconds) && isfinite(dispatchSeconds) && isfinite(rxLatencySeconds)
        rxHostSeconds = dispatchSeconds + rxLatencySeconds;
    end

    applyHostSeconds = nan;
    if isfinite(dispatchSeconds) && isfinite(applyLatencySeconds)
        applyHostSeconds = dispatchSeconds + applyLatencySeconds;
    elseif isfinite(rxHostSeconds) && isfinite(receiveToApplyLatencySeconds)
        applyHostSeconds = rxHostSeconds + receiveToApplyLatencySeconds;
    end

    if ~isfinite(receiveToApplyLatencySeconds) && ...
            isfinite(applyLatencySeconds) && isfinite(rxLatencySeconds)
        receiveToApplyLatencySeconds = applyLatencySeconds - rxLatencySeconds;
    end

    storage.arduinoReadStartSeconds(sampleIndex) = min( ...
        storage.commandDispatchSeconds(sampleIndex, ...
        isfinite(storage.commandDispatchSeconds(sampleIndex, :))), ...
        [], "omitnan");

    storage.arduinoEchoSeconds(sampleIndex, surfaceIndex) = rxHostSeconds;
    storage.arduinoApplySeconds(sampleIndex, surfaceIndex) = applyHostSeconds;
    storage.computerToArduinoRxLatencySeconds(sampleIndex, surfaceIndex) = rxLatencySeconds;
    storage.computerToArduinoApplyLatencySeconds(sampleIndex, surfaceIndex) = applyLatencySeconds;
    storage.arduinoReceiveToApplyLatencySeconds(sampleIndex, surfaceIndex) = receiveToApplyLatencySeconds;
    storage.scheduledToApplyLatencySeconds(sampleIndex, surfaceIndex) = ...
        applyHostSeconds - storage.scheduledTimeSeconds(sampleIndex);
    storage.appliedServoPositions(sampleIndex, surfaceIndex) = appliedPosition;
    storage.appliedEquivalentDegrees(sampleIndex, surfaceIndex) = appliedEquivalentDegrees;
end

for sampleIndex = sampleIndices
    candidateTimes = [ ...
        storage.arduinoEchoSeconds(sampleIndex, :), ...
        storage.arduinoApplySeconds(sampleIndex, :)];

    if any(isfinite(candidateTimes))
        storage.arduinoReadStopSeconds(sampleIndex) = ...
            lastFiniteTimestamp(candidateTimes, storage.commandWriteStopSeconds(sampleIndex));
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
    return;
end

latestImportCsvPath = findLatestMatchingFile(loggerOutputFolder, "arduino_echo_import*.csv");
if strlength(latestImportCsvPath) > 0
    echoImportTable = readtable(latestImportCsvPath);
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
    activeSurfaceNames, ...
    hostTimeOriginUs)
if nargin < 5
    hostTimeOriginUs = 0.0;
end

hostDispatchLog = readtable(hostDispatchCsvPath);
syncRoundTripLog = readtable(syncRoundTripCsvPath);
boardCommandLog = readCommentCsv(boardCommandLogCsvPath);

validateLoggerHostDispatchLog(hostDispatchLog);
validateLoggerSyncRoundTripLog(syncRoundTripLog);
validateLoggerBoardCommandLog(boardCommandLog);

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

[clockSlope, clockIntercept] = estimateBoardToHostClockMapFromCommands( ...
    joinedTable, ...
    syncRoundTripLog);

dispatchUs = double(joinedTable.command_dispatch_us);
rxHostUs = clockSlope .* double(joinedTable.rx_us) + clockIntercept;
applyHostUs = clockSlope .* double(joinedTable.apply_us) + clockIntercept;

rxLatencyUs = rxHostUs - dispatchUs;
applyLatencyUs = applyHostUs - dispatchUs;

minimumRxLatencyUs = min(rxLatencyUs);
minimumApplyLatencyUs = min(applyLatencyUs);
minimumLatencyUs = min([minimumRxLatencyUs; minimumApplyLatencyUs]);

if minimumLatencyUs < 0
    % Re-anchor the fitted clock so matched commands cannot precede dispatch.
    rxHostUs = rxHostUs - minimumLatencyUs;
    applyHostUs = applyHostUs - minimumLatencyUs;
    rxLatencyUs = rxLatencyUs - minimumLatencyUs;
    applyLatencyUs = applyLatencyUs - minimumLatencyUs;
end

echoImportTable = table( ...
    joinedTable.surface_name, ...
    joinedTable.command_sequence, ...
    (rxHostUs - hostTimeOriginUs) ./ 1e6, ...
    rxLatencyUs ./ 1e6, ...
    applyLatencyUs ./ 1e6, ...
    joinedTable.applied_position, ...
    nan(height(joinedTable), 1), ...
    (double(joinedTable.apply_us) - double(joinedTable.rx_us)) ./ 1e6, ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'arduino_echo_time_s', ...
        'computer_to_arduino_rx_latency_s', ...
        'computer_to_arduino_apply_latency_s', ...
        'applied_position', ...
        'applied_equivalent_deg', ...
        'arduino_receive_to_apply_latency_s'});
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
    % Midpoints reduce asymmetric USB/WiFi scheduling error in the clock map.
    hostReferenceUs = 0.5 .* (hostTxUs(midpointMask) + hostRxUs(midpointMask));
    boardReferenceUs = 0.5 .* (boardRxUs(midpointMask) + boardTxUs(midpointMask));
else
    % Legacy captures only include host TX and board RX timestamps.
    hostReferenceUs = hostTxUs(isfinite(hostTxUs) & isfinite(boardRxUs));
    boardReferenceUs = boardRxUs(isfinite(hostTxUs) & isfinite(boardRxUs));
end

if isempty(hostReferenceUs)
    error("Arduino_Test:MissingNanoLoggerSyncTelemetry", ...
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

function [clockSlope, clockIntercept] = estimateBoardToHostClockMapFromCommands( ...
    joinedTable, ...
    syncRoundTripLog)
commandRxUs = double(joinedTable.rx_us);
commandDispatchUs = double(joinedTable.command_dispatch_us);
validCommandMask = isfinite(commandRxUs) & isfinite(commandDispatchUs);

if nnz(validCommandMask) < 2
    [clockSlope, clockIntercept] = estimateBoardToHostClockMap(syncRoundTripLog);
    return;
end

fitCoefficients = polyfit(commandRxUs(validCommandMask), commandDispatchUs(validCommandMask), 1);
clockSlope = fitCoefficients(1);
clockIntercept = fitCoefficients(2) + estimateNanoLoggerOneWayBaselineUs(syncRoundTripLog);
end

function oneWayBaselineUs = estimateNanoLoggerOneWayBaselineUs(syncRoundTripLog)
hostTxUs = double(syncRoundTripLog.host_tx_us);
hostRxUs = double(syncRoundTripLog.host_rx_us);
validRoundTripMask = ...
    isfinite(hostTxUs) & ...
    isfinite(hostRxUs) & ...
    hostRxUs >= hostTxUs;

if ~any(validRoundTripMask)
    oneWayBaselineUs = 0.0;
    return;
end

validIndices = find(validRoundTripMask);
preRunIndices = validIndices;
if numel(validIndices) >= 2
    hostTxGapsUs = diff(hostTxUs(validIndices));
    [maximumGapUs, maximumGapIndex] = max(hostTxGapsUs);
    if isfinite(maximumGapUs) && maximumGapUs > 0
        preRunIndices = validIndices(1:maximumGapIndex);
    end
end

roundTripUs = hostRxUs(preRunIndices) - hostTxUs(preRunIndices);
roundTripUs = roundTripUs(roundTripUs >= 0);
if isempty(roundTripUs)
    oneWayBaselineUs = 0.0;
    return;
end

oneWayBaselineUs = 0.5 .* median(roundTripUs, "omitnan");
if ~isfinite(oneWayBaselineUs)
    oneWayBaselineUs = 0.0;
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
    error("Arduino_Test:MissingArduinoLoggerColumns", ...
        "The %s is missing: %s", ...
        tableLabel, ...
        char(join(missingColumns, ", ")));
end
end

function rawEchoTable = readArduinoEchoImportFile(importConfig)
filePath = importConfig.filePath;
if ~isfile(filePath)
    error("Arduino_Test:MissingArduinoEchoImport", ...
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
        error("Arduino_Test:UnsupportedArduinoEchoImport", ...
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
        error("Arduino_Test:AmbiguousArduinoEchoSurface", ...
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

% Preserve explicit RX/APPLY columns when present.
rxLatencyColumnName = resolveTableVariableName(rawEchoTable, "computer_to_arduino_rx_latency_s", false);
applyLatencyColumnName = resolveTableVariableName(rawEchoTable, "computer_to_arduino_apply_latency_s", false);
receiveToApplyColumnName = resolveTableVariableName(rawEchoTable, "arduino_receive_to_apply_latency_s", false);

% Legacy import path for backward compatibility.
legacyLatencyColumnName = resolveTableVariableName(rawEchoTable, importConfig.latencyColumn, false);
echoTimeColumnName = resolveTableVariableName(rawEchoTable, importConfig.echoTimeColumn, false);

if strlength(rxLatencyColumnName) == 0 && ...
        strlength(legacyLatencyColumnName) == 0 && ...
        strlength(echoTimeColumnName) == 0
    error("Arduino_Test:MissingArduinoEchoTiming", ...
        "arduinoEchoImport must provide RX latency, legacy latency, or echo time.");
end

echoImportTable.computer_to_arduino_rx_latency_s = nan(rowCount, 1);
if strlength(rxLatencyColumnName) > 0
    echoImportTable.computer_to_arduino_rx_latency_s = ...
        reshape(double(rawEchoTable.(char(rxLatencyColumnName))), [], 1);
elseif strlength(legacyLatencyColumnName) > 0
    echoImportTable.computer_to_arduino_rx_latency_s = ...
        reshape(double(rawEchoTable.(char(legacyLatencyColumnName))), [], 1);
end

echoImportTable.computer_to_arduino_apply_latency_s = nan(rowCount, 1);
if strlength(applyLatencyColumnName) > 0
    echoImportTable.computer_to_arduino_apply_latency_s = ...
        reshape(double(rawEchoTable.(char(applyLatencyColumnName))), [], 1);
end

echoImportTable.arduino_receive_to_apply_latency_s = nan(rowCount, 1);
if strlength(receiveToApplyColumnName) > 0
    echoImportTable.arduino_receive_to_apply_latency_s = ...
        reshape(double(rawEchoTable.(char(receiveToApplyColumnName))), [], 1);
elseif any(isfinite(echoImportTable.computer_to_arduino_apply_latency_s)) && ...
        any(isfinite(echoImportTable.computer_to_arduino_rx_latency_s))
    echoImportTable.arduino_receive_to_apply_latency_s = ...
        echoImportTable.computer_to_arduino_apply_latency_s - ...
        echoImportTable.computer_to_arduino_rx_latency_s;
end

% Keep the legacy field as an alias to RX for backward compatibility.
echoImportTable.computer_to_arduino_latency_s = ...
    echoImportTable.computer_to_arduino_rx_latency_s;

echoImportTable.arduino_echo_time_s = nan(rowCount, 1);
if strlength(echoTimeColumnName) > 0
    echoImportTable.arduino_echo_time_s = ...
        reshape(double(rawEchoTable.(char(echoTimeColumnName))), [], 1) + ...
        importConfig.matlabTimeOffsetSeconds;
end

appliedPositionColumnName = resolveTableVariableName(rawEchoTable, importConfig.appliedPositionColumn, false);
echoImportTable.applied_position = nan(rowCount, 1);
if strlength(appliedPositionColumnName) > 0
    echoImportTable.applied_position = reshape(double(rawEchoTable.(char(appliedPositionColumnName))), [], 1);
end

appliedEquivalentDegreesColumnName = ...
    resolveTableVariableName(rawEchoTable, importConfig.appliedEquivalentDegreesColumn, false);
echoImportTable.applied_equivalent_deg = nan(rowCount, 1);
if strlength(appliedEquivalentDegreesColumnName) > 0
    echoImportTable.applied_equivalent_deg = ...
        reshape(double(rawEchoTable.(char(appliedEquivalentDegreesColumnName))), [], 1);
end
end

function fieldValue = getAssignmentField(assignmentsTable, fieldName, rowIndex)
fieldValue = nan;
if ismember(fieldName, string(assignmentsTable.Properties.VariableNames))
    fieldValue = assignmentsTable.(fieldName)(rowIndex);
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

% Keep one lookup row per surface/sequence so vector UDP commands can be
% audited for per-servo drops even though they share one dispatch packet.
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

function [echoAssignments, integritySummary] = assignArduinoEchoToCommands(commandLookup, echoImportTable, config)
echoImportTable = reshapeEchoImportTable(echoImportTable);
uniqueEchoImportTable = deduplicateEchoImportTable(echoImportTable);

% Surface+sequence is the stable cross-clock key between MATLAB dispatch and
% Nano telemetry; timing values are used after the row identity is known.
echoAssignments = commandLookup(:, {'sample_index', 'surface_index', 'surface_name', 'command_sequence', 'command_dispatch_s'});
rowCount = height(echoAssignments);

echoAssignments.arduino_echo_time_s = nan(rowCount, 1);
echoAssignments.computer_to_arduino_rx_latency_s = nan(rowCount, 1);
echoAssignments.computer_to_arduino_apply_latency_s = nan(rowCount, 1);
echoAssignments.arduino_receive_to_apply_latency_s = nan(rowCount, 1);
echoAssignments.applied_position = nan(rowCount, 1);
echoAssignments.applied_equivalent_deg = nan(rowCount, 1);

commandKeys = buildSurfaceSequenceKeys( ...
    echoAssignments.surface_name, ...
    echoAssignments.command_sequence);
telemetryKeys = buildSurfaceSequenceKeys( ...
    uniqueEchoImportTable.surface_name, ...
    uniqueEchoImportTable.command_sequence);
[isMatched, matchedIndices] = ismember(commandKeys, telemetryKeys);

echoAssignments.arduino_echo_time_s(isMatched) = ...
    uniqueEchoImportTable.arduino_echo_time_s(matchedIndices(isMatched));
echoAssignments.computer_to_arduino_rx_latency_s(isMatched) = ...
    uniqueEchoImportTable.computer_to_arduino_rx_latency_s(matchedIndices(isMatched));
echoAssignments.computer_to_arduino_apply_latency_s(isMatched) = ...
    uniqueEchoImportTable.computer_to_arduino_apply_latency_s(matchedIndices(isMatched));
echoAssignments.arduino_receive_to_apply_latency_s(isMatched) = ...
    uniqueEchoImportTable.arduino_receive_to_apply_latency_s(matchedIndices(isMatched));
echoAssignments.applied_position(isMatched) = ...
    uniqueEchoImportTable.applied_position(matchedIndices(isMatched));
echoAssignments.applied_equivalent_deg(isMatched) = ...
    uniqueEchoImportTable.applied_equivalent_deg(matchedIndices(isMatched));

for commandIndex = 1:height(echoAssignments)
    echoAssignments.arduino_echo_time_s(commandIndex) = resolveArduinoEchoTime( ...
        echoAssignments, ...
        commandIndex, ...
        echoAssignments.command_dispatch_s(commandIndex));
end

integritySummary = buildIntegritySummary( ...
    commandLookup, ...
    echoAssignments, ...
    echoImportTable, ...
    config);
end

function arduinoEchoTimeSeconds = resolveArduinoEchoTime(echoImportTable, matchedRow, commandDispatchSeconds)
arduinoEchoTimeSeconds = echoImportTable.arduino_echo_time_s(matchedRow);
if isfinite(arduinoEchoTimeSeconds)
    return;
end

if ismember("computer_to_arduino_rx_latency_s", string(echoImportTable.Properties.VariableNames))
    latencySeconds = echoImportTable.computer_to_arduino_rx_latency_s(matchedRow);
    if isfinite(latencySeconds)
        arduinoEchoTimeSeconds = commandDispatchSeconds + latencySeconds;
        return;
    end
end

if ismember("computer_to_arduino_apply_latency_s", string(echoImportTable.Properties.VariableNames))
    latencySeconds = echoImportTable.computer_to_arduino_apply_latency_s(matchedRow);
elseif ismember("computer_to_arduino_latency_s", string(echoImportTable.Properties.VariableNames))
    latencySeconds = echoImportTable.computer_to_arduino_latency_s(matchedRow);
else
    latencySeconds = nan;
end

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
    error("Arduino_Test:MissingArduinoEchoColumn", ...
        "The Arduino echo import is missing column '%s'.", ...
        char(requestedName));
end

variableName = "";
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
    buildSurfaceVariableNames(surfaceNames, "rx_time_s"), ...
    buildSurfaceVariableNames(surfaceNames, "apply_time_s"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_position"), ...
    buildSurfaceVariableNames(surfaceNames, "applied_equivalent_deg"), ...
    buildSurfaceVariableNames(surfaceNames, "host_scheduling_delay_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_rx_latency_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_apply_latency_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_receive_to_apply_latency_s"), ...
    buildSurfaceVariableNames(surfaceNames, "scheduled_to_apply_latency_s")];

arduinoLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    storage.arduinoApplySeconds(sampleIndices, :), ...
    storage.appliedServoPositions(sampleIndices, :), ...
    storage.appliedEquivalentDegrees(sampleIndices, :), ...
    storage.hostSchedulingDelaySeconds(sampleIndices, :), ...
    storage.computerToArduinoRxLatencySeconds(sampleIndices, :), ...
    storage.computerToArduinoApplyLatencySeconds(sampleIndices, :), ...
    storage.arduinoReceiveToApplyLatencySeconds(sampleIndices, :), ...
    storage.scheduledToApplyLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(arduinoVariableNames));

arduinoLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
arduinoLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
arduinoLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);
arduinoLog.arduino_echo_available = any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2);
arduinoLog.arduino_apply_available = any(isfinite(storage.arduinoApplySeconds(sampleIndices, :)), 2);

rxVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_rx_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_rx_latency_s")];

computerToArduinoRxLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.commandDispatchSeconds(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    storage.computerToArduinoRxLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(rxVariableNames));

computerToArduinoRxLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
computerToArduinoRxLatencyLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
computerToArduinoRxLatencyLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);
computerToArduinoRxLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
computerToArduinoRxLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

hostSchedulingVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "scheduled_time_s"), ...
    buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
    buildSurfaceVariableNames(surfaceNames, "host_scheduling_delay_s")];

hostSchedulingDelayLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    repmat(storage.scheduledTimeSeconds(sampleIndices), 1, numel(surfaceNames)), ...
    storage.commandDispatchSeconds(sampleIndices, :), ...
    storage.hostSchedulingDelaySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(chooseFiniteRowTimes( ...
        storage.commandWriteStopSeconds(sampleIndices), ...
        storage.scheduledTimeSeconds(sampleIndices))), ...
    'VariableNames', cellstr(hostSchedulingVariableNames));

hostSchedulingDelayLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
hostSchedulingDelayLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);

applyVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "command_dispatch_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_apply_s"), ...
    buildSurfaceVariableNames(surfaceNames, "computer_to_arduino_apply_latency_s")];

computerToArduinoApplyLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.commandDispatchSeconds(sampleIndices, :), ...
    storage.arduinoApplySeconds(sampleIndices, :), ...
    storage.computerToArduinoApplyLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(applyVariableNames));

computerToArduinoApplyLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
computerToArduinoApplyLatencyLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
computerToArduinoApplyLatencyLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);
computerToArduinoApplyLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
computerToArduinoApplyLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

receiveToApplyVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_rx_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_apply_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_receive_to_apply_latency_s")];

arduinoReceiveToApplyLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    storage.arduinoEchoSeconds(sampleIndices, :), ...
    storage.arduinoApplySeconds(sampleIndices, :), ...
    storage.arduinoReceiveToApplyLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(receiveToApplyVariableNames));

arduinoReceiveToApplyLatencyLog.scheduled_time_s = storage.scheduledTimeSeconds(sampleIndices);
arduinoReceiveToApplyLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
arduinoReceiveToApplyLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

scheduledToApplyVariableNames = [ ...
    buildSurfaceVariableNames(surfaceNames, "command_sequence"), ...
    buildSurfaceVariableNames(surfaceNames, "scheduled_time_s"), ...
    buildSurfaceVariableNames(surfaceNames, "arduino_apply_s"), ...
    buildSurfaceVariableNames(surfaceNames, "scheduled_to_apply_latency_s")];

scheduledToApplyLatencyLog = array2timetable( ...
    [ ...
    storage.commandSequenceNumbers(sampleIndices, :), ...
    repmat(storage.scheduledTimeSeconds(sampleIndices), 1, numel(surfaceNames)), ...
    storage.arduinoApplySeconds(sampleIndices, :), ...
    storage.scheduledToApplyLatencySeconds(sampleIndices, :)], ...
    'RowTimes', seconds(arduinoRowTimesSeconds), ...
    'VariableNames', cellstr(scheduledToApplyVariableNames));

scheduledToApplyLatencyLog.command_write_start_s = storage.commandWriteStartSeconds(sampleIndices);
scheduledToApplyLatencyLog.command_write_stop_s = storage.commandWriteStopSeconds(sampleIndices);
scheduledToApplyLatencyLog.arduino_read_start_s = storage.arduinoReadStartSeconds(sampleIndices);
scheduledToApplyLatencyLog.arduino_read_stop_s = storage.arduinoReadStopSeconds(sampleIndices);

sampleSummary = table( ...
    storage.scheduledTimeSeconds(sampleIndices), ...
    storage.baseCommandDegrees(sampleIndices), ...
    storage.commandWriteStartSeconds(sampleIndices), ...
    storage.commandWriteStopSeconds(sampleIndices), ...
    storage.arduinoReadStartSeconds(sampleIndices), ...
    storage.arduinoReadStopSeconds(sampleIndices), ...
    any(isfinite(storage.arduinoEchoSeconds(sampleIndices, :)), 2), ...
    any(isfinite(storage.arduinoApplySeconds(sampleIndices, :)), 2), ...
    'VariableNames', { ...
        'scheduled_time_s', ...
        'base_command_deg', ...
        'command_write_start_s', ...
        'command_write_stop_s', ...
        'arduino_read_start_s', ...
        'arduino_read_stop_s', ...
        'arduino_rx_available', ...
        'arduino_apply_available'});

latencySummary = buildComputerToArduinoLatencySummary(storage, config);

logs = struct( ...
    "inputSignal", commandLog, ...
    "arduinoEcho", arduinoLog, ...
    "computerToArduinoLatency", computerToArduinoRxLatencyLog, ...  % legacy alias
    "hostSchedulingDelay", hostSchedulingDelayLog, ...
    "computerToArduinoRxLatency", computerToArduinoRxLatencyLog, ...
    "computerToArduinoApplyLatency", computerToArduinoApplyLatencyLog, ...
    "arduinoReceiveToApplyLatency", arduinoReceiveToApplyLatencyLog, ...
    "scheduledToApplyLatency", scheduledToApplyLatencyLog, ...
    "latencySummary", latencySummary, ...
    "integritySummary", storage.integritySummary, ...
    "profileEvents", storage.profileInfo.profileEvents, ...
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

latencySummary = buildComputerToArduinoLatencySummary(storage, config);
surfaceSummary = [surfaceSummary latencySummary(:, 3:end)];
end

function latencySummary = buildComputerToArduinoLatencySummary(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceCount = numel(config.surfaceNames);
integritySummary = storage.integritySummary;
if isempty(integritySummary)
    integritySummary = buildEmptyIntegritySummary(config);
end

dispatchedCommandCount = integritySummary.DispatchedCommandCount;
matchedRxCommandCount = integritySummary.MatchedRxCommandCount;
unmatchedRxCommandCount = integritySummary.UnmatchedRxCommandCount;
unmatchedRxCommandFraction = integritySummary.UnmatchedRxCommandFraction;
matchedApplyCommandCount = integritySummary.MatchedApplyCommandCount;
unmatchedApplyCommandCount = integritySummary.UnmatchedApplyCommandCount;
unmatchedApplyCommandFraction = integritySummary.UnmatchedApplyCommandFraction;

hostSchedulingSampleCount = zeros(surfaceCount, 1);
hostSchedulingMeanSeconds = nan(surfaceCount, 1);
hostSchedulingStdSeconds = nan(surfaceCount, 1);
hostSchedulingMedianSeconds = nan(surfaceCount, 1);
hostSchedulingP95Seconds = nan(surfaceCount, 1);
hostSchedulingP99Seconds = nan(surfaceCount, 1);
hostSchedulingMaxSeconds = nan(surfaceCount, 1);

rxSampleCount = zeros(surfaceCount, 1);
rxMeanSeconds = nan(surfaceCount, 1);
rxStdSeconds = nan(surfaceCount, 1);
rxMedianSeconds = nan(surfaceCount, 1);
rxP95Seconds = nan(surfaceCount, 1);
rxP99Seconds = nan(surfaceCount, 1);
rxMaxSeconds = nan(surfaceCount, 1);

applySampleCount = zeros(surfaceCount, 1);
applyMeanSeconds = nan(surfaceCount, 1);
applyStdSeconds = nan(surfaceCount, 1);
applyMedianSeconds = nan(surfaceCount, 1);
applyP95Seconds = nan(surfaceCount, 1);
applyP99Seconds = nan(surfaceCount, 1);
applyMaxSeconds = nan(surfaceCount, 1);

receiveToApplySampleCount = zeros(surfaceCount, 1);
receiveToApplyMeanSeconds = nan(surfaceCount, 1);
receiveToApplyStdSeconds = nan(surfaceCount, 1);
receiveToApplyMedianSeconds = nan(surfaceCount, 1);
receiveToApplyP95Seconds = nan(surfaceCount, 1);
receiveToApplyP99Seconds = nan(surfaceCount, 1);
receiveToApplyMaxSeconds = nan(surfaceCount, 1);

scheduledToApplySampleCount = zeros(surfaceCount, 1);
scheduledToApplyMeanSeconds = nan(surfaceCount, 1);
scheduledToApplyStdSeconds = nan(surfaceCount, 1);
scheduledToApplyMedianSeconds = nan(surfaceCount, 1);
scheduledToApplyP95Seconds = nan(surfaceCount, 1);
scheduledToApplyP99Seconds = nan(surfaceCount, 1);
scheduledToApplyMaxSeconds = nan(surfaceCount, 1);

scheduledToApplyUnmatchedCommandCount = unmatchedApplyCommandCount;
scheduledToApplyUnmatchedCommandFraction = unmatchedApplyCommandFraction;

for surfaceIndex = 1:surfaceCount
    hostSchedulingStats = computeLatencyStats(storage.hostSchedulingDelaySeconds(sampleIndices, surfaceIndex));
    rxStats = computeLatencyStats(storage.computerToArduinoRxLatencySeconds(sampleIndices, surfaceIndex));
    applyStats = computeLatencyStats(storage.computerToArduinoApplyLatencySeconds(sampleIndices, surfaceIndex));
    receiveToApplyStats = computeLatencyStats(storage.arduinoReceiveToApplyLatencySeconds(sampleIndices, surfaceIndex));
    scheduledToApplyStats = computeLatencyStats(storage.scheduledToApplyLatencySeconds(sampleIndices, surfaceIndex));

    [hostSchedulingSampleCount(surfaceIndex), hostSchedulingMeanSeconds(surfaceIndex), hostSchedulingStdSeconds(surfaceIndex), ...
        hostSchedulingMedianSeconds(surfaceIndex), hostSchedulingP95Seconds(surfaceIndex), hostSchedulingP99Seconds(surfaceIndex), ...
        hostSchedulingMaxSeconds(surfaceIndex)] = unpackLatencyStats(hostSchedulingStats);
    [rxSampleCount(surfaceIndex), rxMeanSeconds(surfaceIndex), rxStdSeconds(surfaceIndex), ...
        rxMedianSeconds(surfaceIndex), rxP95Seconds(surfaceIndex), rxP99Seconds(surfaceIndex), ...
        rxMaxSeconds(surfaceIndex)] = unpackLatencyStats(rxStats);
    [applySampleCount(surfaceIndex), applyMeanSeconds(surfaceIndex), applyStdSeconds(surfaceIndex), ...
        applyMedianSeconds(surfaceIndex), applyP95Seconds(surfaceIndex), applyP99Seconds(surfaceIndex), ...
        applyMaxSeconds(surfaceIndex)] = unpackLatencyStats(applyStats);
    [receiveToApplySampleCount(surfaceIndex), receiveToApplyMeanSeconds(surfaceIndex), receiveToApplyStdSeconds(surfaceIndex), ...
        receiveToApplyMedianSeconds(surfaceIndex), receiveToApplyP95Seconds(surfaceIndex), receiveToApplyP99Seconds(surfaceIndex), ...
        receiveToApplyMaxSeconds(surfaceIndex)] = unpackLatencyStats(receiveToApplyStats);
    [scheduledToApplySampleCount(surfaceIndex), scheduledToApplyMeanSeconds(surfaceIndex), scheduledToApplyStdSeconds(surfaceIndex), ...
        scheduledToApplyMedianSeconds(surfaceIndex), scheduledToApplyP95Seconds(surfaceIndex), scheduledToApplyP99Seconds(surfaceIndex), ...
        scheduledToApplyMaxSeconds(surfaceIndex)] = unpackLatencyStats(scheduledToApplyStats);
end

latencySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    dispatchedCommandCount, ...
    matchedRxCommandCount, ...
    unmatchedRxCommandCount, ...
    unmatchedRxCommandFraction, ...
    matchedApplyCommandCount, ...
    unmatchedApplyCommandCount, ...
    unmatchedApplyCommandFraction, ...
    hostSchedulingSampleCount, hostSchedulingMeanSeconds, hostSchedulingStdSeconds, ...
    hostSchedulingMedianSeconds, hostSchedulingP95Seconds, hostSchedulingP99Seconds, hostSchedulingMaxSeconds, ...
    rxSampleCount, rxMeanSeconds, rxStdSeconds, rxMedianSeconds, rxP95Seconds, rxP99Seconds, rxMaxSeconds, ...
    applySampleCount, applyMeanSeconds, applyStdSeconds, applyMedianSeconds, applyP95Seconds, applyP99Seconds, applyMaxSeconds, ...
    receiveToApplySampleCount, receiveToApplyMeanSeconds, receiveToApplyStdSeconds, receiveToApplyMedianSeconds, ...
    receiveToApplyP95Seconds, receiveToApplyP99Seconds, receiveToApplyMaxSeconds, ...
    scheduledToApplySampleCount, scheduledToApplyUnmatchedCommandCount, scheduledToApplyUnmatchedCommandFraction, ...
    scheduledToApplyMeanSeconds, scheduledToApplyStdSeconds, scheduledToApplyMedianSeconds, ...
    scheduledToApplyP95Seconds, scheduledToApplyP99Seconds, scheduledToApplyMaxSeconds, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'DispatchedCommandCount', ...
        'MatchedRxCommandCount', ...
        'UnmatchedRxCommandCount', ...
        'UnmatchedRxCommandFraction', ...
        'MatchedApplyCommandCount', ...
        'UnmatchedApplyCommandCount', ...
        'UnmatchedApplyCommandFraction', ...
        'HostSchedulingDelaySampleCount', ...
        'HostSchedulingDelayMean_s', ...
        'HostSchedulingDelayStd_s', ...
        'HostSchedulingDelayMedian_s', ...
        'HostSchedulingDelayP95_s', ...
        'HostSchedulingDelayP99_s', ...
        'HostSchedulingDelayMax_s', ...
        'ComputerToArduinoRxLatencySampleCount', ...
        'ComputerToArduinoRxLatencyMean_s', ...
        'ComputerToArduinoRxLatencyStd_s', ...
        'ComputerToArduinoRxLatencyMedian_s', ...
        'ComputerToArduinoRxLatencyP95_s', ...
        'ComputerToArduinoRxLatencyP99_s', ...
        'ComputerToArduinoRxLatencyMax_s', ...
        'ComputerToArduinoApplyLatencySampleCount', ...
        'ComputerToArduinoApplyLatencyMean_s', ...
        'ComputerToArduinoApplyLatencyStd_s', ...
        'ComputerToArduinoApplyLatencyMedian_s', ...
        'ComputerToArduinoApplyLatencyP95_s', ...
        'ComputerToArduinoApplyLatencyP99_s', ...
        'ComputerToArduinoApplyLatencyMax_s', ...
        'ArduinoReceiveToApplyLatencySampleCount', ...
        'ArduinoReceiveToApplyLatencyMean_s', ...
        'ArduinoReceiveToApplyLatencyStd_s', ...
        'ArduinoReceiveToApplyLatencyMedian_s', ...
        'ArduinoReceiveToApplyLatencyP95_s', ...
        'ArduinoReceiveToApplyLatencyP99_s', ...
        'ArduinoReceiveToApplyLatencyMax_s', ...
        'ScheduledToApplyLatencySampleCount', ...
        'ScheduledToApplyUnmatchedCommandCount', ...
        'ScheduledToApplyUnmatchedCommandFraction', ...
        'ScheduledToApplyLatencyMean_s', ...
        'ScheduledToApplyLatencyStd_s', ...
        'ScheduledToApplyLatencyMedian_s', ...
        'ScheduledToApplyLatencyP95_s', ...
        'ScheduledToApplyLatencyP99_s', ...
        'ScheduledToApplyLatencyMax_s'});
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

%% =============================================================================
% 6) Export, Profile Builders, and Utility Helpers
% =============================================================================
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

% Legacy sheet name kept, but now RX-only by definition.
writetable( ...
    timetableToExportTable(runData.logs.computerToArduinoLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ComputerToArduinoLatency');

writetable( ...
    timetableToExportTable(runData.logs.hostSchedulingDelay), ...
    workbookPath, ...
    'Sheet', ...
    'HostSchedulingDelay');

writetable( ...
    timetableToExportTable(runData.logs.computerToArduinoRxLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ComputerToArduinoRxLatency');

writetable( ...
    timetableToExportTable(runData.logs.computerToArduinoApplyLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ComputerToArduinoApplyLatency');

writetable( ...
    timetableToExportTable(runData.logs.arduinoReceiveToApplyLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ArduinoReceiveToApplyLatency');

writetable( ...
    timetableToExportTable(runData.logs.scheduledToApplyLatency), ...
    workbookPath, ...
    'Sheet', ...
    'ScheduledToApplyLatency');

writetable(runData.logs.latencySummary, workbookPath, 'Sheet', 'LatencySummary');
writetable(runData.logs.integritySummary, workbookPath, 'Sheet', 'IntegritySummary');
writetable(runData.logs.profileEvents, workbookPath, 'Sheet', 'ProfileEvents');
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
    "ArduinoTransport", "OperatingMode", formatSettingValue(config.arduinoTransport.operatingMode); ...
    "ArduinoTransport", "CommandEncoding", formatSettingValue(config.arduinoTransport.commandEncoding); ...
    "ArduinoTransport", "LoggerPort", formatSettingValue(config.arduinoTransport.loggerPort); ...
    "ArduinoTransport", "LoggerProbeTimeoutSeconds", formatSettingValue(config.arduinoTransport.loggerProbeTimeoutSeconds); ...
    "ArduinoTransport", "LoggerTimeoutSeconds", formatSettingValue(config.arduinoTransport.loggerTimeoutSeconds); ...
    "ArduinoTransport", "SyncReplyTimeoutSeconds", formatSettingValue(config.arduinoTransport.syncReplyTimeoutSeconds); ...
    "ArduinoTransport", "SyncCountBeforeRun", formatSettingValue(config.arduinoTransport.syncCountBeforeRun); ...
    "ArduinoTransport", "SyncCountAfterRun", formatSettingValue(config.arduinoTransport.syncCountAfterRun); ...
    "ArduinoTransport", "SyncPauseSeconds", formatSettingValue(config.arduinoTransport.syncPauseSeconds); ...
    "ArduinoTransport", "PostRunSettleSeconds", formatSettingValue(config.arduinoTransport.postRunSettleSeconds); ...
    "ArduinoTransport", "ClearLogsBeforeRun", formatSettingValue(config.arduinoTransport.clearLogsBeforeRun); ...
    "ArduinoTransport", "LoggerOutputFolder", formatSettingValue(config.arduinoTransport.loggerOutputFolder); ...
    "ArduinoTransport", "CaptureSucceeded", formatSettingValue(config.arduinoTransport.captureSucceeded); ...
    "ArduinoTransport", "CaptureMessage", formatSettingValue(config.arduinoTransport.captureMessage); ...
    "ArduinoTransport", "CaptureRowCount", formatSettingValue(config.arduinoTransport.captureRowCount); ...
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
    "Profile", "EventHoldSeconds", formatSettingValue(profile.eventHoldSeconds); ...
    "Profile", "EventNeutralHoldSeconds", formatSettingValue(profile.eventNeutralHoldSeconds); ...
    "Profile", "EventDwellSeconds", formatSettingValue(profile.eventDwellSeconds); ...
    "Profile", "EventRandomJitterSeconds", formatSettingValue(profile.eventRandomJitterSeconds); ...
    "Profile", "RandomSeed", formatSettingValue(profile.randomSeed); ...
    "Profile", "EffectiveRandomSeed", formatSettingValue(profile.effectiveRandomSeed); ...
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

function cleanupResources(commandInterface, config)
if ~isempty(fieldnames(commandInterface)) && config.returnToNeutralOnExit
    try
        moveServosToNeutral(commandInterface, config.servoNeutralPositions);
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
    % Each segment changes one commanded vector, making analyser transitions
    % attributable while still exercising coupled roll/pitch/yaw commands.
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

function echoImportTable = reshapeEchoImportTable(echoImportTable)
if isempty(echoImportTable)
    return;
end

echoImportTable.surface_name = reshape(string(echoImportTable.surface_name), [], 1);
echoImportTable.command_sequence = reshape(double(echoImportTable.command_sequence), [], 1);
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

function integritySummary = buildIntegritySummary(commandLookup, echoAssignments, echoImportTable, config)
surfaceCount = numel(config.surfaceNames);

dispatchedCommandCount = zeros(surfaceCount, 1);
matchedRxCommandCount = zeros(surfaceCount, 1);
matchedApplyCommandCount = zeros(surfaceCount, 1);
unmatchedRxCommandCount = zeros(surfaceCount, 1);
unmatchedRxCommandFraction = nan(surfaceCount, 1);
unmatchedApplyCommandCount = zeros(surfaceCount, 1);
unmatchedApplyCommandFraction = nan(surfaceCount, 1);
duplicateTelemetryKeyCount = zeros(surfaceCount, 1);
duplicateTelemetryKeyFraction = nan(surfaceCount, 1);
unexpectedTelemetryRowCount = zeros(surfaceCount, 1);
unexpectedTelemetryRowFraction = nan(surfaceCount, 1);
nonMonotonicSequenceCount = zeros(surfaceCount, 1);
nonMonotonicSequenceFraction = nan(surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceName = config.surfaceNames(surfaceIndex);
    commandMask = commandLookup.surface_name == surfaceName;
    telemetryMask = echoImportTable.surface_name == surfaceName;
    dispatchSequences = commandLookup.command_sequence(commandMask);
    telemetrySequences = echoImportTable.command_sequence(telemetryMask);

    dispatchedCommandCount(surfaceIndex) = nnz(commandMask);

    matchedRxMask = ...
        echoAssignments.surface_name == surfaceName & ...
        (isfinite(echoAssignments.computer_to_arduino_rx_latency_s) | ...
        isfinite(echoAssignments.arduino_echo_time_s));
    matchedApplyMask = ...
        echoAssignments.surface_name == surfaceName & ...
        isfinite(echoAssignments.computer_to_arduino_apply_latency_s);

    matchedRxCommandCount(surfaceIndex) = nnz(matchedRxMask);
    matchedApplyCommandCount(surfaceIndex) = nnz(matchedApplyMask);
    unmatchedRxCommandCount(surfaceIndex) = ...
        max(0, dispatchedCommandCount(surfaceIndex) - matchedRxCommandCount(surfaceIndex));
    unmatchedApplyCommandCount(surfaceIndex) = ...
        max(0, dispatchedCommandCount(surfaceIndex) - matchedApplyCommandCount(surfaceIndex));

    if dispatchedCommandCount(surfaceIndex) > 0
        unmatchedRxCommandFraction(surfaceIndex) = ...
            unmatchedRxCommandCount(surfaceIndex) ./ dispatchedCommandCount(surfaceIndex);
        unmatchedApplyCommandFraction(surfaceIndex) = ...
            unmatchedApplyCommandCount(surfaceIndex) ./ dispatchedCommandCount(surfaceIndex);
    end

    if ~isempty(telemetrySequences)
        [~, ~, groupIndex] = unique(telemetrySequences, "stable");
        groupCounts = accumarray(groupIndex, 1);
        duplicateTelemetryKeyCount(surfaceIndex) = sum(max(groupCounts - 1, 0));
        duplicateTelemetryKeyFraction(surfaceIndex) = ...
            duplicateTelemetryKeyCount(surfaceIndex) ./ numel(telemetrySequences);

        unexpectedTelemetryRowCount(surfaceIndex) = ...
            sum(~ismember(telemetrySequences, dispatchSequences));
        unexpectedTelemetryRowFraction(surfaceIndex) = ...
            unexpectedTelemetryRowCount(surfaceIndex) ./ numel(telemetrySequences);

        if numel(telemetrySequences) >= 2
            nonMonotonicSequenceCount(surfaceIndex) = ...
                sum(diff(telemetrySequences) <= 0);
            nonMonotonicSequenceFraction(surfaceIndex) = ...
                nonMonotonicSequenceCount(surfaceIndex) ./ (numel(telemetrySequences) - 1);
        end
    end
end

integritySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    dispatchedCommandCount, ...
    matchedRxCommandCount, ...
    matchedApplyCommandCount, ...
    unmatchedRxCommandCount, ...
    unmatchedRxCommandFraction, ...
    unmatchedApplyCommandCount, ...
    unmatchedApplyCommandFraction, ...
    duplicateTelemetryKeyCount, ...
    duplicateTelemetryKeyFraction, ...
    unexpectedTelemetryRowCount, ...
    unexpectedTelemetryRowFraction, ...
    nonMonotonicSequenceCount, ...
    nonMonotonicSequenceFraction, ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'DispatchedCommandCount', ...
        'MatchedRxCommandCount', ...
        'MatchedApplyCommandCount', ...
        'UnmatchedRxCommandCount', ...
        'UnmatchedRxCommandFraction', ...
        'UnmatchedApplyCommandCount', ...
        'UnmatchedApplyCommandFraction', ...
        'DuplicateTelemetryKeyCount', ...
        'DuplicateTelemetryKeyFraction', ...
        'UnexpectedTelemetryRowCount', ...
        'UnexpectedTelemetryRowFraction', ...
        'NonMonotonicSequenceCount', ...
        'NonMonotonicSequenceFraction'});
end

function integritySummary = buildEmptyIntegritySummary(config)
surfaceCount = numel(config.surfaceNames);
integritySummary = table( ...
    config.surfaceNames, ...
    config.activeSurfaceMask, ...
    zeros(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    nan(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    nan(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    nan(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    nan(surfaceCount, 1), ...
    zeros(surfaceCount, 1), ...
    nan(surfaceCount, 1), ...
    'VariableNames', { ...
        'SurfaceName', ...
        'IsActive', ...
        'DispatchedCommandCount', ...
        'MatchedRxCommandCount', ...
        'MatchedApplyCommandCount', ...
        'UnmatchedRxCommandCount', ...
        'UnmatchedRxCommandFraction', ...
        'UnmatchedApplyCommandCount', ...
        'UnmatchedApplyCommandFraction', ...
        'DuplicateTelemetryKeyCount', ...
        'DuplicateTelemetryKeyFraction', ...
        'UnexpectedTelemetryRowCount', ...
        'UnexpectedTelemetryRowFraction', ...
        'NonMonotonicSequenceCount', ...
        'NonMonotonicSequenceFraction'});
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
    error("Arduino_Test:InvalidConfigType", "%s must be a text scalar.", fieldName);
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

function value = getNonnegativeIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "nonnegative"}, char(mfilename), char(fieldName));
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

function commandLine = buildNanoLoggerSetAllCommand(surfaceNames, commandSequenceNumbers, servoPositions, activeSurfaceMask)
activeIndices = find(activeSurfaceMask);
surfaceCount = numel(activeIndices);

if surfaceCount == 0
    error("Arduino_Test:NoActiveSurfaces", "SET_ALL requires at least one active surface.");
end

sampleSequence = commandSequenceNumbers(activeIndices(1));
if ~all(commandSequenceNumbers(activeIndices) == sampleSequence)
    % Guard against sequence gaps from future sample-counter changes.
    error("Arduino_Test:InconsistentSetAllSequence", ...
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
    error("Arduino_Test:NoActiveSurfaces", "Binary vector command requires at least one active surface.");
end

sampleSequence = commandSequenceNumbers(activeIndices(1));
if ~all(commandSequenceNumbers(activeIndices) == sampleSequence)
    error("Arduino_Test:InconsistentBinaryVectorSequence", ...
        "All active surfaces must share the same sample sequence.");
end

surfaceMask = uint8(0);
for surfaceIndex = activeIndices(:).'
    surfaceMask = bitor(surfaceMask, bitshift(uint8(1), surfaceIndex - 1));
end

% The firmware expects normalized positions as little-endian uint16 values
% following a surface bitmask; clipping occurs before encoding.
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
