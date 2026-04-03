function runData = Transmitter_Test(config)
%TRANSMITTER_TEST Execute a wired serial-to-PPM trainer-link latency test.
%   runData = Transmitter_Test(config) drives the Arduino Uno over serial,
%   commands the four logical control surfaces as one compact vector, logs
%   RX_EVENT and COMMIT_EVENT telemetry, and optionally matches imported
%   downstream capture tables for trainer and receiver latency analysis.
%   Default transmitter wiring uses D3 for trainer PPM and D2 for the
%   reference strobe.
arguments
    config (1,1) struct = struct()
end

config = normalizeTransmitterConfig(config);
runPlan = buildTransmitterRunPlan(config);
runData = initializeTransmitterRunData(config);
runData.runInfo.startTime = datetime("now");
runData.runInfo.scheduledDurationSeconds = runPlan.scheduledDurationSeconds;

assignin("base", "TransmitterTestLatestState", struct([]));
assignin("base", "TransmitterTestRunData", runData);

commandInterface = struct();
sigrokSession = createEmptySigrokSession();
cleanupHandle = onCleanup(@() cleanupTransmitterResources(commandInterface, config, sigrokSession));

try
    [commandInterface, arduinoInfo] = connectToTransmitter(config);
    runData.config = config;
    runData.connectionInfo.arduino = arduinoInfo;
    assignin("base", "TransmitterTestRunData", runData);

    if ~arduinoInfo.isConnected
        runData.runInfo.status = "arduino_connection_failed";
        runData.runInfo.reason = arduinoInfo.connectionMessage;
        runData.runInfo.stopTime = datetime("now");
        assignin("base", "TransmitterTestRunData", runData);
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

    assignin("base", "TransmitterTestRunData", runData);
    assignin("base", "TransmitterTestLatestState", buildTransmitterLatestState(storage, config, storage.sampleCount));

    clear cleanupHandle
    cleanupTransmitterResources(commandInterface, config, sigrokSession);
catch executionException
    runData.runInfo.status = "failed";
    runData.runInfo.reason = string(executionException.message);
    runData.runInfo.stopTime = datetime("now");
    assignin("base", "TransmitterTestRunData", runData);
    rethrow(executionException);
end
end

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

validProfileTypes = ["latency_step_train", "latency_isolated_step", "sine", "square", "doublet", "custom", "function"];
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

        % Build and send one datagram for the whole active actuator vector.
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
            loggerSession = appendNanoLoggerDispatchRow( ...
                loggerSession, ...
                surfaceNames(surfaceIndex), ...
                commandSequenceNumbers(surfaceIndex), ...
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
    % Use host and board midpoints when direct round-trip timestamps exist.
    hostReferenceUs = 0.5 .* (hostTxUs(midpointMask) + hostRxUs(midpointMask));
    boardReferenceUs = 0.5 .* (boardRxUs(midpointMask) + boardTxUs(midpointMask));
else
    % Fallback for legacy captures without immediate SYNC_EVENT round trips.
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
    % This should never happen with the current sample-wise counter update,
    % but guard it explicitly.
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
    "baudRate", getPositiveIntegerField(transportConfig, "baudRate", 115200), ...
    "commandEncoding", canonicalizeNanoLoggerCommandEncoding( ...
        getTextScalarField(transportConfig, "commandEncoding", "binary_vector")), ...
    "serialResetSeconds", getNonnegativeScalarField(transportConfig, "serialResetSeconds", 4.0), ...
    "probeTimeoutSeconds", getPositiveScalarField(transportConfig, "probeTimeoutSeconds", 6.0), ...
    "helloRetrySeconds", getPositiveScalarField(transportConfig, "helloRetrySeconds", 0.25), ...
    "linePollPauseSeconds", getPositiveScalarField(transportConfig, "linePollPauseSeconds", 0.01), ...
    "lineIdleTimeoutSeconds", getPositiveScalarField(transportConfig, "lineIdleTimeoutSeconds", 0.25), ...
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
    "frameLengthUs", getPositiveIntegerField(trainerPpmConfig, "frameLengthUs", 22000), ...
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
    config.commandProfile.type = "latency_isolated_step";
end
% Match the controller-workbook latency-step structure explicitly, while
% still respecting the slower Uno trainer cadence needed for frame-based
% PPM commit and downstream observability.
if ~isfield(commandProfileConfig, "randomSeed")
    config.commandProfile.randomSeed = 5;
end
if ~isfield(commandProfileConfig, "sampleTimeSeconds")
    minimumRecommendedSampleTimeSeconds = ...
        max(0.05, 2.0 .* double(config.trainerPpm.frameLengthUs) ./ 1e6 + 0.006);
    config.commandProfile.sampleTimeSeconds = minimumRecommendedSampleTimeSeconds;
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
    "captureDurationSeconds", getOptionalScalarField(logicAnalyzerConfig, "captureDurationSeconds", NaN), ...
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
    "referenceDebounceUs", getPositiveIntegerField(matchingConfig, "referenceDebounceUs", 100), ...
    "ppmChangeThresholdUs", getPositiveIntegerField(matchingConfig, "ppmChangeThresholdUs", 4), ...
    "receiverChangeThresholdUs", getPositiveIntegerField(matchingConfig, "receiverChangeThresholdUs", 8), ...
    "transitionLeadSeconds", getNonnegativeScalarField(matchingConfig, "transitionLeadSeconds", 0.005), ...
    "transitionTargetToleranceUs", getPositiveIntegerField(matchingConfig, "transitionTargetToleranceUs", 80), ...
    "transitionPreviousToleranceUs", getPositiveIntegerField(matchingConfig, "transitionPreviousToleranceUs", 25), ...
    "referenceAssociationWindowSeconds", getPositiveScalarField( ...
        matchingConfig, ...
        "referenceAssociationWindowSeconds", ...
        max(0.02, 1.25 .* double(config.trainerPpm.frameLengthUs) ./ 1e6)), ...
    "maxResponseWindowSeconds", getPositiveScalarField(matchingConfig, "maxResponseWindowSeconds", 0.12));

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
validateattributes(config.servoNeutralPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoNeutralPositions');
validateattributes(config.servoUnitsPerDegree, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoUnitsPerDegree');
validateattributes(config.servoMinimumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMinimumPositions');
validateattributes(config.servoMaximumPositions, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'servoMaximumPositions');
validateattributes(config.commandDeflectionScales, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionScales');
validateattributes(config.commandDeflectionOffsetsDegrees, {"numeric"}, {"real", "finite", "column", "numel", surfaceCount}, char(mfilename), 'commandDeflectionOffsetsDegrees');

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
    desiredDeflectionsDegrees = zeros(1, surfaceCount);
    desiredDeflectionsDegrees(config.activeSurfaceMask.') = ...
        config.commandDeflectionScales(config.activeSurfaceMask).' .* baseCommandDegrees + ...
        config.commandDeflectionOffsetsDegrees(config.activeSurfaceMask).';

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

    assignin("base", "TransmitterTestLatestState", buildTransmitterLatestState(storage, config, sampleIndex));
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
    "referenceStrobeSeconds", nan(sampleCount, surfaceCount), ...
    "trainerPpmSeconds", nan(sampleCount, surfaceCount), ...
    "receiverResponseSeconds", nan(sampleCount, surfaceCount), ...
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
    writeline(serialObject, sprintf("SYNC,%u,%u", uint32(syncIndex), hostNowUs(loggerSession.hostTimer)));
    loggerSession = pauseAndDrainTransmitterTelemetry( ...
        serialObject, ...
        loggerSession, ...
        config.arduinoTransport.syncPauseSeconds, ...
        config.arduinoTransport.linePollPauseSeconds);
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
while true
    loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
    remainingSeconds = targetTimeSeconds - toc(referenceTimer);
    if remainingSeconds <= 0
        return;
    end
    pause(min(pauseSeconds, remainingSeconds));
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

if serialObject.NumBytesAvailable > 0
    loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
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

loggerSession = drainTransmitterTelemetry(serialObject, loggerSession);
writeStopSeconds = max(0, toc(loggerSession.hostTimer) - loggerSession.testStartOffsetSeconds);
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
    if loggerSession.telemetryLineCount + 1 > numel(loggerSession.telemetryLineText)
        error("Transmitter_Test:TelemetryCapacityExceeded", "Serial telemetry log capacity was exceeded.");
    end

    loggerSession.telemetryLineCount = loggerSession.telemetryLineCount + 1;
    telemetryRow = loggerSession.telemetryLineCount;
    loggerSession.telemetryLineText(telemetryRow) = receivedLines(lineIndex);
    loggerSession.telemetryHostRxUs(telemetryRow) = hostRxUs;

    telemetryParts = split(receivedLines(lineIndex), ",");
    if isempty(telemetryParts)
        continue;
    end

    switch telemetryParts(1)
        case "RX_EVENT"
            if numel(telemetryParts) < 8
                continue;
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
                continue;
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
            loggerSession.commitStrobeUs(rowIndex) = str2double(telemetryParts(6));
            loggerSession.commitFrameIndex(rowIndex) = str2double(telemetryParts(7));
            loggerSession.commitPpmUs(rowIndex, :) = ...
                double(str2double(telemetryParts(8:15))).';

        case "SYNC_EVENT"
            if numel(telemetryParts) < 5
                continue;
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
    writeline(serialObject, sprintf("SYNC,%u,%u", syncId, hostNowUs(loggerSession.hostTimer)));
    loggerSession = pauseAndDrainTransmitterTelemetry( ...
        serialObject, ...
        loggerSession, ...
        config.arduinoTransport.syncPauseSeconds, ...
        config.arduinoTransport.linePollPauseSeconds);
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

if loggerSession.boardSyncCount == 0
    error("Transmitter_Test:MissingSyncTelemetry", "No SYNC_EVENT lines were received from the transmitter.");
end

loggerData = struct( ...
    "testStartOffsetUs", double(loggerSession.testStartOffsetUs), ...
    "testStartOffsetSeconds", loggerSession.testStartOffsetSeconds, ...
    "hostDispatchLog", buildHostDispatchLogFromSession(loggerSession), ...
    "boardRxLog", buildBoardRxLogFromSession(loggerSession, config.surfaceNames), ...
    "boardCommitLog", buildBoardCommitLogFromSession(loggerSession), ...
    "boardSyncLog", buildBoardSyncLogFromSession(loggerSession), ...
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

if strlength(config.logicAnalyzer.deviceId) > 0
    validationCommand = strjoin([ ...
        quoteWindowsArgument(sigrokCliPath), ...
        "--driver", quoteWindowsArgument(driverSpec), ...
        "--show"], " ");
    [status, outputText] = runWindowsCommand(validationCommand);
    raiseSigrokUsbAccessErrorIfNeeded(outputText);
    if status ~= 0
        error("Transmitter_Test:SigrokDeviceValidationFailed", "sigrok-cli device validation failed. %s", strtrim(outputText));
    end
else
    [status, outputText] = runWindowsCommand(quoteWindowsArgument(sigrokCliPath) + " --scan");
    raiseSigrokUsbAccessErrorIfNeeded(outputText);
    if status ~= 0 || ~contains(lower(string(outputText)), lower(config.logicAnalyzer.deviceDriver))
        error("Transmitter_Test:SigrokDeviceValidationFailed", "sigrok-cli scan did not confirm driver %s.", char(config.logicAnalyzer.deviceDriver));
    end
end
end

function raiseSigrokUsbAccessErrorIfNeeded(outputText)
outputText = string(outputText);
outputTextLower = lower(outputText);

if contains(outputTextLower, "libusb_error_access")
    error( ...
        "Transmitter_Test:SigrokUsbAccessDenied", ...
        [ ...
        "sigrok-cli could not open the logic analyser because Windows denied USB access (%s). " + ...
        "Close PulseView and any other analyser software, unplug/replug the analyser, and ensure the device uses a libusb-compatible driver such as WinUSB for sigrok."], ...
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
markStartMask = false(size(trainerStates));
if config.trainerPpm.idleHigh
    markStartMask(2:end) = diff(trainerStates) < 0;
else
    markStartMask(2:end) = diff(trainerStates) > 0;
end

markStartSamples = logicState.sampleIndex(markStartMask);
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
markWidthSamples = round(double(config.trainerPpm.markWidthUs) .* logicState.sampleRateHz ./ 1e6);

currentFrameIndex = 1;
currentChannelIndex = 1;
for slotIndex = 1:numel(slotSampleCounts)
    slotCount = slotSampleCounts(slotIndex);
    if slotCount > syncGapThresholdSamples
        currentFrameIndex = currentFrameIndex + 1;
        currentChannelIndex = 1;
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
            % Raw Uno trainer reconstruction yields the inter-mark interval.
            % Add the fixed PPM mark width so sample-based normalization
            % recovers the commanded slot width used by board-side commits.
            sampleCount(rowCount) = slotCount + markWidthSamples;
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

function boardCommitLog = buildBoardCommitLogFromSession(loggerSession)
boardCommitLog = table( ...
    loggerSession.commitSampleSequence(1:loggerSession.commitEventCount), ...
    loggerSession.commitActiveMask(1:loggerSession.commitEventCount), ...
    loggerSession.commitHostRxUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitBoardRxUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitBoardCommitUs(1:loggerSession.commitEventCount), ...
    loggerSession.commitBoardCommitUs(1:loggerSession.commitEventCount) - loggerSession.commitBoardRxUs(1:loggerSession.commitEventCount), ...
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
[clockSlope, clockIntercept] = estimateBoardToHostClockMap(boardSyncLog(:, {'sync_id', 'host_tx_us', 'host_rx_us', 'board_rx_us', 'board_tx_us'}));

rxHostUs = clockSlope .* double(boardRxLog.rx_us) + clockIntercept;
commitHostUs = clockSlope .* double(boardCommitLog.board_commit_us) + clockIntercept;
strobeHostUs = nan(height(boardCommitLog), 1);
strobeMask = isfinite(boardCommitLog.strobe_us) & boardCommitLog.strobe_us > 0;
strobeHostUs(strobeMask) = clockSlope .* double(boardCommitLog.strobe_us(strobeMask)) + clockIntercept;

minimumLatencyUs = 0.0;
if ~isempty(boardRxLog)
    joinedRx = innerjoin( ...
        loggerData.hostDispatchLog(:, {'surface_name', 'command_sequence', 'command_dispatch_us'}), ...
        boardRxLog(:, {'surface_name', 'command_sequence', 'rx_us'}), ...
        'Keys', {'surface_name', 'command_sequence'});
    if ~isempty(joinedRx)
        minimumLatencyUs = min(minimumLatencyUs, ...
            min(clockSlope .* double(joinedRx.rx_us) + clockIntercept - double(joinedRx.command_dispatch_us)));
    end
end

if ~isempty(boardCommitLog)
    dispatchBySample = unique(loggerData.hostDispatchLog(:, {'sample_sequence', 'command_dispatch_us'}), 'rows', 'stable');
    joinedCommit = innerjoin( ...
        dispatchBySample, ...
        boardCommitLog(:, {'sample_sequence', 'board_commit_us'}), ...
        'Keys', "sample_sequence");
    if ~isempty(joinedCommit)
        minimumLatencyUs = min(minimumLatencyUs, ...
            min(clockSlope .* double(joinedCommit.board_commit_us) + clockIntercept - double(joinedCommit.command_dispatch_us)));
    end
end

if minimumLatencyUs < 0
    rxHostUs = rxHostUs - minimumLatencyUs;
    commitHostUs = commitHostUs - minimumLatencyUs;
    strobeHostUs(strobeMask) = strobeHostUs(strobeMask) - minimumLatencyUs;
end

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
    storage.referenceStrobeSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.reference_strobe_s(rowIndex);
    storage.trainerPpmSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.trainer_ppm_s(rowIndex);
    storage.receiverResponseSeconds(sampleIndex, surfaceIndex) = storage.matchedEvents.receiver_response_s(rowIndex);
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
    candidateStop = [candidateStart, storage.referenceStrobeSeconds(sampleIndex, :), storage.receiverResponseSeconds(sampleIndex, :)];
    if any(isfinite(candidateStart))
        storage.boardReadStartSeconds(sampleIndex) = min(candidateStart, [], "omitnan");
    end
    if any(isfinite(candidateStop))
        storage.boardReadStopSeconds(sampleIndex) = max(candidateStop, [], "omitnan");
    end
end

storage.integritySummary = buildTransmitterIntegritySummary(storage, config);
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
        matchedEvents.board_commit_s(rowIndex) = commitUnique.commit_time_s(commitRow);
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

matchedEvents = matchDownstreamPulseCaptures(matchedEvents, trainerPpmCapture, receiverCapture, config);
end

function commitTable = deduplicateCommitTable(commitTable)
if isempty(commitTable)
    return;
end
[~, firstIndex] = unique(double(commitTable.sample_sequence), "stable");
commitTable = commitTable(sort(firstIndex), :);
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
    if referenceIndex > height(referenceCapture)
        return;
    end
    if referenceCapture.time_s(referenceIndex) > searchTime + referenceAssociationWindowSeconds
        continue;
    end
    referenceTime(commitIndex) = referenceCapture.time_s(referenceIndex);
    if ismember("sample_index", string(referenceCapture.Properties.VariableNames))
        referenceSampleIndex(commitIndex) = referenceCapture.sample_index(referenceIndex);
    end
    if ismember("sample_rate_hz", string(referenceCapture.Properties.VariableNames))
        referenceSampleRateHz(commitIndex) = referenceCapture.sample_rate_hz(referenceIndex);
    end
    referenceIndex = referenceIndex + 1;
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
trainerTables = buildPulseTransitionTablesBySurface( ...
    trainerPpmCapture, ...
    config.surfaceNames, ...
    config.matching.ppmChangeThresholdUs);
receiverTables = buildPulseTransitionTablesBySurface( ...
    receiverCapture, ...
    config.surfaceNames, ...
    config.matching.receiverChangeThresholdUs);
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
    expectedDeltaUs = expectedUs - previousExpectedUs(surfaceIdx);
    searchStartSeconds = selectDownstreamSearchAnchorTime( ...
        matchedEvents.board_commit_s(rowIndex), ...
        matchedEvents.reference_strobe_s(rowIndex));

    if matchedEvents.is_observable_downstream(rowIndex)
        [trainerTime, trainerPulse, trainerSampleIndex, trainerSampleRateHz, trainerNextIndex(surfaceIdx)] = findFirstTransitionAfterAnchor( ...
            trainerTables{surfaceIdx}, ...
            trainerNextIndex(surfaceIdx), ...
            searchStartSeconds, ...
            config.matching.maxResponseWindowSeconds, ...
            config.matching.transitionLeadSeconds);
        matchedEvents.trainer_ppm_s(rowIndex) = trainerTime;
        matchedEvents.trainer_ppm_us(rowIndex) = trainerPulse;
        matchedEvents.trainer_sample_index(rowIndex) = trainerSampleIndex;
        if isfinite(trainerSampleRateHz)
            matchedEvents.analyzer_sample_rate_hz(rowIndex) = trainerSampleRateHz;
        end
        if isfinite(trainerSampleIndex) && isfinite(trainerSampleRateHz)
            matchedEvents.trainer_time_from_samples_s(rowIndex) = trainerSampleIndex ./ trainerSampleRateHz;
        end

        receiverStartSeconds = searchStartSeconds;

        [receiverTime, receiverPulse, receiverSampleIndex, receiverSampleRateHz, receiverNextIndex(surfaceIdx)] = findFirstTransitionAfterAnchor( ...
            receiverTables{surfaceIdx}, ...
            receiverNextIndex(surfaceIdx), ...
            receiverStartSeconds, ...
            config.matching.maxResponseWindowSeconds, ...
            config.matching.transitionLeadSeconds);
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
        matchedEvents.computer_to_receiver_latency_s(rowIndex) = ...
            matchedEvents.receiver_response_s(rowIndex) - matchedEvents.command_dispatch_s(rowIndex);
        matchedEvents.scheduled_to_receiver_latency_s(rowIndex) = ...
            matchedEvents.receiver_response_s(rowIndex) - matchedEvents.scheduled_time_s(rowIndex);
    end
    matchedEvents.used_sample_based_timing(rowIndex) = ...
        isfinite(matchedEvents.reference_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.trainer_sample_index(rowIndex)) || ...
        isfinite(matchedEvents.receiver_sample_index(rowIndex));

    previousExpectedUs(surfaceIdx) = expectedUs;
end
end

function searchStartSeconds = selectDownstreamSearchAnchorTime(boardCommitSeconds, referenceSeconds)
if isfinite(referenceSeconds)
    searchStartSeconds = double(referenceSeconds);
elseif isfinite(boardCommitSeconds)
    searchStartSeconds = double(boardCommitSeconds);
else
    searchStartSeconds = NaN;
end
end

function splitTables = splitPulseCaptureBySurface(pulseCapture, surfaceNames)
splitTables = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    splitTables{surfaceIndex} = pulseCapture(pulseCapture.surface_name == surfaceNames(surfaceIndex), :);
end
end

function transitionTables = buildPulseTransitionTablesBySurface(pulseCapture, surfaceNames, transitionThresholdUs)
transitionTables = cell(numel(surfaceNames), 1);
for surfaceIndex = 1:numel(surfaceNames)
    surfaceCapture = pulseCapture(pulseCapture.surface_name == surfaceNames(surfaceIndex), :);
    transitionTables{surfaceIndex} = buildPulseTransitionTable(surfaceCapture, transitionThresholdUs);
end
end

function transitionTable = buildPulseTransitionTable(surfaceCapture, transitionThresholdUs)
transitionTable = table( ...
    strings(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', { ...
        'surface_name', ...
        'time_s', ...
        'pulse_us', ...
        'previous_pulse_us', ...
        'sample_index', ...
        'sample_rate_hz', ...
        'delta_pulse_us'});
if isempty(surfaceCapture) || height(surfaceCapture) < 2
    return;
end

currentPulseUs = double(surfaceCapture.pulse_us(2:end));
previousPulseUs = double(surfaceCapture.pulse_us(1:end-1));
deltaPulseUs = currentPulseUs - previousPulseUs;
keepMask = isfinite(deltaPulseUs) & abs(deltaPulseUs) >= transitionThresholdUs;
if ~any(keepMask)
    return;
end

transitionTable = table( ...
    surfaceCapture.surface_name(2:end), ...
    surfaceCapture.time_s(2:end), ...
    currentPulseUs, ...
    previousPulseUs, ...
    surfaceCapture.sample_index(2:end), ...
    surfaceCapture.sample_rate_hz(2:end), ...
    deltaPulseUs, ...
    'VariableNames', { ...
        'surface_name', ...
        'time_s', ...
        'pulse_us', ...
        'previous_pulse_us', ...
        'sample_index', ...
        'sample_rate_hz', ...
        'delta_pulse_us'});
transitionTable = transitionTable(keepMask, :);
end

function [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz, nextIndex] = findFirstTransitionAfterAnchor( ...
    surfaceCapture, ...
    startIndex, ...
    startTimeSeconds, ...
    maxWindowSeconds, ...
    leadSeconds)
matchedTime = NaN;
matchedPulse = NaN;
matchedSampleIndex = NaN;
matchedSampleRateHz = NaN;
nextIndex = startIndex;
if isempty(surfaceCapture)
    return;
end

rowIndex = max(1, startIndex);
windowStartSeconds = startTimeSeconds - leadSeconds;
windowEndSeconds = startTimeSeconds + maxWindowSeconds;
while rowIndex <= height(surfaceCapture) && surfaceCapture.time_s(rowIndex) < windowStartSeconds
    rowIndex = rowIndex + 1;
end

while rowIndex <= height(surfaceCapture)
    if surfaceCapture.time_s(rowIndex) > windowEndSeconds
        nextIndex = rowIndex;
        break;
    end

    [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz] = ...
        extractMatchedPulseRow(surfaceCapture, rowIndex);
    nextIndex = rowIndex + 1;
    return;
    rowIndex = rowIndex + 1;
end

nextIndex = rowIndex;
end

function [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz, nextIndex] = findFirstDirectionalTransitionAfterAnchor( ...
    surfaceCapture, ...
    startIndex, ...
    startTimeSeconds, ...
    expectedDeltaUs, ...
    maxWindowSeconds, ...
    leadSeconds)
matchedTime = NaN;
matchedPulse = NaN;
matchedSampleIndex = NaN;
matchedSampleRateHz = NaN;
nextIndex = startIndex;
if isempty(surfaceCapture) || ~isfinite(expectedDeltaUs) || expectedDeltaUs == 0
    return;
end

rowIndex = max(1, startIndex);
windowStartSeconds = startTimeSeconds - leadSeconds;
windowEndSeconds = startTimeSeconds + maxWindowSeconds;
while rowIndex <= height(surfaceCapture) && surfaceCapture.time_s(rowIndex) < windowStartSeconds
    rowIndex = rowIndex + 1;
end

expectedDirection = sign(expectedDeltaUs);
while rowIndex <= height(surfaceCapture)
    if surfaceCapture.time_s(rowIndex) > windowEndSeconds
        nextIndex = rowIndex;
        break;
    end

    if sign(surfaceCapture.delta_pulse_us(rowIndex)) == expectedDirection
        [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz] = ...
            extractMatchedPulseRow(surfaceCapture, rowIndex);
        nextIndex = rowIndex + 1;
        return;
    end
    rowIndex = rowIndex + 1;
end

nextIndex = rowIndex;
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
trainerTables = buildPulseTransitionTablesBySurface( ...
    trainerPpmCapture, ...
    config.surfaceNames, ...
    config.matching.ppmChangeThresholdUs);
receiverTables = buildPulseTransitionTablesBySurface( ...
    receiverCapture, ...
    config.surfaceNames, ...
    config.matching.receiverChangeThresholdUs);
trainerNextIndex = ones(numel(config.surfaceNames), 1);
receiverNextIndex = ones(numel(config.surfaceNames), 1);

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
        thresholdUs = config.matching.receiverChangeThresholdUs;
        if abs(deltaPulseUs) < thresholdUs
            continue;
        end

        eventCount = eventCount + 1;
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

        [trainerTime, trainerPulse, ~, ~, trainerNextIndex(surfaceIndex)] = findFirstDirectionalTransitionAfterAnchor( ...
            trainerTables{surfaceIndex}, ...
            trainerNextIndex(surfaceIndex), ...
            anchorTime, ...
            deltaPulseUs, ...
            config.matching.maxResponseWindowSeconds, ...
            config.matching.transitionLeadSeconds);
        if isfinite(trainerTime)
            directEvents.trainer_transition_s(eventCount) = trainerTime;
            directEvents.trainer_transition_us(eventCount) = trainerPulse;
            directEvents.anchor_to_trainer_latency_s(eventCount) = trainerTime - anchorTime;
        end
        [receiverTime, receiverPulse, ~, ~, receiverNextIndex(surfaceIndex)] = findFirstDirectionalTransitionAfterAnchor( ...
            receiverTables{surfaceIndex}, ...
            receiverNextIndex(surfaceIndex), ...
            anchorTime, ...
            deltaPulseUs, ...
            config.matching.maxResponseWindowSeconds, ...
            config.matching.transitionLeadSeconds);
        if isfinite(receiverTime)
            directEvents.receiver_transition_s(eventCount) = receiverTime;
            directEvents.receiver_transition_us(eventCount) = receiverPulse;
            directEvents.anchor_to_receiver_latency_s(eventCount) = receiverTime - anchorTime;
            if isfinite(directEvents.command_dispatch_s(eventCount))
                directEvents.computer_to_receiver_latency_s(eventCount) = ...
                    receiverTime - directEvents.command_dispatch_s(eventCount);
            end
            if isfinite(directEvents.scheduled_time_s(eventCount))
                directEvents.scheduled_to_receiver_latency_s(eventCount) = ...
                    receiverTime - directEvents.scheduled_time_s(eventCount);
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

function [matchedTime, matchedPulse, matchedSampleIndex, matchedSampleRateHz] = extractMatchedPulseRow(surfaceCapture, rowIndex)
matchedTime = surfaceCapture.time_s(rowIndex);
matchedPulse = surfaceCapture.pulse_us(rowIndex);
matchedSampleIndex = NaN;
matchedSampleRateHz = NaN;
if ismember("sample_index", string(surfaceCapture.Properties.VariableNames))
    matchedSampleIndex = surfaceCapture.sample_index(rowIndex);
end
if ismember("sample_rate_hz", string(surfaceCapture.Properties.VariableNames))
    matchedSampleRateHz = surfaceCapture.sample_rate_hz(rowIndex);
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

function logs = buildTransmitterLogs(storage, config)
sampleIndices = 1:storage.sampleCount;
surfaceNames = config.surfaceNames;
rowTimesSeconds = chooseFiniteRowTimes(storage.commandWriteStopSeconds(sampleIndices), storage.scheduledTimeSeconds(sampleIndices));
directLatencyEvents = buildReferenceAnchoredLatencyEvents( ...
    storage.rawLogs.hostDispatchLog, ...
    storage.rawLogs.boardRxLog, ...
    storage.rawLogs.boardCommitLog, ...
    storage.rawLogs.referenceCapture, ...
    storage.rawLogs.trainerPpmCapture, ...
    storage.rawLogs.receiverCapture, ...
    config);

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
    "ppmToReceiverLatency", ppmToReceiverLatency, ...
    "computerToReceiverLatency", computerToReceiverLatency, ...
    "scheduledToReceiverLatency", scheduledToReceiverLatency, ...
    "directLatencyEvents", directLatencyEvents, ...
    "matchedEvents", storage.matchedEvents, ...
    "latencySummary", buildTransmitterLatencySummary(storage, config, directLatencyEvents), ...
    "integritySummary", storage.integritySummary, ...
    "profileEvents", storage.profileInfo.profileEvents, ...
    "sampleSummary", sampleSummary, ...
    "hostDispatchLog", storage.rawLogs.hostDispatchLog, ...
    "serialTelemetryLog", storage.rawLogs.serialTelemetryLog);
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

function surfaceSummary = buildTransmitterSurfaceSummary(storage, config)
surfaceSummary = buildTransmitterIntegritySummary(storage, config);
latencySummary = buildTransmitterLatencySummary(storage, config);
surfaceSummary = [surfaceSummary latencySummary(:, 3:end)];
end

function surfaceSummary = buildTransmitterSurfaceSummaryFromLogs(logs)
surfaceSummary = logs.integritySummary;
surfaceSummary = [surfaceSummary logs.latencySummary(:, 3:end)];
end

function latestState = buildTransmitterLatestState(storage, config, sampleIndex)
surfaceCount = numel(config.surfaceNames);
surfaceStates = repmat(struct( ...
    "name", "", ...
    "desired_deg", NaN, ...
    "command_position", NaN, ...
    "received_position", NaN, ...
    "board_rx_s", NaN, ...
    "board_commit_s", NaN, ...
    "receiver_response_s", NaN), surfaceCount, 1);

for surfaceIndex = 1:surfaceCount
    surfaceStates(surfaceIndex).name = config.surfaceNames(surfaceIndex);
    surfaceStates(surfaceIndex).desired_deg = storage.desiredDeflectionsDegrees(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).command_position = storage.commandedServoPositions(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).received_position = storage.receivedPositionNorm(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).board_rx_s = storage.boardRxSeconds(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).board_commit_s = storage.boardCommitSeconds(sampleIndex, surfaceIndex);
    surfaceStates(surfaceIndex).receiver_response_s = storage.receiverResponseSeconds(sampleIndex, surfaceIndex);
end

latestState = struct( ...
    "sampleIndex", sampleIndex, ...
    "scheduledTimeSeconds", storage.scheduledTimeSeconds(sampleIndex), ...
    "commandWriteTimeSeconds", storage.commandWriteStopSeconds(sampleIndex), ...
    "boardReadTimeSeconds", storage.boardReadStopSeconds(sampleIndex), ...
    "surfaces", surfaceStates);
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
workbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");
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
writeWorkbookSheet(runData.logs.ppmToReceiverLatency, workbookPath, "PpmToReceiverLatency");
writeWorkbookSheet(runData.logs.computerToReceiverLatency, workbookPath, "ComputerToReceiverLatency");
writeWorkbookSheet(runData.logs.scheduledToReceiverLatency, workbookPath, "ScheduledToReceiverLatency");
writeWorkbookSheet(runData.logs.directLatencyEvents, workbookPath, "DirectLatencyEvents");
writeWorkbookSheet(runData.logs.matchedEvents, workbookPath, "MatchedEvents");
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
    "Matching", "PpmChangeThresholdUs", formatSettingValue(config.matching.ppmChangeThresholdUs); ...
    "Matching", "ReceiverChangeThresholdUs", formatSettingValue(config.matching.receiverChangeThresholdUs); ...
    "Matching", "TransitionLeadSeconds", formatSettingValue(config.matching.transitionLeadSeconds); ...
    "Matching", "TransitionTargetToleranceUs", formatSettingValue(config.matching.transitionTargetToleranceUs); ...
    "Matching", "TransitionPreviousToleranceUs", formatSettingValue(config.matching.transitionPreviousToleranceUs); ...
    "Matching", "MaxResponseWindowSeconds", formatSettingValue(config.matching.maxResponseWindowSeconds)};
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
