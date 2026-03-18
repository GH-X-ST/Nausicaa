function artifacts = Nano33IoT_Echo_Client_Example(config)
%NANO33IOT_ECHO_CLIENT_EXAMPLE Drive the standalone Nano UDP echo logger.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

hostTimer = tic;
[socket, remoteAddress] = createUdpSocket(config.ipAddress, config.port, config.timeoutSeconds);
cleanupHandle = onCleanup(@() socket.close());

sendControlBurst(socket, remoteAddress, config.port, "CLEAR_LOGS", 3, 0.02);

dispatchLog = table( ...
    strings(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', { ...
        'surface_name', ...
        'command_sequence', ...
        'command_dispatch_us', ...
        'position_norm'});

for syncIndex = 1:config.syncCountBeforeRun
    sendUdpPayload(socket, remoteAddress, config.port, composeSyncPayload(uint32(syncIndex), hostTimer));
    pause(config.syncPauseSeconds);
end

for planIndex = 1:height(config.commandPlan)
    dispatchUs = hostNowUs(hostTimer);
    commandPayload = compose( ...
        "SET,%s,%u,%.6f", ...
        config.commandPlan.surface_name(planIndex), ...
        uint32(config.commandPlan.command_sequence(planIndex)), ...
        config.commandPlan.position_norm(planIndex));
    sendUdpPayload(socket, remoteAddress, config.port, commandPayload);

    dispatchLog(end + 1, :) = { ...
        config.commandPlan.surface_name(planIndex), ...
        double(config.commandPlan.command_sequence(planIndex)), ...
        double(dispatchUs), ...
        config.commandPlan.position_norm(planIndex)}; %#ok<AGROW>

    pause(config.commandPlan.wait_after_s(planIndex));
end

if config.returnToNeutralAfterRun
    sendControlBurst(socket, remoteAddress, config.port, "SET_NEUTRAL", 2, 0.02);
end

pause(config.postRunSettleSeconds);

for syncIndex = 1:config.syncCountAfterRun
    syncId = uint32(config.syncCountBeforeRun + syncIndex);
    sendUdpPayload(socket, remoteAddress, config.port, composeSyncPayload(syncId, hostTimer));
    pause(config.syncPauseSeconds);
end

[telemetryLines, boardCommandLog, boardSyncLog] = collectTelemetry(socket, config.timeoutSeconds);
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

artifacts = struct( ...
    "dispatchLogPath", string(fullfile(config.outputFolder, "host_dispatch_log.csv")), ...
    "syncRoundTripPath", string(fullfile(config.outputFolder, "host_sync_roundtrip.csv")), ...
    "commandLogPath", string(fullfile(config.outputFolder, "board_command_log.csv")), ...
    "syncLogPath", string(fullfile(config.outputFolder, "board_sync_log.csv")), ...
    "telemetryLogPath", string(fullfile(config.outputFolder, "udp_telemetry_log.txt")));

writetable(dispatchLog, artifacts.dispatchLogPath);
writetable(syncRoundTripLog, artifacts.syncRoundTripPath);
writetable(boardCommandLog, artifacts.commandLogPath);
writetable(boardSyncLog, artifacts.syncLogPath);
if ~isempty(telemetryLines)
    writelines(telemetryLines, artifacts.telemetryLogPath);
end

clear cleanupHandle
socket.close();
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));

config.ipAddress = getTextScalarField(config, "ipAddress", "192.168.0.33");
config.port = getPositiveIntegerField(config, "port", 9500);
config.timeoutSeconds = getPositiveScalarField(config, "timeoutSeconds", 5);
config.syncCountBeforeRun = getNonnegativeIntegerField(config, "syncCountBeforeRun", 10);
config.syncCountAfterRun = getNonnegativeIntegerField(config, "syncCountAfterRun", 10);
config.syncPauseSeconds = getNonnegativeScalarField(config, "syncPauseSeconds", 0.05);
config.postRunSettleSeconds = getNonnegativeScalarField(config, "postRunSettleSeconds", 0.25);
config.returnToNeutralAfterRun = getLogicalField(config, "returnToNeutralAfterRun", true);
config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "output"));
config.commandPlan = normalizeCommandPlan(getFieldOrDefault(config, "commandPlan", table()));
end

function commandPlan = normalizeCommandPlan(commandPlan)
if isempty(commandPlan)
    commandPlan = table( ...
        ["Aileron_L"; "Aileron_L"; "Aileron_L"; "Aileron_L"], ...
        (1:4).', ...
        [0.50; 0.65; 0.35; 0.50], ...
        0.20 .* ones(4, 1), ...
        'VariableNames', {'surface_name', 'command_sequence', 'position_norm', 'wait_after_s'});
    return;
end

if ~istable(commandPlan)
    error("Nano33IoT_Echo_Client_Example:InvalidCommandPlan", "config.commandPlan must be a table.");
end

requiredVariables = ["surface_name", "command_sequence", "position_norm", "wait_after_s"];
missingVariables = requiredVariables(~ismember(requiredVariables, string(commandPlan.Properties.VariableNames)));
if ~isempty(missingVariables)
    error("Nano33IoT_Echo_Client_Example:MissingCommandPlanColumns", ...
        "config.commandPlan is missing: %s", ...
        char(join(missingVariables, ", ")));
end

commandPlan.surface_name = reshape(string(commandPlan.surface_name), [], 1);
commandPlan.command_sequence = reshape(double(commandPlan.command_sequence), [], 1);
commandPlan.position_norm = reshape(double(commandPlan.position_norm), [], 1);
commandPlan.wait_after_s = reshape(double(commandPlan.wait_after_s), [], 1);
end

function [socket, remoteAddress] = createUdpSocket(ipAddress, port, timeoutSeconds)
timeoutMilliseconds = int32(max(1, ceil(timeoutSeconds .* 1000)));
socket = javaObject("java.net.DatagramSocket");
socket.setSoTimeout(timeoutMilliseconds);
socket.setReceiveBufferSize(int32(262144));
remoteAddress = javaMethod("getByName", "java.net.InetAddress", char(ipAddress));
socket.connect(remoteAddress, int32(port));
end

function payload = composeSyncPayload(syncId, hostTimer)
payload = sprintf("SYNC,%u,%u", uint32(syncId), hostNowUs(hostTimer));
end

function sendControlBurst(socket, remoteAddress, port, payloadText, repeatCount, pauseSeconds)
for repeatIndex = 1:repeatCount
    sendUdpPayload(socket, remoteAddress, port, payloadText);
    if repeatIndex < repeatCount && pauseSeconds > 0
        pause(pauseSeconds);
    end
end
end

function sendUdpPayload(socket, remoteAddress, port, payloadText)
payloadBytes = uint8(char(string(payloadText)));
packet = javaObject( ...
    "java.net.DatagramPacket", ...
    int8(payloadBytes), ...
    numel(payloadBytes), ...
    remoteAddress, ...
    int32(port));
socket.send(packet);
end

function [telemetryLines, boardCommandLog, boardSyncLog] = collectTelemetry(socket, maxWaitSeconds)
telemetryLines = strings(0, 1);
collectStart = tic;
lastReceiveElapsedSeconds = 0;
idleTimeoutSeconds = 0.25;

while toc(collectStart) < maxWaitSeconds
    remainingSeconds = maxWaitSeconds - toc(collectStart);
    perReadTimeoutMilliseconds = int32(max(1, ceil(min(0.05, remainingSeconds) .* 1000)));
    socket.setSoTimeout(perReadTimeoutMilliseconds);

    receivePacket = javaObject("java.net.DatagramPacket", int8(zeros(1, 2048)), 2048);
    try
        socket.receive(receivePacket);
        packetBytes = uint8(receivePacket.getData());
        packetLength = double(receivePacket.getLength());
        packetBytes = reshape(packetBytes(1:packetLength), 1, []);
        telemetryLines(end + 1, 1) = string(strtrim(char(packetBytes))); %#ok<AGROW>
        lastReceiveElapsedSeconds = toc(collectStart);
    catch readException
        if ~(contains(string(readException.message), "timed out", 'IgnoreCase', true) || ...
                contains(string(readException.message), "timeout", 'IgnoreCase', true))
            rethrow(readException);
        end

        if toc(collectStart) >= idleTimeoutSeconds && ...
                (toc(collectStart) - lastReceiveElapsedSeconds) >= idleTimeoutSeconds
            break;
        end
    end
end

[boardCommandLog, boardSyncLog] = parseTelemetryLines(telemetryLines);
end

function [boardCommandLog, boardSyncLog] = parseTelemetryLines(telemetryLines)
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

function value = getFieldOrDefault(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName)
    value = structValue.(fieldName);
else
    value = defaultValue;
end
end

function value = getTextScalarField(structValue, fieldName, defaultValue)
value = string(getFieldOrDefault(structValue, fieldName, defaultValue));
if ~isscalar(value)
    error("Nano33IoT_Echo_Client_Example:InvalidTextField", "%s must be scalar text.", fieldName);
end
end

function value = getPositiveIntegerField(structValue, fieldName, defaultValue)
value = round(double(getFieldOrDefault(structValue, fieldName, defaultValue)));
if ~isscalar(value) || ~isfinite(value) || value <= 0
    error("Nano33IoT_Echo_Client_Example:InvalidIntegerField", "%s must be a positive integer.", fieldName);
end
end

function value = getNonnegativeIntegerField(structValue, fieldName, defaultValue)
value = round(double(getFieldOrDefault(structValue, fieldName, defaultValue)));
if ~isscalar(value) || ~isfinite(value) || value < 0
    error("Nano33IoT_Echo_Client_Example:InvalidIntegerField", "%s must be a nonnegative integer.", fieldName);
end
end

function value = getPositiveScalarField(structValue, fieldName, defaultValue)
value = double(getFieldOrDefault(structValue, fieldName, defaultValue));
if ~isscalar(value) || ~isfinite(value) || value <= 0
    error("Nano33IoT_Echo_Client_Example:InvalidScalarField", "%s must be positive.", fieldName);
end
end

function value = getNonnegativeScalarField(structValue, fieldName, defaultValue)
value = double(getFieldOrDefault(structValue, fieldName, defaultValue));
if ~isscalar(value) || ~isfinite(value) || value < 0
    error("Nano33IoT_Echo_Client_Example:InvalidScalarField", "%s must be nonnegative.", fieldName);
end
end

function value = getLogicalField(structValue, fieldName, defaultValue)
value = logical(getFieldOrDefault(structValue, fieldName, defaultValue));
if ~isscalar(value)
    error("Nano33IoT_Echo_Client_Example:InvalidLogicalField", "%s must be logical scalar.", fieldName);
end
end
