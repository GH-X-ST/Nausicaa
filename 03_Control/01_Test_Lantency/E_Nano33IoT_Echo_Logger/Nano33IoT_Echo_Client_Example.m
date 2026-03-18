function artifacts = Nano33IoT_Echo_Client_Example(config)
%NANO33IOT_ECHO_CLIENT_EXAMPLE Drive the standalone Nano echo logger.
%   This example uses tcpclient rather than MATLAB's arduino()/servo()
%   support-package path, because the echo logger sketch is standalone.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end

hostTimer = tic;
client = tcpclient( ...
    char(config.ipAddress), ...
    config.port, ...
    "ConnectTimeout", config.timeoutSeconds, ...
    "Timeout", config.timeoutSeconds);
cleanupHandle = onCleanup(@() cleanupClient(client));

initialGreeting = "";
if client.NumBytesAvailable > 0
    initialGreeting = strtrim(readline(client));
else
    pause(0.25);
    if client.NumBytesAvailable > 0
        initialGreeting = strtrim(readline(client));
    end
end

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

syncRoundTripLog = table( ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    zeros(0, 1), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});

okReply = requestReply(client, "CLEAR_LOGS");
if strlength(okReply) == 0
    error("Nano33IoT_Echo_Client_Example:NoClearReply", "The Nano did not reply to CLEAR_LOGS.");
end

for syncIndex = 1:config.syncCountBeforeRun
    syncRoundTripLog(end + 1, :) = sendSync(client, hostTimer, uint32(syncIndex)); %#ok<AGROW>
    pause(config.syncPauseSeconds);
end

for planIndex = 1:height(config.commandPlan)
    dispatchUs = hostNowUs(hostTimer);
    commandLine = compose( ...
        "SET,%s,%u,%.6f", ...
        config.commandPlan.surface_name(planIndex), ...
        uint32(config.commandPlan.command_sequence(planIndex)), ...
        config.commandPlan.position_norm(planIndex));
    writeline(client, char(commandLine));

    dispatchLog(end + 1, :) = { ...
        config.commandPlan.surface_name(planIndex), ...
        double(config.commandPlan.command_sequence(planIndex)), ...
        double(dispatchUs), ...
        config.commandPlan.position_norm(planIndex)}; %#ok<AGROW>

    pause(config.commandPlan.wait_after_s(planIndex));
end

if config.returnToNeutralAfterRun
    requestReply(client, "SET_NEUTRAL");
end

pause(config.postRunSettleSeconds);

for syncIndex = 1:config.syncCountAfterRun
    syncRoundTripLog(end + 1, :) = sendSync(client, hostTimer, uint32(config.syncCountBeforeRun + syncIndex)); %#ok<AGROW>
    pause(config.syncPauseSeconds);
end

commandDumpLines = requestDump(client, "DUMP_COMMAND_LOG", "#COMMAND_LOG_END");
syncDumpLines = requestDump(client, "DUMP_SYNC_LOG", "#SYNC_LOG_END");

artifacts = struct( ...
    "initialGreeting", initialGreeting, ...
    "dispatchLogPath", string(fullfile(config.outputFolder, "host_dispatch_log.csv")), ...
    "syncRoundTripPath", string(fullfile(config.outputFolder, "host_sync_roundtrip.csv")), ...
    "commandDumpPath", string(fullfile(config.outputFolder, "board_command_log.csv")), ...
    "syncDumpPath", string(fullfile(config.outputFolder, "board_sync_log.csv")));

writetable(dispatchLog, artifacts.dispatchLogPath);
writetable(syncRoundTripLog, artifacts.syncRoundTripPath);
writelines(commandDumpLines, artifacts.commandDumpPath);
writelines(syncDumpLines, artifacts.syncDumpPath);

clear cleanupHandle
cleanupClient(client);
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

function syncRow = sendSync(client, hostTimer, syncId)
hostTxUs = hostNowUs(hostTimer);
writeline(client, sprintf("SYNC,%u,%u", uint32(syncId), uint32(hostTxUs)));
reply = strtrim(readline(client));
hostRxUs = hostNowUs(hostTimer);
reply = readProtocolLine(client, reply);

replyParts = split(reply, ",");
if numel(replyParts) ~= 5 || replyParts(1) ~= "SYNC_REPLY"
    error("Nano33IoT_Echo_Client_Example:InvalidSyncReply", "Unexpected SYNC reply: %s", reply);
end

syncRow = table( ...
    double(str2double(replyParts(2))), ...
    double(str2double(replyParts(3))), ...
    double(hostRxUs), ...
    double(str2double(replyParts(4))), ...
    double(str2double(replyParts(5))), ...
    'VariableNames', { ...
        'sync_id', ...
        'host_tx_us', ...
        'host_rx_us', ...
        'board_rx_us', ...
        'board_tx_us'});
end

function reply = requestReply(client, requestLine)
writeline(client, requestLine);
reply = readProtocolLine(client);
end

function dumpLines = requestDump(client, requestLine, endMarker)
writeline(client, requestLine);
dumpLines = strings(0, 1);

while true
    nextLine = readProtocolLine(client);
    dumpLines(end + 1, 1) = nextLine; %#ok<AGROW>
    if nextLine == endMarker
        return;
    end
end
end

function nextLine = readProtocolLine(client, firstLine)
if nargin >= 2
    nextLine = string(firstLine);
else
    nextLine = string(strtrim(readline(client)));
end

while startsWith(nextLine, "HELLO_REPLY")
    nextLine = string(strtrim(readline(client)));
end
end

function nowUs = hostNowUs(hostTimer)
nowUs = uint32(round(toc(hostTimer) .* 1e6));
end

function cleanupClient(client)
if isempty(client)
    return;
end

try
    clear client %#ok<NASGU>
catch
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
    error("Nano33IoT_Echo_Client_Example:InvalidConfigType", "%s must be a text scalar.", fieldName);
end
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

function value = getNonnegativeIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"numeric"}, {"real", "finite", "scalar", "integer", "nonnegative"}, mfilename, fieldName);
value = double(value);
end

function value = getLogicalField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {"logical", "numeric"}, {"scalar"}, mfilename, fieldName);
value = logical(value);
end
