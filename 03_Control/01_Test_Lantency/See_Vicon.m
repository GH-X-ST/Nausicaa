function runData = See_Vicon(config)
%SEE_VICON Record a baseline Vicon run and export the captured states.
arguments
    config (1,1) struct = struct()
end

config = normalizeConfig(config);
runData = initializeRunData(config);

assignin("base", "ViconLatestState", struct([]));
assignin("base", "ViconRunData", runData);

client = [];
control = struct("figure", [], "statusText", []);
cleanupHandle = onCleanup(@() cleanupResources(client, control));

dllPath = resolveSdkAssembly();
loadSdkAssembly(dllPath);

[client, connectionInfo] = connectToVicon(config, dllPath);
runData.connectionInfo = connectionInfo;
assignin("base", "ViconConnectionInfo", connectionInfo);

if ~connectionInfo.isConnected
    runData.runInfo.status = "connection_failed";
    runData.runInfo.reason = connectionInfo.connectionMessage;
    assignin("base", "ViconRunData", runData);
    return;
end

subjectInfo = discoverSubjects(client, config);
connectionInfo.subjectCount = numel(subjectInfo);
connectionInfo.subjectNames = reshape(string({subjectInfo.name}), [], 1);
connectionInfo.rootSegments = reshape(string({subjectInfo.rootSegmentName}), [], 1);
connectionInfo.segmentNames = reshape(string({subjectInfo.segmentName}), [], 1);
connectionInfo.subjectDiagnostics = subjectSelectionDiagnostics(config, connectionInfo.subjectNames);
runData.connectionInfo = connectionInfo;
assignin("base", "ViconConnectionInfo", connectionInfo);

if isempty(subjectInfo)
    runData.runInfo.status = "no_subjects";
    runData.runInfo.reason = "The connection succeeded, but no selected Vicon subjects are currently visible.";
    assignin("base", "ViconRunData", runData);
    return;
end

control = createControlFigure(config, connectionInfo);
[rawStorage, runInfo] = executeRecordingLoop(client, config, subjectInfo, control);
runData.runInfo = runInfo;

if rawStorage.sampleCount == 0
    assignin("base", "ViconRunData", runData);
    return;
end

runData.subjects = buildSubjectOutputs(rawStorage, subjectInfo);
[workbookPath, roomBoundsFigurePath, autoBoundsFigurePath] = exportRunData(runData);
runData.runInfo.workbookPath = workbookPath;
runData.runInfo.roomBoundsFigurePath = roomBoundsFigurePath;
runData.runInfo.autoBoundsFigurePath = autoBoundsFigurePath;

assignin("base", "ViconRunData", runData);
clear cleanupHandle
cleanupResources(client, control);
end

function config = normalizeConfig(config)
rootFolder = fileparts(mfilename("fullpath"));

config.hostName = getTextScalarField(config, "hostName", "localhost:801");
config.subjectNames = getStringArrayField(config, "subjectNames", strings(0, 1));
config.segmentNames = getFieldOrDefault(config, "segmentNames", struct());
config.streamMode = getFieldOrDefault(config, "streamMode", "ServerPush");
config.axisMapping = getTextScalarField(config, "axisMapping", "ZUp");
config.roomBoundsMeters = getRequiredNumericMatrixField(config, "roomBoundsMeters", [3, 2]);
config.outputFolder = getTextScalarField(config, "outputFolder", fullfile(rootFolder, "B_See_Vicon"));
config.workspaceUpdateDivider = getPositiveIntegerField(config, "workspaceUpdateDivider", 1);
config.runLabel = getTextScalarField(config, "runLabel", "");
config.maxFrames = getPositiveIntegerField(config, "maxFrames", 120000);
config.connectTimeoutSeconds = getPositiveScalarField(config, "connectTimeoutSeconds", 5);
config.connectRetryPauseSeconds = getNonnegativeScalarField(config, "connectRetryPauseSeconds", 0.25);
config.maxConnectionAttempts = getPositiveIntegerField(config, "maxConnectionAttempts", 3);
config.frameWaitTimeoutSeconds = getPositiveScalarField(config, "frameWaitTimeoutSeconds", 2);

if ~any(config.axisMapping == ["XUp", "YUp", "ZUp"])
    error("See_Vicon:InvalidAxisMapping", "axisMapping must be XUp, YUp, or ZUp.");
end

mustHaveValidRoomBounds(config.roomBoundsMeters);

streamModeName = resolveStreamModeName(config.streamMode);
config.streamMode = streamModeName;

if strlength(config.runLabel) == 0
    timeStamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    config.runLabel = timeStamp + "_Test";
end

if ~isfolder(config.outputFolder)
    mkdir(config.outputFolder);
end
end

function runData = initializeRunData(config)
runData = struct( ...
    "config", config, ...
    "connectionInfo", struct(), ...
    "runInfo", struct( ...
        "status", "initialized", ...
        "reason", "", ...
        "sampleCount", 0, ...
        "startTime", NaT, ...
        "stopTime", NaT, ...
        "workbookPath", "", ...
        "roomBoundsFigurePath", "", ...
        "autoBoundsFigurePath", ""), ...
    "subjects", struct([]));
end

function dllPath = resolveSdkAssembly()
rootFolder = fileparts(mfilename("fullpath"));
dllPath = fullfile(rootFolder, "A_Vicon_Example", "dotNET", "ViconDataStreamSDK_DotNET.dll");

if ~isfile(dllPath)
    error("See_Vicon:MissingSdkDll", "Vicon SDK DLL not found at %s.", dllPath);
end
end

function loadSdkAssembly(dllPath)
NET.addAssembly(char(dllPath));
end

function [client, connectionInfo] = connectToVicon(config, dllPath)
client = ViconDataStreamSDK.DotNET.Client();
connectionDiagnostics = strings(config.maxConnectionAttempts + 1, 1);
diagnosticCount = 0;

try
    client.SetConnectionTimeout(uint32(round(config.connectTimeoutSeconds * 1000)));
catch
    diagnosticCount = diagnosticCount + 1;
    connectionDiagnostics(diagnosticCount) = "The SDK client does not expose SetConnectionTimeout in this MATLAB session.";
end

connectStart = tic;
lastResult = "NotAttempted";
for attemptIndex = 1:config.maxConnectionAttempts
    try
        connectOutput = client.Connect(char(config.hostName));
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

isConnected = adaptLogical(client.IsConnected().Connected);
connectionDiagnostics = connectionDiagnostics(1:diagnosticCount);

connectionInfo = struct( ...
    "dllPath", string(dllPath), ...
    "hostName", config.hostName, ...
    "isConnected", isConnected, ...
    "connectAttempts", attemptIndex, ...
    "connectElapsedSeconds", toc(connectStart), ...
    "connectionMessage", lastResult, ...
    "diagnostics", connectionDiagnostics, ...
    "sdkVersion", "", ...
    "streamMode", "", ...
    "axisMapping", config.axisMapping, ...
    "frameRateHz", NaN, ...
    "latencySeconds", NaN, ...
    "subjectCount", 0, ...
    "subjectNames", strings(0, 1), ...
    "rootSegments", strings(0, 1), ...
    "segmentNames", strings(0, 1), ...
    "subjectDiagnostics", strings(0, 1));

if ~isConnected
    return;
end

client.EnableSegmentData();
client.SetBufferSize(uint32(1));
streamMode = resolveStreamMode(config.streamMode);
client.SetStreamMode(streamMode);
applyAxisMapping(client, config.axisMapping);

versionOutput = client.GetVersion();
connectionInfo.sdkVersion = join(string([versionOutput.Major, versionOutput.Minor, versionOutput.Point]), ".");
connectionInfo.streamMode = resolveStreamModeName(config.streamMode);
connectionInfo.axisMapping = getAxisMappingString(client);

frameReady = waitForFrame(client, config.frameWaitTimeoutSeconds);
if frameReady
    connectionInfo.frameRateHz = double(client.GetFrameRate().FrameRateHz);
    connectionInfo.latencySeconds = double(client.GetLatencyTotal().Total);
else
    connectionInfo.diagnostics = [connectionInfo.diagnostics; "No frame arrived during the initial readiness wait."];
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
    error("See_Vicon:InvalidStreamMode", ...
        "Unsupported streamMode '%s'. Use ClientPull, ClientPullPreFetch, or ServerPush.", ...
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

function frameReady = waitForFrame(client, timeoutSeconds)
frameReady = false;
waitStart = tic;
while toc(waitStart) < timeoutSeconds
    getFrameOutput = client.GetFrame();
    if isSuccessResult(getFrameOutput.Result)
        frameReady = true;
        return;
    end
    pause(0.01);
end
end

function subjectInfo = discoverSubjects(client, config)
subjectCount = double(client.GetSubjectCount().SubjectCount);
allSubjects = repmat(struct( ...
    "name", "", ...
    "rootSegmentName", "", ...
    "segmentName", ""), subjectCount, 1);

for subjectIndex = 1:subjectCount
    subjectName = string(client.GetSubjectName(uint32(subjectIndex - 1)).SubjectName);
    rootSegmentName = string(client.GetSubjectRootSegmentName(char(subjectName)).SegmentName);
    allSubjects(subjectIndex).name = subjectName;
    allSubjects(subjectIndex).rootSegmentName = rootSegmentName;
    allSubjects(subjectIndex).segmentName = resolveSegmentName(config.segmentNames, subjectName, rootSegmentName);
end

if isempty(config.subjectNames)
    subjectInfo = allSubjects;
    return;
end

selectedMask = ismember(reshape(string({allSubjects.name}), [], 1), config.subjectNames);
subjectInfo = allSubjects(selectedMask);
end

function diagnostics = subjectSelectionDiagnostics(config, availableSubjectNames)
if isempty(config.subjectNames)
    diagnostics = "Auto-discovery enabled: all visible Vicon subjects will be recorded.";
    return;
end

missingSubjects = setdiff(config.subjectNames, availableSubjectNames);
if isempty(missingSubjects)
    diagnostics = "All requested subjects are available.";
else
    diagnostics = "Requested but not visible: " + join(missingSubjects, ", ");
end
end

function segmentName = resolveSegmentName(segmentConfig, subjectName, defaultSegmentName)
segmentName = string(defaultSegmentName);

if isempty(segmentConfig)
    return;
end

subjectFieldName = matlab.lang.makeValidName(char(subjectName));

if isstruct(segmentConfig) && isfield(segmentConfig, subjectFieldName)
    segmentName = string(segmentConfig.(subjectFieldName));
    return;
end

if istable(segmentConfig)
    variableNames = string(segmentConfig.Properties.VariableNames);
    if all(ismember(["SubjectName", "SegmentName"], variableNames))
        matchesSubject = string(segmentConfig.SubjectName) == subjectName;
        if any(matchesSubject)
            segmentName = string(segmentConfig.SegmentName(find(matchesSubject, 1, "first")));
        end
    end
    return;
end

if iscell(segmentConfig) && size(segmentConfig, 2) >= 2
    subjectColumn = string(segmentConfig(:, 1));
    matchIndex = find(subjectColumn == subjectName, 1, "first");
    if ~isempty(matchIndex)
        segmentName = string(segmentConfig{matchIndex, 2});
    end
end
end

function control = createControlFigure(config, connectionInfo)
control.figure = figure( ...
    "Name", "Vicon Baseline Recorder", ...
    "NumberTitle", "off", ...
    "MenuBar", "none", ...
    "ToolBar", "none", ...
    "HandleVisibility", "callback", ...
    "Position", [200, 200, 500, 220], ...
    "Resize", "off", ...
    "CloseRequestFcn", @(source, ~) setControlCommand(source, "disconnect"));

control.statusText = uicontrol( ...
    control.figure, ...
    "Style", "text", ...
    "HorizontalAlignment", "left", ...
    "Position", [20, 150, 460, 50], ...
    "String", sprintf("Connected to %s\nVisible subjects: %d", config.hostName, connectionInfo.subjectCount));

uicontrol( ...
    control.figure, ...
    "Style", "pushbutton", ...
    "Position", [20, 80, 140, 40], ...
    "String", "Start Recording", ...
    "Callback", @(~, ~) setControlCommand(control.figure, "start"));

uicontrol( ...
    control.figure, ...
    "Style", "pushbutton", ...
    "Position", [180, 80, 140, 40], ...
    "String", "Stop Recording", ...
    "Callback", @(~, ~) setControlCommand(control.figure, "stop"));

uicontrol( ...
    control.figure, ...
    "Style", "pushbutton", ...
    "Position", [340, 80, 140, 40], ...
    "String", "Disconnect", ...
    "Callback", @(~, ~) setControlCommand(control.figure, "disconnect"));

uicontrol( ...
    control.figure, ...
    "Style", "text", ...
    "HorizontalAlignment", "left", ...
    "Position", [20, 20, 460, 40], ...
    "String", "Use Start to begin one recording run. Stop finalizes and exports the run.");

setappdata(control.figure, "RecorderCommand", "idle");
end

function setControlCommand(controlFigure, command)
if isgraphics(controlFigure)
    setappdata(controlFigure, "RecorderCommand", command);
end
end

function [storage, runInfo] = executeRecordingLoop(client, config, subjectInfo, control)
storage = initializeStorage(numel(subjectInfo), config.maxFrames);
runInfo = struct( ...
    "status", "not_started", ...
    "reason", "Recording did not start.", ...
    "sampleCount", 0, ...
    "startTime", NaT, ...
    "stopTime", NaT, ...
    "workbookPath", "", ...
    "roomBoundsFigurePath", "", ...
    "autoBoundsFigurePath", "");

recordingActive = false;
startTic = [];

while isgraphics(control.figure)
    drawnow limitrate;
    command = getappdata(control.figure, "RecorderCommand");

    switch string(command)
        case "start"
            if ~recordingActive && storage.sampleCount == 0
                recordingActive = true;
                startTic = tic;
                runInfo.status = "recording";
                runInfo.reason = "";
                runInfo.startTime = datetime("now", "TimeZone", "local");
                set(control.statusText, "String", "Recording in progress. Press Stop to finalize or Disconnect to abort.");
            end
            setappdata(control.figure, "RecorderCommand", "idle");
        case "stop"
            if recordingActive
                runInfo.status = "completed";
                runInfo.reason = "Recording stopped by the user.";
                break;
            end
            setappdata(control.figure, "RecorderCommand", "idle");
        case "disconnect"
            if recordingActive
                runInfo.status = "disconnected";
                runInfo.reason = "Recording stopped because Disconnect was requested.";
            else
                runInfo.status = "not_started";
                runInfo.reason = "Disconnected before recording started.";
            end
            break;
    end

    if ~recordingActive
        pause(0.02);
        continue;
    end

    getFrameOutput = client.GetFrame();
    if ~isSuccessResult(getFrameOutput.Result)
        pause(0.001);
        continue;
    end

    if storage.sampleCount >= config.maxFrames
        runInfo.status = "max_frames_reached";
        runInfo.reason = "The preallocated sample budget was exhausted before Stop was pressed.";
        break;
    end

    storage.sampleCount = storage.sampleCount + 1;
    sampleIndex = storage.sampleCount;
    storage.timeSeconds(sampleIndex) = toc(startTic);
    storage.frameNumbers(sampleIndex) = double(client.GetFrameNumber().FrameNumber);
    storage.latencySeconds(sampleIndex) = double(client.GetLatencyTotal().Total);
    storage = storeSubjectSamples(storage, client, subjectInfo, sampleIndex);

    if mod(sampleIndex, config.workspaceUpdateDivider) == 0 || sampleIndex == 1
        assignin("base", "ViconLatestState", createLatestStateSnapshot(storage, subjectInfo, sampleIndex));
    end
end

if storage.sampleCount > 0
    assignin("base", "ViconLatestState", createLatestStateSnapshot(storage, subjectInfo, storage.sampleCount));
end

runInfo.sampleCount = storage.sampleCount;
runInfo.stopTime = datetime("now", "TimeZone", "local");
end

function storage = initializeStorage(subjectCount, maxFrames)
subjectStorageTemplate = struct( ...
    "positionMeters", nan(maxFrames, 3), ...
    "eulerRadians", nan(maxFrames, 3), ...
    "quaternionXYZW", nan(maxFrames, 4), ...
    "liveBodyVelocityMps", nan(maxFrames, 3), ...
    "liveBodyRatesRadps", nan(maxFrames, 3), ...
    "isOccluded", true(maxFrames, 1), ...
    "lastValidIndex", 0);

storage = struct( ...
    "sampleCount", 0, ...
    "timeSeconds", nan(maxFrames, 1), ...
    "frameNumbers", nan(maxFrames, 1), ...
    "latencySeconds", nan(maxFrames, 1), ...
    "subjects", repmat(subjectStorageTemplate, subjectCount, 1));
end

function storage = storeSubjectSamples(storage, client, subjectInfo, sampleIndex)
for subjectIndex = 1:numel(subjectInfo)
    subjectName = char(subjectInfo(subjectIndex).name);
    segmentName = char(subjectInfo(subjectIndex).segmentName);

    translationOutput = client.GetSegmentGlobalTranslation(subjectName, segmentName);
    quaternionOutput = client.GetSegmentGlobalRotationQuaternion(subjectName, segmentName);
    eulerOutput = client.GetSegmentGlobalRotationEulerXYZ(subjectName, segmentName);

    isSampleValid = isSuccessOutput(translationOutput) && ...
        isSuccessOutput(quaternionOutput) && ...
        isSuccessOutput(eulerOutput) && ...
        ~adaptLogical(translationOutput.Occluded) && ...
        ~adaptLogical(quaternionOutput.Occluded) && ...
        ~adaptLogical(eulerOutput.Occluded);

    if ~isSampleValid
        storage.subjects(subjectIndex).isOccluded(sampleIndex) = true;
        continue;
    end

    positionMeters = reshape(double(translationOutput.Translation), 1, 3) ./ 1000;
    quaternionXYZW = reshape(double(quaternionOutput.Rotation), 1, 4);
    eulerRadians = reshape(double(eulerOutput.Rotation), 1, 3);

    storage.subjects(subjectIndex).positionMeters(sampleIndex, :) = positionMeters;
    storage.subjects(subjectIndex).quaternionXYZW(sampleIndex, :) = quaternionXYZW;
    storage.subjects(subjectIndex).eulerRadians(sampleIndex, :) = eulerRadians;
    storage.subjects(subjectIndex).isOccluded(sampleIndex) = false;

    previousIndex = storage.subjects(subjectIndex).lastValidIndex;
    if previousIndex > 0
        deltaTime = storage.timeSeconds(sampleIndex) - storage.timeSeconds(previousIndex);
        if deltaTime > 0
            previousPosition = storage.subjects(subjectIndex).positionMeters(previousIndex, :);
            globalVelocity = (positionMeters - previousPosition) ./ deltaTime;
            rotationMatrix = quaternionToRotationMatrix(quaternionXYZW);
            bodyVelocity = (rotationMatrix.' * globalVelocity.').';

            previousQuaternion = storage.subjects(subjectIndex).quaternionXYZW(previousIndex, :);
            previousRotationMatrix = quaternionToRotationMatrix(previousQuaternion);
            rotationDerivative = (rotationMatrix - previousRotationMatrix) ./ deltaTime;
            omegaSkew = 0.5 .* (rotationMatrix.' * rotationDerivative - rotationDerivative.' * rotationMatrix);
            bodyRates = [omegaSkew(3, 2), omegaSkew(1, 3), omegaSkew(2, 1)];

            storage.subjects(subjectIndex).liveBodyVelocityMps(sampleIndex, :) = bodyVelocity;
            storage.subjects(subjectIndex).liveBodyRatesRadps(sampleIndex, :) = bodyRates;
        end
    end

    storage.subjects(subjectIndex).lastValidIndex = sampleIndex;
end
end

function latestState = createLatestStateSnapshot(storage, subjectInfo, sampleIndex)
latestState = repmat(struct( ...
    "name", "", ...
    "segmentName", "", ...
    "time_s", NaN, ...
    "frame_number", NaN, ...
    "latency_s", NaN, ...
    "x_m", NaN, ...
    "y_m", NaN, ...
    "z_m", NaN, ...
    "roll_rad", NaN, ...
    "pitch_rad", NaN, ...
    "yaw_rad", NaN, ...
    "u_mps", NaN, ...
    "v_mps", NaN, ...
    "w_mps", NaN, ...
    "p_radps", NaN, ...
    "q_radps", NaN, ...
    "r_radps", NaN, ...
    "is_occluded", true), numel(subjectInfo), 1);

for subjectIndex = 1:numel(subjectInfo)
    latestState(subjectIndex).name = subjectInfo(subjectIndex).name;
    latestState(subjectIndex).segmentName = subjectInfo(subjectIndex).segmentName;
    latestState(subjectIndex).time_s = storage.timeSeconds(sampleIndex);
    latestState(subjectIndex).frame_number = storage.frameNumbers(sampleIndex);
    latestState(subjectIndex).latency_s = storage.latencySeconds(sampleIndex);
    latestState(subjectIndex).x_m = storage.subjects(subjectIndex).positionMeters(sampleIndex, 1);
    latestState(subjectIndex).y_m = storage.subjects(subjectIndex).positionMeters(sampleIndex, 2);
    latestState(subjectIndex).z_m = storage.subjects(subjectIndex).positionMeters(sampleIndex, 3);
    latestState(subjectIndex).roll_rad = storage.subjects(subjectIndex).eulerRadians(sampleIndex, 1);
    latestState(subjectIndex).pitch_rad = storage.subjects(subjectIndex).eulerRadians(sampleIndex, 2);
    latestState(subjectIndex).yaw_rad = storage.subjects(subjectIndex).eulerRadians(sampleIndex, 3);
    latestState(subjectIndex).u_mps = storage.subjects(subjectIndex).liveBodyVelocityMps(sampleIndex, 1);
    latestState(subjectIndex).v_mps = storage.subjects(subjectIndex).liveBodyVelocityMps(sampleIndex, 2);
    latestState(subjectIndex).w_mps = storage.subjects(subjectIndex).liveBodyVelocityMps(sampleIndex, 3);
    latestState(subjectIndex).p_radps = storage.subjects(subjectIndex).liveBodyRatesRadps(sampleIndex, 1);
    latestState(subjectIndex).q_radps = storage.subjects(subjectIndex).liveBodyRatesRadps(sampleIndex, 2);
    latestState(subjectIndex).r_radps = storage.subjects(subjectIndex).liveBodyRatesRadps(sampleIndex, 3);
    latestState(subjectIndex).is_occluded = storage.subjects(subjectIndex).isOccluded(sampleIndex);
end
end

function subjects = buildSubjectOutputs(storage, subjectInfo)
sampleCount = storage.sampleCount;
subjects = repmat(struct( ...
    "name", "", ...
    "segmentName", "", ...
    "rootSegmentName", "", ...
    "samples", table()), numel(subjectInfo), 1);

timeSeconds = storage.timeSeconds(1:sampleCount);
frameNumbers = storage.frameNumbers(1:sampleCount);
latencySeconds = storage.latencySeconds(1:sampleCount);

for subjectIndex = 1:numel(subjectInfo)
    positionMeters = storage.subjects(subjectIndex).positionMeters(1:sampleCount, :);
    eulerRadians = storage.subjects(subjectIndex).eulerRadians(1:sampleCount, :);
    quaternionXYZW = storage.subjects(subjectIndex).quaternionXYZW(1:sampleCount, :);
    isOccluded = storage.subjects(subjectIndex).isOccluded(1:sampleCount);

    [bodyVelocityMps, bodyRatesRadps] = reconstructRates(timeSeconds, positionMeters, quaternionXYZW, isOccluded);

    subjects(subjectIndex).name = subjectInfo(subjectIndex).name;
    subjects(subjectIndex).segmentName = subjectInfo(subjectIndex).segmentName;
    subjects(subjectIndex).rootSegmentName = subjectInfo(subjectIndex).rootSegmentName;
    subjects(subjectIndex).samples = table( ...
        timeSeconds, ...
        frameNumbers, ...
        latencySeconds, ...
        positionMeters(:, 1), ...
        positionMeters(:, 2), ...
        positionMeters(:, 3), ...
        eulerRadians(:, 1), ...
        eulerRadians(:, 2), ...
        eulerRadians(:, 3), ...
        bodyVelocityMps(:, 1), ...
        bodyVelocityMps(:, 2), ...
        bodyVelocityMps(:, 3), ...
        bodyRatesRadps(:, 1), ...
        bodyRatesRadps(:, 2), ...
        bodyRatesRadps(:, 3), ...
        isOccluded, ...
        "VariableNames", { ...
            "time_s", "frame_number", "latency_s", ...
            "x_m", "y_m", "z_m", ...
            "roll_rad", "pitch_rad", "yaw_rad", ...
            "u_mps", "v_mps", "w_mps", ...
            "p_radps", "q_radps", "r_radps", ...
            "is_occluded"});
end
end

function [bodyVelocityMps, bodyRatesRadps] = reconstructRates(timeSeconds, positionMeters, quaternionXYZW, isOccluded)
sampleCount = numel(timeSeconds);
bodyVelocityMps = nan(sampleCount, 3);
bodyRatesRadps = nan(sampleCount, 3);

validMask = ~isOccluded & all(isfinite(positionMeters), 2) & all(isfinite(quaternionXYZW), 2);
if ~any(validMask)
    return;
end

blockStarts = find(diff([false; validMask]) == 1);
blockStops = find(diff([validMask; false]) == -1);

for blockIndex = 1:numel(blockStarts)
    sampleIndices = blockStarts(blockIndex):blockStops(blockIndex);
    blockTimes = timeSeconds(sampleIndices);
    blockPositions = positionMeters(sampleIndices, :);
    blockQuaternions = quaternionXYZW(sampleIndices, :);

    blockGlobalVelocity = centeredDerivative(blockTimes, blockPositions);
    blockRotationMatrices = cell(numel(sampleIndices), 1);
    for sampleOffset = 1:numel(sampleIndices)
        blockRotationMatrices{sampleOffset} = quaternionToRotationMatrix(blockQuaternions(sampleOffset, :));
        bodyVelocityMps(sampleIndices(sampleOffset), :) = ...
            (blockRotationMatrices{sampleOffset}.' * blockGlobalVelocity(sampleOffset, :).').';
    end

    blockBodyRates = nan(numel(sampleIndices), 3);
    if numel(sampleIndices) >= 2
        for sampleOffset = 1:numel(sampleIndices)
            if sampleOffset == 1
                rotationDerivative = derivativeBetweenMatrices( ...
                    blockRotationMatrices{1}, ...
                    blockRotationMatrices{2}, ...
                    blockTimes(1), ...
                    blockTimes(2));
            elseif sampleOffset == numel(sampleIndices)
                rotationDerivative = derivativeBetweenMatrices( ...
                    blockRotationMatrices{end - 1}, ...
                    blockRotationMatrices{end}, ...
                    blockTimes(end - 1), ...
                    blockTimes(end));
            else
                rotationDerivative = derivativeBetweenMatrices( ...
                    blockRotationMatrices{sampleOffset - 1}, ...
                    blockRotationMatrices{sampleOffset + 1}, ...
                    blockTimes(sampleOffset - 1), ...
                    blockTimes(sampleOffset + 1));
            end

            omegaSkew = 0.5 .* ( ...
                blockRotationMatrices{sampleOffset}.' * rotationDerivative - ...
                rotationDerivative.' * blockRotationMatrices{sampleOffset});
            blockBodyRates(sampleOffset, :) = [omegaSkew(3, 2), omegaSkew(1, 3), omegaSkew(2, 1)];
        end
    end

    bodyRatesRadps(sampleIndices, :) = blockBodyRates;
end
end

function derivativeValues = centeredDerivative(timeSeconds, sampleValues)
sampleCount = size(sampleValues, 1);
derivativeValues = nan(sampleCount, size(sampleValues, 2));

if sampleCount < 2
    return;
end

for sampleIndex = 1:sampleCount
    if sampleIndex == 1
        deltaTime = timeSeconds(2) - timeSeconds(1);
        if deltaTime > 0
            derivativeValues(sampleIndex, :) = (sampleValues(2, :) - sampleValues(1, :)) ./ deltaTime;
        end
    elseif sampleIndex == sampleCount
        deltaTime = timeSeconds(end) - timeSeconds(end - 1);
        if deltaTime > 0
            derivativeValues(sampleIndex, :) = (sampleValues(end, :) - sampleValues(end - 1, :)) ./ deltaTime;
        end
    else
        deltaTime = timeSeconds(sampleIndex + 1) - timeSeconds(sampleIndex - 1);
        if deltaTime > 0
            derivativeValues(sampleIndex, :) = ...
                (sampleValues(sampleIndex + 1, :) - sampleValues(sampleIndex - 1, :)) ./ deltaTime;
        end
    end
end
end

function rotationDerivative = derivativeBetweenMatrices(firstMatrix, secondMatrix, firstTime, secondTime)
deltaTime = secondTime - firstTime;
if deltaTime <= 0
    rotationDerivative = nan(3, 3);
else
    rotationDerivative = (secondMatrix - firstMatrix) ./ deltaTime;
end
end

function rotationMatrix = quaternionToRotationMatrix(quaternionXYZW)
qx = quaternionXYZW(1);
qy = quaternionXYZW(2);
qz = quaternionXYZW(3);
qw = quaternionXYZW(4);

quaternionNorm = sqrt(qx.^2 + qy.^2 + qz.^2 + qw.^2);
if quaternionNorm <= eps
    rotationMatrix = eye(3);
    return;
end

qx = qx ./ quaternionNorm;
qy = qy ./ quaternionNorm;
qz = qz ./ quaternionNorm;
qw = qw ./ quaternionNorm;

rotationMatrix = [ ...
    1 - 2 .* (qy.^2 + qz.^2), 2 .* (qx .* qy - qz .* qw), 2 .* (qx .* qz + qy .* qw); ...
    2 .* (qx .* qy + qz .* qw), 1 - 2 .* (qx.^2 + qz.^2), 2 .* (qy .* qz - qx .* qw); ...
    2 .* (qx .* qz - qy .* qw), 2 .* (qy .* qz + qx .* qw), 1 - 2 .* (qx.^2 + qy.^2)];
end

function [workbookPath, roomBoundsFigurePath, autoBoundsFigurePath] = exportRunData(runData)
workbookPath = fullfile(runData.config.outputFolder, runData.config.runLabel + ".xlsx");
metadataCell = {
    "Run Label", char(runData.config.runLabel); ...
    "Host Name", char(runData.config.hostName); ...
    "Run Status", char(runData.runInfo.status); ...
    "Run Reason", char(runData.runInfo.reason); ...
    "Connection Successful", runData.connectionInfo.isConnected; ...
    "Connection Message", char(runData.connectionInfo.connectionMessage); ...
    "SDK Version", char(runData.connectionInfo.sdkVersion); ...
    "Stream Mode", char(runData.connectionInfo.streamMode); ...
    "Axis Mapping", char(runData.connectionInfo.axisMapping); ...
    "Initial Frame Rate (Hz)", runData.connectionInfo.frameRateHz; ...
    "Initial Latency (s)", runData.connectionInfo.latencySeconds; ...
    "Sample Count", runData.runInfo.sampleCount; ...
    "Start Time", char(string(runData.runInfo.startTime)); ...
    "Stop Time", char(string(runData.runInfo.stopTime)); ...
    "Room Bounds (m)", mat2str(runData.config.roomBoundsMeters); ...
    "Subject Diagnostics", char(join(runData.connectionInfo.subjectDiagnostics, " | ")); ...
    "Fixed Bounds Source", "Configured roomBoundsMeters (the SDK baseline does not expose capture-volume edges directly)."};
writecell(metadataCell, workbookPath, "Sheet", "Metadata");

usedSheetNames = strings(numel(runData.subjects) + 1, 1);
usedSheetNames(1) = "Metadata";
usedSheetCount = 1;
for subjectIndex = 1:numel(runData.subjects)
    sheetName = makeWorksheetName(runData.subjects(subjectIndex).name, usedSheetNames(1:usedSheetCount));
    usedSheetCount = usedSheetCount + 1;
    usedSheetNames(usedSheetCount) = sheetName;
    writetable(runData.subjects(subjectIndex).samples, workbookPath, "Sheet", sheetName);
end

roomBoundsFigurePath = fullfile(runData.config.outputFolder, runData.config.runLabel + "_RoomBounds.png");
autoBoundsFigurePath = fullfile(runData.config.outputFolder, runData.config.runLabel + "_AutoBounds.png");
exportTrajectoryPlot(runData, runData.config.roomBoundsMeters, roomBoundsFigurePath, "Trajectory in Configured Vicon Bounds");
exportTrajectoryPlot(runData, autoBoundsFromData(runData), autoBoundsFigurePath, "Trajectory with Auto-Tight Bounds");
end

function exportTrajectoryPlot(runData, axesBounds, figurePath, figureTitle)
figureHandle = figure( ...
    "Visible", "off", ...
    "Color", "white", ...
    "Name", char(figureTitle));

axesHandle = axes(figureHandle);
hold(axesHandle, "on");
grid(axesHandle, "on");
box(axesHandle, "on");
view(axesHandle, 3);

lineColors = lines(max(1, numel(runData.subjects)));
hasTrajectory = false;
for subjectIndex = 1:numel(runData.subjects)
    samples = runData.subjects(subjectIndex).samples;
    validMask = ~samples.is_occluded & all(isfinite(samples{:, ["x_m", "y_m", "z_m"]}), 2);
    if any(validMask)
        hasTrajectory = true;
        plot3( ...
            axesHandle, ...
            samples.x_m(validMask), ...
            samples.y_m(validMask), ...
            samples.z_m(validMask), ...
            "LineWidth", 1.5, ...
            "Color", lineColors(subjectIndex, :), ...
            "DisplayName", char(runData.subjects(subjectIndex).name));
    end
end

plotBoundsBox(axesHandle, axesBounds);
axis(axesHandle, "equal");
xlabel(axesHandle, "x [m]");
ylabel(axesHandle, "y [m]");
zlabel(axesHandle, "z [m]");
xlim(axesHandle, axesBounds(1, :));
ylim(axesHandle, axesBounds(2, :));
zlim(axesHandle, axesBounds(3, :));
title(axesHandle, sprintf("%s\n%s", figureTitle, runData.config.runLabel), "Interpreter", "none");
if hasTrajectory
    legend(axesHandle, "Location", "bestoutside");
end
exportgraphics(figureHandle, figurePath, "Resolution", 300);
close(figureHandle);
end

function axesBounds = autoBoundsFromData(runData)
pointBlocks = cell(numel(runData.subjects), 1);
pointCount = 0;

for subjectIndex = 1:numel(runData.subjects)
    samples = runData.subjects(subjectIndex).samples;
    validMask = ~samples.is_occluded & all(isfinite(samples{:, ["x_m", "y_m", "z_m"]}), 2);
    pointBlocks{subjectIndex} = samples{validMask, ["x_m", "y_m", "z_m"]};
    pointCount = pointCount + size(pointBlocks{subjectIndex}, 1);
end

if pointCount == 0
    axesBounds = runData.config.roomBoundsMeters;
    return;
end

allPoints = zeros(pointCount, 3);
cursor = 1;
for subjectIndex = 1:numel(pointBlocks)
    blockSize = size(pointBlocks{subjectIndex}, 1);
    if blockSize == 0
        continue;
    end
    allPoints(cursor:cursor + blockSize - 1, :) = pointBlocks{subjectIndex};
    cursor = cursor + blockSize;
end

minimumPoint = min(allPoints, [], 1);
maximumPoint = max(allPoints, [], 1);
span = maximumPoint - minimumPoint;
padding = max(0.05 .* span, 0.05);
axesBounds = [minimumPoint - padding; maximumPoint + padding].';
end

function plotBoundsBox(axesHandle, axesBounds)
xValues = axesBounds(1, :);
yValues = axesBounds(2, :);
zValues = axesBounds(3, :);

vertices = [ ...
    xValues(1), yValues(1), zValues(1); ...
    xValues(2), yValues(1), zValues(1); ...
    xValues(2), yValues(2), zValues(1); ...
    xValues(1), yValues(2), zValues(1); ...
    xValues(1), yValues(1), zValues(2); ...
    xValues(2), yValues(1), zValues(2); ...
    xValues(2), yValues(2), zValues(2); ...
    xValues(1), yValues(2), zValues(2)];

edgePairs = [ ...
    1, 2; 2, 3; 3, 4; 4, 1; ...
    5, 6; 6, 7; 7, 8; 8, 5; ...
    1, 5; 2, 6; 3, 7; 4, 8];

for edgeIndex = 1:size(edgePairs, 1)
    edgeVertices = vertices(edgePairs(edgeIndex, :), :);
    plot3(axesHandle, edgeVertices(:, 1), edgeVertices(:, 2), edgeVertices(:, 3), ...
        "k--", "LineWidth", 0.8, "HandleVisibility", "off");
end
end

function sheetName = makeWorksheetName(baseName, usedSheetNames)
sheetName = regexprep(char(baseName), '[:\\/?*\\[\\]]', "_");
sheetName = strtrim(sheetName);
if isempty(sheetName)
    sheetName = "Subject";
end

sheetName = string(sheetName);
if strlength(sheetName) > 31
    sheetName = extractBefore(sheetName, 32);
end

candidate = sheetName;
suffixIndex = 1;
while any(candidate == usedSheetNames)
    suffix = "_" + suffixIndex;
    maxBaseLength = 31 - strlength(suffix);
    candidate = extractBefore(sheetName + "", maxBaseLength + 1) + suffix;
    suffixIndex = suffixIndex + 1;
end
sheetName = candidate;
end

function cleanupResources(client, control)
if ~isempty(client)
    try
        if adaptLogical(client.IsConnected().Connected)
            client.Disconnect();
        end
    catch
    end
end

if isfield(control, "figure") && ~isempty(control.figure) && isgraphics(control.figure)
    delete(control.figure);
end
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

function value = getFieldOrDefault(config, fieldName, defaultValue)
if isfield(config, fieldName)
    value = config.(fieldName);
else
    value = defaultValue;
end
end

function value = getTextScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
if ischar(value)
    value = string(value);
end

if ~(isstring(value) && isscalar(value))
    error("See_Vicon:InvalidConfigType", "%s must be a text scalar.", fieldName);
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
    error("See_Vicon:InvalidConfigType", "%s must be text, a string array, or a cell array of character vectors.", fieldName);
end

value = reshape(value, [], 1);
end

function value = getRequiredNumericMatrixField(config, fieldName, expectedSize)
if ~isfield(config, fieldName)
    error("See_Vicon:MissingConfigField", "%s is required.", fieldName);
end

value = config.(fieldName);
validateattributes(value, {'numeric'}, {'real', 'finite', 'size', expectedSize}, mfilename, fieldName);
end

function value = getPositiveIntegerField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar', 'integer', 'positive'}, mfilename, fieldName);
value = double(value);
end

function value = getPositiveScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar', 'positive'}, mfilename, fieldName);
value = double(value);
end

function value = getNonnegativeScalarField(config, fieldName, defaultValue)
value = getFieldOrDefault(config, fieldName, defaultValue);
validateattributes(value, {'numeric'}, {'real', 'finite', 'scalar', 'nonnegative'}, mfilename, fieldName);
value = double(value);
end

function mustHaveValidRoomBounds(bounds)
if any(bounds(:, 1) >= bounds(:, 2))
    error("See_Vicon:InvalidRoomBounds", ...
        "roomBoundsMeters must be [xmin xmax; ymin ymax; zmin zmax] with min < max in each row.");
end
end
