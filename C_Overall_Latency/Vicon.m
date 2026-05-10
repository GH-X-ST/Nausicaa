classdef Vicon < handle
    %VICON Lightweight Vicon DataStream tracker for raw pose and surface state.

    properties
        config
        connectionInfo
        subjectInfo
    end

    properties (Access = private)
        client
        neutralSurfaceEulerRad
        neutralBodyEulerRad
        lastSurfaceAngleRad
        lastSurfaceCaptureS
        filteredSurfaceAngleRad
        filteredSurfaceCaptureS
        isClosed = false
    end

    methods
        function obj = Vicon(config)
            if nargin < 1 || isempty(config)
                config = struct();
            end
            obj.config = obj.normalizeConfig(config);
            obj.connectionInfo = struct();
            obj.subjectInfo = table();
            obj.neutralSurfaceEulerRad = nan(numel(obj.config.surfaceSubjectNames), 3);
            obj.neutralBodyEulerRad = nan(1, 3);
            obj.lastSurfaceAngleRad = nan(1, numel(obj.config.surfaceSubjectNames));
            obj.lastSurfaceCaptureS = nan(1, numel(obj.config.surfaceSubjectNames));
            obj.filteredSurfaceAngleRad = nan(1, numel(obj.config.surfaceSubjectNames));
            obj.filteredSurfaceCaptureS = nan(1, numel(obj.config.surfaceSubjectNames));
            obj.connect();
        end

        function calibrateNeutral(obj, runClock, durationS)
            validateattributes(durationS, {'numeric'}, {'real', 'finite', 'scalar', 'positive'}, mfilename, 'durationS');
            surfaceCount = numel(obj.config.surfaceSubjectNames);
            surfaceSamples = cell(surfaceCount, 1);
            bodySamples = nan(0, 3);
            stopTimeS = toc(runClock) + durationS;

            while toc(runClock) < stopTimeS
                frameData = obj.readFrame(runClock);
                if ~frameData.frameOk
                    pause(0.002);
                    continue;
                end

                for surfaceIndex = 1:surfaceCount
                    subjectName = obj.config.surfaceSubjectNames(surfaceIndex);
                    pose = obj.findPose(frameData.poses, subjectName);
                    if ~isempty(pose) && ~pose.is_occluded && all(isfinite(pose.eulerRad))
                        surfaceSamples{surfaceIndex}(end + 1, :) = pose.eulerRad;
                    end
                end

                if strlength(obj.config.bodySubjectName) > 0
                    bodyPose = obj.findPose(frameData.poses, obj.config.bodySubjectName);
                    if ~isempty(bodyPose) && ~bodyPose.is_occluded && all(isfinite(bodyPose.eulerRad))
                        bodySamples(end + 1, :) = bodyPose.eulerRad; %#ok<AGROW>
                    end
                end
                pause(0.002);
            end

            for surfaceIndex = 1:surfaceCount
                if ~isempty(surfaceSamples{surfaceIndex})
                    obj.neutralSurfaceEulerRad(surfaceIndex, :) = Vicon.angularMeanRad(surfaceSamples{surfaceIndex});
                end
            end

            if ~isempty(bodySamples)
                obj.neutralBodyEulerRad = Vicon.angularMeanRad(bodySamples);
            end
        end

        function [rawRows, stateRows, viconSample] = readLatest(obj, runClock)
            frameData = obj.readFrame(runClock);
            if ~frameData.frameOk
                rawRows = obj.emptyRawTable();
                stateRows = obj.emptyStateTable();
                viconSample = obj.emptySample();
                return;
            end

            rawRows = obj.buildRawRows(frameData);
            stateRows = obj.buildStateRows(frameData);
            viconSample = obj.buildSample(frameData, rawRows, stateRows);
        end

        function close(obj)
            if obj.isClosed
                return;
            end
            obj.isClosed = true;
            if ~isempty(obj.client)
                try
                    if obj.adaptLogical(obj.client.IsConnected().Connected)
                        obj.client.Disconnect();
                    end
                catch
                end
            end
        end
    end

    methods (Access = private)
        function connect(obj)
            dllPath = Vicon.resolveSdkAssembly(obj.config.sdkDllPath);
            NET.addAssembly(char(dllPath));

            obj.client = ViconDataStreamSDK.DotNET.Client();
            obj.connectionInfo = struct( ...
                "dllPath", string(dllPath), ...
                "hostName", obj.config.hostAndPort, ...
                "isConnected", false, ...
                "connectionMessage", "", ...
                "streamMode", obj.config.streamMode, ...
                "axisMapping", obj.config.axisMapping, ...
                "frameRateHz", NaN, ...
                "latencySeconds", NaN, ...
                "rawSubjectNames", obj.config.rawSubjectNames, ...
                "surfaceSubjectNames", obj.config.surfaceSubjectNames, ...
                "stateFilterEnabled", obj.config.stateFilterEnabled, ...
                "stateFilterCutoffHz", obj.config.stateFilterCutoffHz);

            try
                obj.client.SetConnectionTimeout(uint32(round(obj.config.connectTimeoutSeconds * 1000)));
            catch
            end

            lastResult = "not attempted";
            for attemptIndex = 1:obj.config.maxConnectionAttempts
                try
                    connectOutput = obj.client.Connect(char(obj.config.hostAndPort));
                    lastResult = obj.netValueToString(connectOutput.Result);
                catch connectionException
                    lastResult = string(connectionException.message);
                end

                if obj.adaptLogical(obj.client.IsConnected().Connected)
                    break;
                end
                pause(obj.config.connectRetryPauseSeconds);
            end

            obj.connectionInfo.isConnected = obj.adaptLogical(obj.client.IsConnected().Connected);
            obj.connectionInfo.connectionMessage = lastResult;
            if ~obj.connectionInfo.isConnected
                error("Vicon:ConnectionFailed", ...
                    "Unable to connect to Vicon host %s: %s", ...
                    char(obj.config.hostAndPort), char(lastResult));
            end

            obj.client.EnableSegmentData();
            obj.client.SetBufferSize(uint32(1));
            obj.client.SetStreamMode(obj.resolveStreamMode(obj.config.streamMode));
            obj.applyAxisMapping(obj.config.axisMapping);
            obj.connectionInfo.initialFrameReady = obj.waitForFrame(obj.config.frameWaitTimeoutSeconds);
            obj.subjectInfo = obj.discoverSubjects();
            obj.requireConfiguredSurfaceSubjects();

            try
                obj.connectionInfo.frameRateHz = double(obj.client.GetFrameRate().FrameRateHz);
                obj.connectionInfo.latencySeconds = double(obj.client.GetLatencyTotal().Total);
            catch
            end
        end

        function frameData = readFrame(obj, runClock)
            frameData = struct( ...
                "frameOk", false, ...
                "frameNumber", NaN, ...
                "tReadHostS", NaN, ...
                "viconLatencyS", NaN, ...
                "tCaptureHostS", NaN, ...
                "poses", struct([]));

            if isempty(obj.client) || obj.isClosed
                return;
            end

            try
                getFrameOutput = obj.client.GetFrame();
                if ~obj.isSuccessResult(getFrameOutput.Result)
                    return;
                end
                frameData.tReadHostS = toc(runClock);
                frameData.viconLatencyS = double(obj.client.GetLatencyTotal().Total);
                frameData.tCaptureHostS = frameData.tReadHostS - frameData.viconLatencyS;
                frameData.frameNumber = double(obj.client.GetFrameNumber().FrameNumber);
                frameData.frameOk = true;
            catch
                return;
            end

            poses = repmat(obj.emptyPose(), height(obj.subjectInfo), 1);
            for rowIndex = 1:height(obj.subjectInfo)
                poses(rowIndex) = obj.readSubjectPose(obj.subjectInfo(rowIndex, :));
            end
            frameData.poses = poses;
        end

        function pose = readSubjectPose(obj, subjectRow)
            pose = obj.emptyPose();
            pose.subjectName = string(subjectRow.subject_name);
            pose.segmentName = string(subjectRow.segment_name);

            if ~logical(subjectRow.is_available)
                pose.is_occluded = true;
                return;
            end

            subjectName = char(subjectRow.subject_name);
            segmentName = char(subjectRow.segment_name);
            try
                translationOutput = obj.client.GetSegmentGlobalTranslation(subjectName, segmentName);
                quaternionOutput = obj.client.GetSegmentGlobalRotationQuaternion(subjectName, segmentName);
                eulerOutput = obj.client.GetSegmentGlobalRotationEulerXYZ(subjectName, segmentName);

                isValid = obj.isSuccessOutput(translationOutput) && ...
                    obj.isSuccessOutput(quaternionOutput) && ...
                    obj.isSuccessOutput(eulerOutput) && ...
                    ~obj.adaptLogical(translationOutput.Occluded) && ...
                    ~obj.adaptLogical(quaternionOutput.Occluded) && ...
                    ~obj.adaptLogical(eulerOutput.Occluded);

                pose.is_occluded = ~isValid;
                if ~isValid
                    return;
                end

                pose.positionM = reshape(double(translationOutput.Translation), 1, 3) ./ 1000;
                pose.quaternionXYZW = reshape(double(quaternionOutput.Rotation), 1, 4);
                pose.eulerRad = reshape(double(eulerOutput.Rotation), 1, 3);
            catch
                pose.is_occluded = true;
            end
        end

        function rawRows = buildRawRows(obj, frameData)
            rowCount = numel(frameData.poses);
            rawRows = obj.emptyRawTable();
            if rowCount == 0
                return;
            end

            rawRows = table( ...
                repmat(frameData.frameNumber, rowCount, 1), ...
                repmat(frameData.tReadHostS, rowCount, 1), ...
                repmat(frameData.viconLatencyS, rowCount, 1), ...
                repmat(frameData.tCaptureHostS, rowCount, 1), ...
                strings(rowCount, 1), strings(rowCount, 1), ...
                nan(rowCount, 1), nan(rowCount, 1), nan(rowCount, 1), ...
                nan(rowCount, 1), nan(rowCount, 1), nan(rowCount, 1), nan(rowCount, 1), ...
                nan(rowCount, 1), nan(rowCount, 1), nan(rowCount, 1), true(rowCount, 1), ...
                'VariableNames', rawRows.Properties.VariableNames);

            for rowIndex = 1:rowCount
                pose = frameData.poses(rowIndex);
                rawRows.subject_name(rowIndex) = pose.subjectName;
                rawRows.segment_name(rowIndex) = pose.segmentName;
                rawRows.x_m(rowIndex) = pose.positionM(1);
                rawRows.y_m(rowIndex) = pose.positionM(2);
                rawRows.z_m(rowIndex) = pose.positionM(3);
                rawRows.qx(rowIndex) = pose.quaternionXYZW(1);
                rawRows.qy(rowIndex) = pose.quaternionXYZW(2);
                rawRows.qz(rowIndex) = pose.quaternionXYZW(3);
                rawRows.qw(rowIndex) = pose.quaternionXYZW(4);
                rawRows.roll_rad(rowIndex) = pose.eulerRad(1);
                rawRows.pitch_rad(rowIndex) = pose.eulerRad(2);
                rawRows.yaw_rad(rowIndex) = pose.eulerRad(3);
                rawRows.is_occluded(rowIndex) = pose.is_occluded;
            end
        end

        function stateRows = buildStateRows(obj, frameData)
            surfaceCount = numel(obj.config.surfaceSubjectNames);
            stateRows = obj.emptyStateTable();
            if surfaceCount == 0
                return;
            end

            stateRows = table( ...
                repmat(frameData.frameNumber, surfaceCount, 1), ...
                repmat(frameData.tCaptureHostS, surfaceCount, 1), ...
                zeros(surfaceCount, 1), ...
                strings(surfaceCount, 1), zeros(surfaceCount, 1), strings(surfaceCount, 1), ...
                nan(surfaceCount, 1), nan(surfaceCount, 1), ...
                nan(surfaceCount, 1), nan(surfaceCount, 1), ...
                nan(surfaceCount, 1), nan(surfaceCount, 1), ...
                repmat(obj.config.stateFilterEnabled, surfaceCount, 1), ...
                repmat(obj.config.stateFilterCutoffHz, surfaceCount, 1), ...
                strings(surfaceCount, 1), ...
                'VariableNames', stateRows.Properties.VariableNames);

            bodyPose = [];
            bodyNeutral = nan(1, 3);
            if strlength(obj.config.bodySubjectName) > 0
                bodyPose = obj.findPose(frameData.poses, obj.config.bodySubjectName);
                bodyNeutral = obj.neutralBodyEulerRad;
            end

            stateReadyHostS = frameData.tReadHostS;
            for surfaceIndex = 1:surfaceCount
                surfaceName = obj.config.surfaceSubjectNames(surfaceIndex);
                pose = obj.findPose(frameData.poses, surfaceName);
                stateRows.state_ready_host_s(surfaceIndex) = stateReadyHostS;
                stateRows.subject_name(surfaceIndex) = surfaceName;
                stateRows.surface_index(surfaceIndex) = surfaceIndex;
                stateRows.surface_name(surfaceIndex) = surfaceName;

                [rawAngleRad, qualityFlag] = obj.estimateSurfaceAngle(surfaceIndex, pose, bodyPose, bodyNeutral);
                angleRad = obj.applyStateFilter(surfaceIndex, rawAngleRad, frameData.tCaptureHostS, qualityFlag);
                rateRadps = obj.estimateSurfaceRate(surfaceIndex, angleRad, frameData.tCaptureHostS);

                stateRows.raw_surface_angle_rad(surfaceIndex) = rawAngleRad;
                stateRows.raw_surface_angle_deg(surfaceIndex) = rad2deg(rawAngleRad);
                stateRows.surface_angle_rad(surfaceIndex) = angleRad;
                stateRows.surface_angle_deg(surfaceIndex) = rad2deg(angleRad);
                stateRows.surface_rate_radps(surfaceIndex) = rateRadps;
                stateRows.surface_rate_degps(surfaceIndex) = rad2deg(rateRadps);
                stateRows.state_filter_enabled(surfaceIndex) = obj.config.stateFilterEnabled;
                stateRows.state_filter_cutoff_hz(surfaceIndex) = obj.config.stateFilterCutoffHz;
                stateRows.quality_flag(surfaceIndex) = qualityFlag;
            end
        end

        function angleRad = applyStateFilter(obj, surfaceIndex, rawAngleRad, tCaptureHostS, qualityFlag)
            angleRad = rawAngleRad;
            if ~obj.config.stateFilterEnabled
                return;
            end
            if string(qualityFlag) ~= "ok" || ~isfinite(rawAngleRad) || ~isfinite(tCaptureHostS)
                angleRad = NaN;
                return;
            end

            previousAngle = obj.filteredSurfaceAngleRad(surfaceIndex);
            previousTime = obj.filteredSurfaceCaptureS(surfaceIndex);
            if ~isfinite(previousAngle) || ~isfinite(previousTime) || tCaptureHostS <= previousTime
                obj.filteredSurfaceAngleRad(surfaceIndex) = rawAngleRad;
                obj.filteredSurfaceCaptureS(surfaceIndex) = tCaptureHostS;
                angleRad = rawAngleRad;
                return;
            end

            dtS = tCaptureHostS - previousTime;
            tauS = 1 ./ (2 .* pi .* obj.config.stateFilterCutoffHz);
            alpha = 1 - exp(-dtS ./ tauS);
            alpha = min(max(alpha, 0), 1);
            deltaRad = obj.wrapToPiLocal(rawAngleRad - previousAngle);
            angleRad = obj.wrapToPiLocal(previousAngle + alpha .* deltaRad);
            obj.filteredSurfaceAngleRad(surfaceIndex) = angleRad;
            obj.filteredSurfaceCaptureS(surfaceIndex) = tCaptureHostS;
        end

        function [angleRad, qualityFlag] = estimateSurfaceAngle(obj, surfaceIndex, pose, bodyPose, bodyNeutral)
            angleRad = NaN;
            qualityFlag = "ok";

            if isempty(pose)
                qualityFlag = "missing_subject";
                return;
            end
            if pose.is_occluded || ~all(isfinite(pose.eulerRad))
                qualityFlag = "occluded";
                return;
            end
            if any(~isfinite(obj.neutralSurfaceEulerRad(surfaceIndex, :)))
                qualityFlag = "missing_neutral";
                return;
            end

            axisIndex = obj.axisNameToIndex(obj.config.surfaceEulerAxes(surfaceIndex));
            currentEuler = pose.eulerRad;
            neutralEuler = obj.neutralSurfaceEulerRad(surfaceIndex, :);

            if ~isempty(bodyPose) && ~bodyPose.is_occluded && all(isfinite(bodyPose.eulerRad)) && all(isfinite(bodyNeutral))
                currentEuler = currentEuler - bodyPose.eulerRad;
                neutralEuler = neutralEuler - bodyNeutral;
            end

            angleRad = obj.wrapToPiLocal(currentEuler(axisIndex) - neutralEuler(axisIndex));
        end

        function rateRadps = estimateSurfaceRate(obj, surfaceIndex, angleRad, tCaptureHostS)
            rateRadps = NaN;
            if ~isfinite(angleRad) || ~isfinite(tCaptureHostS)
                return;
            end

            previousAngle = obj.lastSurfaceAngleRad(surfaceIndex);
            previousTime = obj.lastSurfaceCaptureS(surfaceIndex);
            if isfinite(previousAngle) && isfinite(previousTime) && tCaptureHostS > previousTime
                rateRadps = obj.wrapToPiLocal(angleRad - previousAngle) ./ (tCaptureHostS - previousTime);
            end

            obj.lastSurfaceAngleRad(surfaceIndex) = angleRad;
            obj.lastSurfaceCaptureS(surfaceIndex) = tCaptureHostS;
        end

        function sample = buildSample(obj, frameData, rawRows, stateRows)
            sample = obj.emptySample();
            sample.frame_number = frameData.frameNumber;
            sample.t_read_host_s = frameData.tReadHostS;
            sample.t_capture_host_s = frameData.tCaptureHostS;
            sample.surfaceNames = reshape(string(stateRows.surface_name), 1, []);
            sample.raw_surface_angle_rad = reshape(double(stateRows.raw_surface_angle_rad), 1, []);
            sample.raw_surface_angle_deg = reshape(double(stateRows.raw_surface_angle_deg), 1, []);
            sample.surface_angle_rad = reshape(double(stateRows.surface_angle_rad), 1, []);
            sample.surface_angle_deg = reshape(double(stateRows.surface_angle_deg), 1, []);
            sample.state_filter_enabled = obj.config.stateFilterEnabled;
            sample.state_filter_cutoff_hz = obj.config.stateFilterCutoffHz;
            sample.qualityFlags = reshape(string(stateRows.quality_flag), 1, []);
            sample.rawTable = rawRows;
            sample.stateTable = stateRows;
        end

        function subjectInfo = discoverSubjects(obj)
            availableNames = strings(0, 1);
            rootSegments = strings(0, 1);
            try
                subjectCount = double(obj.client.GetSubjectCount().SubjectCount);
                availableNames = strings(subjectCount, 1);
                rootSegments = strings(subjectCount, 1);
                for subjectIndex = 1:subjectCount
                    subjectName = string(obj.client.GetSubjectName(uint32(subjectIndex - 1)).SubjectName);
                    availableNames(subjectIndex) = subjectName;
                    rootSegments(subjectIndex) = string(obj.client.GetSubjectRootSegmentName(char(subjectName)).SegmentName);
                end
            catch
            end

            validSubjectRows = ~ismissing(availableNames) & strlength(availableNames) > 0;
            availableNames = availableNames(validSubjectRows);
            rootSegments = rootSegments(validSubjectRows);
            obj.connectionInfo.availableSubjectNames = reshape(availableNames, 1, []);

            rawNames = unique([reshape(obj.config.rawSubjectNames, [], 1); reshape(obj.config.surfaceSubjectNames, [], 1); obj.config.bodySubjectName], "stable");
            rawNames = rawNames(strlength(rawNames) > 0);
            rowCount = numel(rawNames);
            subjectInfo = table( ...
                rawNames, strings(rowCount, 1), false(rowCount, 1), ...
                'VariableNames', {'subject_name', 'segment_name', 'is_available'});

            for rowIndex = 1:rowCount
                matchIndex = find(availableNames == rawNames(rowIndex), 1, "first");
                if isempty(matchIndex)
                    subjectInfo.segment_name(rowIndex) = "";
                    subjectInfo.is_available(rowIndex) = false;
                else
                    subjectInfo.segment_name(rowIndex) = rootSegments(matchIndex);
                    subjectInfo.is_available(rowIndex) = true;
                end
            end
        end

        function requireConfiguredSurfaceSubjects(obj)
            requiredNames = reshape(obj.config.surfaceSubjectNames, [], 1);
            availableNames = reshape(obj.connectionInfo.availableSubjectNames, [], 1);
            missingNames = requiredNames(~ismember(requiredNames, availableNames));
            if isempty(missingNames)
                return;
            end

            error("Vicon:MissingRequiredSurfaceSubjects", ...
                "Missing required Vicon surface subject(s): %s. Available Vicon subject(s): %s.", ...
                char(Vicon.formatNameList(missingNames)), ...
                char(Vicon.formatNameList(availableNames)));
        end

        function pose = findPose(~, poses, subjectName)
            pose = [];
            if isempty(poses)
                return;
            end
            names = reshape(string({poses.subjectName}), [], 1);
            matchIndex = find(names == string(subjectName), 1, "first");
            if ~isempty(matchIndex)
                pose = poses(matchIndex);
            end
        end

        function streamMode = resolveStreamMode(~, streamModeName)
            streamModeName = string(streamModeName);
            switch streamModeName
                case "ClientPull"
                    streamMode = ViconDataStreamSDK.DotNET.StreamMode.ClientPull;
                case "ClientPullPreFetch"
                    streamMode = ViconDataStreamSDK.DotNET.StreamMode.ClientPullPreFetch;
                otherwise
                    streamMode = ViconDataStreamSDK.DotNET.StreamMode.ServerPush;
            end
        end

        function applyAxisMapping(obj, axisMapping)
            switch string(axisMapping)
                case "XUp"
                    obj.client.SetAxisMapping( ...
                        ViconDataStreamSDK.DotNET.Direction.Up, ...
                        ViconDataStreamSDK.DotNET.Direction.Forward, ...
                        ViconDataStreamSDK.DotNET.Direction.Left);
                case "YUp"
                    obj.client.SetAxisMapping( ...
                        ViconDataStreamSDK.DotNET.Direction.Forward, ...
                        ViconDataStreamSDK.DotNET.Direction.Up, ...
                        ViconDataStreamSDK.DotNET.Direction.Right);
                otherwise
                    obj.client.SetAxisMapping( ...
                        ViconDataStreamSDK.DotNET.Direction.Forward, ...
                        ViconDataStreamSDK.DotNET.Direction.Left, ...
                        ViconDataStreamSDK.DotNET.Direction.Up);
            end
        end

        function frameReady = waitForFrame(obj, timeoutSeconds)
            frameReady = false;
            waitStart = tic;
            while toc(waitStart) < timeoutSeconds
                getFrameOutput = obj.client.GetFrame();
                if obj.isSuccessResult(getFrameOutput.Result)
                    frameReady = true;
                    return;
                end
                pause(0.01);
            end
        end
    end

    methods (Static, Access = private)
        function config = normalizeConfig(config)
            config.hostName = Vicon.getFieldOrDefault(config, "hostName", "192.168.0.100:801");
            config.port = Vicon.getFieldOrDefault(config, "port", 801);
            if isfield(config, "hostAndPort")
                config.hostAndPort = string(config.hostAndPort);
            elseif any(char(string(config.hostName)) == ':')
                config.hostAndPort = string(config.hostName);
                hostParts = split(string(config.hostName), ":");
                config.hostName = hostParts(1);
                parsedPort = str2double(hostParts(end));
                if isfinite(parsedPort) && parsedPort > 0
                    config.port = parsedPort;
                end
            else
                config.hostAndPort = string(config.hostName) + ":" + string(config.port);
            end
            config.rawSubjectNames = reshape(string(Vicon.getFieldOrDefault( ...
                config, "rawSubjectNames", ["Aileron_L", "Aileron_R", "Rudder", "Elevator"])), 1, []);
            config.surfaceSubjectNames = reshape(string(Vicon.getFieldOrDefault( ...
                config, "surfaceSubjectNames", ["Aileron_L", "Aileron_R", "Rudder", "Elevator"])), 1, []);
            config.surfaceEulerAxes = reshape(string(Vicon.getFieldOrDefault(config, "surfaceEulerAxes", ["X", "X", "Z", "X"])), 1, []);
            config.surfaceEulerAxes = Vicon.resizeRow(config.surfaceEulerAxes, numel(config.surfaceSubjectNames), "X");
            config.bodySubjectName = string(Vicon.getFieldOrDefault(config, "bodySubjectName", ""));
            config.axisMapping = string(Vicon.getFieldOrDefault(config, "axisMapping", "ZUp"));
            config.streamMode = string(Vicon.getFieldOrDefault(config, "streamMode", "ServerPush"));
            config.connectTimeoutSeconds = double(Vicon.getFieldOrDefault(config, "connectTimeoutSeconds", 5.0));
            config.connectRetryPauseSeconds = double(Vicon.getFieldOrDefault(config, "connectRetryPauseSeconds", 0.25));
            config.maxConnectionAttempts = double(Vicon.getFieldOrDefault(config, "maxConnectionAttempts", 3));
            config.frameWaitTimeoutSeconds = double(Vicon.getFieldOrDefault(config, "frameWaitTimeoutSeconds", 2.0));
            config.sdkDllPath = string(Vicon.getFieldOrDefault(config, "sdkDllPath", ""));
            config.stateFilterEnabled = logical(Vicon.getFieldOrDefault(config, "stateFilterEnabled", false));
            config.stateFilterCutoffHz = double(Vicon.getFieldOrDefault(config, "stateFilterCutoffHz", 20.0));
            if ~isfinite(config.stateFilterCutoffHz) || config.stateFilterCutoffHz <= 0
                error("Vicon:InvalidFilterConfig", "stateFilterCutoffHz must be positive and finite.");
            end
        end

        function dllPath = resolveSdkAssembly(configuredPath)
            if strlength(string(configuredPath)) > 0
                dllPath = string(configuredPath);
                if isfile(dllPath)
                    return;
                end
            end
            currentFolder = fileparts(mfilename("fullpath"));
            repoRoot = fileparts(currentFolder);
            dllPath = fullfile(repoRoot, "B_Test_Lantency", "A_Vicon_Example", "dotNET", "ViconDataStreamSDK_DotNET.dll");
            if ~isfile(dllPath)
                error("Vicon:MissingSdkDll", "Vicon SDK DLL not found at %s.", dllPath);
            end
        end

        function pose = emptyPose()
            pose = struct( ...
                "subjectName", "", ...
                "segmentName", "", ...
                "positionM", nan(1, 3), ...
                "quaternionXYZW", nan(1, 4), ...
                "eulerRad", nan(1, 3), ...
                "is_occluded", true);
        end

        function sample = emptySample()
            sample = struct( ...
                "frame_number", NaN, ...
                "t_read_host_s", NaN, ...
                "t_capture_host_s", NaN, ...
                "surfaceNames", strings(1, 0), ...
                "raw_surface_angle_rad", nan(1, 0), ...
                "raw_surface_angle_deg", nan(1, 0), ...
                "surface_angle_rad", nan(1, 0), ...
                "surface_angle_deg", nan(1, 0), ...
                "state_filter_enabled", false, ...
                "state_filter_cutoff_hz", NaN, ...
                "qualityFlags", strings(1, 0), ...
                "rawTable", Vicon.emptyRawTable(), ...
                "stateTable", Vicon.emptyStateTable());
        end

        function tableOut = emptyRawTable()
            tableOut = table( ...
                zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), strings(0, 1), ...
                zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
                zeros(0, 1), zeros(0, 1), zeros(0, 1), false(0, 1), ...
                'VariableNames', {'frame_number', 't_read_host_s', 'vicon_latency_s', 't_capture_host_s', ...
                'subject_name', 'segment_name', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw', ...
                'roll_rad', 'pitch_rad', 'yaw_rad', 'is_occluded'});
        end

        function tableOut = emptyStateTable()
            tableOut = table( ...
                zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), zeros(0, 1), strings(0, 1), ...
                zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
                zeros(0, 1), zeros(0, 1), false(0, 1), zeros(0, 1), strings(0, 1), ...
                'VariableNames', {'frame_number', 't_capture_host_s', 'state_ready_host_s', 'subject_name', ...
                'surface_index', 'surface_name', 'raw_surface_angle_rad', 'raw_surface_angle_deg', ...
                'surface_angle_rad', 'surface_angle_deg', 'surface_rate_radps', 'surface_rate_degps', ...
                'state_filter_enabled', 'state_filter_cutoff_hz', 'quality_flag'});
        end

        function success = isSuccessOutput(output)
            success = true;
            if isprop(output, "Result")
                success = Vicon.isSuccessResult(output.Result);
            end
        end

        function success = isSuccessResult(resultValue)
            success = Vicon.netValueToString(resultValue) == "Success";
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
                value = strcmpi(Vicon.netValueToString(netLogical), "True");
            end
        end

        function listText = formatNameList(names)
            names = reshape(string(names), [], 1);
            names(ismissing(names)) = "<missing>";
            names = names(strlength(names) > 0);
            if isempty(names)
                listText = "<none>";
            else
                listText = join(names, ", ");
            end
        end

        function value = getFieldOrDefault(config, fieldName, defaultValue)
            if isstruct(config) && isfield(config, fieldName)
                value = config.(fieldName);
            else
                value = defaultValue;
            end
        end

        function row = resizeRow(row, targetCount, fillValue)
            row = reshape(row, 1, []);
            if numel(row) < targetCount
                row = [row, repmat(fillValue, 1, targetCount - numel(row))];
            elseif numel(row) > targetCount
                row = row(1:targetCount);
            end
        end

        function axisIndex = axisNameToIndex(axisName)
            switch upper(string(axisName))
                case "Y"
                    axisIndex = 2;
                case "Z"
                    axisIndex = 3;
                otherwise
                    axisIndex = 1;
            end
        end

        function angleRad = wrapToPiLocal(angleRad)
            angleRad = mod(angleRad + pi, 2 * pi) - pi;
        end

        function meanRad = angularMeanRad(angleRowsRad)
            meanRad = atan2(mean(sin(angleRowsRad), 1, "omitnan"), mean(cos(angleRowsRad), 1, "omitnan"));
        end
    end
end
