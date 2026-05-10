%RUN_NAUSICAA_VICON_MOTION_TEST Characterize single-body Vicon pose fluctuation.
% Use this when Vicon Tracker has one rigid body named "Nausicaa".
addpath(fileparts(mfilename("fullpath")));

baseCfg = defaultOverallLatencyConfig("latency");

testCfg = struct();
testCfg.subjectName = "Nausicaa";
testCfg.testMode = "static_then_dynamic";
testCfg.samplePauseSeconds = 0.002;
testCfg.trendWindowSeconds = 0.50;
testCfg.outputRoot = fullfile("C_Overall_Latency", "data", "vicon_characterization");
testCfg.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_nausicaa_vicon_motion";
testCfg.phaseSchedule = buildPhaseSchedule(testCfg.testMode);
testCfg.durationSeconds = double(max(testCfg.phaseSchedule.end_s));

viconCfg = struct();
viconCfg.hostName = baseCfg.viconHostName;
viconCfg.port = baseCfg.viconPort;
viconCfg.rawSubjectNames = testCfg.subjectName;
viconCfg.surfaceSubjectNames = strings(1, 0);
viconCfg.surfaceEulerAxes = strings(1, 0);
viconCfg.bodySubjectName = testCfg.subjectName;
viconCfg.axisMapping = "ZUp";
viconCfg.streamMode = "ServerPush";

outputFolder = fullfile(testCfg.outputRoot, testCfg.runLabel);
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

fprintf("\nNausicaa Vicon motion characterization\n");
fprintf("Subject: %s\n", testCfg.subjectName);
fprintf("Test mode: %s\n", testCfg.testMode);
fprintf("Duration: %.1f s\n", testCfg.durationSeconds);
fprintf("Output folder:\n%s\n", outputFolder);
fprintf("\nPhase schedule:\n");
disp(testCfg.phaseSchedule);

runClock = tic;
tracker = [];
rawChunks = {};
captureStartS = NaN;
try
    tracker = Vicon(viconCfg);
    captureStartS = toc(runClock);
    captureStopS = captureStartS + testCfg.durationSeconds;

    fprintf("Capture started. Follow the phase schedule above.\n");
    while toc(runClock) < captureStopS
        [rawRows, ~, ~] = tracker.readLatest(runClock);
        if ~isempty(rawRows)
            rawRows = rawRows(string(rawRows.subject_name) == testCfg.subjectName, :);
            if ~isempty(rawRows)
                rawChunks{end + 1, 1} = rawRows; %#ok<SAGROW>
            end
        end
        pause(testCfg.samplePauseSeconds);
    end
catch captureError
    if ~isempty(tracker)
        tracker.close();
    end
    rethrow(captureError);
end

if ~isempty(tracker)
    tracker.close();
end

if isempty(rawChunks)
    error("Run_Nausicaa_Vicon_Motion_Test:NoData", ...
        "No Vicon rows were captured for subject '%s'.", testCfg.subjectName);
end

rawLog = vertcat(rawChunks{:});
result = analyzeNausicaaMotion(rawLog, testCfg, captureStartS);

rawCsv = string(fullfile(outputFolder, "nausicaa_vicon_raw.csv"));
processedCsv = string(fullfile(outputFolder, "nausicaa_vicon_processed.csv"));
phaseSummaryCsv = string(fullfile(outputFolder, "nausicaa_vicon_phase_summary.csv"));
filterCsv = string(fullfile(outputFolder, "nausicaa_vicon_filter_candidates.csv"));
resultMat = string(fullfile(outputFolder, "nausicaa_vicon_motion_result.mat"));

writetable(result.rawLog, rawCsv);
writetable(result.processedLog, processedCsv);
writetable(result.phaseSummary, phaseSummaryCsv);
writetable(result.filterCandidates, filterCsv);
save(resultMat, "result", "testCfg", "viconCfg");

fprintf("\nVicon motion characterization complete.\n");
fprintf("Raw log:\n%s\n", rawCsv);
fprintf("Phase summary:\n%s\n", phaseSummaryCsv);
fprintf("Filter candidates:\n%s\n", filterCsv);
if isfield(result, "recommendedFilter")
    fprintf("Recommended starting filter: one-pole low-pass at %.1f Hz, estimated delay %.1f ms.\n", ...
        result.recommendedFilter.cutoff_hz, result.recommendedFilter.estimated_delay_ms);
end

function phaseSchedule = buildPhaseSchedule(testMode)
testMode = lower(string(testMode));
switch testMode
    case "static_only"
        phaseSchedule = phaseTable( ...
            ["static_reference"], ...
            ["static_reference"], ...
            [0], [60], ...
            ["place the glider rigidly on a stand and do not touch it"]);
    case "dynamic_only"
        phaseSchedule = phaseTable( ...
            ["hand_static_start"; "slow_hand_motion"; "fast_run_motion"; "jump_motion"; "hand_static_end"], ...
            ["hand_static"; "dynamic_slow"; "dynamic_fast"; "dynamic_jump"; "hand_static"], ...
            [0; 15; 35; 55; 75], ...
            [15; 35; 55; 75; 90], ...
            ["hold the glider by hand as still as possible"; ...
             "walk or move smoothly with the glider"; ...
             "run through the tracking volume"; ...
             "jump or excite the body motion"; ...
             "hold the glider by hand as still as possible"]);
    otherwise
        phaseSchedule = phaseTable( ...
            ["static_reference"; "pickup_transition"; "hand_static_start"; ...
             "slow_hand_motion"; "fast_run_motion"; "jump_motion"; "hand_static_end"], ...
            ["static_reference"; "transition"; "hand_static"; ...
             "dynamic_slow"; "dynamic_fast"; "dynamic_jump"; "hand_static"], ...
            [0; 60; 70; 85; 105; 125; 145], ...
            [60; 70; 85; 105; 125; 145; 160], ...
            ["place the glider rigidly on a stand and do not touch it"; ...
             "pick up the glider and prepare for hand-held motion"; ...
             "hold the glider by hand as still as possible"; ...
             "walk or move smoothly with the glider"; ...
             "run through the tracking volume"; ...
             "jump or excite the body motion"; ...
             "hold the glider by hand as still as possible"]);
end
end

function schedule = phaseTable(phaseName, phaseType, startS, endS, instruction)
schedule = table( ...
    reshape(string(phaseName), [], 1), ...
    reshape(string(phaseType), [], 1), ...
    reshape(double(startS), [], 1), ...
    reshape(double(endS), [], 1), ...
    reshape(string(instruction), [], 1), ...
    'VariableNames', {'phase_name', 'phase_type', 'start_s', 'end_s', 'instruction'});
end

function result = analyzeNausicaaMotion(rawLog, testCfg, captureStartS)
rawLog = sortrows(rawLog, "t_capture_host_s");
rawLog.t_elapsed_s = double(rawLog.t_capture_host_s) - captureStartS;
rawLog.phase_name = assignPhase(rawLog.t_elapsed_s, testCfg.phaseSchedule);
rawLog.phase_type = assignPhaseType(rawLog.t_elapsed_s, testCfg.phaseSchedule);

validRows = rawLog(~logical(rawLog.is_occluded), :);
if isempty(validRows)
    error("Run_Nausicaa_Vicon_Motion_Test:NoValidData", ...
        "All captured rows were occluded.");
end

processedLog = buildProcessedLog(validRows);
processedLog.phase_name = assignPhase(processedLog.t_elapsed_s, testCfg.phaseSchedule);
processedLog.phase_type = assignPhaseType(processedLog.t_elapsed_s, testCfg.phaseSchedule);

phaseSummary = summarizePhases(rawLog, processedLog, testCfg);
filterCandidates = evaluateFilterCandidates(processedLog, testCfg);
recommendedFilter = chooseRecommendedFilter(filterCandidates);

result = struct();
result.rawLog = rawLog;
result.processedLog = processedLog;
result.phaseSummary = phaseSummary;
result.filterCandidates = filterCandidates;
result.recommendedFilter = recommendedFilter;
result.notes = "Filter recommendation uses static_reference noise plus dynamic high-frequency residual and derivative-spike penalties.";
end

function phaseName = assignPhase(tElapsedS, phaseSchedule)
tElapsedS = double(tElapsedS);
phaseName = strings(numel(tElapsedS), 1);
phaseName(:) = "unlabeled";
for phaseIndex = 1:height(phaseSchedule)
    mask = tElapsedS >= double(phaseSchedule.start_s(phaseIndex)) & ...
        tElapsedS < double(phaseSchedule.end_s(phaseIndex));
    phaseName(mask) = string(phaseSchedule.phase_name(phaseIndex));
end
end

function phaseType = assignPhaseType(tElapsedS, phaseSchedule)
tElapsedS = double(tElapsedS);
phaseType = strings(numel(tElapsedS), 1);
phaseType(:) = "unlabeled";
for phaseIndex = 1:height(phaseSchedule)
    mask = tElapsedS >= double(phaseSchedule.start_s(phaseIndex)) & ...
        tElapsedS < double(phaseSchedule.end_s(phaseIndex));
    phaseType(mask) = string(phaseSchedule.phase_type(phaseIndex));
end
end

function processedLog = buildProcessedLog(validRows)
t = double(validRows.t_elapsed_s);
x = double(validRows.x_m);
y = double(validRows.y_m);
z = double(validRows.z_m);
roll = unwrap(double(validRows.roll_rad));
pitch = unwrap(double(validRows.pitch_rad));
yaw = unwrap(double(validRows.yaw_rad));

dt = [NaN; diff(t)];
vx = finiteDerivative(x, t);
vy = finiteDerivative(y, t);
vz = finiteDerivative(z, t);
rollRate = finiteDerivative(roll, t);
pitchRate = finiteDerivative(pitch, t);
yawRate = finiteDerivative(yaw, t);

processedLog = table( ...
    double(validRows.frame_number), double(validRows.t_read_host_s), ...
    double(validRows.vicon_latency_s), double(validRows.t_capture_host_s), t, dt, ...
    x, y, z, ...
    1000 .* x, 1000 .* y, 1000 .* z, ...
    roll, pitch, yaw, ...
    rad2deg(roll), rad2deg(pitch), rad2deg(yaw), ...
    vx, vy, vz, rollRate, pitchRate, yawRate, ...
    rad2deg(rollRate), rad2deg(pitchRate), rad2deg(yawRate), ...
    'VariableNames', {'frame_number', 't_read_host_s', 'vicon_latency_s', ...
    't_capture_host_s', 't_elapsed_s', 'dt_s', ...
    'x_m', 'y_m', 'z_m', 'x_mm', 'y_mm', 'z_mm', ...
    'roll_rad', 'pitch_rad', 'yaw_rad', 'roll_deg', 'pitch_deg', 'yaw_deg', ...
    'vx_mps', 'vy_mps', 'vz_mps', 'roll_rate_radps', 'pitch_rate_radps', 'yaw_rate_radps', ...
    'roll_rate_degps', 'pitch_rate_degps', 'yaw_rate_degps'});
end

function derivative = finiteDerivative(value, timeS)
value = double(value);
timeS = double(timeS);
derivative = nan(size(value));
dt = diff(timeS);
dv = diff(value);
valid = isfinite(dt) & dt > 0 & isfinite(dv);
derivative(2:end) = NaN;
derivative(find(valid) + 1) = dv(valid) ./ dt(valid); %#ok<FNDSB>
end

function phaseSummary = summarizePhases(rawLog, processedLog, testCfg)
phaseSummary = table();
for phaseIndex = 1:height(testCfg.phaseSchedule)
    phaseName = string(testCfg.phaseSchedule.phase_name(phaseIndex));
    phaseType = string(testCfg.phaseSchedule.phase_type(phaseIndex));
    rawRows = rawLog(string(rawLog.phase_name) == phaseName, :);
    rows = processedLog(string(processedLog.phase_name) == phaseName, :);
    if isempty(rawRows)
        continue;
    end

    validCount = height(rows);
    rawCount = height(rawRows);
    occludedCount = sum(logical(rawRows.is_occluded));
    durationS = double(testCfg.phaseSchedule.end_s(phaseIndex)) - double(testCfg.phaseSchedule.start_s(phaseIndex));
    medianDtS = median(double(rows.dt_s), "omitnan");
    frameRateHz = 1 ./ medianDtS;

    [posResidualRmsMm, rollResidualRmsDeg, pitchResidualRmsDeg, yawResidualRmsDeg] = ...
        residualMetrics(rows, testCfg.trendWindowSeconds);

    phaseSummary = [phaseSummary; table( ... %#ok<AGROW>
        phaseName, phaseType, string(testCfg.phaseSchedule.instruction(phaseIndex)), durationS, ...
        rawCount, validCount, occludedCount, occludedCount ./ max(rawCount, 1), ...
        medianDtS, frameRateHz, ...
        stdFinite(rows.x_mm), stdFinite(rows.y_mm), stdFinite(rows.z_mm), ...
        stdFinite(rows.roll_deg), stdFinite(rows.pitch_deg), stdFinite(rows.yaw_deg), ...
        posResidualRmsMm, rollResidualRmsDeg, pitchResidualRmsDeg, yawResidualRmsDeg, ...
        rmsFinite([rows.vx_mps, rows.vy_mps, rows.vz_mps]), ...
        rmsFinite([rows.roll_rate_degps, rows.pitch_rate_degps, rows.yaw_rate_degps]), ...
        'VariableNames', {'phase_name', 'phase_type', 'instruction', 'duration_s', ...
        'raw_sample_count', 'valid_sample_count', 'occluded_sample_count', 'occluded_fraction', ...
        'median_dt_s', 'frame_rate_hz', ...
        'x_std_mm', 'y_std_mm', 'z_std_mm', ...
        'roll_std_deg', 'pitch_std_deg', 'yaw_std_deg', ...
        'position_high_freq_rms_mm', 'roll_high_freq_rms_deg', ...
        'pitch_high_freq_rms_deg', 'yaw_high_freq_rms_deg', ...
        'velocity_rms_mps', 'angular_rate_rms_degps'})];
end
end

function filterCandidates = evaluateFilterCandidates(processedLog, testCfg)
cutoffHz = [5; 8; 10; 12; 15; 20];
staticRows = processedLog(string(processedLog.phase_type) == "static_reference", :);
staticSource = "static_reference";
if isempty(staticRows)
    staticRows = processedLog(contains(string(processedLog.phase_name), "static"), :);
    staticSource = "any_static_named_phase";
end
if isempty(staticRows)
    staticRows = processedLog;
    staticSource = "all_valid_rows_fallback";
end
dynamicRows = processedLog(startsWith(string(processedLog.phase_type), "dynamic_"), :);
dynamicSource = "dynamic_phases";
if isempty(dynamicRows)
    dynamicRows = processedLog(~contains(string(processedLog.phase_type), "static_reference"), :);
    dynamicSource = "non_static_rows_fallback";
end
if isempty(dynamicRows)
    dynamicRows = processedLog;
    dynamicSource = "all_valid_rows_fallback";
end

dtS = median(double(processedLog.dt_s), "omitnan");
if ~isfinite(dtS) || dtS <= 0
    dtS = 0.01;
end

rawPositionRmsMm = combinedStd(staticRows, ["x_mm", "y_mm", "z_mm"]);
rawEulerRmsDeg = combinedStd(staticRows, ["roll_deg", "pitch_deg", "yaw_deg"]);
[rawDynamicPositionResidualRmsMm, rawDynamicEulerResidualRmsDeg] = dynamicResidualMetrics(dynamicRows, testCfg.trendWindowSeconds);
[rawDynamicVelocityP95Mps, rawDynamicAngularRateP95Degps] = dynamicRateSpikeMetrics(dynamicRows);

filterCandidates = table();
for cutoffIndex = 1:numel(cutoffHz)
    fc = cutoffHz(cutoffIndex);
    filteredStatic = applyPoseLowpass(staticRows, dtS, fc);
    filteredDynamic = applyPoseLowpass(dynamicRows, dtS, fc);

    positionRmsMm = combinedStd(filteredStatic, ["x_mm", "y_mm", "z_mm"]);
    eulerRmsDeg = combinedStd(filteredStatic, ["roll_deg", "pitch_deg", "yaw_deg"]);
    [dynamicPositionResidualRmsMm, dynamicEulerResidualRmsDeg] = ...
        dynamicResidualMetrics(filteredDynamic, testCfg.trendWindowSeconds);
    [dynamicVelocityP95Mps, dynamicAngularRateP95Degps] = dynamicRateSpikeMetrics(filteredDynamic);

    staticPositionRatio = safeRatio(positionRmsMm, rawPositionRmsMm);
    staticEulerRatio = safeRatio(eulerRmsDeg, rawEulerRmsDeg);
    dynamicPositionRatio = safeRatio(dynamicPositionResidualRmsMm, rawDynamicPositionResidualRmsMm);
    dynamicEulerRatio = safeRatio(dynamicEulerResidualRmsDeg, rawDynamicEulerResidualRmsDeg);
    dynamicVelocityRatio = safeRatio(dynamicVelocityP95Mps, rawDynamicVelocityP95Mps);
    dynamicAngularRateRatio = safeRatio(dynamicAngularRateP95Degps, rawDynamicAngularRateP95Degps);

    estimatedDelayMs = 1000 ./ (2 .* pi .* fc);
    score = weightedScore( ...
        [staticPositionRatio, staticEulerRatio, dynamicPositionRatio, ...
        dynamicEulerRatio, dynamicVelocityRatio, dynamicAngularRateRatio], ...
        [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]) + 0.02 .* estimatedDelayMs;

    filterCandidates = [filterCandidates; table( ... %#ok<AGROW>
        fc, string(staticSource), height(staticRows), string(dynamicSource), height(dynamicRows), ...
        estimatedDelayMs, positionRmsMm, eulerRmsDeg, ...
        dynamicPositionResidualRmsMm, dynamicEulerResidualRmsDeg, ...
        dynamicVelocityP95Mps, dynamicAngularRateP95Degps, ...
        safeReduction(rawPositionRmsMm, positionRmsMm), ...
        safeReduction(rawEulerRmsDeg, eulerRmsDeg), ...
        safeReduction(rawDynamicPositionResidualRmsMm, dynamicPositionResidualRmsMm), ...
        safeReduction(rawDynamicEulerResidualRmsDeg, dynamicEulerResidualRmsDeg), ...
        safeReduction(rawDynamicVelocityP95Mps, dynamicVelocityP95Mps), ...
        safeReduction(rawDynamicAngularRateP95Degps, dynamicAngularRateP95Degps), ...
        score, false, ...
        'VariableNames', {'cutoff_hz', ...
        'static_filter_design_source', 'static_filter_design_sample_count', ...
        'dynamic_validation_source', 'dynamic_validation_sample_count', ...
        'estimated_delay_ms', 'static_position_rms_mm', 'static_euler_rms_deg', ...
        'dynamic_position_high_freq_rms_mm', 'dynamic_euler_high_freq_rms_deg', ...
        'dynamic_velocity_p95_mps', 'dynamic_angular_rate_p95_degps', ...
        'static_position_noise_reduction_fraction', 'static_euler_noise_reduction_fraction', ...
        'dynamic_position_residual_reduction_fraction', 'dynamic_euler_residual_reduction_fraction', ...
        'dynamic_velocity_spike_reduction_fraction', 'dynamic_angular_rate_spike_reduction_fraction', ...
        'selection_score', 'recommended'})];
end

candidateMask = filterCandidates.estimated_delay_ms <= 20 & isfinite(filterCandidates.selection_score);
if ~any(candidateMask)
    candidateMask = filterCandidates.estimated_delay_ms <= 20;
end
if any(candidateMask)
    candidateRows = filterCandidates(candidateMask, :);
    [~, bestLocalIndex] = min(candidateRows.selection_score);
    originalIndex = find(candidateMask);
    filterCandidates.recommended(originalIndex(bestLocalIndex)) = true;
end
end

function recommendedFilter = chooseRecommendedFilter(filterCandidates)
recommendedFilter = struct("cutoff_hz", NaN, "estimated_delay_ms", NaN);
rowIndex = find(logical(filterCandidates.recommended), 1, "first");
if isempty(rowIndex)
    return;
end
recommendedFilter.cutoff_hz = double(filterCandidates.cutoff_hz(rowIndex));
recommendedFilter.estimated_delay_ms = double(filterCandidates.estimated_delay_ms(rowIndex));
end

function [posResidualRmsMm, rollRmsDeg, pitchRmsDeg, yawRmsDeg] = residualMetrics(rows, trendWindowSeconds)
if isempty(rows)
    posResidualRmsMm = NaN;
    rollRmsDeg = NaN;
    pitchRmsDeg = NaN;
    yawRmsDeg = NaN;
    return;
end

dtS = median(double(rows.dt_s), "omitnan");
if ~isfinite(dtS) || dtS <= 0
    dtS = 0.01;
end
windowSamples = max(3, round(trendWindowSeconds ./ dtS));

rx = double(rows.x_mm) - localMovingMean(double(rows.x_mm), windowSamples);
ry = double(rows.y_mm) - localMovingMean(double(rows.y_mm), windowSamples);
rz = double(rows.z_mm) - localMovingMean(double(rows.z_mm), windowSamples);
rr = double(rows.roll_deg) - localMovingMean(double(rows.roll_deg), windowSamples);
rp = double(rows.pitch_deg) - localMovingMean(double(rows.pitch_deg), windowSamples);
ryaw = double(rows.yaw_deg) - localMovingMean(double(rows.yaw_deg), windowSamples);

posResidualRmsMm = rmsFinite([rx, ry, rz]);
rollRmsDeg = rmsFinite(rr);
pitchRmsDeg = rmsFinite(rp);
yawRmsDeg = rmsFinite(ryaw);
end

function y = localMovingMean(x, windowSamples)
x = reshape(double(x), [], 1);
windowSamples = max(1, round(windowSamples));
kernel = ones(windowSamples, 1);
valid = isfinite(x);
x0 = x;
x0(~valid) = 0;
counts = conv(double(valid), kernel, "same");
sums = conv(x0, kernel, "same");
y = sums ./ counts;
y(counts == 0) = NaN;
end

function filteredRows = applyPoseLowpass(rows, dtS, cutoffHz)
filteredRows = rows;
if isempty(filteredRows)
    return;
end

filteredRows.x_mm = onePoleLowpass(rows.x_mm, dtS, cutoffHz);
filteredRows.y_mm = onePoleLowpass(rows.y_mm, dtS, cutoffHz);
filteredRows.z_mm = onePoleLowpass(rows.z_mm, dtS, cutoffHz);
filteredRows.x_m = filteredRows.x_mm ./ 1000;
filteredRows.y_m = filteredRows.y_mm ./ 1000;
filteredRows.z_m = filteredRows.z_mm ./ 1000;

filteredRows.roll_deg = onePoleLowpass(rows.roll_deg, dtS, cutoffHz);
filteredRows.pitch_deg = onePoleLowpass(rows.pitch_deg, dtS, cutoffHz);
filteredRows.yaw_deg = onePoleLowpass(rows.yaw_deg, dtS, cutoffHz);
filteredRows.roll_rad = deg2rad(filteredRows.roll_deg);
filteredRows.pitch_rad = deg2rad(filteredRows.pitch_deg);
filteredRows.yaw_rad = deg2rad(filteredRows.yaw_deg);

t = double(filteredRows.t_elapsed_s);
filteredRows.vx_mps = finiteDerivative(filteredRows.x_m, t);
filteredRows.vy_mps = finiteDerivative(filteredRows.y_m, t);
filteredRows.vz_mps = finiteDerivative(filteredRows.z_m, t);
filteredRows.roll_rate_degps = finiteDerivative(filteredRows.roll_deg, t);
filteredRows.pitch_rate_degps = finiteDerivative(filteredRows.pitch_deg, t);
filteredRows.yaw_rate_degps = finiteDerivative(filteredRows.yaw_deg, t);
filteredRows.roll_rate_radps = deg2rad(filteredRows.roll_rate_degps);
filteredRows.pitch_rate_radps = deg2rad(filteredRows.pitch_rate_degps);
filteredRows.yaw_rate_radps = deg2rad(filteredRows.yaw_rate_degps);
end

function [positionResidualRmsMm, eulerResidualRmsDeg] = dynamicResidualMetrics(rows, trendWindowSeconds)
[positionResidualRmsMm, rollResidualRmsDeg, pitchResidualRmsDeg, yawResidualRmsDeg] = ...
    residualMetrics(rows, trendWindowSeconds);
eulerResidualRmsDeg = combinedScalarRms([rollResidualRmsDeg, pitchResidualRmsDeg, yawResidualRmsDeg]);
end

function [velocityP95Mps, angularRateP95Degps] = dynamicRateSpikeMetrics(rows)
if isempty(rows)
    velocityP95Mps = NaN;
    angularRateP95Degps = NaN;
    return;
end
velocityMagnitudeMps = sqrt(double(rows.vx_mps) .^ 2 + double(rows.vy_mps) .^ 2 + double(rows.vz_mps) .^ 2);
angularRateMagnitudeDegps = sqrt(double(rows.roll_rate_degps) .^ 2 + ...
    double(rows.pitch_rate_degps) .^ 2 + double(rows.yaw_rate_degps) .^ 2);
velocityP95Mps = percentileFinite(velocityMagnitudeMps, 95);
angularRateP95Degps = percentileFinite(angularRateMagnitudeDegps, 95);
end

function y = onePoleLowpass(x, dtS, cutoffHz)
x = reshape(double(x), [], 1);
y = nan(size(x));
tauS = 1 ./ (2 .* pi .* cutoffHz);
alpha = dtS ./ (tauS + dtS);
lastY = NaN;
for index = 1:numel(x)
    if ~isfinite(x(index))
        y(index) = lastY;
        continue;
    end
    if ~isfinite(lastY)
        lastY = x(index);
    else
        lastY = lastY + alpha .* (x(index) - lastY);
    end
    y(index) = lastY;
end
end

function value = weightedScore(components, weights)
components = double(components);
weights = double(weights);
valid = isfinite(components) & isfinite(weights) & weights > 0;
if ~any(valid)
    value = Inf;
else
    value = sum(components(valid) .* weights(valid)) ./ sum(weights(valid));
end
end

function value = combinedScalarRms(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = sqrt(sum(values .^ 2));
end
end

function value = percentileFinite(values, percentile)
values = sort(values(isfinite(values)));
if isempty(values)
    value = NaN;
    return;
end
if numel(values) == 1
    value = values(1);
    return;
end
index = 1 + (numel(values) - 1) .* percentile ./ 100;
lowerIndex = floor(index);
upperIndex = ceil(index);
if lowerIndex == upperIndex
    value = values(lowerIndex);
else
    fraction = index - lowerIndex;
    value = values(lowerIndex) + fraction .* (values(upperIndex) - values(lowerIndex));
end
end

function value = combinedStd(rows, variableNames)
stdValues = nan(1, numel(variableNames));
for index = 1:numel(variableNames)
    stdValues(index) = stdFinite(double(rows.(variableNames(index))));
end
value = sqrt(sum(stdValues(isfinite(stdValues)) .^ 2));
if isempty(stdValues(isfinite(stdValues)))
    value = NaN;
end
end

function value = stdFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = std(values, 0);
end
end

function value = rmsFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = sqrt(mean(values .^ 2));
end
end

function value = safeRatio(numerator, denominator)
if ~isfinite(numerator) || ~isfinite(denominator) || denominator <= 0
    value = NaN;
else
    value = numerator ./ denominator;
end
end

function value = safeReduction(rawValue, filteredValue)
if ~isfinite(rawValue) || ~isfinite(filteredValue) || rawValue <= 0
    value = NaN;
else
    value = 1 - filteredValue ./ rawValue;
end
end
