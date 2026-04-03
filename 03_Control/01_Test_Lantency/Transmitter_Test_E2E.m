function [runData, e2e] = Transmitter_Test_E2E(config)
% TRANSMITTER_TEST_E2E Strict shared-clock E2E latency in MATLAB.
%   Default behavior runs real hardware via Transmitter_Test, then computes
%   strict end-to-end tables from board logs + shared-clock Sigrok captures.
%   [runData,e2e] = Transmitter_Test_E2E(struct("seed",1)); 
arguments
    config (1,1) struct = struct()
end

runHardware = getField(config, "runHardware", true);
transmitterConfig = getField(config, "transmitterConfig", struct());
loggerFolder = string(getField(config, "loggerFolder", ""));
frameWindowSeconds = double(getField(config, "frameWindowSeconds", 0.025));
maxCommitAssociationSeconds = double(getField(config, "maxCommitAssociationSeconds", 0.25));
maxRxAssociationSeconds = double(getField(config, "maxRxAssociationSeconds", 0.08));
outputPrefix = string(getField(config, "outputPrefix", "e2e_shared_clock"));
seed = double(getField(config, "seed", 1));
enforceArduinoStepTrainDefaults = logical(getField(config, "enforceArduinoStepTrainDefaults", true));
recordLeadSeconds = max(0, double(getField(config, "recordLeadSeconds", 10.0)));
recordLagSeconds = max(0, double(getField(config, "recordLagSeconds", 10.0)));
commandActiveSeconds = max(0, double(getField(config, "commandActiveSeconds", 59.0)));
quality = buildQualityConfig(config, frameWindowSeconds, maxRxAssociationSeconds);

if runHardware
    transmitterConfig = applyComparableProfileDefaults( ...
        transmitterConfig, ...
        seed, ...
        enforceArduinoStepTrainDefaults, ...
        recordLeadSeconds, ...
        recordLagSeconds, ...
        commandActiveSeconds);
    runData = Transmitter_Test(transmitterConfig);
    logs = runData.logs;
    loggerFolder = string(runData.artifacts.loggerFolderPath);
else
    runData = struct();
    if strlength(loggerFolder) == 0
        loggerFolder = resolveLatestLoggerFolder("D_Transmitter_Test");
    end
    logs = struct();
    logs.hostDispatchLog = readtable(fullfile(loggerFolder, "host_dispatch_log.csv"));
    logs.boardRxLog = readtable(fullfile(loggerFolder, "board_rx_log.csv"));
    logs.boardCommitLog = readtable(fullfile(loggerFolder, "board_commit_log.csv"));
    logs.referenceCapture = readtable(fullfile(loggerFolder, "reference_capture.csv"));
    logs.trainerPpmCapture = readtable(fullfile(loggerFolder, "trainer_ppm_capture.csv"));
    logs.receiverCapture = readtable(fullfile(loggerFolder, "receiver_capture.csv"));
end

e2e = computeE2E(logs, frameWindowSeconds, maxCommitAssociationSeconds, maxRxAssociationSeconds, quality);
e2e.outputPaths = exportE2E(e2e, loggerFolder, outputPrefix);
if runHardware
    runData.e2e = e2e;
end

fprintf("Transmitter_Test_E2E summary\n");
fprintf("  Logger folder: %s\n", char(loggerFolder));
fprintf("  Anchor rows: %d\n", height(e2e.anchorTable));
fprintf("  Event rows: %d\n", height(e2e.eventLatency));
fprintf("  Strict raw rows: %d\n", sum(e2e.eventLatency.is_true_e2e_raw));
fprintf("  Strict true-E2E rows: %d\n", height(e2e.trueE2EEvents));
fprintf("  Excluded non-realistic rows: %d\n", sum(e2e.eventLatency.excluded_non_realistic));
end

function e2e = computeE2E(logs, frameWindowSeconds, maxCommitAssociationSeconds, maxRxAssociationSeconds, quality)
dispatch = logs.hostDispatchLog;
dispatch.surface_name = string(dispatch.surface_name);
dispatch = sortrows(dispatch, {'sample_index', 'surface_name', 'command_sequence'});

rx = logs.boardRxLog;
rx.surface_name = string(rx.surface_name);

commit = logs.boardCommitLog;
ref = sortrows(logs.referenceCapture, "time_s");
trainer = logs.trainerPpmCapture;
trainer.surface_name = string(trainer.surface_name);
receiver = logs.receiverCapture;
receiver.surface_name = string(receiver.surface_name);

dispatchBySample = aggregateDispatchBySample(dispatch);

extraCommitNames = strings(0, 1);
if ismember("receive_to_commit_us", string(commit.Properties.VariableNames))
    extraCommitNames = "receive_to_commit_us";
end
commitSel = selectCandidateByDispatch( ...
    dispatchBySample, ...
    commit, ...
    "sample_sequence", ...
    "command_dispatch_s", ...
    "commit_time_s", ...
    maxCommitAssociationSeconds, ...
    extraCommitNames);

anchorTable = dispatchBySample;
[okCommit, commitIdx] = ismember(anchorTable.sample_sequence, commitSel.sample_sequence);
anchorTable.commit_time_s = nan(height(anchorTable), 1);
anchorTable.commit_valid_for_anchor = false(height(anchorTable), 1);
anchorTable.receive_to_commit_us = nan(height(anchorTable), 1);
for i = 1:height(anchorTable)
    if ~okCommit(i)
        continue;
    end
    j = commitIdx(i);
    anchorTable.commit_time_s(i) = commitSel.commit_time_s(j);
    anchorTable.commit_valid_for_anchor(i) = commitSel.candidate_in_window(j);
    if ismember("receive_to_commit_us", string(commitSel.Properties.VariableNames))
        anchorTable.receive_to_commit_us(i) = commitSel.receive_to_commit_us(j);
    end
end
anchorTable.commit_time_s(~anchorTable.commit_valid_for_anchor) = NaN;
anchorTable.receive_to_commit_us(~anchorTable.commit_valid_for_anchor) = NaN;
anchorTable.candidate_time_s = anchorTable.command_dispatch_s;
useCommit = anchorTable.commit_valid_for_anchor & isfinite(anchorTable.commit_time_s);
anchorTable.candidate_time_s(useCommit) = anchorTable.commit_time_s(useCommit);

refTime = double(ref.time_s);
trainerGlobal = sortrows(trainer(:, {'time_s', 'sample_index', 'sample_rate_hz'}), 'time_s');
trainerGlobalTime = double(trainerGlobal.time_s);
anchorTable.anchor_time_s = nan(height(anchorTable), 1);
anchorTable.anchor_source = strings(height(anchorTable), 1);
anchorTable.anchor_sample_index = nan(height(anchorTable), 1);
anchorTable.anchor_sample_rate_hz = nan(height(anchorTable), 1);
refNext = 1;
trNext = 1;
for i = 1:height(anchorTable)
    ct = anchorTable.candidate_time_s(i);
    [ri, refNext] = findNearestWithinWindow(refTime, ct, refNext, frameWindowSeconds);
    if ri > 0
        anchorTable.anchor_time_s(i) = ref.time_s(ri);
        anchorTable.anchor_source(i) = "D4";
        anchorTable.anchor_sample_index(i) = ref.sample_index(ri);
        anchorTable.anchor_sample_rate_hz(i) = ref.sample_rate_hz(ri);
        continue;
    end
    [ti, trNext] = findNearestWithinWindow(trainerGlobalTime, ct, trNext, frameWindowSeconds);
    if ti > 0
        anchorTable.anchor_time_s(i) = trainerGlobal.time_s(ti);
        anchorTable.anchor_source(i) = "D5";
        anchorTable.anchor_sample_index(i) = trainerGlobal.sample_index(ti);
        anchorTable.anchor_sample_rate_hz(i) = trainerGlobal.sample_rate_hz(ti);
    end
end
anchorTable.anchor_matched = isfinite(anchorTable.anchor_time_s);

events = dispatch;
rxSel = selectCandidateByDispatch( ...
    events, ...
    rx, ...
    ["surface_name", "command_sequence"], ...
    "command_dispatch_s", ...
    "rx_time_s", ...
    maxRxAssociationSeconds, ...
    strings(0, 1));
rxSel.board_rx_s = nan(height(rxSel), 1);
rxSel.board_rx_s(rxSel.candidate_in_window) = rxSel.rx_time_s(rxSel.candidate_in_window);

[okRx, rxIdx] = ismember(buildCompositeKey(events, ["surface_name", "command_sequence"]), ...
                         buildCompositeKey(rxSel, ["surface_name", "command_sequence"]));
events.board_rx_s = nan(height(events), 1);
events.board_rx_s(okRx) = rxSel.board_rx_s(rxIdx(okRx));

[okA, aIdx] = ismember(events.sample_sequence, anchorTable.sample_sequence);
events.commit_time_s = nan(height(events), 1);
events.receive_to_commit_us = nan(height(events), 1);
events.commit_valid_for_anchor = false(height(events), 1);
events.anchor_time_s = nan(height(events), 1);
events.anchor_source = strings(height(events), 1);
events.anchor_sample_index = nan(height(events), 1);
events.anchor_sample_rate_hz = nan(height(events), 1);
for i = 1:height(events)
    if ~okA(i)
        continue;
    end
    j = aIdx(i);
    events.commit_time_s(i) = anchorTable.commit_time_s(j);
    events.receive_to_commit_us(i) = anchorTable.receive_to_commit_us(j);
    events.commit_valid_for_anchor(i) = anchorTable.commit_valid_for_anchor(j);
    events.anchor_time_s(i) = anchorTable.anchor_time_s(j);
    events.anchor_source(i) = anchorTable.anchor_source(j);
    events.anchor_sample_index(i) = anchorTable.anchor_sample_index(j);
    events.anchor_sample_rate_hz(i) = anchorTable.anchor_sample_rate_hz(j);
end

surfaceNames = unique(events.surface_name, "stable");
trainerBySurface = cell(numel(surfaceNames), 1);
receiverBySurface = cell(numel(surfaceNames), 1);
trainerNext = ones(numel(surfaceNames), 1);
receiverNext = ones(numel(surfaceNames), 1);
for s = 1:numel(surfaceNames)
    trainerBySurface{s} = sortrows(trainer(trainer.surface_name == surfaceNames(s), :), "time_s");
    receiverBySurface{s} = sortrows(receiver(receiver.surface_name == surfaceNames(s), :), "time_s");
end

events.ppm_time_s = nan(height(events), 1);
events.ppm_pulse_us = nan(height(events), 1);
events.ppm_sample_index = nan(height(events), 1);
events.receiver_time_s = nan(height(events), 1);
events.receiver_pulse_us = nan(height(events), 1);
events.receiver_sample_index = nan(height(events), 1);

rowOrder = sortrows([(1:height(events)).', events.sample_index, events.sample_sequence], [2, 3, 1]);
for r = 1:size(rowOrder, 1)
    i = rowOrder(r, 1);
    at = events.anchor_time_s(i);
    if ~isfinite(at)
        continue;
    end
    s = find(surfaceNames == events.surface_name(i), 1, "first");
    if isempty(s)
        continue;
    end
    [trRow, trainerNext(s)] = findFirstAfterWithinWindow(trainerBySurface{s}.time_s, at, trainerNext(s), frameWindowSeconds);
    if trRow > 0
        events.ppm_time_s(i) = trainerBySurface{s}.time_s(trRow);
        events.ppm_pulse_us(i) = trainerBySurface{s}.pulse_us(trRow);
        events.ppm_sample_index(i) = trainerBySurface{s}.sample_index(trRow);
    end
    [rxRow, receiverNext(s)] = findFirstAfterWithinWindow(receiverBySurface{s}.time_s, at, receiverNext(s), frameWindowSeconds);
    if rxRow > 0
        events.receiver_time_s(i) = receiverBySurface{s}.time_s(rxRow);
        events.receiver_pulse_us(i) = receiverBySurface{s}.pulse_us(rxRow);
        events.receiver_sample_index(i) = receiverBySurface{s}.sample_index(rxRow);
    end
end

events.host_scheduling_delay_s = events.command_dispatch_s - events.scheduled_time_s;
events.dispatch_to_rx_latency_s = events.board_rx_s - events.command_dispatch_s;
events.dispatch_to_commit_latency_s = events.commit_time_s - events.command_dispatch_s;
events.rx_to_commit_latency_s = events.commit_time_s - events.board_rx_s;
rtcMask = isfinite(events.receive_to_commit_us);
events.rx_to_commit_latency_s(rtcMask) = events.receive_to_commit_us(rtcMask) ./ 1e6;
events.commit_to_anchor_latency_s = events.anchor_time_s - events.commit_time_s;
events.dispatch_to_anchor_latency_s = events.anchor_time_s - events.command_dispatch_s;
events.anchor_to_ppm_latency_s = events.ppm_time_s - events.anchor_time_s;
events.anchor_to_receiver_latency_s = events.receiver_time_s - events.anchor_time_s;
events.ppm_to_receiver_latency_s = events.receiver_time_s - events.ppm_time_s;
events.dispatch_to_ppm_latency_s = events.ppm_time_s - events.command_dispatch_s;
events.scheduled_to_ppm_latency_s = events.ppm_time_s - events.scheduled_time_s;
events.dispatch_to_receiver_latency_s = events.receiver_time_s - events.command_dispatch_s;
events.scheduled_to_receiver_latency_s = events.receiver_time_s - events.scheduled_time_s;

strictMaskRaw = events.commit_valid_for_anchor & isfinite(events.command_dispatch_s) & ...
    isfinite(events.commit_time_s) & isfinite(events.anchor_time_s) & ...
    isfinite(events.ppm_time_s) & isfinite(events.receiver_time_s);
events.is_true_e2e_raw = strictMaskRaw;
events.is_true_e2e = strictMaskRaw;
events.true_dispatch_to_ppm_latency_s = nan(height(events), 1);
events.true_scheduled_to_ppm_latency_s = nan(height(events), 1);
events.true_ppm_to_receiver_latency_s = nan(height(events), 1);
events.true_dispatch_to_receiver_latency_s = nan(height(events), 1);
events.true_scheduled_to_receiver_latency_s = nan(height(events), 1);
events.true_chain_residual_s = nan(height(events), 1);
events.true_dispatch_to_ppm_latency_s(strictMaskRaw) = events.dispatch_to_ppm_latency_s(strictMaskRaw);
events.true_scheduled_to_ppm_latency_s(strictMaskRaw) = events.scheduled_to_ppm_latency_s(strictMaskRaw);
events.true_ppm_to_receiver_latency_s(strictMaskRaw) = events.ppm_to_receiver_latency_s(strictMaskRaw);
events.true_dispatch_to_receiver_latency_s(strictMaskRaw) = events.dispatch_to_receiver_latency_s(strictMaskRaw);
events.true_scheduled_to_receiver_latency_s(strictMaskRaw) = events.scheduled_to_receiver_latency_s(strictMaskRaw);
chainSum = events.dispatch_to_commit_latency_s + events.commit_to_anchor_latency_s + events.anchor_to_receiver_latency_s;
events.true_chain_residual_s(strictMaskRaw) = events.dispatch_to_receiver_latency_s(strictMaskRaw) - chainSum(strictMaskRaw);

[qualityMask, qualityReason] = evaluateEventQuality(events, strictMaskRaw, quality);
events.is_realistic_event = qualityMask;
events.non_realistic_reason = qualityReason;
events.excluded_non_realistic = strictMaskRaw & ~qualityMask;
if quality.filterEnabled
    events.is_true_e2e = strictMaskRaw & qualityMask;
    trueFields = { ...
        'true_dispatch_to_ppm_latency_s', ...
        'true_scheduled_to_ppm_latency_s', ...
        'true_ppm_to_receiver_latency_s', ...
        'true_dispatch_to_receiver_latency_s', ...
        'true_scheduled_to_receiver_latency_s', ...
        'true_chain_residual_s'};
    for k = 1:numel(trueFields)
        fieldName = trueFields{k};
        fieldValues = events.(fieldName);
        fieldValues(~events.is_true_e2e) = NaN;
        events.(fieldName) = fieldValues;
    end
end

events.ppm_matched = isfinite(events.ppm_time_s);
events.receiver_matched = isfinite(events.receiver_time_s);

e2e = struct();
e2e.anchorTable = anchorTable;
e2e.eventLatency = events;
e2e.anchorToPpm = events(isfinite(events.anchor_to_ppm_latency_s), :);
e2e.anchorToReceiver = events(isfinite(events.anchor_to_receiver_latency_s), :);
e2e.trueE2EEvents = events(events.is_true_e2e, :);
e2e.qualitySummary = buildQualitySummary(events);
e2e.surfaceSummary = buildSurfaceSummary(events);
e2e.overallSummary = buildOverallSummary(events);
end

function dispatchBySample = aggregateDispatchBySample(dispatch)
[groupId, sampleSequence] = findgroups(dispatch.sample_sequence);
dispatchBySample = table();
dispatchBySample.sample_sequence = sampleSequence;
dispatchBySample.sample_index = splitapply(@min, dispatch.sample_index, groupId);
dispatchBySample.scheduled_time_s = splitapply(@medianFinite, dispatch.scheduled_time_s, groupId);
dispatchBySample.command_dispatch_s = splitapply(@medianFinite, dispatch.command_dispatch_s, groupId);
dispatchBySample = sortrows(dispatchBySample, {'sample_index', 'sample_sequence'});
end

function summary = buildSurfaceSummary(events)
surfaceNames = unique(events.surface_name, "stable");
metricNames = [ ...
    "dispatch_to_rx_latency_s", "dispatch_to_commit_latency_s", "rx_to_commit_latency_s", ...
    "commit_to_anchor_latency_s", "dispatch_to_anchor_latency_s", "anchor_to_ppm_latency_s", ...
    "anchor_to_receiver_latency_s", "ppm_to_receiver_latency_s", ...
    "dispatch_to_ppm_latency_s", "dispatch_to_receiver_latency_s", ...
    "scheduled_to_ppm_latency_s", "scheduled_to_receiver_latency_s", ...
    "true_dispatch_to_ppm_latency_s", "true_scheduled_to_ppm_latency_s", ...
    "true_ppm_to_receiver_latency_s", "true_dispatch_to_receiver_latency_s", ...
    "true_scheduled_to_receiver_latency_s", "true_chain_residual_s"];

rows = cell(numel(surfaceNames), 1);
for i = 1:numel(surfaceNames)
    t = events(events.surface_name == surfaceNames(i), :);
    row = table(surfaceNames(i), height(t), sum(isfinite(t.anchor_time_s)), ...
        sum(isfinite(t.ppm_time_s)), sum(isfinite(t.receiver_time_s)), ...
        'VariableNames', {'surface_name', 'dispatch_rows', 'anchor_matches', 'ppm_matches', 'receiver_matches'});
    for m = 1:numel(metricNames)
        metric = metricNames(m);
        metricName = char(metric);
        v = t.(metricName);
        v = v(isfinite(v));
        row.(char(metric + "_count")) = numel(v);
        row.(char(metric + "_median_s")) = medianOrNaN(v);
        row.(char(metric + "_p95_s")) = percentileOrNaN(v, 95);
        row.(char(metric + "_p99_s")) = percentileOrNaN(v, 99);
        row.(char(metric + "_max_s")) = maxOrNaN(v);
    end
    rows{i} = row;
end
summary = vertcat(rows{:});
end

function summary = buildOverallSummary(events)
metricNames = [ ...
    "dispatch_to_rx_latency_s", "dispatch_to_commit_latency_s", "rx_to_commit_latency_s", ...
    "commit_to_anchor_latency_s", "dispatch_to_anchor_latency_s", "anchor_to_ppm_latency_s", ...
    "anchor_to_receiver_latency_s", "ppm_to_receiver_latency_s", ...
    "dispatch_to_ppm_latency_s", "dispatch_to_receiver_latency_s", ...
    "scheduled_to_ppm_latency_s", "scheduled_to_receiver_latency_s", ...
    "true_dispatch_to_ppm_latency_s", "true_scheduled_to_ppm_latency_s", ...
    "true_ppm_to_receiver_latency_s", "true_dispatch_to_receiver_latency_s", ...
    "true_scheduled_to_receiver_latency_s", "true_chain_residual_s"];
rows = cell(numel(metricNames), 1);
for i = 1:numel(metricNames)
    metric = metricNames(i);
    v = events.(char(metric));
    v = v(isfinite(v));
    rows{i} = table(metric, numel(v), medianOrNaN(v), percentileOrNaN(v, 95), ...
        percentileOrNaN(v, 99), maxOrNaN(v), ...
        'VariableNames', {'metric', 'count', 'median_s', 'p95_s', 'p99_s', 'max_s'});
end
summary = vertcat(rows{:});
end

function paths = exportE2E(e2e, loggerFolder, prefix)
paths = struct();
paths.anchorTableCsv = fullfile(loggerFolder, prefix + "_anchor_table.csv");
paths.eventLatencyCsv = fullfile(loggerFolder, prefix + "_event_latency.csv");
paths.anchorToPpmCsv = fullfile(loggerFolder, prefix + "_anchor_to_ppm.csv");
paths.anchorToReceiverCsv = fullfile(loggerFolder, prefix + "_anchor_to_receiver.csv");
paths.trueE2ECsv = fullfile(loggerFolder, prefix + "_true_e2e_events.csv");
paths.qualitySummaryCsv = fullfile(loggerFolder, prefix + "_quality_summary.csv");
paths.surfaceSummaryCsv = fullfile(loggerFolder, prefix + "_surface_summary.csv");
paths.overallSummaryCsv = fullfile(loggerFolder, prefix + "_overall_summary.csv");
writetable(e2e.anchorTable, paths.anchorTableCsv);
writetable(e2e.eventLatency, paths.eventLatencyCsv);
writetable(e2e.anchorToPpm, paths.anchorToPpmCsv);
writetable(e2e.anchorToReceiver, paths.anchorToReceiverCsv);
writetable(e2e.trueE2EEvents, paths.trueE2ECsv);
writetable(e2e.qualitySummary, paths.qualitySummaryCsv);
writetable(e2e.surfaceSummary, paths.surfaceSummaryCsv);
writetable(e2e.overallSummary, paths.overallSummaryCsv);
end

function t = selectCandidateByDispatch(dispatchFrame, candidateFrame, keyNames, dispatchTimeName, candidateTimeName, maxAssocSeconds, extraNames)
if ischar(keyNames) || isStringScalar(keyNames)
    keyNames = string(keyNames);
else
    keyNames = string(keyNames);
end
if nargin < 8
    extraNames = strings(0, 1);
end
extraNames = string(extraNames);

dispatchKey = buildCompositeKey(dispatchFrame, keyNames);
[~, ia] = unique(dispatchKey, "stable");
dispatchColumns = [cellstr(keyNames(:).'), {char(dispatchTimeName)}];
dispatchUnique = dispatchFrame(ia, dispatchColumns);
dispatchKey = dispatchKey(ia);

t = dispatchUnique;
t.(char(candidateTimeName)) = nan(height(t), 1);
t.candidate_delta_s = nan(height(t), 1);
t.candidate_in_window = false(height(t), 1);
t.candidate_finite = false(height(t), 1);
for n = 1:numel(extraNames)
    extraName = char(extraNames(n));
    if ismember(string(extraName), string(candidateFrame.Properties.VariableNames))
        t.(extraName) = nan(height(t), 1);
    end
end

candidateKey = buildCompositeKey(candidateFrame, keyNames);
for i = 1:height(t)
    mask = candidateKey == dispatchKey(i);
    if ~any(mask)
        continue;
    end
    c = candidateFrame(mask, :);
    ct = double(c.(char(candidateTimeName)));
    dt = double(t.(char(dispatchTimeName))(i));
    delta = ct - dt;
    finiteMask = isfinite(ct);
    inMask = finiteMask & isfinite(delta) & delta >= -0.005 & delta <= maxAssocSeconds;
    if any(inMask)
        idx = find(inMask);
        [~, j] = min(abs(delta(inMask)));
        chosen = idx(j);
    else
        idx = find(finiteMask);
        if isempty(idx)
            continue;
        end
        [~, j] = min(ct(finiteMask));
        chosen = idx(j);
    end
    t.(char(candidateTimeName))(i) = ct(chosen);
    t.candidate_delta_s(i) = delta(chosen);
    t.candidate_in_window(i) = inMask(chosen);
    t.candidate_finite(i) = finiteMask(chosen);
    for n = 1:numel(extraNames)
        nm = char(extraNames(n));
        if ismember(string(nm), string(c.Properties.VariableNames))
            t.(nm)(i) = double(c.(nm)(chosen));
        end
    end
end
end

function keys = buildCompositeKey(tableData, keyNames)
keyNames = string(keyNames);
keys = strings(height(tableData), 1);
for i = 1:height(tableData)
    parts = strings(1, numel(keyNames));
    for k = 1:numel(keyNames)
        value = tableData.(char(keyNames(k)))(i);
        parts(k) = string(value);
    end
    keys(i) = join(parts, "|");
end
end

function [idx, nextIdx] = findNearestWithinWindow(sortedTime, centerTime, startIdx, windowSeconds)
idx = 0;
nextIdx = startIdx;
if startIdx > numel(sortedTime) || ~isfinite(centerTime)
    return;
end
left = find(sortedTime >= centerTime - windowSeconds, 1, "first");
if isempty(left)
    return;
end
left = max(left, startIdx);
right = find(sortedTime <= centerTime + windowSeconds, 1, "last");
if isempty(right) || right < left
    return;
end
cand = left:right;
[~, j] = min(abs(sortedTime(cand) - centerTime));
idx = cand(j);
nextIdx = idx + 1;
end

function [idx, nextIdx] = findFirstAfterWithinWindow(sortedTime, anchorTime, startIdx, windowSeconds)
idx = 0;
nextIdx = startIdx;
if startIdx > numel(sortedTime) || ~isfinite(anchorTime)
    return;
end
candidate = find(sortedTime >= anchorTime, 1, "first");
if isempty(candidate)
    return;
end
candidate = max(candidate, startIdx);
if candidate > numel(sortedTime)
    return;
end
if sortedTime(candidate) > anchorTime + windowSeconds
    return;
end
idx = candidate;
nextIdx = idx + 1;
end

function value = getField(config, name, defaultValue)
if isfield(config, name)
    value = config.(name);
else
    value = defaultValue;
end
end

function value = medianOrNaN(v)
if isempty(v)
    value = NaN;
else
    value = median(v);
end
end

function value = percentileOrNaN(v, p)
if isempty(v)
    value = NaN;
else
    value = prctile(v, p);
end
end

function value = maxOrNaN(v)
if isempty(v)
    value = NaN;
else
    value = max(v);
end
end

function value = medianFinite(v)
v = v(isfinite(v));
if isempty(v)
    value = NaN;
else
    value = median(v);
end
end

function quality = buildQualityConfig(config, frameWindowSeconds, maxRxAssociationSeconds)
qualityConfig = getField(config, "quality", struct());
quality = struct();
quality.filterEnabled = logical(getField( ...
    qualityConfig, ...
    "filterEnabled", ...
    getField(config, "filterNonRealistic", true)));
quality.allowSmallNegativeSeconds = max(0, double(getField(qualityConfig, "allowSmallNegativeSeconds", 5e-4)));
quality.maxDispatchToRxSeconds = max(0, double(getField(qualityConfig, "maxDispatchToRxSeconds", maxRxAssociationSeconds)));
quality.maxRxToCommitSeconds = max(0, double(getField(qualityConfig, "maxRxToCommitSeconds", 0.03)));
quality.maxCommitToAnchorSeconds = max(0, double(getField(qualityConfig, "maxCommitToAnchorSeconds", frameWindowSeconds)));
quality.maxAnchorToPpmSeconds = max(0, double(getField(qualityConfig, "maxAnchorToPpmSeconds", frameWindowSeconds)));
quality.maxAnchorToReceiverSeconds = max(0, double(getField(qualityConfig, "maxAnchorToReceiverSeconds", frameWindowSeconds)));
quality.maxPpmToReceiverSeconds = max(0, double(getField(qualityConfig, "maxPpmToReceiverSeconds", frameWindowSeconds)));
quality.maxDispatchToReceiverSeconds = max(0, double(getField(qualityConfig, "maxDispatchToReceiverSeconds", 0.20)));
quality.maxChainResidualSeconds = max(0, double(getField(qualityConfig, "maxChainResidualSeconds", 0.002)));
quality.requirePpmBeforeReceiver = logical(getField(qualityConfig, "requirePpmBeforeReceiver", true));
quality.requireMonotonicChain = logical(getField(qualityConfig, "requireMonotonicChain", true));
end

function [qualityMask, reason] = evaluateEventQuality(events, strictMaskRaw, quality)
rowCount = height(events);
qualityMask = strictMaskRaw;
reason = strings(rowCount, 1);
if rowCount == 0
    return;
end

epsNeg = quality.allowSmallNegativeSeconds;
candidateMask = strictMaskRaw;

dispatchToRxOk = isfinite(events.dispatch_to_rx_latency_s) & ...
    events.dispatch_to_rx_latency_s >= -epsNeg & ...
    events.dispatch_to_rx_latency_s <= quality.maxDispatchToRxSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~dispatchToRxOk, "dispatch_to_rx_out_of_range");

rxToCommitOk = isfinite(events.rx_to_commit_latency_s) & ...
    events.rx_to_commit_latency_s >= -epsNeg & ...
    events.rx_to_commit_latency_s <= quality.maxRxToCommitSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~rxToCommitOk, "rx_to_commit_out_of_range");

commitToAnchorOk = isfinite(events.commit_to_anchor_latency_s) & ...
    events.commit_to_anchor_latency_s >= -epsNeg & ...
    events.commit_to_anchor_latency_s <= quality.maxCommitToAnchorSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~commitToAnchorOk, "commit_to_anchor_out_of_range");

anchorToPpmOk = isfinite(events.anchor_to_ppm_latency_s) & ...
    events.anchor_to_ppm_latency_s >= -epsNeg & ...
    events.anchor_to_ppm_latency_s <= quality.maxAnchorToPpmSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~anchorToPpmOk, "anchor_to_ppm_out_of_range");

anchorToReceiverOk = isfinite(events.anchor_to_receiver_latency_s) & ...
    events.anchor_to_receiver_latency_s >= -epsNeg & ...
    events.anchor_to_receiver_latency_s <= quality.maxAnchorToReceiverSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~anchorToReceiverOk, "anchor_to_receiver_out_of_range");

ppmToReceiverOk = isfinite(events.ppm_to_receiver_latency_s) & ...
    events.ppm_to_receiver_latency_s >= -epsNeg & ...
    events.ppm_to_receiver_latency_s <= quality.maxPpmToReceiverSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~ppmToReceiverOk, "ppm_to_receiver_out_of_range");

dispatchToReceiverOk = isfinite(events.dispatch_to_receiver_latency_s) & ...
    events.dispatch_to_receiver_latency_s >= -epsNeg & ...
    events.dispatch_to_receiver_latency_s <= quality.maxDispatchToReceiverSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~dispatchToReceiverOk, "dispatch_to_receiver_out_of_range");

chainResidualOk = isfinite(events.true_chain_residual_s) & ...
    abs(events.true_chain_residual_s) <= quality.maxChainResidualSeconds;
[qualityMask, reason] = applyQualityFailure( ...
    qualityMask, reason, candidateMask & ~chainResidualOk, "chain_residual_out_of_range");

if quality.requireMonotonicChain
    monotonicOk = ...
        events.board_rx_s >= (events.command_dispatch_s - epsNeg) & ...
        events.commit_time_s >= (events.board_rx_s - epsNeg) & ...
        events.anchor_time_s >= (events.commit_time_s - epsNeg) & ...
        events.ppm_time_s >= (events.anchor_time_s - epsNeg) & ...
        events.receiver_time_s >= (events.anchor_time_s - epsNeg);
    [qualityMask, reason] = applyQualityFailure( ...
        qualityMask, reason, candidateMask & ~monotonicOk, "non_monotonic_chain");
end

if quality.requirePpmBeforeReceiver
    ppmBeforeReceiverOk = events.receiver_time_s >= (events.ppm_time_s - epsNeg);
    [qualityMask, reason] = applyQualityFailure( ...
        qualityMask, reason, candidateMask & ~ppmBeforeReceiverOk, "receiver_before_ppm");
end

reason(candidateMask & qualityMask & reason == "") = "ok";
reason(~candidateMask) = "not_strict_e2e";
end

function [qualityMask, reason] = applyQualityFailure(qualityMask, reason, failMask, reasonLabel)
qualityMask(failMask) = false;
newReasonMask = failMask & reason == "";
reason(newReasonMask) = string(reasonLabel);
end

function summary = buildQualitySummary(events)
summary = table( ...
    ["total_events"; "strict_true_e2e_raw"; "strict_true_e2e_final"; "excluded_non_realistic"], ...
    [height(events); sum(events.is_true_e2e_raw); sum(events.is_true_e2e); sum(events.excluded_non_realistic)], ...
    'VariableNames', {'Metric', 'Count'});

excludedReasons = events.non_realistic_reason(events.excluded_non_realistic);
excludedReasons = excludedReasons(strlength(excludedReasons) > 0);
if isempty(excludedReasons)
    return;
end

reasonLabels = unique(excludedReasons, "stable");
reasonRows = table();
reasonRows.Metric = "excluded_reason:" + reasonLabels;
reasonRows.Count = zeros(numel(reasonLabels), 1);
for i = 1:numel(reasonLabels)
    reasonRows.Count(i) = sum(excludedReasons == reasonLabels(i));
end
summary = [summary; reasonRows];
end

function loggerFolder = resolveLatestLoggerFolder(rootFolder)
folderInfo = dir(fullfile(rootFolder, "*_TransmitterLogger"));
if isempty(folderInfo)
    error("Transmitter_Test_E2E:MissingLoggerFolder", "No *_TransmitterLogger folder found in %s.", rootFolder);
end
[~, idx] = max([folderInfo.datenum]);
loggerFolder = string(fullfile(folderInfo(idx).folder, folderInfo(idx).name));
end

function transmitterConfig = applyComparableProfileDefaults( ...
    transmitterConfig, ...
    seed, ...
    enforceArduinoStepTrainDefaults, ...
    recordLeadSeconds, ...
    recordLagSeconds, ...
    commandActiveSeconds)
if ~isfield(transmitterConfig, "commandProfile") || isempty(transmitterConfig.commandProfile)
    transmitterConfig.commandProfile = struct();
end

if enforceArduinoStepTrainDefaults
    % Match Arduino_Test latency_step_train defaults exactly for direct
    % seed-to-seed reference comparison, while preserving 20 ms cadence.
    % Recording window control:
    %   preCommandNeutralSeconds  -> lead-in recording time before command.
    %   durationSeconds           -> active command segment length.
    %   postCommandNeutralSeconds -> tail recording time after command.
    transmitterConfig.commandProfile.type = "latency_step_train";
    transmitterConfig.commandProfile.sampleTimeSeconds = 0.02;
    transmitterConfig.commandProfile.preCommandNeutralSeconds = recordLeadSeconds;
    transmitterConfig.commandProfile.postCommandNeutralSeconds = recordLagSeconds;
    transmitterConfig.commandProfile.durationSeconds = commandActiveSeconds;
    transmitterConfig.commandProfile.amplitudeDegrees = 45.0;
    transmitterConfig.commandProfile.offsetDegrees = 0.0;
    transmitterConfig.commandProfile.frequencyHz = 0.5;
    transmitterConfig.commandProfile.phaseDegrees = 90.0;
    transmitterConfig.commandProfile.doubletHoldSeconds = 0.5;
    transmitterConfig.commandProfile.eventHoldSeconds = 0.20;
    transmitterConfig.commandProfile.eventNeutralHoldSeconds = 0.10;
    transmitterConfig.commandProfile.eventDwellSeconds = 0.60;
    transmitterConfig.commandProfile.eventRandomJitterSeconds = 0.05;

    if isfinite(seed)
        transmitterConfig.commandProfile.randomSeed = seed;
    else
        transmitterConfig.commandProfile.randomSeed = 5;
    end
    return;
end

if ~isfield(transmitterConfig.commandProfile, "type")
    transmitterConfig.commandProfile.type = "latency_step_train";
end
if ~isfield(transmitterConfig.commandProfile, "sampleTimeSeconds")
    transmitterConfig.commandProfile.sampleTimeSeconds = 0.02;
end
if ~isfield(transmitterConfig.commandProfile, "randomSeed")
    if isfinite(seed)
        transmitterConfig.commandProfile.randomSeed = seed;
    else
        transmitterConfig.commandProfile.randomSeed = 5;
    end
end
end
