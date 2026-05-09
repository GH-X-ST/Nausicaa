%RUN_DEFLECTION_TEST Run a five-case deflection calibration batch.
addpath(fileparts(mfilename("fullpath")));

batchTimestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
batchFolder = fullfile("C_Overall_Latency", "data", "processed", batchTimestamp + "_deflection_batch");
if ~isfolder(batchFolder)
    mkdir(batchFolder);
end

scenarios = deflectionBatchScenarios();
surfaceSummary = table();
lookupSamples = table();
interRunPauseSeconds = 8.0;
batterySwapAfterScenario = 3;
batterySwapPauseSeconds = 150.0;

for scenarioIndex = 1:numel(scenarios)
    scenario = scenarios(scenarioIndex);
    cfg = defaultOverallLatencyConfig("deflection");
    cfg.runLabel = batchTimestamp + "_deflection_" + scenario.name;
    cfg.enabledSurfaceIndices = 1:4;
    cfg.deflectionHoldSeconds = scenario.holdSeconds;
    cfg.deflectionRampLevels = scenario.rampLevels;
    cfg.neutralLeadSeconds = scenario.neutralLeadSeconds;
    cfg.neutralTailSeconds = scenario.neutralTailSeconds;
    cfg.commandDtSeconds = scenario.commandDtSeconds;

    fprintf("\n[%d/%d] Deflection scenario: %s\n", scenarioIndex, numel(scenarios), scenario.name);
    fprintf("  holdSeconds = %.2f, commandDtSeconds = %.3f\n", cfg.deflectionHoldSeconds, cfg.commandDtSeconds);
    fprintf("  rampLevels = %s\n", mat2str(cfg.deflectionRampLevels));

    runData = Latency_Test(cfg);
    result = Post_Analysis("deflection", runData.outputFiles.runDataMat);

    surfaceSummary = [surfaceSummary; makeSurfaceSummaryRows(scenarioIndex, scenario, runData, result)]; %#ok<AGROW>
    lookupSamples = [lookupSamples; makeLookupRows(scenarioIndex, scenario, result)]; %#ok<AGROW>

    fprintf("  raw run: %s\n", runData.outputFiles.runDataMat);
    fprintf("  calibration: %s\n", result.outputFiles.calibrationMat);

    if scenarioIndex < numel(scenarios)
        if scenarioIndex == batterySwapAfterScenario
            fprintf("\nBattery swap pause before scenario %d.\n", scenarioIndex + 1);
            fprintf("Swap the battery now. The next scenario will start in %.0f seconds.\n", batterySwapPauseSeconds);
            pauseWithCountdown(batterySwapPauseSeconds, "battery swap");
        else
            fprintf("  waiting %.1f s before next scenario\n", interRunPauseSeconds);
            pause(interRunPauseSeconds);
        end
    end
end

lookupSummary = summarizeLookupSamples(lookupSamples);

surfaceSummaryFile = string(fullfile(batchFolder, "batch_surface_summary.csv"));
lookupSamplesFile = string(fullfile(batchFolder, "batch_lookup_samples.csv"));
lookupSummaryFile = string(fullfile(batchFolder, "batch_lookup_summary.csv"));
batchMatFile = string(fullfile(batchFolder, "batch_results.mat"));

writetable(surfaceSummary, surfaceSummaryFile);
writetable(lookupSamples, lookupSamplesFile);
writetable(lookupSummary, lookupSummaryFile);
save(batchMatFile, "scenarios", "surfaceSummary", "lookupSamples", "lookupSummary");

fprintf("\nDeflection batch complete.\n");
fprintf("Batch folder:\n%s\n", batchFolder);
fprintf("Surface summary:\n%s\n", surfaceSummaryFile);
fprintf("Lookup summary:\n%s\n", lookupSummaryFile);

function scenarios = deflectionBatchScenarios()
scenarios = struct( ...
    "name", {}, ...
    "holdSeconds", {}, ...
    "rampLevels", {}, ...
    "commandDtSeconds", {}, ...
    "neutralLeadSeconds", {}, ...
    "neutralTailSeconds", {}, ...
    "description", {});

scenarios(end + 1) = scenarioRow( ...
    "fast_coarse", 1.25, [0, 0.15, 0.30, 0.50, 0.75, 1.00], 0.015, ...
    "Short holds with aileron-informed coarse steps above the effective input gap.");
scenarios(end + 1) = scenarioRow( ...
    "medium_nominal", 1.50, [0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00], 0.02, ...
    "Moderate holds using the aileron deadband estimate as the first useful increment.");
scenarios(end + 1) = scenarioRow( ...
    "baseline_nominal", 2.00, [0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00], 0.025, ...
    "Baseline calibration with command points matched to the aileron response.");
scenarios(end + 1) = scenarioRow( ...
    "slow_nominal", 2.50, [0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00], 0.030, ...
    "Longer holds at the aileron-informed command levels to reduce settling sensitivity.");
scenarios(end + 1) = scenarioRow( ...
    "fine_deadband", 2.00, [0, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00], 0.020, ...
    "Low-end aileron gap check around 0.05 to 0.15, then regular calibration levels.");
end

function scenario = scenarioRow(name, holdSeconds, rampLevels, commandDtSeconds, description)
scenario = struct( ...
    "name", string(name), ...
    "holdSeconds", holdSeconds, ...
    "rampLevels", rampLevels, ...
    "commandDtSeconds", commandDtSeconds, ...
    "neutralLeadSeconds", 5.0, ...
    "neutralTailSeconds", 5.0, ...
    "description", string(description));
end

function pauseWithCountdown(durationSeconds, labelText)
remainingSeconds = round(durationSeconds);
while remainingSeconds > 0
    if remainingSeconds == round(durationSeconds) || remainingSeconds <= 10 || mod(remainingSeconds, 30) == 0
        fprintf("  %s pause: %d s remaining\n", labelText, remainingSeconds);
    end
    pause(1.0);
    remainingSeconds = remainingSeconds - 1;
end
fprintf("  %s pause complete.\n", labelText);
end

function rows = makeSurfaceSummaryRows(scenarioIndex, scenario, runData, result)
summary = result.summaryTable;
surfaceCount = height(summary);
rows = table( ...
    repmat(scenarioIndex, surfaceCount, 1), ...
    repmat(string(scenario.name), surfaceCount, 1), ...
    repmat(string(scenario.description), surfaceCount, 1), ...
    repmat(scenario.holdSeconds, surfaceCount, 1), ...
    repmat(scenario.commandDtSeconds, surfaceCount, 1), ...
    repmat(string(mat2str(scenario.rampLevels)), surfaceCount, 1), ...
    summary.surface_index, string(summary.surface_name), ...
    summary.positive_full_valid, summary.negative_full_valid, ...
    summary.min_deflection_deg, summary.max_deflection_deg, ...
    summary.positive_range_deg, summary.negative_range_deg, ...
    readSummaryColumn(summary, "lookup_rmse_deg"), ...
    readSummaryColumn(summary, "linear_fit_rmse_deg"), ...
    readSummaryColumn(summary, "return_hysteresis_deg"), ...
    readSummaryColumn(summary, "repeatability_std_deg"), ...
    repmat(string(runData.outputFiles.runDataMat), surfaceCount, 1), ...
    repmat(string(result.outputFiles.calibrationMat), surfaceCount, 1), ...
    'VariableNames', {'scenario_index', 'scenario_name', 'scenario_description', ...
    'hold_seconds', 'command_dt_seconds', 'ramp_levels', ...
    'surface_index', 'surface_name', 'positive_full_valid', 'negative_full_valid', ...
    'min_deflection_deg', 'max_deflection_deg', 'positive_range_deg', 'negative_range_deg', ...
    'lookup_rmse_deg', 'linear_fit_rmse_deg', 'return_hysteresis_deg', ...
    'repeatability_std_deg', 'run_data_mat', 'calibration_mat'});
end

function rows = makeLookupRows(scenarioIndex, scenario, result)
lookup = result.calibration.lookupTable;
rowCount = height(lookup);
rows = table( ...
    repmat(scenarioIndex, rowCount, 1), ...
    repmat(string(scenario.name), rowCount, 1), ...
    repmat(scenario.holdSeconds, rowCount, 1), ...
    repmat(scenario.commandDtSeconds, rowCount, 1), ...
    lookup.surface_index, string(lookup.surface_name), ...
    lookup.command_level_norm, lookup.lookup_deflection_deg, lookup.lookup_std_deg, ...
    lookup.lookup_min_deg, lookup.lookup_max_deg, lookup.valid_hold_count, ...
    'VariableNames', {'scenario_index', 'scenario_name', 'hold_seconds', 'command_dt_seconds', ...
    'surface_index', 'surface_name', 'command_level_norm', 'lookup_deflection_deg', ...
    'lookup_std_deg', 'lookup_min_deg', 'lookup_max_deg', 'valid_hold_count'});
end

function summary = summarizeLookupSamples(samples)
summary = table();
if isempty(samples)
    return;
end

surfaces = unique(string(samples.surface_name), "stable");
for surfaceIndex = 1:numel(surfaces)
    surfaceRows = samples(string(samples.surface_name) == surfaces(surfaceIndex), :);
    levels = unique(double(surfaceRows.command_level_norm));
    levels = sort(levels);
    for levelIndex = 1:numel(levels)
        levelRows = surfaceRows(abs(double(surfaceRows.command_level_norm) - levels(levelIndex)) < 10 * eps, :);
        deflections = double(levelRows.lookup_deflection_deg);
        stdValues = double(levelRows.lookup_std_deg);
        summary = [summary; table( ... %#ok<AGROW>
            double(levelRows.surface_index(1)), surfaces(surfaceIndex), levels(levelIndex), ...
            height(levelRows), median(deflections, "omitnan"), mean(deflections, "omitnan"), ...
            std(deflections, 0, "omitnan"), minFinite(deflections), maxFinite(deflections), ...
            median(stdValues, "omitnan"), ...
            'VariableNames', {'surface_index', 'surface_name', 'command_level_norm', ...
            'scenario_count', 'median_deflection_deg', 'mean_deflection_deg', ...
            'between_scenario_std_deg', 'min_deflection_deg', 'max_deflection_deg', ...
            'median_within_hold_std_deg'})];
    end
end

summary = addDiscreteStepColumns(summary);
end

function summary = addDiscreteStepColumns(summary)
summary.step_from_previous_norm = nan(height(summary), 1);
summary.step_from_previous_deg = nan(height(summary), 1);
summary.abs_step_from_previous_deg = nan(height(summary), 1);

surfaces = unique(string(summary.surface_name), "stable");
for surfaceIndex = 1:numel(surfaces)
    rowMask = string(summary.surface_name) == surfaces(surfaceIndex);
    rowIndices = find(rowMask);
    [~, order] = sort(double(summary.command_level_norm(rowMask)));
    rowIndices = rowIndices(order);

    commands = double(summary.command_level_norm(rowIndices));
    deflections = double(summary.median_deflection_deg(rowIndices));
    summary.step_from_previous_norm(rowIndices(2:end)) = diff(commands);
    summary.step_from_previous_deg(rowIndices(2:end)) = diff(deflections);
    summary.abs_step_from_previous_deg(rowIndices(2:end)) = abs(diff(deflections));
end
end

function values = readSummaryColumn(summary, variableName)
if ismember(variableName, string(summary.Properties.VariableNames))
    values = double(summary.(variableName));
else
    values = nan(height(summary), 1);
end
end

function value = minFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = min(values);
end
end

function value = maxFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end
