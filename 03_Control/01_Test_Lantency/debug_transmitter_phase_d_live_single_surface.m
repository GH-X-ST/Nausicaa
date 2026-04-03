% DEBUG_TRANSMITTER_PHASE_D_LIVE_SINGLE_SURFACE
% Phase D: live end-to-end single-surface run with sigrok automation.
%
% Preconditions:
%   - Phase B manual D2->D3 check passed.
%   - Phase C import-only matching passed.

config = struct();
config.runLabel = "PhaseD_LiveSingle_" + string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
config.commandMode = "single";
config.singleSurfaceName = "Aileron_R";
config.commandProfile = struct( ...
    "type", "latency_isolated_step", ...
    "preCommandNeutralSeconds", 1.0, ...
    "durationSeconds", 8.0, ...
    "postCommandNeutralSeconds", 1.0, ...
    "randomSeed", 5);
config.logicAnalyzer = struct( ...
    "enabled", true, ...
    "mode", "sigrok_auto");
config.matching = struct( ...
    "truthSubsetMode", "off");
config.debug = struct( ...
    "strictDownstreamMatching", true);

runData = Transmitter_Test(config);
surfaceSummary = runData.surfaceSummary(runData.surfaceSummary.SurfaceName == config.singleSurfaceName, :);

disp(runData.logs.alignmentSummary);
disp(surfaceSummary(:, { ...
    'SurfaceName', ...
    'MatchedReceiverCount', ...
    'LatencySummarySource'}));

if surfaceSummary.MatchedReceiverCount <= 0
    error("Phase D failed: direct receiver matching is zero for %s.", config.singleSurfaceName);
end
if surfaceSummary.LatencySummarySource ~= "reference_anchored_direct_events"
    error("Phase D failed: unexpected latency summary source '%s'.", surfaceSummary.LatencySummarySource);
end

disp("Phase D pass: live single-surface direct matching is healthy.");
