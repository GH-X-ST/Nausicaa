% DEBUG_TRANSMITTER_PHASE_C_IMPORT_ONLY
% Phase C: import-only MATLAB matching using previously captured CSV files.
%
% Before running:
%   - Point capture paths below to manually verified files from Phase B.
%   - Keep truth-subset disabled to force direct matching validation.

referenceCapturePath = "";
trainerPpmCapturePath = "";
receiverCapturePath = "";

config = struct();
config.runLabel = "PhaseC_ImportOnly_" + string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
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
    "mode", "import_only");
config.matching = struct( ...
    "truthSubsetMode", "off");
config.debug = struct( ...
    "strictDownstreamMatching", true);

if strlength(referenceCapturePath) > 0
    config.logicAnalyzer.referenceCapturePath = referenceCapturePath;
end
if strlength(trainerPpmCapturePath) > 0
    config.logicAnalyzer.trainerPpmCapturePath = trainerPpmCapturePath;
end
if strlength(receiverCapturePath) > 0
    config.logicAnalyzer.receiverCapturePath = receiverCapturePath;
end

runData = Transmitter_Test(config);
surfaceSummary = runData.surfaceSummary(runData.surfaceSummary.SurfaceName == config.singleSurfaceName, :);

disp(runData.logs.alignmentSummary);
disp(surfaceSummary(:, { ...
    'SurfaceName', ...
    'MatchedReceiverCount', ...
    'LatencySummarySource'}));

if surfaceSummary.MatchedReceiverCount <= 0
    error("Phase C failed: direct receiver matching is zero for %s.", config.singleSurfaceName);
end
if surfaceSummary.LatencySummarySource ~= "reference_anchored_direct_events"
    error("Phase C failed: unexpected latency summary source '%s'.", surfaceSummary.LatencySummarySource);
end

disp("Phase C pass: import-only direct matching is healthy.");
