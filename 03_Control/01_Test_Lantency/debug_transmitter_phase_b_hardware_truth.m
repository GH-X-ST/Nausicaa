% DEBUG_TRANSMITTER_PHASE_B_HARDWARE_TRUTH
% Phase B: run command stimulus while capturing D2 (reference) and D3
% (trainer PPM) manually on the logic analyzer.
%
% Hardware setup:
%   - Disconnect servos.
%   - Probe D2 and D3 only (plus ground).
%   - Capture manually in PulseView/sigrok.
%
% Acceptance criterion:
%   D2 -> changed D3 trainer slot <= 0.025 s.

config = struct();
config.runLabel = "PhaseB_D2D3_" + string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
config.commandMode = "single";
config.singleSurfaceName = "Aileron_R";
config.commandProfile = struct( ...
    "type", "latency_isolated_step", ...
    "preCommandNeutralSeconds", 1.0, ...
    "durationSeconds", 8.0, ...
    "postCommandNeutralSeconds", 1.0, ...
    "randomSeed", 5);

% Keep analyzer automation off in MATLAB for this phase.
config.logicAnalyzer = struct( ...
    "enabled", false);
config.matching = struct( ...
    "truthSubsetMode", "off");
config.debug = struct( ...
    "strictDownstreamMatching", true);

runData = Transmitter_Test(config);
disp(runData.surfaceSummary(:, { ...
    'SurfaceName', ...
    'MatchedCommitCount', ...
    'LatencySummarySource'}));
disp("Phase B complete. Verify D2->D3 manually in analyzer capture before Phase C.");
