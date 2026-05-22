%RUN_LATENCY_POST_ANALYSIS Analyze the latest latency run using measured response fractions.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Locate latest raw latency run
% 2) Configure crossing-based latency analysis
% 3) Report generated result file
% ==========================================================================
addpath(fileparts(mfilename("fullpath")));

%% =========================================================================
% 1) Locate latest raw latency run
% ==========================================================================
rawRunFile = latestRunDataFile("latency");

%% =========================================================================
% 2) Configure crossing-based latency analysis
% ==========================================================================
% Thresholds reject small or ambiguous responses while preserving bang-bang surface onsets.
analysisConfig = struct( ...
    "minLatencyResponseDeg", 2.0, ...
    "minLatencyCommandStepNorm", 0.20, ...
    "latencyBaselineWindowSeconds", 0.15, ...
    "latencyBaselineGapSeconds", 0.02, ...
    "latencyFinalWindowSeconds", 0.15, ...
    "latencyEndpointAverageMethod", "median");
result = Post_Analysis("latency", rawRunFile, analysisConfig);

%% =========================================================================
% 3) Report generated result file
% ==========================================================================
fprintf("Latency analysis complete.\n");
if isfield(result.analysisInfo, "viconStateFilterEnabled") && result.analysisInfo.viconStateFilterEnabled
    fprintf("Vicon state filter used: one-pole low-pass %.1f Hz.\n", ...
        result.analysisInfo.viconStateFilterCutoffHz);
end
fprintf("Latency result file:\n%s\n", result.outputFiles.latencyMat);
