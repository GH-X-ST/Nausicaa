%RUN_LATENCY_TEST Run Vicon-corrected bang-bang latency data collection.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Filter and timing configuration
% 2) Runtime execution through the shared command path
% ==========================================================================
addpath(fileparts(mfilename("fullpath")));

%% =========================================================================
% 1) Filter and timing configuration
% ==========================================================================
cfg = defaultOverallLatencyConfig("latency");

% The 20 Hz one-pole Vicon filter is the measured low-delay feedback candidate from the motion test.
cfg.viconStateFilterEnabled = true;
cfg.viconStateFilterCutoffHz = 20.0;
cfg.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_latency_bang_bang_filter20hz";

cfg.neutralLeadSeconds = 2.0;
cfg.neutralTailSeconds = 2.0;
cfg.eventHoldSeconds = 0.60;
cfg.latencyBangBangAmplitudeNorm = 0.70;
cfg.latencyRepetitionsPerSurface = 10;
% Each repetition uses positive step, neutral return, negative step, neutral return per surface.
eventsPerSurfaceRepetition = 4;
cfg.activeCommandSeconds = cfg.eventHoldSeconds * ...
    numel(cfg.surfaceOrder) * eventsPerSurfaceRepetition * cfg.latencyRepetitionsPerSurface;

%% =========================================================================
% 2) Runtime execution through the shared command path
% ==========================================================================
runData = Latency_Test(cfg);

fprintf("Latency run saved to:\n%s\n", runData.outputFiles.runDataMat);
