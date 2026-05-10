%RUN_LATENCY_TEST Run Vicon-corrected bang-bang latency data collection.
addpath(fileparts(mfilename("fullpath")));

cfg = defaultOverallLatencyConfig("latency");

cfg.viconStateFilterEnabled = true;
cfg.viconStateFilterCutoffHz = 20.0;
cfg.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_latency_bang_bang_filter20hz";

cfg.neutralLeadSeconds = 2.0;
cfg.neutralTailSeconds = 2.0;
cfg.eventHoldSeconds = 0.60;
cfg.latencyBangBangAmplitudeNorm = 0.70;
cfg.latencyRepetitionsPerSurface = 10;
eventsPerSurfaceRepetition = 4; % +step, neutral, -step, neutral.
cfg.activeCommandSeconds = cfg.eventHoldSeconds * ...
    numel(cfg.surfaceOrder) * eventsPerSurfaceRepetition * cfg.latencyRepetitionsPerSurface;

runData = Latency_Test(cfg);

fprintf("Latency run saved to:\n%s\n", runData.outputFiles.runDataMat);
