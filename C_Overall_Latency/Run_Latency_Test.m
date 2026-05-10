%RUN_LATENCY_TEST Run Vicon-corrected bang-bang latency data collection.
addpath(fileparts(mfilename("fullpath")));

cfg = defaultOverallLatencyConfig("latency");

cfg.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_latency_bang_bang";

% First hardware check: 30 s gives roughly three full all-surface passes.
% For the full run, set activeCommandSeconds = 90.0.
cfg.activeCommandSeconds = 30.0;
cfg.neutralLeadSeconds = 2.0;
cfg.neutralTailSeconds = 2.0;
cfg.eventHoldSeconds = 0.60;
cfg.latencyBangBangAmplitudeNorm = 0.70;
cfg.latencyRepetitionsPerSurface = 10;

runData = Latency_Test(cfg);

fprintf("Latency run saved to:\n%s\n", runData.outputFiles.runDataMat);
