%RUN_DEFLECTION_TEST Run surface deflection calibration data collection.
addpath(fileparts(mfilename("fullpath")));

cfg = defaultOverallLatencyConfig("deflection");

% First hardware check: use one surface and short holds.
% For the full run, set enabledSurfaceIndices = 1:4 and deflectionHoldSeconds = 0.75.
cfg.enabledSurfaceIndices = 1:4;
cfg.deflectionHoldSeconds = 2.0;
cfg.deflectionRampLevels = [0, 0.20, 0.40, 0.60, 0.80, 1.00];
cfg.neutralLeadSeconds = 5.0;
cfg.neutralTailSeconds = 5.0;

runData = Latency_Test(cfg);

fprintf("Deflection run saved to:\n%s\n", runData.outputFiles.runDataMat);
