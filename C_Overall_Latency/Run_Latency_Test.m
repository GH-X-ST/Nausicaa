%RUN_LATENCY_TEST Run Vicon-corrected physical latency data collection.
addpath(fileparts(mfilename("fullpath")));

cfg = defaultOverallLatencyConfig("latency");
cfg.calibrationFile = latestCalibrationFile();

% First hardware check: keep activeCommandSeconds short.
% For the full run, set activeCommandSeconds = 90.0.
cfg.activeCommandSeconds = 10.0;
cfg.neutralLeadSeconds = 2.0;
cfg.neutralTailSeconds = 2.0;
cfg.eventHoldSeconds = 0.50;

runData = Latency_Test(cfg);

fprintf("Latency run saved to:\n%s\n", runData.outputFiles.runDataMat);

