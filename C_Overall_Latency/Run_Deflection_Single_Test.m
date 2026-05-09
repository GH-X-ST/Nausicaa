%RUN_DEFLECTION_SINGLE_TEST Run one deflection scenario before the full batch.
addpath(fileparts(mfilename("fullpath")));

scenario = struct( ...
    "name", "single_fast_coarse", ...
    "holdSeconds", 1.25, ...
    "rampLevels", [0, 0.15, 0.30, 0.50, 0.75, 1.00], ...
    "commandDtSeconds", 0.015, ...
    "neutralLeadSeconds", 5.0, ...
    "neutralTailSeconds", 5.0);

cfg = defaultOverallLatencyConfig("deflection");
cfg.runLabel = string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + "_deflection_" + scenario.name;
cfg.enabledSurfaceIndices = 1:4;
cfg.deflectionHoldSeconds = scenario.holdSeconds;
cfg.deflectionRampLevels = scenario.rampLevels;
cfg.commandDtSeconds = scenario.commandDtSeconds;
cfg.neutralLeadSeconds = scenario.neutralLeadSeconds;
cfg.neutralTailSeconds = scenario.neutralTailSeconds;

fprintf("\nSingle deflection scenario: %s\n", scenario.name);
fprintf("  receiverChannelSurfaceOrder = %s\n", strjoin(string(cfg.receiverChannelSurfaceOrder), ", "));
fprintf("  surfaceEulerAxes = %s\n", strjoin(string(cfg.surfaceEulerAxes), ", "));
fprintf("  holdSeconds = %.2f, commandDtSeconds = %.3f\n", cfg.deflectionHoldSeconds, cfg.commandDtSeconds);
fprintf("  rampLevels = %s\n", mat2str(cfg.deflectionRampLevels));

runData = Latency_Test(cfg);
result = Post_Analysis("deflection", runData.outputFiles.runDataMat);

fprintf("\nSingle deflection run complete.\n");
fprintf("Raw run:\n%s\n", runData.outputFiles.runDataMat);
fprintf("Calibration:\n%s\n", result.outputFiles.calibrationMat);
disp(result.summaryTable);
