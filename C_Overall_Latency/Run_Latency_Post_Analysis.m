%RUN_LATENCY_POST_ANALYSIS Analyze the latest latency run using latest calibration.
addpath(fileparts(mfilename("fullpath")));

rawRunFile = latestRunDataFile("latency");
calibrationFile = latestCalibrationFile();
result = Post_Analysis("latency", rawRunFile, calibrationFile);

fprintf("Latency analysis complete.\n");
fprintf("Latency result file:\n%s\n", result.outputFiles.latencyMat);

