%RUN_DEFLECTION_POST_ANALYSIS Analyze the latest deflection run.
addpath(fileparts(mfilename("fullpath")));

rawRunFile = latestRunDataFile("deflection");
result = Post_Analysis("deflection", rawRunFile);

fprintf("Deflection analysis complete.\n");
fprintf("Calibration file:\n%s\n", result.outputFiles.calibrationMat);

