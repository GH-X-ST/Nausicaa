%RUN_DEFLECTION_POST_ANALYSIS Analyze the latest deflection run.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Locate latest raw deflection run
% 2) Run offline calibration analysis
% ==========================================================================
addpath(fileparts(mfilename("fullpath")));

%% =========================================================================
% 1) Locate latest raw deflection run
% ==========================================================================
% The newest timestamped raw deflection run is treated as the data source for this wrapper.
rawRunFile = latestRunDataFile("deflection");

%% =========================================================================
% 2) Run offline calibration analysis
% ==========================================================================
result = Post_Analysis("deflection", rawRunFile);

fprintf("Deflection analysis complete.\n");
fprintf("Calibration file:\n%s\n", result.outputFiles.calibrationMat);
