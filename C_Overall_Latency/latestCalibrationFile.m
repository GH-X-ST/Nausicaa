function calibrationFile = latestCalibrationFile()
%LATESTCALIBRATIONFILE Locate the newest processed surface_calibration.mat.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Search generated deflection-analysis folders
% ==========================================================================
%% =========================================================================
% 1) Search generated deflection-analysis folders
% ==========================================================================
root = fullfile("C_Overall_Latency", "data", "processed");
% Processed calibration files are offline products; runtime loops never load or write them.
matches = dir(fullfile(root, "**", "surface_calibration.mat"));
if isempty(matches)
    error("latestCalibrationFile:NoCalibration", ...
        "No surface_calibration.mat found. Run Run_Deflection_Post_Analysis.m first.");
end

[~, idx] = max([matches.datenum]);
calibrationFile = string(fullfile(matches(idx).folder, matches(idx).name));
end
