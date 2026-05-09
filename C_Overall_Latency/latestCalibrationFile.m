function calibrationFile = latestCalibrationFile()
%LATESTCALIBRATIONFILE Return the newest processed surface_calibration.mat.
root = fullfile("C_Overall_Latency", "data", "processed");
matches = dir(fullfile(root, "**", "surface_calibration.mat"));
if isempty(matches)
    error("latestCalibrationFile:NoCalibration", ...
        "No surface_calibration.mat found. Run Run_Deflection_Post_Analysis.m first.");
end

[~, idx] = max([matches.datenum]);
calibrationFile = string(fullfile(matches(idx).folder, matches(idx).name));
end

