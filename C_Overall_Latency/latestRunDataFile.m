function rawRunFile = latestRunDataFile(mode)
%LATESTRUNDATAFILE Return the newest raw run_data.mat for a mode.
arguments
    mode (1,1) string
end

root = fullfile("C_Overall_Latency", "data", "raw");
matches = dir(fullfile(root, "**", "run_data.mat"));
if isempty(matches)
    error("latestRunDataFile:NoRuns", "No run_data.mat files found under %s.", root);
end

keep = false(numel(matches), 1);
for k = 1:numel(matches)
    keep(k) = contains(string(matches(k).folder), "_" + mode);
end
matches = matches(keep);
if isempty(matches)
    error("latestRunDataFile:NoModeRuns", "No %s run_data.mat files found under %s.", mode, root);
end

[~, idx] = max([matches.datenum]);
rawRunFile = string(fullfile(matches(idx).folder, matches(idx).name));
end

