function cfg = defaultOverallLatencyConfig(mode)
%DEFAULTOVERALLLATENCYCONFIG Shared hardware defaults for overall latency tests.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Hardware communication defaults
% 2) Surface ordering, sign, and Vicon-axis assumptions
% 3) Command timing and analysis defaults
% ==========================================================================
arguments
    mode (1,1) string {mustBeMember(mode, ["deflection", "latency"])}
end

cfg = struct();
cfg.mode = mode;

%% =========================================================================
% 1) Hardware communication defaults
% ==========================================================================
cfg.serialPort = "COM11";
cfg.baudRate = 1000000;

cfg.viconHostName = "192.168.0.100:801";
cfg.viconPort = 801;

%% =========================================================================
% 2) Surface ordering, sign, and Vicon-axis assumptions
% ==========================================================================
cfg.surfaceNames = ["Aileron_L", "Aileron_R", "Rudder", "Elevator"];
cfg.surfaceOrder = cfg.surfaceNames;
cfg.viconRawSubjectNames = cfg.surfaceNames;
cfg.viconSurfaceSubjectNames = cfg.surfaceNames;

% Receiver channels are physical transmitter order; command vectors remain in surfaceOrder.
cfg.receiverChannelSurfaceOrder = ["Aileron_R", "Aileron_L", "Rudder", "Elevator"];

% Euler axes and servo signs are hardware sign-check assumptions saved with runData.
cfg.surfaceEulerAxes = ["X", "X", "Z", "X"];
cfg.servoSigns = [1, -1, 1, -1];

%% =========================================================================
% 3) Command timing and analysis defaults
% ==========================================================================
cfg.commandDtSeconds = 0.02;
cfg.neutralDurationSeconds = 5.0;
cfg.viconStateFilterEnabled = false;
cfg.viconStateFilterCutoffHz = 20.0;
cfg.randomSeed = 2;
cfg.saveCsv = true;
cfg.makePlots = false;

cfg.eventHoldSeconds = 0.60;
cfg.latencyProfileMode = "bang_bang_measured_response_fraction";
cfg.latencyBangBangAmplitudeNorm = 0.70;
cfg.latencyRepetitionsPerSurface = 4;
end

