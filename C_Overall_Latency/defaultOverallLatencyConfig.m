function cfg = defaultOverallLatencyConfig(mode)
%DEFAULTOVERALLLATENCYCONFIG Shared hardware defaults for overall latency tests.
arguments
    mode (1,1) string {mustBeMember(mode, ["deflection", "latency"])}
end

cfg = struct();
cfg.mode = mode;

cfg.serialPort = "COM11";
cfg.baudRate = 1000000;

cfg.viconHostName = "192.168.0.100:801";
cfg.viconPort = 801;

cfg.surfaceNames = ["Aileron_L", "Aileron_R", "Rudder", "Elevator"];
cfg.surfaceOrder = cfg.surfaceNames;
cfg.viconRawSubjectNames = cfg.surfaceNames;
cfg.viconSurfaceSubjectNames = cfg.surfaceNames;

cfg.receiverChannelSurfaceOrder = ["Aileron_R", "Aileron_L", "Rudder", "Elevator"];

cfg.surfaceEulerAxes = ["X", "X", "Z", "X"];
cfg.servoSigns = [1, -1, 1, -1];

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

