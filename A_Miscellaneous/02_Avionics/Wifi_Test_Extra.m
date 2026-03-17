function Wifi_test_extra(config)
%WIFI_TEST_EXTRA Drive four Nano 33 IoT servos with a sine-wave command.
%   Wifi_test_extra() connects to the Arduino Nano 33 IoT over WiFi,
%   commands the four configured servos with the same sine-wave input, and
%   returns them to neutral at the end of the test.
arguments
    config.ipAddress (1,1) string = "192.168.0.33"
    config.boardName (1,1) string = "Nano33IoT"
    config.neutralPosition (1,1) double {mustBeFinite, mustBeGreaterThanOrEqual(config.neutralPosition, 0), mustBeLessThanOrEqual(config.neutralPosition, 1)} = 0.5
    config.amplitudePosition (1,1) double {mustBeFinite, mustBeNonnegative} = 60/180
    config.frequencyHz (1,1) double {mustBeFinite, mustBePositive} = 5
    config.sampleTimeSeconds (1,1) double {mustBeFinite, mustBePositive} = 0.02
    config.durationSeconds (1,1) double {mustBeFinite, mustBePositive} = 10.0
    config.pinNames (4,1) string = ["D9"; "D10"; "D11"; "D12"]
end

maximumAmplitude = min(config.neutralPosition, 1 - config.neutralPosition);
if config.amplitudePosition > maximumAmplitude
    error("Wifi_test_extra:AmplitudeOutOfRange", ...
        "amplitudePosition must be less than or equal to %.3f for neutralPosition %.3f.", ...
        maximumAmplitude, ...
        config.neutralPosition);
end

arduinoObject = arduino(char(config.ipAddress), char(config.boardName));
servoObjects = cell(numel(config.pinNames), 1);

for servoIndex = 1:numel(config.pinNames)
    servoObjects{servoIndex} = servo(arduinoObject, char(config.pinNames(servoIndex)));
end

cleanupHandle = onCleanup(@() returnServosToNeutral(servoObjects, config.neutralPosition));
returnServosToNeutral(servoObjects, config.neutralPosition);
pause(1.0);

timeVectorSeconds = 0:config.sampleTimeSeconds:config.durationSeconds;

for timeIndex = 1:numel(timeVectorSeconds)
    commandPosition = config.neutralPosition + ...
        config.amplitudePosition .* sin(2 .* pi .* config.frequencyHz .* timeVectorSeconds(timeIndex));
    commandPosition = min(max(commandPosition, 0.0), 1.0);

    for servoIndex = 1:numel(servoObjects)
        writePosition(servoObjects{servoIndex}, commandPosition);
    end

    pause(config.sampleTimeSeconds);
end

returnServosToNeutral(servoObjects, config.neutralPosition);
pause(1.0);
clear cleanupHandle
end

function returnServosToNeutral(servoObjects, neutralPosition)
for servoIndex = 1:numel(servoObjects)
    try
        writePosition(servoObjects{servoIndex}, neutralPosition);
    catch
    end
end
end
