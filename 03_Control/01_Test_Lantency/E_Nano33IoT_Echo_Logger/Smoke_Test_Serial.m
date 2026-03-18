port = "COM5";
baudRate = 115200;
resetSeconds = 4.0;
probeTimeoutSeconds = 6.0;
helloRetrySeconds = 0.25;

serialObject = serialport(char(port), baudRate);
cleanupSerial = onCleanup(@() delete(serialObject));
configureTerminator(serialObject, "LF");

pause(resetSeconds);
flush(serialObject);

fprintf("Opened %s at %d baud.\n", char(port), baudRate);

probeStart = tic;
lastHelloSendSeconds = -inf;
responseLine = "";

while toc(probeStart) < probeTimeoutSeconds
    elapsedSeconds = toc(probeStart);
    if elapsedSeconds - lastHelloSendSeconds >= helloRetrySeconds
        writeline(serialObject, "HELLO");
        lastHelloSendSeconds = elapsedSeconds;
    end

    if serialObject.NumBytesAvailable > 0
        responseLine = string(strtrim(readline(serialObject)));
        fprintf("Received: %s\n", char(responseLine));
        if startsWith(responseLine, "HELLO_EVENT")
            break;
        end
    else
        pause(0.01);
    end
end

if strlength(responseLine) == 0 || ~startsWith(responseLine, "HELLO_EVENT")
    error("Smoke_Test_Serial:NoHelloEvent", ...
        "No HELLO_EVENT was received from %s within %.1f s.", ...
        char(port), ...
        probeTimeoutSeconds);
end

writeline(serialObject, "STATUS");
pause(0.10);
while serialObject.NumBytesAvailable > 0
    fprintf("Received: %s\n", char(string(strtrim(readline(serialObject)))));
end
